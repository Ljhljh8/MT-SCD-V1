from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_, DropPath
except Exception:
    def trunc_normal_(tensor, std=0.02):
        return nn.init.trunc_normal_(tensor, std=std)

    class DropPath(nn.Identity):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

try:
    from timm.models.registry import register_model
except Exception:
    def register_model(fn):
        return fn

from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d, make_lif_node
# from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d, make_lif_node

class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        del bias
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


def _is_dendfadc(module: nn.Module) -> bool:
    return isinstance(module, DendFADCConv2d)


class SepConv(nn.Module):
    """Inverted separable convolution with optional DendFADC depthwise 7x7."""

    def __init__(
        self,
        dim,
        detach_reset,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        use_dendfadc=False,
        branch_num=4,
        fadc_fs_cfg=None,
    ):
        super().__init__()
        del act2_layer
        med_channels = int(expansion_ratio * dim)
        self.lif1 = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = make_lif_node(tau=2.0, detach_reset=detach_reset)

        if use_dendfadc:
            # Safe replacement: input has passed lif2 and is a spike tensor.
            self.dwconv = DendFADCConv2d(
                in_channels=med_channels,
                out_channels=med_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=med_channels,
                bias=bias,
                branch_num=branch_num,
                detach_reset=detach_reset,
                fs_cfg=fadc_fs_cfg,
            )
        else:
            self.dwconv = nn.Conv2d(
                med_channels,
                med_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=med_channels,
                bias=bias,
            )
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        #x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        if _is_dendfadc(self.dwconv):
            x = self.dwconv(x)
            _, _, _, H2, W2 = x.shape
            x = self.bn2(self.pwconv2(x.flatten(0, 1))).reshape(T, B, -1, H2, W2)
        else:
            x = self.dwconv(x.flatten(0, 1))
            _, _, H2, W2 = x.shape
            x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H2, W2)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        detach_reset,
        mlp_ratio=4.0,
        use_dendfadc=False,
        replace_sepconv_dw=True,
        replace_ffn_conv=True,
        branch_num=4,
        fadc_fs_cfg=None,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.Conv = SepConv(
            dim=dim,
            detach_reset=detach_reset,
            use_dendfadc=use_dendfadc and replace_sepconv_dw,
            branch_num=branch_num,
            fadc_fs_cfg=fadc_fs_cfg,
        )

        self.lif1 = make_lif_node(tau=2.0, detach_reset=detach_reset)
        if use_dendfadc and replace_ffn_conv:
            # Safe replacement: input is self.lif1(x), hence spike-based.
            self.conv1 = DendFADCConv2d(
                dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
                branch_num=branch_num,
                detach_reset=detach_reset,
                fs_cfg=fadc_fs_cfg,
            )
        else:
            self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.lif2 = make_lif_node(tau=2.0, detach_reset=detach_reset)
        if use_dendfadc and replace_ffn_conv:
            # Safe replacement: input is self.lif2(x), hence spike-based.
            self.conv2 = DendFADCConv2d(
                hidden_dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
                branch_num=branch_num,
                detach_reset=detach_reset,
                fs_cfg=fadc_fs_cfg,
            )
        else:
            self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def _conv_bn(self, conv, bn, x, out_channels):
        T, B, _, _, _ = x.shape
        if _is_dendfadc(conv):
            x = conv(x)
            _, _, _, H, W = x.shape
            x = bn(x.flatten(0, 1)).reshape(T, B, out_channels, H, W)
        else:
            _, _, _, H, W = x.shape
            x = bn(conv(x.flatten(0, 1))).reshape(T, B, out_channels, H, W)
        return x.contiguous()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self._conv_bn(self.conv1, self.bn1, self.lif1(x), 4 * C)
        x = self._conv_bn(self.conv2, self.bn2, self.lif2(x), C)
        x = x_feat + x
        return x


class MS_MLP(nn.Module):
    """Token/channel MLP. Conv1d is intentionally not replaced by DendFADC."""

    def __init__(
        self,
        in_features,
        detach_reset,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        layer=0,
    ):
        super().__init__()
        del drop, layer
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        return x


class MS_Attention_RepConv(nn.Module):
    """Attention RepConv path is kept unchanged by default."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        detach_reset=True,
    ):
        super().__init__()
        del qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.q_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.k_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.v_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.attn_lif = make_lif_node(tau=2.0, v_threshold=0.5, detach_reset=detach_reset)
        self.proj_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


class MS_Attention_3D_RepConv(nn.Module):
    """3D attention path is kept unchanged by default."""

    def __init__(
        self,
        dim,
        sim_mode,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        detach_reset=True,
    ):
        super().__init__()
        del qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.sim_mode = sim_mode

        self.head_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.q_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.k_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.v_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)

        if self.sim_mode == "dot":
            self.attn_lif = make_lif_node(
                tau=2.0, v_threshold=0.5, v_reset=0.0, detach_reset=detach_reset
            )
        elif self.sim_mode == "hamming":
            self.attn_bn = nn.BatchNorm2d(dim)
            self.attn_lif = make_lif_node(
                tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=detach_reset
            )
        else:
            raise NotImplementedError

        self.proj_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = T * H * W

        x = self.head_lif(x)
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q)
        q = (
            q.permute(1, 0, 3, 4, 2)
            .flatten(1, 3)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = self.k_lif(k)
        k = (
            k.permute(1, 0, 3, 4, 2)
            .flatten(1, 3)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = self.v_lif(v)
        v = (
            v.permute(1, 0, 3, 4, 2)
            .flatten(1, 3)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        if self.sim_mode == "dot":
            x = k.transpose(-2, -1) @ v
            x = (q @ x) * self.scale
        elif self.sim_mode == "hamming":
            x = (2 * k - 1).transpose(-2, -1) @ v
            x = (2 * q - 1) @ x
            x = x / (2 * self.dim)
        else:
            raise NotImplementedError

        x = (
            x.permute(0, 2, 1, 3)
            .reshape(B, N, C)
            .reshape(B, T, H, W, C)
            .permute(1, 0, 4, 2, 3)
            .contiguous()
        )
        if self.sim_mode == "hamming":
            x = self.attn_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        x = self.attn_lif(x)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        detach_reset,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        att_mode="2D",
        replace_attention_repconv=False,
    ):
        super().__init__()
        del norm_layer
        if replace_attention_repconv:
            raise NotImplementedError(
                "replace_attention_repconv is reserved for a second-stage experiment; "
                "default False keeps q/k/v/proj RepConv unchanged."
            )

        if att_mode == "2D":
            self.attn = MS_Attention_RepConv(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                detach_reset=detach_reset,
            )
        elif att_mode == "3D_dot":
            self.attn = MS_Attention_3D_RepConv(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                detach_reset=detach_reset,
                sim_mode="dot",
            )
        elif att_mode == "3D_ham":
            self.attn = MS_Attention_3D_RepConv(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                detach_reset=detach_reset,
                sim_mode="hamming",
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            detach_reset=detach_reset,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        detach_reset,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        use_dendfadc=False,
        replace_first_layer=False,
        branch_num=4,
        fadc_fs_cfg=None,
    ):
        super().__init__()
        self.SN_CLS = False
        if not first_layer:
            self.encode_lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
            self.SN_CLS = True
        should_replace = use_dendfadc and (not first_layer or replace_first_layer)
        if should_replace:
            # Safe for non-first layers because encode_lif has already produced spikes.
            self.encode_conv = DendFADCConv2d(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=1,
                bias=True,
                branch_num=branch_num,
                detach_reset=detach_reset,
                fs_cfg=fadc_fs_cfg,
                SN_CLS=self.SN_CLS,
            )
        else:
            self.encode_conv = nn.Conv2d(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        self.encode_bn = nn.BatchNorm2d(embed_dims)


    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        if _is_dendfadc(self.encode_conv):
            x = self.encode_conv(x)
            _, _, _, H, W = x.shape
            x = self.encode_bn(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        else:
            x = self.encode_conv(x.flatten(0, 1))
            _, _, H, W = x.shape
            x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class Spiking_vit_MetaFormer(nn.Module):
    def __init__(
        self,
        detach_reset,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=(64, 128, 256, 360),
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=8,
        sr_ratios=1,
        kd=False,
        att_mode="2D",
        use_dendfadc: bool = True,
        replace_downsample: bool = True,
        replace_convblock: bool = True,
        replace_sepconv_dw: bool = True,
        replace_attention_repconv: bool = False,
        replace_first_layer: bool = True,
        branch_num: int = 4,
        fadc_fs_cfg: dict = None,
    ):
        super().__init__()
        del img_size_h, img_size_w, patch_size
        if len(embed_dim) != 5:
            raise ValueError(
                "SNN_Models_DendFADC uses the current 5-stage MetaFormer layout; "
                "pass four embed_dim values."
            )
        self.num_classes = num_classes
        self.depths = depths
        self.T = 1

        total_depth = depths if isinstance(depths, int) else sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            detach_reset=detach_reset,
            use_dendfadc=use_dendfadc and replace_downsample,
            replace_first_layer=replace_first_layer,
            branch_num=branch_num,
            fadc_fs_cfg=fadc_fs_cfg,
        )
        self.ConvBlock1_1 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[0] // 2,
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                    use_dendfadc=use_dendfadc and replace_convblock,
                    replace_sepconv_dw=replace_sepconv_dw,
                    replace_ffn_conv=True,
                    branch_num=branch_num,
                    fadc_fs_cfg=fadc_fs_cfg,
                )
            ]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            detach_reset=detach_reset,
            use_dendfadc=use_dendfadc and replace_downsample,
            replace_first_layer=replace_first_layer,
            branch_num=branch_num,
            fadc_fs_cfg=fadc_fs_cfg,
        )
        self.ConvBlock1_2 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[0],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                    use_dendfadc=use_dendfadc and replace_convblock,
                    replace_sepconv_dw=replace_sepconv_dw,
                    replace_ffn_conv=True,
                    branch_num=branch_num,
                    fadc_fs_cfg=fadc_fs_cfg,
                )
            ]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            detach_reset=detach_reset,
            use_dendfadc=use_dendfadc and replace_downsample,
            replace_first_layer=replace_first_layer,
            branch_num=branch_num,
            fadc_fs_cfg=fadc_fs_cfg,
        )
        self.ConvBlock2_1 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[1],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                    use_dendfadc=use_dendfadc and replace_convblock,
                    replace_sepconv_dw=replace_sepconv_dw,
                    replace_ffn_conv=True,
                    branch_num=branch_num,
                    fadc_fs_cfg=fadc_fs_cfg,
                )
            ]
        )
        self.ConvBlock2_2 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[1],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                    use_dendfadc=use_dendfadc and replace_convblock,
                    replace_sepconv_dw=replace_sepconv_dw,
                    replace_ffn_conv=True,
                    branch_num=branch_num,
                    fadc_fs_cfg=fadc_fs_cfg,
                )
            ]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            detach_reset=detach_reset,
            use_dendfadc=use_dendfadc and replace_downsample,
            replace_first_layer=replace_first_layer,
            branch_num=branch_num,
            fadc_fs_cfg=fadc_fs_cfg,
        )
        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    detach_reset=detach_reset,
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    att_mode=att_mode,
                    replace_attention_repconv=replace_attention_repconv,
                )
                for j in range(1)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            detach_reset=detach_reset,
            use_dendfadc=use_dendfadc and replace_downsample,
            replace_first_layer=replace_first_layer,
            branch_num=branch_num,
            fadc_fs_cfg=fadc_fs_cfg,
        )
        self.block4 = nn.ModuleList(
            [
                MS_Block(
                    detach_reset=detach_reset,
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    att_mode=att_mode,
                    replace_attention_repconv=replace_attention_repconv,
                )
                for j in range(1)
            ]
        )
        self.downsample5 = MS_DownSampling(
            in_channels=embed_dim[3],
            embed_dims=embed_dim[4],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            detach_reset=detach_reset,
        )

        self.block5 = nn.ModuleList(
            [
                MS_Block(
                    detach_reset=detach_reset,
                    dim=embed_dim[4],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    att_mode=att_mode,
                )
                for j in range(1)
            ]
        )
        self.lif = make_lif_node(tau=2.0, detach_reset=detach_reset)
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        self.kd = kd
        if self.kd:
            self.head_kd = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        fs = []

        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        fs.append(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        fs.append(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)
        fs.append(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        fs.append(x)

        x = self.downsample5(x)
        for blk in self.block5:
            x = blk(x)
        fs.append(x)

        return fs

    def forward(self, x):
        # x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        features = self.forward_features(x)
        # x = features[-1].flatten(3).mean(3)
        # x_lif = self.lif(x)
        # x = self.head(x_lif).mean(0)
        # if self.kd:
        #     x_kd = self.head_kd(x_lif).mean(0)
        #     if self.training:
        #         return x, x_kd
        #     return (x + x_kd) / 2
        return features


@register_model
def metaspikformer_8_256(**kwargs):
    model = Spiking_vit_MetaFormer(
        img_size_h=512,
        img_size_w=512,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=13,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def metaspikformer_8_384(**kwargs):
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[96, 192, 384, 480],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def metaspikformer_8_512(**kwargs):
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def metaspikformer_8_768(**kwargs):
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[192, 384, 768, 960],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode, MTSCDPRDNIIFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant, Quant4
try:
    from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d
except Exception:
    modulated_deform_conv2d = None

def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def build_base_offset(kernel_size: int, device=None, dtype=None) -> torch.Tensor:
    """Build DCNv2 base offset order: [y0, x0, y1, x1, ...]."""
    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an int for build_base_offset")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    coords = []
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            coords.extend([y, x])
    return torch.tensor(coords, device=device, dtype=dtype)


def _build_base_offset_2d(
    kernel_size: Tuple[int, int],
    device=None,
    dtype=None,
) -> torch.Tensor:
    kh, kw = kernel_size
    if kh <= 0 or kw <= 0 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("DendFADCConv2d currently requires positive odd kernel sizes")
    if kh == kw:
        return build_base_offset(kh, device=device, dtype=dtype)
    rh, rw = kh // 2, kw // 2
    coords = []
    for y in range(-rh, rh + 1):
        for x in range(-rw, rw + 1):
            coords.extend([y, x])
    return torch.tensor(coords, device=device, dtype=dtype)


def _generate_laplacian_pyramid(
    input_tensor: torch.Tensor,
    num_levels: int,
    size_align: bool = True,
    mode: str = "bilinear",
) -> List[torch.Tensor]:
    pyramid = []
    current_tensor = input_tensor
    _, _, H, W = current_tensor.shape
    align_corners = (H % 2) == 1
    for _ in range(num_levels):
        _, _, h, w = current_tensor.shape
        downsampled_tensor = F.interpolate(
            current_tensor,
            (h // 2 + h % 2, w // 2 + w % 2),
            mode=mode,
            align_corners=align_corners,
        )
        if size_align:
            upsampled_tensor = F.interpolate(
                downsampled_tensor, (H, W), mode=mode, align_corners=align_corners
            )
            laplacian = (
                F.interpolate(current_tensor, (H, W), mode=mode, align_corners=align_corners)
                - upsampled_tensor
            )
        else:
            upsampled_tensor = F.interpolate(
                downsampled_tensor, (h, w), mode=mode, align_corners=align_corners
            )
            laplacian = current_tensor - upsampled_tensor
        pyramid.append(laplacian)
        current_tensor = downsampled_tensor
    if size_align:
        current_tensor = F.interpolate(
            current_tensor, (H, W), mode=mode, align_corners=align_corners
        )
    pyramid.append(current_tensor)
    return pyramid


class FrequencySelection(nn.Module):
    """FreqSelect-style dendritic frequency decomposition driven by optional K maps."""

    def __init__(
        self,
        in_channels: int,
        k_list: Sequence[int] = (2, 4),
        lowfreq_att: bool = False,
        fs_feat: str = "feat",
        lp_type: str = "freq",
        act: str = "sigmoid",
        spatial: str = "conv",
        spatial_group: int = 1,
        spatial_kernel: int = 3,
        init: str = "zero",
        global_selection: bool = False,
    ):
        super().__init__()
        del fs_feat, spatial, spatial_kernel, init
        self.in_channels = in_channels
        self.k_list = list(k_list)
        self.lowfreq_att = lowfreq_att
        self.lp_type = lp_type
        self.act = act
        self.global_selection = global_selection
        if spatial_group > 64:
            spatial_group = in_channels
        if in_channels % spatial_group != 0:
            spatial_group = 1
        self.spatial_group = spatial_group

        self.lp_list = nn.ModuleList()
        if self.lp_type == "avgpool":
            for k in self.k_list:
                left = k // 2
                right = k - 1 - left
                self.lp_list.append(
                    nn.Sequential(
                        nn.ReplicationPad2d((left, right, left, right)),
                        nn.AvgPool2d(kernel_size=k, padding=0, stride=1),
                    )
                )
        elif self.lp_type in ("freq", "laplacian"):
            pass
        else:
            raise NotImplementedError(f"Unsupported lp_type: {self.lp_type}")

    def _k_to_list(
        self,
        K,
        b: int,
        h: int,
        w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[List[torch.Tensor]]:
        if K is None:
            return None
        if isinstance(K, torch.Tensor):
            if K.dim() == 5:
                maps = [K[i] for i in range(K.shape[0])]
            elif K.dim() == 4:
                expected = len(self.k_list) + (1 if self.lowfreq_att else 0)
                if K.shape[1] == expected * self.spatial_group:
                    maps = list(torch.split(K, self.spatial_group, dim=1))
                else:
                    maps = [K]
            else:
                raise ValueError("K tensor must have shape [N,TB,G,H,W] or [TB,N*G,H,W]")
        elif isinstance(K, (list, tuple)):
            maps = list(K)
        else:
            raise TypeError("K must be None, Tensor, list, or tuple")

        out = []
        for item in maps:
            item = item.to(device=device, dtype=dtype)
            if item.dim() == 3:
                item = item.unsqueeze(1)
            if item.shape[-2:] != (h, w):
                item = F.interpolate(item, size=(h, w), mode="bilinear", align_corners=False)
            if item.shape[0] != b:
                if item.shape[0] == 1:
                    item = item.expand(b, -1, -1, -1)
                else:
                    raise ValueError(f"K batch dimension {item.shape[0]} does not match {b}")
            if item.shape[1] == 1 and self.spatial_group != 1:
                item = item.expand(-1, self.spatial_group, -1, -1)
            if item.shape[1] != self.spatial_group:
                raise ValueError(
                    f"K channel dimension must be 1 or spatial_group={self.spatial_group}, "
                    f"got {item.shape[1]}"
                )
            out.append(item)
        return out

    def _apply_k(self, part: torch.Tensor, maps: Optional[List[torch.Tensor]], idx: int) -> torch.Tensor:
        if maps is None or idx >= len(maps):
            return part
        b, _, h, w = part.shape
        weight = maps[idx]
        part_grouped = part.reshape(b, self.spatial_group, -1, h, w)
        return (weight.reshape(b, self.spatial_group, -1, h, w) * part_grouped).reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor, K=None, att_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        del att_feat
        b, _, h, w = x.shape
        original_dtype = x.dtype
        k_maps = self._k_to_list(K, b, h, w, x.dtype, x.device)
        x_list = []

        if self.lp_type == "avgpool":
            pre_x = x
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(pre_x)
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(self._apply_k(high_part, k_maps, idx))
            if self.lowfreq_att:
                x_list.append(self._apply_k(pre_x, k_maps, len(self.k_list)))
            else:
                x_list.append(pre_x)

        elif self.lp_type == "laplacian":
            pyramids = _generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
            for idx in range(len(self.k_list)):
                x_list.append(self._apply_k(pyramids[idx], k_maps, idx))
            if self.lowfreq_att:
                x_list.append(self._apply_k(pyramids[-1], k_maps, len(self.k_list)))
            else:
                x_list.append(pyramids[-1])

        elif self.lp_type == "freq":
            if x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
                if k_maps is not None:
                    k_maps = [item.float() for item in k_maps]

            pre_x = x.clone()
            x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device, dtype=x.dtype)
                h0 = round(h / 2 - h / (2 * freq))
                h1 = round(h / 2 + h / (2 * freq))
                w0 = round(w / 2 - w / (2 * freq))
                w1 = round(w / 2 + w / (2 * freq))
                mask[:, :, h0:h1, w0:w1] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm="ortho").real
                high_part = pre_x - low_part
                pre_x = low_part
                x_list.append(self._apply_k(high_part, k_maps, idx))
            if self.lowfreq_att:
                x_list.append(self._apply_k(pre_x, k_maps, len(self.k_list)))
            else:
                x_list.append(pre_x)
        else:
            raise NotImplementedError(f"Unsupported lp_type: {self.lp_type}")

        out = sum(x_list)
        return out.to(dtype=original_dtype) if out.dtype != original_dtype else out


class OmniAttention(nn.Module):
    """Attention generator for AdaKern low/high kernel modulation."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 1,
        min_channel: int = 16,
    ):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        att = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return torch.sigmoid(att / self.temperature)

    def get_kernel_attention(self, x):
        att = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(att / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class DendSNLayer(nn.Module):
    """Single, cleaned dendritic frequency selection + soma spike layer."""

    def __init__(
        self,
        in_channels: int,
        branch_num: int = 4,
        detach_reset: bool = True,
        v_th: float = 1.0,
        fs_cfg: Optional[dict] = None,
    ):
        super().__init__()
        cfg = dict(
            k_list=[2, 4],
            lowfreq_att=False,
            lp_type="freq",
            act="sigmoid",
            spatial="conv",
            spatial_group=1,
        )
        if fs_cfg is not None:
            cfg.update(fs_cfg)
        self.frequency_selection = FrequencySelection(in_channels, **cfg)

        self.branch_num = branch_num

    def forward(self, x: torch.Tensor, K=None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("DendSNLayer expects [T, B, C, H, W]")
        T, B, C, H, W = x.shape
        y = self.frequency_selection(x.flatten(0, 1).contiguous(), K)
        y = y.reshape(T, B, C, H, W).contiguous()
        return self.lif(y)


class DendFADCConv2d(nn.Module):
    """Tree dendrite-soma-synapse FADC convolution.

    The module accepts [T, B, C, H, W], applies FreqSelect and LIF to produce
    binary spikes, then uses the spikes to predict AdaDR offset, mask, K_next,
    and AdaKern dynamic weights before a DCNv2-style convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        branch_num: int = 4,
        detach_reset: bool = True,
        deform_groups: int = 1,
        padding_mode: str = "repeat",
        kernel_decompose: Optional[str] = "both",
        pre_fs: bool = True,
        fs_cfg: Optional[dict] = None,
        use_dct: bool = False,
        use_zero_dilation: bool = False,
        calculate_next_k: bool = True,
        v_th: float = 1.0,
        reduction: float = 1.0 / 16.0,
        SN_CLS : bool = False,
        Down_K : bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if deform_groups <= 0:
            raise ValueError("deform_groups must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.branch_num = branch_num
        self.padding_mode = padding_mode
        self.kernel_decompose = kernel_decompose
        self.pre_fs = pre_fs
        self.use_dct = use_dct
        self.use_zero_dilation = use_zero_dilation
        self.calculate_next_k = calculate_next_k
        self.SN_CLS = SN_CLS
        self.Down_K = Down_K

        kh, kw = self.kernel_size
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("DendFADCConv2d requires odd kernel sizes")

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        if padding_mode == "zero":
            self.PAD = nn.ZeroPad2d(self.padding)
            offset_padding = (0, 0)
            deform_padding = (0, 0)
        elif padding_mode == "repeat":
            self.PAD = nn.ReplicationPad2d(self.padding[0])
            offset_padding = (0, 0)
            deform_padding = (0, 0)
        elif padding_mode in ("identity", "none", None):
            self.PAD = nn.Identity()
            offset_padding = self.padding
            deform_padding = self.padding
        else:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")
        self._deform_padding = deform_padding

        cfg = dict(
            k_list=[2, 4],
            lowfreq_att=False,
            lp_type="freq",
            act="sigmoid",
            spatial="conv",
            spatial_group=1,
        )
        if fs_cfg is not None:
            cfg.update(fs_cfg)
        self.fs_cfg = cfg
        self.dendrite = FrequencySelection(in_channels, **cfg) if pre_fs else None
        if SN_CLS:
            # self.lif = make_lif_node(
            #     tau=2.0,
            #     v_threshold=v_th,
            #     v_reset=0.0,
            #     detach_reset=detach_reset,
            # )
            # self.lif = MTSCDPRDNIIFNode()
            self.lif = Q_IFNode(surrogate_function=Quant())

        if kh > 1 or kw > 1:
            self.conv_offset = nn.Conv2d(
                in_channels,
                deform_groups,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=offset_padding,
                dilation=1,
                bias=True,
            )
            self.conv_mask = nn.Conv2d(
                in_channels,
                deform_groups * kh * kw,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=offset_padding,
                dilation=1,
                bias=True,
            )
            base_offset = _build_base_offset_2d(self.kernel_size)
            self.register_buffer("dilated_offset", base_offset.view(1, 1, -1, 1, 1))
        else:
            self.conv_offset = None
            self.conv_mask = None
            self.register_buffer("dilated_offset", torch.zeros(1, 1, 2, 1, 1))

        spatial_group = cfg.get("spatial_group", 1)
        if spatial_group > 64:
            spatial_group = in_channels
        if in_channels % spatial_group != 0:
            spatial_group = 1
        self.spatial_group = spatial_group
        self.k_map_count = len(cfg.get("k_list", [])) + (1 if cfg.get("lowfreq_att", False) else 0)
        if calculate_next_k and self.k_map_count > 0:
            if self.Down_K == True:
                self.freq_weight_conv = nn.Conv2d(
                    in_channels,
                    self.k_map_count * self.spatial_group,
                    kernel_size=3,
                    stride=2,     # stride=self.stride,
                    padding=1,
                    groups=self.spatial_group,
                    bias=False,
                )
            else:
                self.freq_weight_conv = nn.Conv2d(
                    in_channels,
                    self.k_map_count * self.spatial_group,
                    kernel_size=3,
                    stride=1,  # stride=self.stride,
                    padding=1,
                    groups=self.spatial_group,
                    bias=False,
                )
        else:
            self.freq_weight_conv = None

        if kernel_decompose == "both":
            self.OMNI_ATT1 = OmniAttention(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
                reduction=reduction,
                kernel_num=1,
                min_channel=16,
            )
            self.OMNI_ATT2 = OmniAttention(
                in_channels,
                out_channels,
                kernel_size=kh if use_dct else 1,
                groups=groups,
                reduction=reduction,
                kernel_num=1,
                min_channel=16,
            )
        elif kernel_decompose in ("high", "low"):
            self.OMNI_ATT = OmniAttention(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
                reduction=reduction,
                kernel_num=1,
                min_channel=16,
            )
        elif kernel_decompose in (None, "none"):
            pass
        else:
            raise ValueError(f"Unsupported kernel_decompose: {kernel_decompose}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.conv_offset is not None:
            nn.init.constant_(self.conv_offset.weight, 0)
            init_value = (self.dilation[0] - 1) / max(float(self.dilation[0]), 1.0) + 1e-4
            nn.init.constant_(self.conv_offset.bias, init_value)
        if self.conv_mask is not None:
            nn.init.constant_(self.conv_mask.weight, 0)
            nn.init.constant_(self.conv_mask.bias, 0)
        if self.freq_weight_conv is not None:
            nn.init.constant_(self.freq_weight_conv.weight, 0)

    def _sp_act(self, value: torch.Tensor) -> torch.Tensor:
        act = self.fs_cfg.get("act", "sigmoid")
        if act == "sigmoid":
            return value.sigmoid() * 2.0
        if act == "softmax":
            return value.softmax(dim=1) * value.shape[1]
        raise NotImplementedError(f"Unsupported K activation: {act}")

    def _calculate_k_next(self, x_spike: torch.Tensor):
        if self.freq_weight_conv is None:
            return None
        out = self.freq_weight_conv(x_spike)
        maps = torch.split(out, self.spatial_group, dim=1)
        return [self._sp_act(item) for item in maps]

    def _reshape_group_attention(
        self,
        c_att,
        f_att,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        if not torch.is_tensor(c_att):
            c_scale = 1.0
        else:
            c_scale = c_att.to(device=device, dtype=dtype).reshape(
                batch, self.groups, in_per_group, 1, 1
            ).unsqueeze(2)
        if not torch.is_tensor(f_att):
            f_scale = 1.0
        else:
            f_scale = f_att.to(device=device, dtype=dtype).reshape(
                batch, self.groups, out_per_group, 1, 1
            ).unsqueeze(3)
        return c_scale, f_scale

    def _adaptive_weight(self, x_spike: torch.Tensor) -> torch.Tensor:
        b = x_spike.shape[0]
        kh, kw = self.kernel_size
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        weight = self.weight.reshape(1, self.groups, out_per_group, in_per_group, kh, kw)
        weight = weight.expand(b, -1, -1, -1, -1, -1)
        weight_mean = weight.mean(dim=(-1, -2), keepdim=True)
        weight_res = weight - weight_mean

        if hasattr(self, "OMNI_ATT1") and hasattr(self, "OMNI_ATT2"):
            c_att1, f_att1, _, _ = self.OMNI_ATT1(x_spike)
            c_att2, f_att2, spatial_att2, _ = self.OMNI_ATT2(x_spike)
            c_scale1, f_scale1 = self._reshape_group_attention(
                c_att1, f_att1, b, x_spike.dtype, x_spike.device
            )
            c_scale2, f_scale2 = self._reshape_group_attention(
                c_att2, f_att2, b, x_spike.dtype, x_spike.device
            )
            if self.use_dct:
                try:
                    import torch_dct as dct
                except Exception as exc:
                    raise ImportError("torch_dct is required when use_dct=True") from exc
                res_flat = weight_res.reshape(-1, in_per_group, kh, kw)
                dct_coeff = dct.dct_2d(res_flat)
                if torch.is_tensor(spatial_att2):
                    spatial = spatial_att2.reshape(b, 1, 1, 1, kh, kw)
                    spatial = spatial.expand(-1, self.groups, out_per_group, in_per_group, -1, -1)
                    dct_coeff = dct_coeff.reshape_as(weight_res) * (spatial * 2.0)
                    dct_coeff = dct_coeff.reshape(-1, in_per_group, kh, kw)
                weight_res = dct.idct_2d(dct_coeff).reshape_as(weight_res)
            adaptive = weight_mean * (c_scale1 * 2.0) * (f_scale1 * 2.0) + weight_res * (
                c_scale2 * 2.0
            ) * (f_scale2 * 2.0)
        elif hasattr(self, "OMNI_ATT"):
            c_att, f_att, _, _ = self.OMNI_ATT(x_spike)
            c_scale, f_scale = self._reshape_group_attention(
                c_att, f_att, b, x_spike.dtype, x_spike.device
            )
            if self.kernel_decompose == "high":
                adaptive = weight_mean + weight_res * (c_scale * 2.0) * (f_scale * 2.0)
            elif self.kernel_decompose == "low":
                adaptive = weight_mean * (c_scale * 2.0) * (f_scale * 2.0) + weight_res
            else:
                adaptive = weight
        else:
            adaptive = weight

        return adaptive.reshape(b * self.out_channels, in_per_group, kh, kw).contiguous()

    def _fallback_group_conv(
        self,
        x_grouped: torch.Tensor,
        adaptive_weight: torch.Tensor,
        bias,
        batch: int,
    ) -> torch.Tensor:
        return F.conv2d(
            x_grouped,
            adaptive_weight,
            bias=bias,
            stride=self.stride,
            padding=self._deform_padding,
            dilation=(1, 1),
            groups=self.groups * batch,
        )

    def forward(self, x: torch.Tensor, K=None, return_k: bool = True):
        if x.dim() != 5:
            raise ValueError("DendFADCConv2d expects input shape [T, B, C, H, W]")
        T, B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        tb = T * B
        x_flat = x.flatten(0, 1).contiguous()

        if self.dendrite is not None:
            x_dend = self.dendrite(x_flat, K)
        else:
            x_dend = x_flat
        x_dend = x_dend.reshape(T, B, C, H, W).contiguous()
        if hasattr(self, "lif"):
            x_dend = self.lif(x_dend)
        x_spike = x_dend.flatten(0, 1).contiguous()

        K_next = self._calculate_k_next(x_spike)
        adaptive_weight = self._adaptive_weight(x_spike)
        bias = self.bias.repeat(tb) if self.bias is not None else None

        if self.conv_offset is None:
            x_grouped = x_spike.reshape(1, tb * C, H, W).contiguous()
            y = self._fallback_group_conv(x_grouped, adaptive_weight, bias, tb)
            _, _, h_out, w_out = y.shape
            y = y.reshape(T, B, self.out_channels, h_out, w_out).contiguous()
            return (y, K_next) if return_k else y

        offset_source = self.PAD(x_spike)
        offset_factor = self.conv_offset(offset_source)
        if self.use_zero_dilation:
            offset_factor = (F.relu(offset_factor + 1.0, inplace=False) - 1.0) * self.dilation[0]
        else:
            offset_factor = offset_factor.abs() * self.dilation[0]

        _, _, h_out, w_out = offset_factor.shape
        base = self.dilated_offset.to(device=x.device, dtype=x.dtype)
        offset = offset_factor.reshape(tb, self.deform_groups, -1, h_out, w_out) * base
        offset = offset.reshape(1, tb * self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1], h_out, w_out)

        x_pad = self.PAD(x_spike)
        mask = self.conv_mask(x_pad).sigmoid()
        mask = mask.reshape(1, tb * self.deform_groups * self.kernel_size[0] * self.kernel_size[1], h_out, w_out)

        x_grouped = x_pad.reshape(1, tb * C, x_pad.shape[-2], x_pad.shape[-1]).contiguous()
        if modulated_deform_conv2d is not None:
            y = modulated_deform_conv2d(
                x_grouped,
                offset.contiguous(),
                mask.contiguous(),
                adaptive_weight,
                bias,
                self.stride,
                self._deform_padding,
                (1, 1),
                self.groups * tb,
                self.deform_groups * tb,
            )
        else:
            y = self._fallback_group_conv(x_grouped, adaptive_weight, bias, tb)

        y = y.reshape(T, B, self.out_channels, y.shape[-2], y.shape[-1]).contiguous()
        return (y, K_next) if return_k else y


class DendFADCConvBNActWrapper(nn.Module):
    """Optional wrapper for call sites that want conv + BN in one module."""

    def __init__(self, conv: DendFADCConv2d, bn: Optional[nn.Module] = None):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor, K=None, return_k: bool = False):
        if return_k:
            y, k_next = self.conv(x, K=K, return_k=True)
        else:
            y = self.conv(x, K=K, return_k=False)
            k_next = None
        if self.bn is not None:
            T, B, C, H, W = y.shape
            y = self.bn(y.flatten(0, 1)).reshape(T, B, C, H, W).contiguous()
        return (y, k_next) if return_k else y

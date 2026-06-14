
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack
import torch
import torch.nn as nn
import math
from contextlib import nullcontext
amp_off = torch.cuda.amp.autocast(enabled=False) if torch.is_autocast_enabled() else nullcontext()
from typing import List, Tuple, Dict
class Synapse_Weight_Adjustment(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Synapse_Weight_Adjustment, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention  # 指定通道注意力计算函数

        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, groups=1,bias=True)
        self.func_filter = self.get_filter_attention

        self._initialize_weights()  # 权重初始化
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        # 更新softmax温度参数，用于调整注意力分布的平滑程度
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0
    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention
    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x)

class Frequency_Adjustment(nn.Module):
    def __init__(self, in_channels, k_list=[2], lowfreq_att=True, fs_feat='feat',  lp_type='freq',
                 spatial_group=1,  global_selection=False, lam_range=(1.5, 2.0),):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.global_selection = global_selection
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # lambda 取值范围
        self.lam_min, self.lam_max = lam_range

        if spatial_group > 64: spatial_group = in_channels
        self.spatial_group = spatial_group  # 空间分组数

        self._mask_cache: Dict[Tuple[str, int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @torch.no_grad()
    def _build_center_square_masks(self, H: int, W: int, device, dtype) -> torch.Tensor:
        key1 = ('LP', H, W, device, dtype)
        key2 = ('bandN', H, W, device, dtype)
        if key1 in self._mask_cache:
            return self._mask_cache[key1],self._mask_cache[key2]

        masks = torch.zeros((len(self.k_list), 1, 1, H, W), device=device, dtype=dtype)
        cy, cx = H // 2, W // 2
        for i, freq in enumerate(self.k_list):
            hy = int(round(H / (2 * freq)))
            hx = int(round(W / (2 * freq)))
            y0, y1 = max(cy - hy, 0), min(cy + hy, H)
            x0, x1 = max(cx - hx, 0), min(cx + hx, W)
            masks[i, 0, 0, y0:y1, x0:x1] = 1.0
        M = len(self.k_list)
        bands = torch.zeros((M + 1, 1, 1, H, W), device=device, dtype=torch.float)
        lp = masks.float()
        bands[0] = 1.0 - lp[0]
        for i in range(1, M):
            bands[i] = lp[i - 1] - lp[i]  # 只会产生 {0,1}，不需要 clamp
        bands[M] = lp[-1]
        self._mask_cache[key1] = masks
        self._mask_cache[key2] = bands
        return masks, bands

    def _apply_group_weight(self, x_band: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # K = torch.cat(K, dim=1)
        N, T, B, C, H, W = x_band.shape
        G = self.spatial_group
        assert C % G == 0, "C must be divisible by spatial_group"
        # Knbhw = K.permute(1, 0, 2, 3).unsqueeze(1).unsqueeze(3) # (N,1,B,1,H,W)
        # Knbhw = Knbhw.expand(N, T, B, 1, H, W)
        Knbhw = K.permute(1, 0, 2, 3)
        Knbhw = Knbhw.reshape(-1, T, B, H, W).unsqueeze(3)  # (N, T, B, 1, H, W)
        if G == 1:
            out = (x_band * Knbhw).sum(dim=0)  # sum over N -> (T,B,C,H,W)
            return out
        xg = x_band.view(N, T, B, G, C // G, H, W)
        Kg = Knbhw.expand(N, T, B, G, 1, H, W)
        out = (xg * Kg).sum(dim=0).view(T, B, C, H, W)
        return out

        self.lp_list = nn.ModuleList()
        if self.lp_type == 'avgpool':
            pass
        elif self.lp_type == 'laplacian':
            pass
        elif self.lp_type == 'freq':
            pass
        else:
            raise NotImplementedError

    def forward(self, x, K, att_feat=None):
        T,B,C,H,W = x.shape
        device, dtype = x.device, x.dtype
        x = x.flatten(0, 1)  # t*b, c, h, w
        if self.lp_type == 'avgpool':
            pass
        elif self.lp_type == 'laplacian':
            pass
        elif self.lp_type == 'freq':
            pass
        elif self.lp_type == 'freq_fast':
            dtype = x.dtype
            x = x.float()
            x_ori = x # (TB,C,H,W)
            with amp_off:
                x_fft = torch.fft.fft2(x_ori.to(torch.float32), norm='ortho')
                x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))  # 频谱居中
                masks, bands = self._build_center_square_masks(H, W, device, x_fft.dtype)
                # 广播 -> (N,TB,C,H,W)
                low_fft = x_fft.unsqueeze(0) * masks  # (N,TB,C,H,W)

            low_fft = torch.fft.ifftshift(low_fft, dim=(-2, -1))
            low_spatial = torch.fft.ifft2(low_fft, norm='ortho').real  # (N,TB,C,H,W)

            x_spatial = x_ori.unsqueeze(0)                               # (1,TB,C,H,W)
            E = torch.cat([x_spatial, low_spatial], dim=0)            # (N+1,TB,C,H,W)
            high_bands = E[:-1] - E[1:]                               # (N,TB,C,H,W)

            if K is None:
                out = high_bands.sum(dim=0).view(T, B, C, H, W)
            else:
                out = self._apply_group_weight(high_bands.view(len(self.k_list), T, B, C, H, W), K)
            # 低频残差（可选）叠加
            if self.lowfreq_att:
                if K is None:
                    out = out + low_spatial[-1].view(T, B, C, H, W)
                else:
                    K_low = K[:, -1, :, :].unsqueeze(0).unsqueeze(2)  # (1,B,1,H,W)
                    K_low = K_low.expand(T, B, 1, H, W)
                    out = out + low_spatial[-1].view(T, B, C, H, W) * K_low.unsqueeze(2)
            else:
                out = out + low_spatial[-1].view(T, B, C, H, W)
            out = out.type(dtype)
            return out
        elif self.lp_type == 'Dwt_fast':
            pass
class Dend_soma(ModulatedDeformConv2d):
    _version = 2
    def __init__(self, *args,
                 offset_freq=None,
                 padding_mode='repeat',
                 kernel_decompose='both',
                 conv_type='conv',
                 pre_fs=True,
                 epsilon=1e-4,
                 use_zero_dilation=False,
                 K_pool=False,
                 Calcuate_K=True,
                 AdaKern = True,
                 T_Pool = False,
                 reduction = 1/16,
                 v_th=1.0,
                 k_act='sigmoid',
                 fs_cfg={
                     'k_list': [2, 4],
                     'fs_feat': 'feat',
                     'lowfreq_att': False,
                     'lp_type': 'freq_fast',  # 将特征图转成高频/低频过程使用的方法，
                     'spatial_group': 1,
                 },
                 **kwargs):
        super().__init__(*args, **kwargs)  # 初始化父类 ModulatedDeformConv2d (mmcv)
        if padding_mode == 'zero':
            self.PAD = nn.ZeroPad2d(self.kernel_size[0] // 2)  # 零填充，pad大小=核的一半
        elif padding_mode == 'repeat':
            self.PAD = nn.ReplicationPad2d(self.kernel_size[0] // 2)  # 边界复制填充
        else:
            self.PAD = nn.Identity()  # 无填充

        self.lif = MultiStepLIFNode(tau=1.5, v_threshold=v_th, detach_reset=True, backend='cupy')
        # self.lif = MultiStepLIFNode_AMP(v_threshold=v_th, detach_reset=True, backend='torch')
        self.fs_cfg = fs_cfg
        self.Calcuate_K = Calcuate_K
        self.AdaKern = AdaKern
        self.k_act = k_act
        self.T_Pool = T_Pool
        self.K_pool = K_pool
        if fs_cfg is not None:
            self.FS = Frequency_Adjustment(self.in_channels, **fs_cfg)
        self.spatial_group = 1  # 空间分组数,既K调制矩阵的通道维度

        if Calcuate_K:
            if self.spatial_group > 64: spatial_group = self.in_channels
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(self.fs_cfg['k_list'])
            if self.fs_cfg['lowfreq_att']:  _n += 1  # 如果对低频也使用注意力，则总卷积个数=频段数+1
            if self.K_pool==True and self.T_Pool == False:
                # self.freq_weight_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=_n * self.spatial_group,
                #                    kernel_size=3, stride=2, groups=self.spatial_group, padding=3 // 2, bias=False)
                self.freq_weight_conv = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2,padding=3 // 2,
                              groups=self.in_channels,dilation=1,bias=False),
                nn.Conv2d(self.in_channels, _n * self.spatial_group, kernel_size=1, stride=1, padding=0,
                          groups=1, dilation=1, bias=False)
            )
                self.freq_weight_conv[1].weight.data.zero_()
            elif self.K_pool==False and self.T_Pool == False:
                # self.freq_weight_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=_n * self.spatial_group,
                #                    kernel_size=3, stride=1, groups=self.spatial_group, padding=3 // 2, bias=False)
                self.freq_weight_conv = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=3 // 2,
                              groups=self.in_channels, dilation=1, bias=False),
                    nn.Conv2d(self.in_channels, _n * self.spatial_group, kernel_size=1, stride=1, padding=0,
                              groups=1, dilation=1, bias=False)
                )
                self.freq_weight_conv[1].weight.data.zero_()
            elif self.K_pool==True and self.T_Pool == True:
                self.freq_weight_conv = nn.Sequential(
                    nn.Conv3d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=3 // 2,
                              groups=self.in_channels, dilation=1, bias=False),
                    nn.Conv3d(self.in_channels, _n * self.spatial_group, kernel_size=1, stride=1, padding=0,
                              groups=1, dilation=1, bias=False)
                )
                self.freq_weight_conv[1].weight.data.zero_()
            else:
                raise NotImplementedError
        self.kernel_decompose = kernel_decompose  # 卷积核分解模式
        if self.AdaKern:
            # 根据kernel_decompose创建对应的Synapse_Weight_Adjustment实例
            if kernel_decompose == 'both':
                # 两套注意力：一个针对低频核(恒用1x1核)，一个针对高频核(若use_dct则核大小=k，否则1x1)
                self.OMNI_ATT1 = Synapse_Weight_Adjustment(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                               groups=1, reduction=reduction, kernel_num=1, min_channel=16)
                self.OMNI_ATT2 = Synapse_Weight_Adjustment(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                               groups=1, reduction=reduction, kernel_num=1, min_channel=16)
            elif kernel_decompose == 'high':
                # 仅调整高频部分
                self.OMNI_ATT = Synapse_Weight_Adjustment(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                              groups=1, reduction=reduction, kernel_num=1, min_channel=16)
            elif kernel_decompose == 'low':
                # 仅调整低频部分
                self.OMNI_ATT = Synapse_Weight_Adjustment(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                              groups=1, reduction=reduction, kernel_num=1, min_channel=16)
        self.conv_type = conv_type

        if conv_type == 'conv' and self.kernel_size[0] > 1:
            self.conv_offset = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size[0] // 2  if isinstance(self.PAD, nn.Identity) else 0,
                    groups=self.in_channels,
                    dilation=1,
                    bias=False),
                nn.Conv2d(
                    self.in_channels,
                    self.deform_groups,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    dilation=1,
                    bias=True)
            )
        if self.kernel_size[0] > 1:
            self.conv_mask = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding if isinstance(self.PAD, nn.Identity) else 0,
                    groups=self.in_channels,
                    dilation=1,
                    bias=False),
                nn.Conv2d(
                    self.in_channels,
                    self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    dilation=1,
                    bias=True)
            )
            self.offset_freq = offset_freq
            assert self.offset_freq is None
            offset = [-1, -1, -1, 0, -1, 1,
                      0, -1, 0, 0, 0, 1,
                      1, -1, 1, 0, 1, 1]
            offset = torch.Tensor(offset)
            self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None]))  # B, G, 18, 1, 1


        self.pre_fs = pre_fs
        self.epsilon = epsilon
        self.use_zero_dilation = use_zero_dilation
        self.init_weights()

    def sp_act(self, freq_weight):
        # 空间权重激活函数：根据配置将卷积输出的权重归一化
        if self.k_act == 'sigmoid':
            # sigmoid输出0~1，乘2得到0~2范围
            freq_weight = freq_weight.sigmoid() * 2
        elif self.k_act== 'softmax':
            # softmax输出每组权重和为1，再乘以通道数freq_weight.shape[1]，确保权重平均值为1
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight
    def Calcuate_K_ForNextSn(self, x):
        out = self.freq_weight_conv(x)  # 输出 shape: (B, total_out*spatial_group, H, W)
        freq_weight = torch.split(out, self.spatial_group, dim=1)  # 拆分得到每段 (B, spatial_group, H, W)
        freq_weight = [self.sp_act(p) * 2 for p in freq_weight]
        freq_weight = torch.cat(freq_weight,dim=1)
        return freq_weight

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            if self.conv_type == 'conv':
                self.conv_offset[1].weight.data.zero_()
                self.conv_offset[1].bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + self.epsilon)
        if hasattr(self, 'conv_mask'):
            self.conv_mask[1].weight.data.zero_()
            self.conv_mask[1].bias.data.zero_()
        if hasattr(self, 'conv_mask_mean_level'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()
    def forward(self, x, K):
        T, N, C, H, W = x.shape
        # Step1: 频率选择
        x = self.FS(x, K)  # 在offset预测之前先对输入特征应用频率选择
        x = self.lif(x.reshape(T, N, C, H, W)).flatten(0, 1)  # t*b, c, h, w
        # x_soma = self.lif(x,tau_feature)
        # x = x_soma.mean(dim=0)  # b, c, h, w
        if self.Calcuate_K:
            if self.T_Pool:
                x = x.reshape(T, N, C, H, W).permute(1, 2, 0, 3, 4)  #  N, C, T, H, W
                K = self.Calcuate_K_ForNextSn(x)
                K = K.permute(2, 0, 1, 3, 4).flatten(0, 1)  #  T, N, C, H, W -> T*N, C, H, W
            else:
                K = self.Calcuate_K_ForNextSn(x)

        else:
            K = None
        # Step2: 计算AdaKern所需注意力系数
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            c_att1, f_att1 = self.OMNI_ATT1(x)  # 低频核的通道和滤波器注意力
            c_att2, f_att2 = self.OMNI_ATT2(x)  # 高频核的通道、滤波器、空间注意力
        elif hasattr(self, 'OMNI_ATT'):
            # 若只使用单一OmniAttention（'high'或'low'模式）
            c_att, f_att = self.OMNI_ATT(x)  # 通道和滤波器注意力
        if self.kernel_size[0] > 1:
            # Step3: 预测偏移 offset
            if self.conv_type == 'conv':
                offset = self.conv_offset(self.PAD(x))
            elif self.conv_type == 'multifreqband':
                offset = self.conv_offset(x)
            if self.use_zero_dilation:  # 偏移后处理：将offset值限制为非负，并结合use_zero_dilation策略
                offset = (F.relu(offset + 1, inplace=True) - 1) * self.dilation[0]  # ensure > 0
            else:
                offset = offset.abs() * self.dilation[0]

            # Step4: 将 offset 因子映射为实际偏移量
            tb, _, h, w = offset.shape
            offset = offset.reshape(tb, self.deform_groups, -1, h, w) * self.dilated_offset

            # Step5: 计算 modulation 掩模
            x = self.PAD(x)
            mask = self.conv_mask(x)  # 预测mask权重 (TB, k*k, H_out, W_out)
            mask = mask.sigmoid()  # Sigmoid使mask范围在(0,1)

            # Step6: 应用AdaKern注意力调整卷积核权重
            if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
                # 将offset, mask, x在批维度展开，与卷积weight批次一一对应
                offset = offset.reshape(1, tb * (2 * self.kernel_size[0] * self.kernel_size[0]), h, w)# (1, TB*2k^2, H, W)
                mask = mask.reshape(1, tb * (self.kernel_size[0] * self.kernel_size[0]), h, w)# (1, TB*k^2, H, W)
                # 把 (T,B,C,...) 堆成 (N=1, C=TB*C, H, W)
                x_soma = x.reshape(1, -1, x.size(-2), x.size(-1))
                # x_soma = x_soma if isinstance(self.PAD, nn.Identity) else self.PAD(x_soma)

                # 准备卷积核权重：复制权重到大小 (B, C_out, C_in, k, k)
                adaptive_weight = self.weight.unsqueeze(0).repeat(tb, 1, 1, 1, 1)  # tb, c_out, c_in, k, k
                # 计算卷积核全局平均 (低频部分) 和残差 (高频部分)
                adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)  # (tb, C_out, C_in, 1, 1)
                _, c_out, c_in, k, k = adaptive_weight.shape
                if c_out != c_in:
                    adaptive_weight_res = adaptive_weight - adaptive_weight_mean  # (B, C_out, C_in, K, K)
                    adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (
                            f_att1.unsqueeze(2) * 2) + adaptive_weight_res * (c_att2.unsqueeze(1) * 2) * (
                                              f_att2.unsqueeze(2) * 2)
                else:
                    adaptive_weight = adaptive_weight_mean * (2 * c_att1.unsqueeze(2)) + (
                            adaptive_weight - adaptive_weight_mean) * (2 * c_att2.unsqueeze(2))


                adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, k, k).contiguous()

                # 处理偏置：如果有偏置，则也复制batch次；无偏置则保持None
                if self.bias is not None:
                    bias = self.bias.repeat(tb)
                else:
                    bias = self.bias

            return x_soma, K, offset, mask, adaptive_weight, bias, self.PAD



# class AdaptiveDilatedConv(ModulatedDeformConv2d):
#     _version = 2
#     def __init__(self, *args,
#                  offset_freq=None,  # deprecated （已弃用）控制offset的频率分量
#                  padding_mode='repeat',  # 填充模式，可选 'zero' 或 'repeat'（默认使用边界复制填充）
#                  kernel_decompose='both',  # 卷积核分解模式，可选 'both' (高低频都调整), 'high', 'low', 或 None
#                  conv_type='conv',  # 偏移卷积类型，'conv'表示标准卷积预测offset，'multifreqband'表示在freq分支下使用
#                  sp_att=False,  # 是否使用额外的空间attention与掩模结合
#                  pre_fs=True,  # False, use dilation 若True则在offset预测前应用FreqSelect，否则在之后基于offset特征图应用
#                  epsilon=1e-4,  # 在初始化offset偏置时用的一个小值，防止初始为0
#                  use_zero_dilation=False,  # 若True则采用(ReLU(offset+1)-1)*D的方式保证offset非负；False则直接取abs(offset)*D
#                  use_dct=False,  # 是否在AdaKern高频调整中使用DCT变换
#                  qkv_att = False,
#                  K_pool=False,
#                  Calcuate_K=True,
#                  AdaKern = True,
#                  reduction = 1/16,
#                  v_th=1.0,
#                  fs_cfg={  # FrequencySelection子模块的配置字典
#                      'k_list': [2, 4],  # FS模块频率划分尺度，例如3个频段（实际会得到4段含最低频）
#                      'fs_feat': 'feat',  # FS模块注意力生成所用的特征类型：'feat'表示直接用输入特征
#                      'lowfreq_att': False,  # FS模块低频部分是否也是用注意力
#                      'lp_type': 'laplacian',  # 将特征图转成高频/低频过程使用的方法，
#                      # 'lp_type':'laplacian',
#
#                      'act': 'sigmoid',  # 空间权重激活函数：根据配置将卷积输出的权重归一化
#                      'spatial_group': 1,  # 空间卷积的groups参数，默认为1(降低参数量)（若=输入通道数则相当于深度卷积）,
#                      # 类似于分组卷积，并不是指频段数量，指的是在生成某一个频段对应的空间权重时的卷积
#                  },
#                  **kwargs):
#         super().__init__(*args, **kwargs)  # 初始化父类 ModulatedDeformConv2d (mmcv)
#         if padding_mode == 'zero':
#             self.PAD = nn.ZeroPad2d(self.kernel_size[0] // 2)  # 零填充，pad大小=核的一半
#         elif padding_mode == 'repeat':
#             self.PAD = nn.ReplicationPad2d(self.kernel_size[0] // 2)  # 边界复制填充
#         else:
#             self.PAD = nn.Identity()  # 无填充
#
#         self.lif = MultiStepLIFNode(tau=1.5, v_threshold=v_th, detach_reset=True, backend='cupy')
#         self.fs_cfg = fs_cfg
#         self.Calcuate_K = Calcuate_K
#         self.AdaKern = AdaKern
#         if fs_cfg is not None:
#             if pre_fs:  # pre_fs=True：在卷积偏移预测前对特征做频率选择
#                 self.FS = FrequencySelection(self.in_channels, **fs_cfg)
#             else:  # pre_fs=False：在offset预测后（基于offset特征图）应用频率选择，这里先构造FS，输入通道为1
#                 self.FS = FrequencySelection(1, **fs_cfg)  # use dilation # 准备对偏移频图使用
#         self.spatial_group = 1  # 空间分组数,既K调制矩阵的通道维度
#
#         if Calcuate_K:
#             # 若空间分组数过大则设为输入通道数，确保每组至少1通道
#             if self.spatial_group > 64: spatial_group = self.in_channels
#             # 根据spatial参数选择 生成空间频率权重的方式，这里仅实现了'conv'方式
#             self.freq_weight_conv_list = nn.ModuleList()
#             _n = len(self.fs_cfg['k_list'])
#             if self.fs_cfg['lowfreq_att']:  _n += 1  # 如果对低频也使用注意力，则总卷积个数=频段数+1
#             # # 优化思路: 融合为单卷积
#             if K_pool :
#                 self.freq_weight_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=_n * self.spatial_group,
#                                    kernel_size=3, stride=2, groups=self.spatial_group, padding=3 // 2, bias=False)
#                 self.freq_weight_conv.weight.data.zero_()
#             else:
#                 self.freq_weight_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=_n * self.spatial_group,
#                                    kernel_size=3, stride=1, groups=self.spatial_group, padding=3 // 2, bias=False)
#                 self.freq_weight_conv.weight.data.zero_()
#
#         self.kernel_decompose = kernel_decompose  # 卷积核分解模式
#         if self.AdaKern:
#             # 根据kernel_decompose创建对应的OmniAttention实例
#             if kernel_decompose == 'both':
#                 # 两套注意力：一个针对低频核(恒用1x1核)，一个针对高频核(若use_dct则核大小=k，否则1x1)
#                 self.OMNI_ATT1 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
#                                                groups=self.groups, reduction=reduction, kernel_num=1, min_channel=16)
#                 self.OMNI_ATT2 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels,
#                                                kernel_size=1, groups=self.groups,
#                                                reduction=reduction, kernel_num=1, min_channel=16)
#             elif kernel_decompose == 'high':
#                 # 仅调整高频部分
#                 self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
#                                               groups=1, reduction=reduction, kernel_num=1, min_channel=16)
#             elif kernel_decompose == 'low':
#                 # 仅调整低频部分
#                 self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
#                                               groups=1, reduction=reduction, kernel_num=1, min_channel=16)
#         self.conv_type = conv_type
#
#         if conv_type == 'conv' and self.kernel_size[0] > 1:
#             # conv类型：使用一个标准卷积来预测 offset 因子，每个位置输出 deform_groups*1 个偏移因子
#             # deform_groups指的是用于控制 offset 和 modulation mask 的分组数量，它不会影响卷积核本身的分组结构
#             # deform_groups=2，kernel_size=3x3 那么需要预测的偏移是：2 * 9 * deform_groups = 18 * 2 = 36
#             # 调制 mask 的数量是：1 * 9 * deform_groups = 9 * 2 = 18
#             self.conv_offset = nn.Conv2d(
#                 self.in_channels,
#                 self.deform_groups * 1,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
#                 dilation=1,
#                 bias=True)
#         # conv_type若为其他（如'multifreqband'）可在后续forward里特殊处理，这里略过
#         # 掩模预测卷积：输出 deform_groups * 1 * k * k 个值，即每个采样点一个mask
#         if self.kernel_size[0] > 1:
#             self.conv_mask = nn.Conv2d(
#                 self.in_channels,
#                 self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
#                 dilation=1,
#                 bias=True)
#
#             self.offset_freq = offset_freq
#             assert self.offset_freq is None  # 当前实现未使用 offset_freq 功能
#             # An offset is like [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
#             offset = [-1, -1, -1, 0, -1, 1,
#                       0, -1, 0, 0, 0, 1,
#                       1, -1, 1, 0, 1, 1]
#             offset = torch.Tensor(offset)
#             # 将 offset 注册为buffer (常数，不参与训练)，reshape为(1,1,18,1,1) 便于与预测偏移因子相乘
#             self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None]))  # B, G, 18, 1, 1
#
#
#         self.pre_fs = pre_fs
#         self.epsilon = epsilon
#         self.use_zero_dilation = use_zero_dilation
#         self.init_weights()
#
#     def sp_act(self, freq_weight):
#         # 空间权重激活函数：根据配置将卷积输出的权重归一化
#         if self.fs_cfg['act'] == 'sigmoid':
#             # sigmoid输出0~1，乘2得到0~2范围
#             freq_weight = freq_weight.sigmoid() * 2
#         elif self.fs_cfg['act'] == 'softmax':
#             # softmax输出每组权重和为1，再乘以通道数freq_weight.shape[1]，确保权重平均值为1
#             freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
#         else:
#             raise NotImplementedError
#         return freq_weight
#     def Calcuate_K_ForNextSn(self, x):
#         out = self.freq_weight_conv(x)  # 输出 shape: (B, total_out*spatial_group, H, W)
#         freq_weight = torch.split(out, self.spatial_group, dim=1)  # 拆分得到每段 (B, spatial_group, H, W)
#         freq_weight = [self.sp_act(p) * 2 for p in freq_weight]
#         return freq_weight
#
#     def init_weights(self):
#         super().init_weights()  # 调用父类初始化，父类会将 self.weight 正常初始化
#         if hasattr(self, 'conv_offset'):
#             if self.conv_type == 'conv':
#                 # 偏移卷积权重初始化为0，偏置初始化为 (dilation-1)/dilation + epsilon
#                 self.conv_offset.weight.data.zero_()
#                 self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + self.epsilon)
#         if hasattr(self, 'conv_mask'):
#             # 掩模卷积权重和偏置初始化为0，使初始mask输出为0，经sigmoid后约为0.5
#             self.conv_mask.weight.data.zero_()
#             self.conv_mask.bias.data.zero_()
#         if hasattr(self, 'conv_mask_mean_level'):
#             # 额外mask注意力卷积也初始化为0
#             self.conv_mask.weight.data.zero_()
#             self.conv_mask.bias.data.zero_()
#     def forward(self, x, K):
#         T, N, C, H, W = x.shape
#
#         # Step1: 频率选择（若设置pre_fs=True）
#         x = self.FS(x, K)  # 在offset预测之前先对输入特征应用频率选择
#         # x = self.lif(x.reshape(T, N, C, H, W)).flatten(0,1)  # t*b, c, h, w
#         x_soma = self.lif(x)
#         x = x_soma.mean(dim=0)  # b, c, h, w
#         if self.Calcuate_K:
#             K = self.Calcuate_K_ForNextSn(x)
#         else:
#             K = None
#         # Step2: 计算AdaKern所需注意力系数
#         if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
#             c_att1, f_att1 = self.OMNI_ATT1(x)  # 低频核的通道和滤波器注意力
#             c_att2, f_att2 = self.OMNI_ATT2(x)  # 高频核的通道、滤波器、空间注意力
#         elif hasattr(self, 'OMNI_ATT'):
#             # 若只使用单一OmniAttention（'high'或'low'模式）
#             c_att, f_att = self.OMNI_ATT(x)  # 通道和滤波器注意力
#         if self.kernel_size[0] > 1:
#             # Step3: 预测偏移 offset
#             if self.conv_type == 'conv':
#                 offset = self.conv_offset(self.PAD(x))
#             elif self.conv_type == 'multifreqband':
#                 offset = self.conv_offset(x)
#             if self.use_zero_dilation:  # 偏移后处理：将offset值限制为非负，并结合use_zero_dilation策略
#                 offset = (F.relu(offset + 1, inplace=True) - 1) * self.dilation[0]  # ensure > 0
#             else:
#                 offset = offset.abs() * self.dilation[0]
#
#             # Step4: 将 offset 因子映射为实际偏移量
#             b, _, h, w = offset.shape
#             offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
#
#             # Step5: 计算 modulation 掩模
#             x = self.PAD(x)  # 按照padding_mode对输入填充
#             mask = self.conv_mask(x)  # 预测mask权重 (B, k*k, H_out, W_out)
#             mask = mask.sigmoid()  # Sigmoid使mask范围在(0,1)
#
#             # Step6: 应用AdaKern注意力调整卷积核权重
#             if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
#                 # 有两套OmniAttention的情况 ('both')
#                 # 将offset, mask, x在批维度展开，与卷积weight批次一一对应
#                 offset = offset.reshape(1, b * (2 * self.kernel_size[0] * self.kernel_size[0]), h, w).expand(T, -1, -1, -1)# (T, B*2k^2, H, W)
#                 mask = mask.reshape(1, b * (self.kernel_size[0] * self.kernel_size[0]), h, w).expand(T, -1, -1, -1)# (T, B*k^2, H, W)
#                 # 把 (T,B,C,...) 堆成 (N=T, C=B*C, H, W)
#                 x_soma = x_soma.reshape(T, -1, x_soma.size(-2), x_soma.size(-1))
#                 x_soma = x_soma if isinstance(self.PAD, nn.Identity) else self.PAD(x_soma)
#
#                 # 准备卷积核权重：复制权重到大小 (B, C_out, C_in, k, k)
#                 adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # b, c_out, c_in, k, k
#                 # 计算卷积核全局平均 (低频部分) 和残差 (高频部分)
#                 adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)  # (B, C_out, C_in, 1, 1)
#                 adaptive_weight_res = adaptive_weight - adaptive_weight_mean  # (B, C_out, C_in, K, K)
#
#                 _, c_out, c_in, k, k = adaptive_weight.shape
#                 adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (
#                             f_att1.unsqueeze(2) * 2) + adaptive_weight_res * (c_att2.unsqueeze(1) * 2) * (
#                                               f_att2.unsqueeze(2) * 2)
#                 adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, k, k).contiguous()
#
#                 # 处理偏置：如果有偏置，则也复制batch次；无偏置则保持None
#                 if self.bias is not None:
#                     bias = self.bias.repeat(b)
#                 else:
#                     bias = self.bias
#
#             return x_soma, K, offset, mask, adaptive_weight, bias, self.PAD
#
#         else:
#             if hasattr(self, 'OMNI_ATT') :
#                 # 准备卷积核权重：复制权重到大小 (B, C_out, C_in, k, k)
#
#                 x_soma = x_soma.reshape(T, -1, x_soma.size(-2), x_soma.size(-1)).contiguous()
#                 x_soma = x_soma if isinstance(self.PAD, nn.Identity) else self.PAD(x_soma)
#
#                 adaptive_weight = self.weight.unsqueeze(0).repeat(N, 1, 1, 1, 1)  # b, c_out, c_in, k, k
#                 adaptive_weight = adaptive_weight * (c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2)
#                 _, c_out, c_in, k, k = adaptive_weight.shape
#                 adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, k, k)
#                 # 处理偏置：如果有偏置，则也复制batch次；无偏置则保持None
#                 if self.bias is not None:
#                     bias = self.bias.repeat(N)
#                 else:
#                     bias = self.bias
#             return x_soma, K, None, None, adaptive_weight, bias, self.PAD






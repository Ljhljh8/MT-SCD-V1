"""
Task-calibrated dendritic FADC convolution for MT-SCD.

This file is a conservative drop-in candidate for
`models/dendsn_lifFADC_Snn_v2.py` in the MT-SCD repository.

Design boundary:
    - It keeps the original public convolution interface used by
      `DendriticScaleAdapter`: input [T, B, C, H, W], output [T, B, C_out, H, W]
      plus optional K_next.
    - It does not merge PDCA, decoder guidance, or losses into the dendritic
      neuron. PDCA remains a downstream consumer of K/features.
    - The new semantic-transition calibration path is disabled by default.
      When disabled, the behavior is intentionally close to the existing file:
      FrequencySelection -> optional soma node -> K_next/offset/mask/adaptive conv.

Notes on gradients and numerical safety:
    - No CUDA extension is added. mmcv's modulated_deform_conv2d is used only if
      it is already available; otherwise a grouped conv fallback is used.
    - No new hard integer activation is introduced. `round` is used only to build
      fixed FFT mask indices from static H/W/frequency values; no STE is needed
      because these indices are not learnable.
    - `clamp_min`/`max` are used only for scalar numeric guards such as a softmax
      temperature lower bound. No straight-through estimator is used there.
    - The optional task-calibrated gate uses continuous softmax/sigmoid-style
      operations and standard PyTorch autograd.
    - Stateful soma nodes are not silently reset inside forward by default.
      Use `reset_state()` explicitly, or set `reset_before_forward=True` only when
      the caller intentionally wants per-forward resetting.
"""

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmseg.Qtrick_architecture.clock_driven.neuron import MTSCDPRDNIIFNode
except Exception:  # pragma: no cover - only used for standalone shape sanity tests.
    class _IdentitySomaNode(nn.Identity):
        """Fallback soma node used only when the repository-specific mmseg node is unavailable."""

        pass

    MTSCDPRDNIIFNode = _IdentitySomaNode

try:
    from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d
except Exception:  # pragma: no cover - expected in minimal CPU environments.
    modulated_deform_conv2d = None


TensorOrKList = Union[torch.Tensor, Sequence[torch.Tensor]]


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (int(value), int(value))


def _build_base_offset_2d(kernel_size: Tuple[int, int], device=None, dtype=None) -> torch.Tensor:
    kh, kw = kernel_size
    if kh <= 0 or kw <= 0 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("DendFADCConv2d requires positive odd kernel sizes")
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
        down = F.interpolate(
            current_tensor,
            (h // 2 + h % 2, w // 2 + w % 2),
            mode=mode,
            align_corners=align_corners,
        )
        if size_align:
            up = F.interpolate(down, (H, W), mode=mode, align_corners=align_corners)
            lap = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=align_corners) - up
        else:
            up = F.interpolate(down, (h, w), mode=mode, align_corners=align_corners)
            lap = current_tensor - up
        pyramid.append(lap)
        current_tensor = down
    if size_align:
        current_tensor = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=align_corners)
    pyramid.append(current_tensor)
    return pyramid


def _shift_phase_nearest(x: torch.Tensor, direction: int) -> torch.Tensor:
    """Shift [T,B,C,H,W] along T using nearest endpoint padding."""
    if x.shape[0] == 1:
        return x
    if direction < 0:
        return torch.cat([x[:1], x[:-1]], dim=0)
    if direction > 0:
        return torch.cat([x[1:], x[-1:]], dim=0)
    return x


class FrequencySelection(nn.Module):
    """Phase-wise frequency decomposition with optional K modulation.

    Input:
        x: Tensor with shape [T*B, C, H, W].
        K: None, a Tensor, or a list/tuple of tensors. Each K map is aligned to
           [T*B, spatial_group, H, W] and used to modulate one frequency branch.

    Output:
        Tensor with shape [T*B, C, H, W]. The output channel/spatial shape is
        identical to `x`.
    """

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
        self.in_channels = int(in_channels)
        self.k_list = list(k_list)
        self.lowfreq_att = bool(lowfreq_att)
        self.lp_type = str(lp_type)
        self.act = str(act)
        self.global_selection = bool(global_selection)

        spatial_group = int(spatial_group)
        if spatial_group > 64:
            spatial_group = self.in_channels
        if spatial_group <= 0 or self.in_channels % spatial_group != 0:
            spatial_group = 1
        self.spatial_group = spatial_group

        self.lp_list = nn.ModuleList()
        if self.lp_type == "avgpool":
            for k in self.k_list:
                left = int(k) // 2
                right = int(k) - 1 - left
                self.lp_list.append(
                    nn.Sequential(
                        nn.ReplicationPad2d((left, right, left, right)),
                        nn.AvgPool2d(kernel_size=int(k), padding=0, stride=1),
                    )
                )
        elif self.lp_type in ("freq", "laplacian"):
            pass
        else:
            raise NotImplementedError(f"Unsupported lp_type: {self.lp_type}")

    def _k_to_list(
        self,
        K: Optional[TensorOrKList],
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
            if item.dim() != 4:
                raise ValueError(f"Each K item must be 3D or 4D, got shape {tuple(item.shape)}")
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
        weight = weight.reshape(b, self.spatial_group, -1, h, w)
        return (weight * part_grouped).reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor, K: Optional[TensorOrKList] = None, att_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        del att_feat
        if x.dim() != 4:
            raise ValueError(f"FrequencySelection expects [T*B,C,H,W], got {tuple(x.shape)}")
        b, _, h, w = x.shape
        original_dtype = x.dtype
        k_maps = self._k_to_list(K, b, h, w, x.dtype, x.device)
        parts: List[torch.Tensor] = []

        if self.lp_type == "avgpool":
            pre_x = x
            for idx, avg in enumerate(self.lp_list):
                low = avg(pre_x)
                high = pre_x - low
                pre_x = low
                parts.append(self._apply_k(high, k_maps, idx))
            parts.append(self._apply_k(pre_x, k_maps, len(self.k_list)) if self.lowfreq_att else pre_x)

        elif self.lp_type == "laplacian":
            pyramids = _generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
            for idx in range(len(self.k_list)):
                parts.append(self._apply_k(pyramids[idx], k_maps, idx))
            parts.append(self._apply_k(pyramids[-1], k_maps, len(self.k_list)) if self.lowfreq_att else pyramids[-1])

        elif self.lp_type == "freq":
            # FFT is evaluated in fp32 under AMP to reduce non-finite risk.
            # The `round` operations below create fixed integer frequency-window
            # boundaries from static H/W/frequency values; they are not learnable,
            # so no STE or surrogate gradient is involved.
            if x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
                if k_maps is not None:
                    k_maps = [item.float() for item in k_maps]
            pre_x = x.clone()
            x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
            for idx, freq in enumerate(self.k_list):
                freq = int(freq)
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device, dtype=x.dtype)
                h0 = round(h / 2 - h / (2 * freq))
                h1 = round(h / 2 + h / (2 * freq))
                w0 = round(w / 2 - w / (2 * freq))
                w1 = round(w / 2 + w / (2 * freq))
                mask[:, :, h0:h1, w0:w1] = 1.0
                low = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm="ortho").real
                high = pre_x - low
                pre_x = low
                parts.append(self._apply_k(high, k_maps, idx))
            parts.append(self._apply_k(pre_x, k_maps, len(self.k_list)) if self.lowfreq_att else pre_x)
        else:
            raise NotImplementedError(f"Unsupported lp_type: {self.lp_type}")

        out = sum(parts)
        return out.to(dtype=original_dtype) if out.dtype != original_dtype else out


class OmniAttention(nn.Module):
    """Lightweight attention generator for adaptive kernel decomposition.

    Input:
        x: Tensor with shape [T*B, C_in, H, W].

    Output:
        A tuple `(channel_att, filter_att, spatial_att, kernel_att)`. Each item
        is either a Tensor broadcastable to grouped dynamic kernels or scalar
        `1.0` when the corresponding attention is skipped.
    """

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
        attention_channel = max(int(in_planes * reduction), int(min_channel))
        self.kernel_size = int(kernel_size)
        self.kernel_num = int(kernel_num)
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

        if self.kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, self.kernel_size * self.kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if self.kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, self.kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self) -> None:
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

    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        att = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return torch.sigmoid(att / self.temperature)

    def get_kernel_attention(self, x: torch.Tensor) -> torch.Tensor:
        att = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(att / self.temperature, dim=1)

    def forward(self, x: torch.Tensor):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class SemanticTransitionCalibrator(nn.Module):
    """Optional MTSCD-specific dendritic evidence calibration.

    This module stays inside the dendritic neuron. It does not read PDCA outputs
    and does not produce a final change map.

    Input:
        x: Tensor with shape [T, B, C, H, W], where T can be 3, 6, or any
           positive number of remote-sensing phases.

    Output:
        y: Tensor with shape [T, B, C, H, W].
        gates: dict containing continuous internal evidence maps flattened as
               [T*B, 1, H, W]: `stable`, `transition`, and `noise`.

    Mechanism:
        - stable evidence pulls each phase toward a detached/attached temporal
          consensus feature.
        - transition evidence emphasizes temporal high-pass residuals.
        - noise evidence suppresses local high-frequency residuals that often
          correlate with misregistration, shadow, seasonal texture, or sensor
          artifacts. This is an internal variable, not a supervised change map.

    Detach behavior:
        `detach_context=True` stops gradients from the context phase features
        while preserving gradients through the current phase. This is exposed to
        avoid hidden cross-phase coupling under DDP/AMP.
    """

    def __init__(
        self,
        in_channels: int,
        gate_kernel_size: int = 3,
        detach_context: bool = True,
        use_noise_suppression: bool = True,
        residual_init: float = 0.0,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        if gate_kernel_size <= 0 or gate_kernel_size % 2 == 0:
            raise ValueError("gate_kernel_size must be a positive odd integer")
        self.in_channels = int(in_channels)
        self.detach_context = bool(detach_context)
        self.use_noise_suppression = bool(use_noise_suppression)
        # Scalar guard only; this is not an activation clamp and does not need STE.
        self.gate_temperature = max(float(gate_temperature), 1e-4)
        self.gate = nn.Conv2d(4, 3, gate_kernel_size, padding=gate_kernel_size // 2, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gate.weight, 0.0)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if x.dim() != 5:
            raise ValueError(f"SemanticTransitionCalibrator expects [T,B,C,H,W], got {tuple(x.shape)}")
        T, B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={C}")

        original_dtype = x.dtype
        xf = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        prev = _shift_phase_nearest(xf, direction=-1)
        nxt = _shift_phase_nearest(xf, direction=1)
        temporal_mean = xf.mean(dim=0, keepdim=True).expand_as(xf)
        if self.detach_context:
            prev = prev.detach()
            nxt = nxt.detach()
            temporal_mean = temporal_mean.detach()

        local_mean = F.avg_pool2d(xf.flatten(0, 1), kernel_size=3, stride=1, padding=1)
        local_mean = local_mean.reshape(T, B, C, H, W)

        stable_branch = temporal_mean - xf
        transition_branch = xf - 0.5 * (prev + nxt)    # [t1 - (t1+t2)/2, t2 - (t1+t3)/2, t3 - (t2+t3)
        noise_branch = xf - local_mean

        delta_prev = (xf - prev).abs().mean(dim=2, keepdim=True)  # t1-t1, t2-t1, t3-t2
        delta_next = (xf - nxt).abs().mean(dim=2, keepdim=True)   # t1-t2, t2-t3, t3-t3
        center_energy = xf.abs().mean(dim=2, keepdim=True)
        noise_energy = noise_branch.abs().mean(dim=2, keepdim=True)
        stats = torch.cat([delta_prev, delta_next, center_energy, noise_energy], dim=2)
        stats = stats.flatten(0, 1).contiguous()
        stats = torch.nan_to_num(stats, nan=0.0, posinf=1e4, neginf=-1e4)

        logits = self.gate(stats.float()) / self.gate_temperature
        weights = F.softmax(logits, dim=1).to(dtype=xf.dtype)
        stable_w, transition_w, noise_w = torch.split(weights, 1, dim=1)
        stable_w_5d = stable_w.reshape(T, B, 1, H, W)
        transition_w_5d = transition_w.reshape(T, B, 1, H, W)
        noise_w_5d = noise_w.reshape(T, B, 1, H, W)

        correction = stable_w_5d * stable_branch + transition_w_5d * transition_branch
        if self.use_noise_suppression:
            correction = correction - noise_w_5d * noise_branch

        y = xf + self.res_scale.to(dtype=xf.dtype) * correction
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        gates = {
            "stable": stable_w.detach() if self.detach_context else stable_w,
            "transition": transition_w.detach() if self.detach_context else transition_w,
            "noise": noise_w.detach() if self.detach_context else noise_w,
        }
        return y.to(dtype=original_dtype), gates


class DendFADCConv2d(nn.Module):
    """Dendritic FADC convolution with optional task-calibrated evidence gating.

    Input:
        x: Tensor with shape [T, B, C_in, H, W]. T is the remote-sensing phase
           axis and is not hard-coded to 3.
        K: Optional K tensor/list from an earlier scale. It is used only by
           FrequencySelection and optional K update; it is not treated as a final
           change map.
        return_k: If True, return `(y, K_next)`, otherwise return `y`.

    Output:
        y: Tensor with shape [T, B, C_out, H_out, W_out]. For the MT-SCD encoder
           call path with stride=1 and padding=kernel_size//2, H_out=H and W_out=W.
        K_next: None or list of K maps. Each map has shape [T*B, spatial_group,
                H_k, W_k]. H_k/W_k depend on `Down_K` and `freq_weight_conv` stride.

    Default behavior:
        `task_calibrated=False`, so the new MTSCD-specific calibration path is
        disabled. This keeps behavior close to the old implementation.

    Stateful neuron safety:
        If `SN_CLS=True`, the internal soma node may be stateful. This class does
        not auto-reset it by default. Call `reset_state()` explicitly between
        batches/validation phases if your framework does not already call
        spikingjelly/mmseg `reset_net`.
    """

    _STC_FS_KEYS = {
        "task_calibrated",
        "stc_detach_context",
        "stc_detach_k_gate",
        "stc_update_k_from_prev",
        "stc_modulate_k",
        "stc_residual_init",
        "stc_k_scale_init",
        "stc_gate_kernel_size",
        "stc_gate_temperature",
        "stc_use_noise_suppression",
        "reset_before_forward",
    }

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
        SN_CLS: bool = False,
        Down_K: bool = True,
        task_calibrated: bool = False,
        stc_detach_context: bool = True,
        stc_detach_k_gate: bool = True,
        stc_update_k_from_prev: bool = False,
        stc_modulate_k: bool = True,
        stc_residual_init: float = 0.0,
        stc_k_scale_init: float = 0.0,
        stc_gate_kernel_size: int = 3,
        stc_gate_temperature: float = 1.0,
        stc_use_noise_suppression: bool = True,
        reset_before_forward: bool = False,
    ):
        super().__init__()
        del branch_num, detach_reset, v_th

        fs_cfg = dict(fs_cfg or {})
        stc_overrides = {key: fs_cfg.pop(key) for key in list(fs_cfg.keys()) if key in self._STC_FS_KEYS}
        task_calibrated = bool(stc_overrides.get("task_calibrated", task_calibrated))
        stc_detach_context = bool(stc_overrides.get("stc_detach_context", stc_detach_context))
        stc_detach_k_gate = bool(stc_overrides.get("stc_detach_k_gate", stc_detach_k_gate))
        stc_update_k_from_prev = bool(stc_overrides.get("stc_update_k_from_prev", stc_update_k_from_prev))
        stc_modulate_k = bool(stc_overrides.get("stc_modulate_k", stc_modulate_k))
        stc_residual_init = float(stc_overrides.get("stc_residual_init", stc_residual_init))
        stc_k_scale_init = float(stc_overrides.get("stc_k_scale_init", stc_k_scale_init))
        stc_gate_kernel_size = int(stc_overrides.get("stc_gate_kernel_size", stc_gate_kernel_size))
        stc_gate_temperature = float(stc_overrides.get("stc_gate_temperature", stc_gate_temperature))
        stc_use_noise_suppression = bool(stc_overrides.get("stc_use_noise_suppression", stc_use_noise_suppression))
        reset_before_forward = bool(stc_overrides.get("reset_before_forward", reset_before_forward))

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if deform_groups <= 0:
            raise ValueError("deform_groups must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = int(groups)
        self.deform_groups = int(deform_groups)
        self.padding_mode = padding_mode
        self.kernel_decompose = kernel_decompose
        self.pre_fs = bool(pre_fs)
        self.use_dct = bool(use_dct)
        self.use_zero_dilation = bool(use_zero_dilation)
        self.calculate_next_k = bool(calculate_next_k)
        self.SN_CLS = bool(SN_CLS)
        self.Down_K = bool(Down_K)
        self.task_calibrated = bool(task_calibrated)
        self.stc_detach_k_gate = bool(stc_detach_k_gate)
        self.stc_update_k_from_prev = bool(stc_update_k_from_prev)
        self.stc_modulate_k = bool(stc_modulate_k)
        self.reset_before_forward = bool(reset_before_forward)
        if self.task_calibrated:
            self.stc_k_scale = nn.Parameter(torch.tensor(float(stc_k_scale_init)))
        else:
            # Keep the default-off path checkpoint-compatible: no persistent
            # parameter/buffer is added when the new mechanism is disabled.
            self.register_buffer("stc_k_scale", torch.tensor(0.0), persistent=False)

        kh, kw = self.kernel_size
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("DendFADCConv2d requires odd kernel sizes")

        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels // self.groups, kh, kw))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None

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
        cfg.update(fs_cfg)
        self.fs_cfg = cfg
        self.dendrite = FrequencySelection(self.in_channels, **cfg) if self.pre_fs else None
        self.stc_calibrator = (
            SemanticTransitionCalibrator(
                self.in_channels,
                gate_kernel_size=stc_gate_kernel_size,
                detach_context=stc_detach_context,
                use_noise_suppression=stc_use_noise_suppression,
                residual_init=stc_residual_init,
                gate_temperature=stc_gate_temperature,
            )
            if self.task_calibrated
            else None
        )

        if self.SN_CLS:
            self.lif = MTSCDPRDNIIFNode()

        if kh > 1 or kw > 1:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=offset_padding,
                dilation=1,
                bias=True,
            )
            self.conv_mask = nn.Conv2d(
                self.in_channels,
                self.deform_groups * kh * kw,
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

        spatial_group = int(cfg.get("spatial_group", 1))
        if spatial_group > 64:
            spatial_group = self.in_channels
        if spatial_group <= 0 or self.in_channels % spatial_group != 0:
            spatial_group = 1
        self.spatial_group = spatial_group
        self.k_map_count = len(cfg.get("k_list", [])) + (1 if cfg.get("lowfreq_att", False) else 0)
        if self.calculate_next_k and self.k_map_count > 0:
            self.freq_weight_conv = nn.Conv2d(
                self.in_channels,
                self.k_map_count * self.spatial_group,
                kernel_size=3,
                stride=2 if self.Down_K else 1,
                padding=1,
                groups=self.spatial_group,
                bias=False,
            )
        else:
            self.freq_weight_conv = None

        if kernel_decompose == "both":
            self.OMNI_ATT1 = OmniAttention(
                self.in_channels, self.out_channels, kernel_size=1, groups=self.groups,
                reduction=reduction, kernel_num=1, min_channel=16,
            )
            self.OMNI_ATT2 = OmniAttention(
                self.in_channels, self.out_channels, kernel_size=kh if self.use_dct else 1,
                groups=self.groups, reduction=reduction, kernel_num=1, min_channel=16,
            )
        elif kernel_decompose in ("high", "low"):
            self.OMNI_ATT = OmniAttention(
                self.in_channels, self.out_channels, kernel_size=1, groups=self.groups,
                reduction=reduction, kernel_num=1, min_channel=16,
            )
        elif kernel_decompose in (None, "none"):
            pass
        else:
            raise ValueError(f"Unsupported kernel_decompose: {kernel_decompose}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.conv_offset is not None:
            nn.init.constant_(self.conv_offset.weight, 0.0)
            init_value = (self.dilation[0] - 1) / max(float(self.dilation[0]), 1.0) + 1e-4
            nn.init.constant_(self.conv_offset.bias, init_value)
        if self.conv_mask is not None:
            nn.init.constant_(self.conv_mask.weight, 0.0)
            nn.init.constant_(self.conv_mask.bias, 0.0)
        if self.freq_weight_conv is not None:
            nn.init.constant_(self.freq_weight_conv.weight, 0.0)

    def reset_state(self) -> None:
        """Explicitly reset stateful child neurons if they expose `reset` or `reset_state`.

        This method is intentionally not called in `forward` unless
        `reset_before_forward=True` is explicitly set. This avoids hiding
        cross-batch state behavior from the training/validation protocol.
        """
        for module in self.children():
            for name in ("reset_state", "reset"):
                fn = getattr(module, name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        pass
                    break

    def _sp_act(self, value: torch.Tensor) -> torch.Tensor:
        act = self.fs_cfg.get("act", "sigmoid")
        if act == "sigmoid":
            return value.sigmoid() * 2.0
        if act == "softmax":
            return value.softmax(dim=1) * value.shape[1]
        raise NotImplementedError(f"Unsupported K activation: {act}")

    def _align_gate(self, gate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gate = gate.to(device=target.device, dtype=target.dtype)
        if self.stc_detach_k_gate:
            gate = gate.detach()
        if gate.shape[-2:] != target.shape[-2:]:
            gate = F.interpolate(gate, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if gate.shape[0] != target.shape[0]:
            raise ValueError(f"Gate batch {gate.shape[0]} does not match target batch {target.shape[0]}")
        if gate.shape[1] == 1 and target.shape[1] != 1:
            gate = gate.expand(-1, target.shape[1], -1, -1)
        return gate

    def _prev_k_to_list(self, K: Optional[TensorOrKList], ref: torch.Tensor) -> Optional[List[torch.Tensor]]:
        if K is None or self.dendrite is None:
            return None
        return self.dendrite._k_to_list(K, ref.shape[0], ref.shape[-2], ref.shape[-1], ref.dtype, ref.device)

    def _calculate_k_next(
        self,
        x_spike: torch.Tensor,
        K_prev: Optional[TensorOrKList] = None,
        stc_gates: Optional[dict] = None,
    ) -> Optional[List[torch.Tensor]]:
        if self.freq_weight_conv is None:
            return None
        out = self.freq_weight_conv(x_spike)
        maps = [self._sp_act(item) for item in torch.split(out, self.spatial_group, dim=1)]
        if not self.task_calibrated or stc_gates is None or not self.stc_modulate_k:
            return maps

        transition_gate = self._align_gate(stc_gates["transition"], maps[0])
        stable_gate = self._align_gate(stc_gates["stable"], maps[0])
        k_scale = self.stc_k_scale.to(device=maps[0].device, dtype=maps[0].dtype)
        prev_maps = self._prev_k_to_list(K_prev, maps[0]) if self.stc_update_k_from_prev else None

        updated = []
        for idx, candidate in enumerate(maps):
            trans = transition_gate
            stab = stable_gate
            if trans.shape[-2:] != candidate.shape[-2:]:
                trans = F.interpolate(trans, size=candidate.shape[-2:], mode="bilinear", align_corners=False)
                stab = F.interpolate(stab, size=candidate.shape[-2:], mode="bilinear", align_corners=False)
            if trans.shape[1] == 1 and candidate.shape[1] != 1:
                trans = trans.expand(-1, candidate.shape[1], -1, -1)
                stab = stab.expand(-1, candidate.shape[1], -1, -1)

            # Continuous K modulation. No hard threshold, no integer activation,
            # and therefore no STE/surrogate path is introduced here.
            candidate = candidate * (1.0 + k_scale * (2.0 * trans - 1.0))

            if prev_maps is not None and idx < len(prev_maps):
                prev = prev_maps[idx]
                if prev.shape[-2:] != candidate.shape[-2:]:
                    prev = F.interpolate(prev, size=candidate.shape[-2:], mode="bilinear", align_corners=False)
                if prev.shape[1] == 1 and candidate.shape[1] != 1:
                    prev = prev.expand(-1, candidate.shape[1], -1, -1)
                if self.stc_detach_k_gate:
                    prev = prev.detach()
                candidate = stab * prev + trans * candidate

            updated.append(torch.nan_to_num(candidate, nan=1.0, posinf=2.0, neginf=0.0))
        return updated

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
            c_scale = c_att.to(device=device, dtype=dtype).reshape(batch, self.groups, in_per_group, 1, 1).unsqueeze(2)
        if not torch.is_tensor(f_att):
            f_scale = 1.0
        else:
            f_scale = f_att.to(device=device, dtype=dtype).reshape(batch, self.groups, out_per_group, 1, 1).unsqueeze(3)
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
            c_scale1, f_scale1 = self._reshape_group_attention(c_att1, f_att1, b, x_spike.dtype, x_spike.device)
            c_scale2, f_scale2 = self._reshape_group_attention(c_att2, f_att2, b, x_spike.dtype, x_spike.device)
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
            adaptive = weight_mean * (c_scale1 * 2.0) * (f_scale1 * 2.0) + weight_res * (c_scale2 * 2.0) * (f_scale2 * 2.0)
        elif hasattr(self, "OMNI_ATT"):
            c_att, f_att, _, _ = self.OMNI_ATT(x_spike)
            c_scale, f_scale = self._reshape_group_attention(c_att, f_att, b, x_spike.dtype, x_spike.device)
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
        bias: Optional[torch.Tensor],
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

    def forward(self, x: torch.Tensor, K: Optional[TensorOrKList] = None, return_k: bool = True):
        if self.reset_before_forward:
            self.reset_state()
        if x.dim() != 5:
            raise ValueError(f"DendFADCConv2d expects [T,B,C,H,W], got {tuple(x.shape)}")
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

        stc_gates = None
        if self.stc_calibrator is not None:
            x_dend, stc_gates = self.stc_calibrator(x_dend)

        if hasattr(self, "lif"):
            x_dend = self.lif(x_dend)
        x_spike = x_dend.flatten(0, 1).contiguous()

        K_next = self._calculate_k_next(x_spike, K_prev=K, stc_gates=stc_gates)
        adaptive_weight = self._adaptive_weight(x_spike)
        bias = self.bias.repeat(tb) if self.bias is not None else None

        if self.conv_offset is None:
            x_grouped = x_spike.reshape(1, tb * C, H, W).contiguous()
            y = self._fallback_group_conv(x_grouped, adaptive_weight, bias, tb)
            y = y.reshape(T, B, self.out_channels, y.shape[-2], y.shape[-1]).contiguous()
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
    """Optional wrapper for DendFADCConv2d followed by 2D normalization.

    Input:
        x: Tensor with shape [T, B, C_in, H, W].
        K: Optional K tensor/list passed to the wrapped DendFADCConv2d.
        return_k: If True, return `(y, K_next)`, otherwise return `y`.

    Output:
        y: Tensor with shape [T, B, C_out, H_out, W_out]. If `bn` is provided,
           it is applied on flattened [T*B, C_out, H_out, W_out] and reshaped
           back to [T, B, C_out, H_out, W_out].
    """

    def __init__(self, conv: DendFADCConv2d, bn: Optional[nn.Module] = None):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor, K: Optional[TensorOrKList] = None, return_k: bool = False):
        if return_k:
            y, k_next = self.conv(x, K=K, return_k=True)
        else:
            y = self.conv(x, K=K, return_k=False)
            k_next = None
        if self.bn is not None:
            T, B, C, H, W = y.shape
            y = self.bn(y.flatten(0, 1)).reshape(T, B, C, H, W).contiguous()
        return (y, k_next) if return_k else y


def _shape_sanity_test() -> None:
    torch.manual_seed(7)
    for T in (3, 6):
        x = torch.randn(T, 2, 8, 16, 16)
        layer = DendFADCConv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            padding=1,
            groups=8,
            deform_groups=1,
            bias=False,
            kernel_decompose=None,
            SN_CLS=False,
            Down_K=False,
            task_calibrated=True,
            stc_detach_context=True,
            stc_residual_init=0.0,
            stc_k_scale_init=0.0,
        )
        y, k_next = layer(x, K=None, return_k=True)
        assert y.shape == x.shape, (T, y.shape, x.shape)
        assert isinstance(k_next, list) and len(k_next) == 2
        assert all(k.shape[:2] == (T * 2, 1) for k in k_next)
        assert torch.isfinite(y).all()
        assert all(torch.isfinite(k).all() for k in k_next)
    print("shape sanity test passed for T=3 and T=6")


if __name__ == "__main__":
    _shape_sanity_test()

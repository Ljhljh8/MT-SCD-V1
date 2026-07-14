import math
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant
from models.dendsn_lifFADC_Snn_v2 import FrequencySelection


INTEGRATION_KERNEL_BY_SCALE = {1: 5, 2: 5, 3: 3}
DIRECTION_DILATION_BY_SCALE = {1: 2, 2: 1, 3: 1}

# The five modes are intentionally mutually exclusive. Each ablation changes
# one causal factor while preserving the trainable parameter/state-dict layout.
ABLATION_CONFIGS = {
    "full": {
        "routing_mode": "local",
        "descriptor_mask": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "basis_mode": "directional",
    },
    "uniform_route": {
        "routing_mode": "uniform",
        "descriptor_mask": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "basis_mode": "directional",
    },
    "global_route": {
        "routing_mode": "global",
        "descriptor_mask": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "basis_mode": "directional",
    },
    "no_axis_descriptor": {
        # Keep anisotropy magnitude a, but remove the axis coordinates q1/q2.
        "routing_mode": "local",
        "descriptor_mask": (1.0, 1.0, 1.0, 0.0, 0.0, 1.0),
        "basis_mode": "directional",
    },
    "isotropic_direction_pool": {
        "routing_mode": "local",
        "descriptor_mask": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "basis_mode": "isotropic_direction_pool",
    },
}


def _to_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


@contextmanager
def _fp32_context(value: torch.Tensor):
    if value.is_cuda:
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


class AnalyticStructureDescriptor(nn.Module):
    def __init__(self, integration_kernel: int, eps: float = 1e-6):
        super().__init__()
        if integration_kernel <= 0 or integration_kernel % 2 == 0:
            raise ValueError("integration_kernel must be a positive odd integer")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.integration_kernel = int(integration_kernel)
        self.eps = float(eps)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous())

    def _average(self, value: torch.Tensor) -> torch.Tensor:
        pad = self.integration_kernel // 2
        value = F.pad(value, (pad, pad, pad, pad), mode="replicate")
        return F.avg_pool2d(value, kernel_size=self.integration_kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("AnalyticStructureDescriptor expects [B,C,H,W]")
        with _fp32_context(x):
            x = x.float()
            channels = x.shape[1]
            mean_c = self._average(x)
            second_c = self._average(x.square())
            mu = mean_c.mean(dim=1, keepdim=True)
            second = second_c.mean(dim=1, keepdim=True)
            variance = (second_c - mean_c.square()).clamp_min(0.0).mean(dim=1, keepdim=True)
            nu = variance / (second + self.eps)

            padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
            sobel_x = self.sobel_x.expand(channels, 1, 3, 3).contiguous()
            sobel_y = self.sobel_y.expand(channels, 1, 3, 3).contiguous()
            gx = F.conv2d(padded, sobel_x, padding=0, groups=channels)
            gy = F.conv2d(padded, sobel_y, padding=0, groups=channels)

            jxx = self._average(gx.square()).mean(dim=1, keepdim=True)
            jyy = self._average(gy.square()).mean(dim=1, keepdim=True)
            jxy = self._average(gx * gy).mean(dim=1, keepdim=True)
            trace = jxx + jyy
            e = trace / (trace + second + self.eps)
            q1 = (jxx - jyy) / (trace + self.eps)
            q2 = (2.0 * jxy) / (trace + self.eps)
            anisotropy = torch.sqrt(q1.square() + q2.square() + self.eps)
            return torch.cat((mu, nu, e, q1, q2, anisotropy), dim=1)


class SharedAxialDirectionalDWConv(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.channels = int(channels)
        self.dilation = int(dilation)
        self.local_dw = nn.Conv2d(channels, channels, 3, groups=channels, bias=False, padding=0)
        self.canonical = nn.Parameter(torch.empty(channels, 2))
        self.region_dw = nn.Conv2d(channels, channels, 5, groups=channels, bias=False, padding=0)

        templates = torch.zeros(4, 3, 3, 3, dtype=torch.float32)
        positions = (
            ((1, 0), (1, 1), (1, 2)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 0), (1, 1), (2, 2)),
            ((0, 2), (1, 1), (2, 0)),
        )
        for direction, direction_positions in enumerate(positions):
            for coefficient, (row, column) in enumerate(direction_positions):
                templates[direction, coefficient, row, column] = 1.0
        self.register_buffer("direction_templates", templates)
        nn.init.uniform_(self.canonical, -1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))

    def _directional_kernels(self) -> torch.Tensor:
        coefficients = torch.stack(
            (self.canonical[:, 0], self.canonical[:, 1], self.canonical[:, 0]),
            dim=1,
        )
        kernels = torch.einsum("cp,dpij->dcij", coefficients, self.direction_templates)
        return kernels.unsqueeze(2)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("SharedAxialDirectionalDWConv expects [B,C,H,W]")
        outputs = [self.local_dw(F.pad(x, (1, 1, 1, 1), mode="replicate"))]
        directional_input = F.pad(
            x,
            (self.dilation, self.dilation, self.dilation, self.dilation),
            mode="replicate",
        )
        kernels = self._directional_kernels()
        for direction in range(4):
            outputs.append(
                F.conv2d(
                    directional_input,
                    kernels[direction],
                    padding=0,
                    dilation=self.dilation,
                    groups=self.channels,
                )
            )
        outputs.append(self.region_dw(F.pad(x, (2, 2, 2, 2), mode="replicate")))
        return tuple(outputs)


class DendStructureRoutedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
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
        scale_index: int = 1,
        ablation_mode: str = "full",
    ):
        super().__init__()
        del branch_num, detach_reset, deform_groups, kernel_decompose, use_dct
        del use_zero_dilation, v_th, reduction
        if in_channels != out_channels:
            raise ValueError("DendStructureRoutedConv2d requires equal input/output channels")
        if groups != in_channels:
            raise ValueError("DendStructureRoutedConv2d requires depthwise groups")
        if _to_pair(kernel_size) != (3, 3):
            raise ValueError("DendStructureRoutedConv2d requires kernel_size=3")
        if _to_pair(stride) != (1, 1) or _to_pair(dilation) != (1, 1):
            raise ValueError("DendStructureRoutedConv2d requires stride=dilation=1")
        if _to_pair(padding) != (1, 1) or padding_mode != "repeat":
            raise ValueError("DendStructureRoutedConv2d requires replicate padding=1")
        if bias:
            raise ValueError("DendStructureRoutedConv2d requires bias=False")
        if not pre_fs or not calculate_next_k:
            raise ValueError("FrequencySelection and K_next must remain enabled")
        if scale_index not in INTEGRATION_KERNEL_BY_SCALE:
            raise ValueError("scale_index must be one of 1, 2, 3")
        if ablation_mode not in ABLATION_CONFIGS:
            valid_modes = ", ".join(sorted(ABLATION_CONFIGS))
            raise ValueError(
                "Unsupported ablation_mode %r; expected one of: %s"
                % (ablation_mode, valid_modes)
            )

        ablation_cfg = ABLATION_CONFIGS[ablation_mode]
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.scale_index = int(scale_index)
        self.Down_K = bool(Down_K)
        self.ablation_mode = str(ablation_mode)
        self.routing_mode = str(ablation_cfg["routing_mode"])
        self.basis_mode = str(ablation_cfg["basis_mode"])
        descriptor_mask = torch.tensor(
            ablation_cfg["descriptor_mask"], dtype=torch.float32
        ).view(1, 6, 1, 1)
        self.register_buffer("descriptor_mask", descriptor_mask, persistent=False)

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
        self.frequency_selection = FrequencySelection(in_channels, **cfg)
        if SN_CLS:
            self.lif = Q_IFNode(surrogate_function=Quant())

        spatial_group = cfg.get("spatial_group", 1)
        if spatial_group > 64:
            spatial_group = in_channels
        if in_channels % spatial_group != 0:
            spatial_group = 1
        self.spatial_group = spatial_group
        self.k_map_count = len(cfg.get("k_list", [])) + (1 if cfg.get("lowfreq_att", False) else 0)
        self.freq_weight_conv = nn.Conv2d(
            in_channels,
            self.k_map_count * self.spatial_group,
            kernel_size=3,
            stride=2 if self.Down_K else 1,
            padding=1,
            groups=self.spatial_group,
            bias=False,
        )

        self.descriptor = AnalyticStructureDescriptor(INTEGRATION_KERNEL_BY_SCALE[scale_index])
        self.router = nn.Conv2d(6, 6, kernel_size=1, bias=True)
        self.spatial_bases = SharedAxialDirectionalDWConv(
            in_channels,
            DIRECTION_DILATION_BY_SCALE[scale_index],
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.freq_weight_conv.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def _sp_act(self, value: torch.Tensor) -> torch.Tensor:
        act = self.fs_cfg.get("act", "sigmoid")
        if act == "sigmoid":
            return value.sigmoid() * 2.0
        if act == "softmax":
            return value.softmax(dim=1) * value.shape[1]
        raise NotImplementedError("Unsupported K activation: %s" % act)

    def _calculate_k_next(self, x_spike: torch.Tensor):
        out = self.freq_weight_conv(x_spike)
        maps = torch.split(out, self.spatial_group, dim=1)
        return [self._sp_act(item) for item in maps]

    def _compute_routing_gates(self, structure: torch.Tensor) -> torch.Tensor:
        # Keep the full/global/uniform descriptor path byte-for-byte equivalent
        # to V1; only the no-axis ablation applies a descriptor mask.
        route_input = structure
        if self.ablation_mode == "no_axis_descriptor":
            descriptor_mask = self.descriptor_mask.to(dtype=structure.dtype)
            route_input = structure * descriptor_mask
        if self.routing_mode == "global":
            route_input = route_input.mean(dim=(-2, -1), keepdim=True)

        with _fp32_context(route_input):
            logits = self.router(route_input.float())
            if self.routing_mode == "uniform":
                # Keep the router in the graph while making every gate exactly 1/6.
                logits = logits * 0.0
            gates = torch.softmax(logits.float(), dim=1)
        return gates

    def _prepare_basis_responses(
        self, responses: Sequence[torch.Tensor]
    ) -> Sequence[torch.Tensor]:
        if len(responses) != 6:
            raise ValueError("Expected exactly six structured basis responses")
        if self.basis_mode != "isotropic_direction_pool":
            return responses

        # Collapse orientation selectivity without removing trainable parameters,
        # convolution branches, or router channels. The average is an
        # orientation-agnostic context response derived from the same canonical
        # kernel; it adds only the explicit response-averaging arithmetic.
        isotropic = (
            responses[1] + responses[2] + responses[3] + responses[4]
        ) / 4.0
        return (
            responses[0], isotropic, isotropic, isotropic, isotropic, responses[5]
        )

    def extra_repr(self) -> str:
        return "scale_index=%d, ablation_mode=%r" % (
            self.scale_index,
            self.ablation_mode,
        )

    def forward(self, x: torch.Tensor, K=None, return_k: bool = True):
        if x.ndim != 5:
            raise ValueError("DendStructureRoutedConv2d expects [N,B,C,H,W]")
        phases, batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError("Expected %d input channels, got %d" % (self.in_channels, channels))

        x_dend = self.frequency_selection(x.flatten(0, 1).contiguous(), K)
        x_dend = x_dend.reshape(phases, batch, channels, height, width).contiguous()
        if hasattr(self, "lif"):
            x_dend = self.lif(x_dend)
        x_spike = x_dend.flatten(0, 1).contiguous()
        k_next = self._calculate_k_next(x_spike)

        with torch.no_grad():
            structure = self.descriptor(x_spike.detach().float())
        gates = self._compute_routing_gates(structure)
        gates = gates.to(dtype=x_spike.dtype)

        responses = self.spatial_bases(x_spike)
        responses = self._prepare_basis_responses(responses)
        fused = gates[:, 0:1] * responses[0]
        for index in range(1, 6):
            fused = fused + gates[:, index:index + 1] * responses[index]
        residual = self.pointwise(fused)
        residual = residual.reshape(phases, batch, self.out_channels, height, width).contiguous()
        return (residual, k_next) if return_k else residual

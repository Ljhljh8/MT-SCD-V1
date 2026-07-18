"""Bounded residual Structure-Routed Convolution V3.

V3 keeps the complete V1 full path and adds only a zero-initialized residual
calibration of the six routing logits.  The calibration reads the detached,
FP32 V1 analytic descriptor; it never reads ``x_spike``, ``K``/``K_next``,
cross-phase features, or decoder state.

The six original experiments and two follow-up controls share one
parameter/state layout.  ``v3_6`` is
the complete bounded hypothesis; ``v3_1`` is an intentionally unbounded linear
reparameterization control.
"""

import math
from contextlib import contextmanager
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant
from models.dendsn_lifFADC_Snn_v2 import FrequencySelection


INTEGRATION_KERNEL_BY_SCALE = {1: 5, 2: 5, 3: 3}
DIRECTION_DILATION_BY_SCALE = {1: 2, 2: 1, 3: 1}
V3_EPSILON = 0.25

BRANCH_NAMES = (
    "local",
    "horizontal",
    "vertical",
    "main_diag",
    "anti_diag",
    "region",
)


class V3ModeConfig(NamedTuple):
    bounded: bool
    branch_mask: Tuple[float, float, float, float, float, float]
    active_stages: Tuple[int, ...]


# Insertion order is the experiment registration contract.
V3_MODE_CONFIGS = {
    # Linear residual control: same inputs and topology, but no tanh bound.
    "v3_1": V3ModeConfig(
        False,
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (1, 2, 3),
    ),
    # Bounded calibration of only the four directional branches.
    "v3_2": V3ModeConfig(True, (0.0, 1.0, 1.0, 1.0, 1.0, 0.0), (1, 2, 3)),
    # Bounded calibration of only the local and region branches.
    "v3_3": V3ModeConfig(True, (1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1, 2, 3)),
    # Bounded all-branch calibration only at the first active Encoder Stage.
    "v3_4": V3ModeConfig(True, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), (1,)),
    # Bounded all-branch calibration only at the later active Encoder Stages.
    "v3_5": V3ModeConfig(True, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), (2, 3)),
    # V3 full: bounded all-branch calibration at every active Encoder Stage.
    "v3_6": V3ModeConfig(True, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), (1, 2, 3)),
    # V3-7: direction-only linear control at all active scales.
    # This is an unbounded linear reparameterization control for V3-2.
    "v3_7": V3ModeConfig(False, (0.0, 1.0, 1.0, 1.0, 1.0, 0.0), (1, 2, 3)),
    # V3-8: bounded direction-only calibration only at scale_index=1.
    # Scales 2 and 3 retain the same parameters/computation graph,
    # but their calibration contribution is multiplied by zero.
    "v3_8": V3ModeConfig(True, (0.0, 1.0, 1.0, 1.0, 1.0, 0.0), (1,)),

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


class AnalyticStructureDescriptorV1(nn.Module):
    """The parameter-free six-channel analytic descriptor used by V1 full."""

    output_channels = 6

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
        return F.avg_pool2d(
            value,
            kernel_size=self.integration_kernel,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("AnalyticStructureDescriptorV1 expects [B,C,H,W]")

        # This boundary is intentional: the descriptor reads structure but does
        # not provide a gradient path through which the backbone can shape it.
        with torch.no_grad():
            with _fp32_context(x):
                value = x.detach().float()
                channels = value.shape[1]
                mean_c = self._average(value)
                second_c = self._average(value.square())
                mu = mean_c.mean(dim=1, keepdim=True)
                second = second_c.mean(dim=1, keepdim=True)
                variance = (
                    (second_c - mean_c.square())
                    .clamp_min(0.0)
                    .mean(dim=1, keepdim=True)
                )
                nu = variance / (second + self.eps)

                padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
                sobel_x = self.sobel_x.expand(channels, 1, 3, 3).contiguous()
                sobel_y = self.sobel_y.expand(channels, 1, 3, 3).contiguous()
                gx = F.conv2d(padded, sobel_x, padding=0, groups=channels)
                gy = F.conv2d(padded, sobel_y, padding=0, groups=channels)

                jxx = self._average(gx.square()).mean(dim=1, keepdim=True)
                jyy = self._average(gy.square()).mean(dim=1, keepdim=True)
                jxy = self._average(gx * gy).mean(dim=1, keepdim=True)
                trace = jxx + jyy
                edge_energy = trace / (trace + second + self.eps)
                q1 = (jxx - jyy) / (trace + self.eps)
                q2 = (2.0 * jxy) / (trace + self.eps)
                anisotropy = torch.sqrt(q1.square() + q2.square() + self.eps)
                descriptor = torch.cat(
                    (mu, nu, edge_energy, q1, q2, anisotropy),
                    dim=1,
                )
                return descriptor.detach()


class SharedAxialDirectionalDWConv(nn.Module):
    """The unchanged V1 local, directional, and region spatial bases."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.channels = int(channels)
        self.dilation = int(dilation)
        self.local_dw = nn.Conv2d(
            channels,
            channels,
            3,
            groups=channels,
            bias=False,
            padding=0,
        )
        # One canonical [a_c, b_c, a_c] kernel is shared by every direction.
        self.canonical = nn.Parameter(torch.empty(channels, 2))
        self.region_dw = nn.Conv2d(
            channels,
            channels,
            5,
            groups=channels,
            bias=False,
            padding=0,
        )

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
        nn.init.uniform_(
            self.canonical,
            -1.0 / math.sqrt(3.0),
            1.0 / math.sqrt(3.0),
        )

    def _directional_kernels(self) -> torch.Tensor:
        coefficients = torch.stack(
            (
                self.canonical[:, 0],
                self.canonical[:, 1],
                self.canonical[:, 0],
            ),
            dim=1,
        )
        kernels = torch.einsum(
            "cp,dpij->dcij",
            coefficients,
            self.direction_templates,
        )
        return kernels.unsqueeze(2)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("SharedAxialDirectionalDWConv expects [B,C,H,W]")

        outputs = [
            self.local_dw(F.pad(x, (1, 1, 1, 1), mode="replicate"))
        ]
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
        outputs.append(
            self.region_dw(F.pad(x, (2, 2, 2, 2), mode="replicate"))
        )
        return tuple(outputs)


class DendStructureRoutedConv2dV3(nn.Module):
    """Drop-in V1 RouteConv plus a preregistered residual-router calibration."""

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
        v3_mode: str = "v3_6",
    ):
        super().__init__()
        del branch_num, detach_reset, deform_groups, kernel_decompose, use_dct
        del use_zero_dilation, v_th, reduction

        if in_channels != out_channels:
            raise ValueError(
                "DendStructureRoutedConv2dV3 requires equal input/output channels"
            )
        if groups != in_channels:
            raise ValueError(
                "DendStructureRoutedConv2dV3 requires depthwise groups"
            )
        if _to_pair(kernel_size) != (3, 3):
            raise ValueError("DendStructureRoutedConv2dV3 requires kernel_size=3")
        if _to_pair(stride) != (1, 1) or _to_pair(dilation) != (1, 1):
            raise ValueError(
                "DendStructureRoutedConv2dV3 requires stride=dilation=1"
            )
        if _to_pair(padding) != (1, 1) or padding_mode != "repeat":
            raise ValueError(
                "DendStructureRoutedConv2dV3 requires replicate padding=1"
            )
        if bias:
            raise ValueError("DendStructureRoutedConv2dV3 requires bias=False")
        if not pre_fs or not calculate_next_k:
            raise ValueError("FrequencySelection and K_next must remain enabled")
        if scale_index not in INTEGRATION_KERNEL_BY_SCALE:
            raise ValueError("scale_index must be one of 1, 2, 3")

        v3_mode = str(v3_mode).lower()
        if v3_mode not in V3_MODE_CONFIGS:
            valid_modes = ", ".join(V3_MODE_CONFIGS)
            raise ValueError(
                "Unsupported v3_mode %r; expected one of: %s"
                % (v3_mode, valid_modes)
            )

        mode_config = V3_MODE_CONFIGS[v3_mode]
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.scale_index = int(scale_index)
        self.Down_K = bool(Down_K)
        self.v3_mode = v3_mode
        self.calibration_bounded = bool(mode_config.bounded)
        self.epsilon = float(V3_EPSILON)

        branch_mask = torch.tensor(
            mode_config.branch_mask,
            dtype=torch.float32,
        ).view(1, 6, 1, 1)
        stage_mask = torch.tensor(
            float(self.scale_index in mode_config.active_stages),
            dtype=torch.float32,
        ).view(1, 1, 1, 1)
        self.register_buffer("calibration_branch_mask", branch_mask, persistent=False)
        self.register_buffer("calibration_stage_mask", stage_mask, persistent=False)

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
        self.k_map_count = len(cfg.get("k_list", [])) + (
            1 if cfg.get("lowfreq_att", False) else 0
        )
        self.freq_weight_conv = nn.Conv2d(
            in_channels,
            self.k_map_count * self.spatial_group,
            kernel_size=3,
            stride=2 if self.Down_K else 1,
            padding=1,
            groups=self.spatial_group,
            bias=False,
        )

        # Preserve the V1 construction order for every common trainable tensor.
        self.descriptor = AnalyticStructureDescriptorV1(
            INTEGRATION_KERNEL_BY_SCALE[scale_index]
        )
        self.router = nn.Conv2d(6, 6, kernel_size=1, bias=True)
        self.spatial_bases = SharedAxialDirectionalDWConv(
            in_channels,
            DIRECTION_DILATION_BY_SCALE[scale_index],
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        # Constructing an extra Conv2d normally advances the global RNG and
        # changes all later Stage/downstream initializations.  Forking the CPU
        # RNG makes every V1-common parameter identical under the same seed.
        with torch.random.fork_rng(devices=[]):
            self.calibration = nn.Conv2d(6, 6, kernel_size=1, bias=True)

        nn.init.zeros_(self.freq_weight_conv.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

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

    def _center_active_branches(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.calibration_branch_mask.to(
            device=value.device,
            dtype=value.dtype,
        )
        active_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        active_mean = (value * mask).sum(dim=1, keepdim=True) / active_count
        return (value - active_mean) * mask

    def _routing_terms(
        self,
        structure: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if structure.ndim != 4 or structure.shape[1] != 6:
            raise ValueError("structure must have shape [B,6,H,W]")

        # Only S_v1 enters both learnable mappings.  S_v1 is detached, while
        # router and calibration parameters remain in the autograd graph.
        with _fp32_context(structure):
            route_input = structure.detach().float()
            base_logits = self.router(route_input)
            calibration_raw = self.calibration(route_input)
            calibration_signal = (
                torch.tanh(calibration_raw)
                if self.calibration_bounded
                else calibration_raw
            )
            centered = self._center_active_branches(calibration_signal)
            stage_mask = self.calibration_stage_mask.to(
                device=centered.device,
                dtype=centered.dtype,
            )
            delta_logits = self.epsilon * stage_mask * centered
            calibrated_logits = base_logits + delta_logits
        return base_logits, calibration_raw, delta_logits, calibrated_logits

    def routing_diagnostics(self, structure: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return routing tensors for checkpoint diagnostics without side effects."""

        base_logits, calibration_raw, delta_logits, logits = self._routing_terms(
            structure
        )
        return {
            "base_logits": base_logits,
            "calibration_raw": calibration_raw,
            "delta_logits": delta_logits,
            "logits": logits,
            "base_gates": torch.softmax(base_logits.float(), dim=1),
            "gates": torch.softmax(logits.float(), dim=1),
        }

    def _compute_routing_gates(self, structure: torch.Tensor) -> torch.Tensor:
        _, _, _, logits = self._routing_terms(structure)
        return torch.softmax(logits.float(), dim=1)

    def extra_repr(self) -> str:
        active = bool(self.calibration_stage_mask.item())
        return (
            "scale_index=%d, v3_mode=%r, epsilon=%.2f, "
            "bounded=%r, stage_active=%r"
            % (
                self.scale_index,
                self.v3_mode,
                self.epsilon,
                self.calibration_bounded,
                active,
            )
        )

    def forward(self, x: torch.Tensor, K=None, return_k: bool = True):
        if x.ndim != 5:
            raise ValueError("DendStructureRoutedConv2dV3 expects [N,B,C,H,W]")
        phases, batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                "Expected %d input channels, got %d"
                % (self.in_channels, channels)
            )

        # Keep the V1 FrequencySelection -> inner Q_IF -> K_next order intact.
        x_dend = self.frequency_selection(
            x.flatten(0, 1).contiguous(),
            K,
        )
        x_dend = x_dend.reshape(
            phases,
            batch,
            channels,
            height,
            width,
        ).contiguous()
        if hasattr(self, "lif"):
            x_dend = self.lif(x_dend)
        x_spike = x_dend.flatten(0, 1).contiguous()
        k_next = self._calculate_k_next(x_spike)

        structure = self.descriptor(x_spike)
        gates = self._compute_routing_gates(structure).to(dtype=x_spike.dtype)

        responses = self.spatial_bases(x_spike)
        fused = gates[:, 0:1] * responses[0]
        for index in range(1, 6):
            fused = fused + gates[:, index:index + 1] * responses[index]
        residual = self.pointwise(fused)
        residual = residual.reshape(
            phases,
            batch,
            self.out_channels,
            height,
            width,
        ).contiguous()
        return (residual, k_next) if return_k else residual


__all__ = [
    "AnalyticStructureDescriptorV1",
    "SharedAxialDirectionalDWConv",
    "DendStructureRoutedConv2dV3",
    "V3ModeConfig",
    "V3_MODE_CONFIGS",
    "V3_EPSILON",
    "BRANCH_NAMES",
]
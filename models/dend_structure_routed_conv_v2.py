"""Structure-routed depthwise convolution V2 for the MT-SCD Encoder.

The module keeps the V1 frequency-state interface and exposes six controlled
RouteConv experiments through ``v2_mode``.  It is intended to live at
``models/dend_structure_routed_conv_v2.py`` in MT-SCD-V1.

The analytic descriptor reads each flattened phase independently.  It has no
trainable parameters, detaches its input, and always returns FP32 maps.  The
spatial path fuses local, directional, and region depthwise responses before a
single shared pointwise projection.
"""

import math
from contextlib import contextmanager
from typing import Dict, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant
from models.dendsn_lifFADC_Snn_v2 import FrequencySelection


# DESIGN_HYPOTHESIS: the short span preserves the corresponding V1 span;
# dual-scale modes add exactly one longer predefined span.
DIRECTION_DILATIONS_BY_SCALE = {
    1: (2, 3),
    2: (1, 2),
    3: (1, 2),
}
DESCRIPTOR_WINDOWS_BY_SCALE = {
    1: (5, 7),
    2: (3, 5),
    3: (3, 5),
}


class V2ModeConfig(NamedTuple):
    descriptor: str
    routing: str
    directional_spans: int


# Insertion order is part of the experiment registration contract.
V2_MODE_CONFIGS = {
    "v2_1": V2ModeConfig("multiscale_mean", "flat6", 1),
    "v2_2": V2ModeConfig("robust", "flat6", 1),
    "v2_3": V2ModeConfig("robust", "factorized_single", 1),
    "v2_4": V2ModeConfig("robust", "flat10", 2),
    "v2_5": V2ModeConfig("robust", "factorized_dual_uniform", 2),
    "v2_6": V2ModeConfig("robust", "factorized_dual", 2),
}
SIX_BRANCH_NAMES = (
    "local", "horizontal", "vertical", "main_diag", "anti_diag", "region"
)
TEN_BRANCH_NAMES = (
    "local",
    "horizontal_short",
    "horizontal_long",
    "vertical_short",
    "vertical_long",
    "main_diag_short",
    "main_diag_long",
    "anti_diag_short",
    "anti_diag_long",
    "region",
)


def _to_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


@contextmanager
def _fp32_context(value: torch.Tensor):
    # The project pins PyTorch 2.1, whose device-generic autocast context
    # covers both CUDA AMP and CPU BF16 autocast.
    with torch.autocast(device_type=value.device.type, enabled=False):
        yield


class AnalyticStructureDescriptorV2(nn.Module):
    """Parameter-free two-scale descriptor with symmetric channel aggregation.

    The fixed 20-channel layout is:

    - short and long, eight maps each: mean, RMS, normalized variance,
      support ratio, edge strength, tangent q1, tangent q2, coherence;
    - cross-scale, four maps: direction consistency, direction reliability,
      edge contrast, and structure uncertainty.

    ``robust=False`` keeps the same layout but zeros RMS and support ratio and
    uses channel-mean structure tensors.  ``robust=True`` additionally uses
    gradient-energy weighting.  Every operation is invariant to a permutation
    of the input-channel index.
    """

    output_channels = 20

    def __init__(self, scale_index: int, robust: bool = True, eps: float = 1e-6):
        super().__init__()
        if scale_index not in DESCRIPTOR_WINDOWS_BY_SCALE:
            raise ValueError("scale_index must be one of 1, 2, 3")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.scale_index = int(scale_index)
        self.windows = DESCRIPTOR_WINDOWS_BY_SCALE[self.scale_index]
        self.robust = bool(robust)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer(
            "sobel_y", sobel_x.transpose(-1, -2).contiguous()
        )

    @staticmethod
    def _average(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        padded = F.pad(value, (pad, pad, pad, pad), mode="replicate")
        return F.avg_pool2d(
            padded,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def _spatial_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = x.shape[1]
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        sobel_x = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sobel_y = self.sobel_y.to(device=x.device, dtype=torch.float32)
        gx = F.conv2d(
            padded,
            sobel_x.expand(channels, 1, 3, 3).contiguous(),
            padding=0,
            groups=channels,
        )
        gy = F.conv2d(
            padded,
            sobel_y.expand(channels, 1, 3, 3).contiguous(),
            padding=0,
            groups=channels,
        )
        return gx, gy

    def _participation_ratio(self, energy: torch.Tensor) -> torch.Tensor:
        channels = energy.shape[1]
        energy_sum = energy.sum(dim=1, keepdim=True)
        energy_square_sum = energy.square().sum(dim=1, keepdim=True)
        ratio = energy_sum.square() / (
            float(channels) * energy_square_sum + self.eps
        )
        ratio = torch.where(
            energy_sum > self.eps,
            ratio,
            torch.zeros_like(ratio),
        )
        return ratio.clamp_(0.0, 1.0)

    def _scale_descriptor(
        self,
        x: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
        kernel_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        mean_c = self._average(x, kernel_size)
        second_c = self._average(x.square(), kernel_size).clamp_min_(0.0)
        variance_c = (second_c - mean_c.square()).clamp_min_(0.0)

        mean_level = mean_c.mean(dim=1, keepdim=True)
        normalized_variance = (
            variance_c / (second_c + self.eps)
        ).mean(dim=1, keepdim=True)

        jxx_c = self._average(gx.square(), kernel_size)
        jyy_c = self._average(gy.square(), kernel_size)
        jxy_c = self._average(gx * gy, kernel_size)
        structure_energy_c = (jxx_c + jyy_c).clamp_min_(0.0)

        if self.robust:
            rms_level = torch.sqrt(
                second_c.mean(dim=1, keepdim=True).clamp_min_(0.0)
            )
            support_ratio = self._participation_ratio(structure_energy_c)
            weight_denominator = structure_energy_c.sum(dim=1, keepdim=True)
            channel_weights = torch.where(
                weight_denominator > self.eps,
                structure_energy_c / (weight_denominator + self.eps),
                torch.zeros_like(structure_energy_c),
            )
            jxx = (channel_weights * jxx_c).sum(dim=1, keepdim=True)
            jyy = (channel_weights * jyy_c).sum(dim=1, keepdim=True)
            jxy = (channel_weights * jxy_c).sum(dim=1, keepdim=True)
            signal_energy = (
                channel_weights * second_c
            ).sum(dim=1, keepdim=True)
        else:
            rms_level = torch.zeros_like(mean_level)
            support_ratio = torch.zeros_like(mean_level)
            jxx = jxx_c.mean(dim=1, keepdim=True)
            jyy = jyy_c.mean(dim=1, keepdim=True)
            jxy = jxy_c.mean(dim=1, keepdim=True)
            signal_energy = second_c.mean(dim=1, keepdim=True)

        trace = (jxx + jyy).clamp_min_(0.0)
        edge_strength = trace / (trace + signal_energy + self.eps)

        # The dominant tensor eigenvector follows the gradient normal.  The
        # minus sign rotates its double-angle representation to the tangent,
        # i.e. the support direction used by the directional convolution.
        tangent_q1 = -(jxx - jyy) / (trace + self.eps)
        tangent_q2 = -(2.0 * jxy) / (trace + self.eps)
        coherence = torch.sqrt(
            (tangent_q1.square() + tangent_q2.square()).clamp_min_(0.0)
        ).clamp_(0.0, 1.0)

        return (
            mean_level,
            rms_level,
            normalized_variance,
            support_ratio,
            edge_strength,
            tangent_q1,
            tangent_q2,
            coherence,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "AnalyticStructureDescriptorV2 expects [B,C,H,W]"
            )
        if x.shape[1] <= 0:
            raise ValueError("AnalyticStructureDescriptorV2 requires channels")

        with torch.no_grad():
            with _fp32_context(x):
                value = x.detach().float()
                gx, gy = self._spatial_gradients(value)
                short = self._scale_descriptor(
                    value, gx, gy, self.windows[0]
                )
                long = self._scale_descriptor(
                    value, gx, gy, self.windows[1]
                )

                short_coherence = short[7]
                long_coherence = long[7]
                reliability = torch.sqrt(
                    (short_coherence * long_coherence).clamp_min_(0.0)
                ).clamp_(0.0, 1.0)
                axis_dot = short[5] * long[5] + short[6] * long[6]
                axis_cosine = axis_dot / (
                    short_coherence * long_coherence + self.eps
                )
                axis_cosine = axis_cosine.clamp_(-1.0, 1.0)
                direction_consistency = (
                    0.5 * (axis_cosine + 1.0) * reliability
                )
                scale_edge_contrast = (short[4] - long[4]) / (
                    short[4] + long[4] + self.eps
                )
                structure_uncertainty = torch.maximum(
                    short[4], long[4]
                ) * (1.0 - direction_consistency)

                descriptor = torch.cat(
                    short
                    + long
                    + (
                        direction_consistency,
                        reliability,
                        scale_edge_contrast,
                        structure_uncertainty,
                    ),
                    dim=1,
                )
                descriptor = torch.nan_to_num(
                    descriptor,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                return descriptor.detach()


class SharedMultiScaleDirectionalDWConv(nn.Module):
    """Local, region, and shared-canonical directional depthwise bases."""

    def __init__(self, channels: int, scale_index: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if scale_index not in DIRECTION_DILATIONS_BY_SCALE:
            raise ValueError("scale_index must be one of 1, 2, 3")
        self.channels = int(channels)
        self.scale_index = int(scale_index)
        self.direction_dilations = DIRECTION_DILATIONS_BY_SCALE[scale_index]

        self.local_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            bias=False,
            padding=0,
        )
        self.region_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            groups=channels,
            bias=False,
            padding=0,
        )
        # One trainable tensor is shared by all four axes and both spans.
        self.canonical = nn.Parameter(torch.empty(channels, 2))

        templates = torch.zeros(4, 3, 3, 3, dtype=torch.float32)
        positions = (
            ((1, 0), (1, 1), (1, 2)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 0), (1, 1), (2, 2)),
            ((0, 2), (1, 1), (2, 0)),
        )
        for orientation, orientation_positions in enumerate(positions):
            for coefficient, (row, column) in enumerate(
                orientation_positions
            ):
                templates[orientation, coefficient, row, column] = 1.0
        self.register_buffer("direction_templates", templates)

        bound = 1.0 / math.sqrt(3.0)
        nn.init.uniform_(self.canonical, -bound, bound)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "SharedMultiScaleDirectionalDWConv expects [B,C,H,W]"
            )

    def _directional_kernel(self, orientation: int) -> torch.Tensor:
        if orientation not in (0, 1, 2, 3):
            raise ValueError("orientation must be one of 0, 1, 2, 3")
        coefficients = torch.stack(
            (
                self.canonical[:, 0],
                self.canonical[:, 1],
                self.canonical[:, 0],
            ),
            dim=1,
        )
        template = self.direction_templates[orientation]
        kernel = torch.einsum("cp,pij->cij", coefficients, template)
        return kernel.unsqueeze(1)

    def local(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return self.local_dw(padded)

    def region(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        padded = F.pad(x, (2, 2, 2, 2), mode="replicate")
        return self.region_dw(padded)

    def directional(
        self,
        x: torch.Tensor,
        orientation: int,
        span: int,
    ) -> torch.Tensor:
        self._validate_input(x)
        if span not in (0, 1):
            raise ValueError("span must be 0 (short) or 1 (long)")
        dilation = self.direction_dilations[span]
        padded = F.pad(
            x,
            (dilation, dilation, dilation, dilation),
            mode="replicate",
        )
        return F.conv2d(
            padded,
            self._directional_kernel(orientation),
            padding=0,
            dilation=dilation,
            groups=self.channels,
        )


class DendStructureRoutedConv2dV2(nn.Module):
    """Drop-in Encoder RouteConv with six preregistered V2 modes."""

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
        v2_mode: str = "v2_6",
    ):
        super().__init__()
        del branch_num, detach_reset, deform_groups, kernel_decompose, use_dct
        del use_zero_dilation, v_th, reduction

        if in_channels != out_channels:
            raise ValueError(
                "DendStructureRoutedConv2dV2 requires equal input/output channels"
            )
        if groups != in_channels:
            raise ValueError(
                "DendStructureRoutedConv2dV2 requires depthwise groups"
            )
        if _to_pair(kernel_size) != (3, 3):
            raise ValueError(
                "DendStructureRoutedConv2dV2 requires kernel_size=3"
            )
        if _to_pair(stride) != (1, 1) or _to_pair(dilation) != (1, 1):
            raise ValueError(
                "DendStructureRoutedConv2dV2 requires stride=dilation=1"
            )
        if _to_pair(padding) != (1, 1) or padding_mode != "repeat":
            raise ValueError(
                "DendStructureRoutedConv2dV2 requires replicate padding=1"
            )
        if bias:
            raise ValueError("DendStructureRoutedConv2dV2 requires bias=False")
        if not pre_fs or not calculate_next_k:
            raise ValueError("FrequencySelection and K_next must remain enabled")
        if scale_index not in DIRECTION_DILATIONS_BY_SCALE:
            raise ValueError("scale_index must be one of 1, 2, 3")
        if v2_mode not in V2_MODE_CONFIGS:
            valid_modes = ", ".join(V2_MODE_CONFIGS)
            raise ValueError(
                "Unsupported v2_mode %r; expected one of: %s"
                % (v2_mode, valid_modes)
            )

        mode_config = V2_MODE_CONFIGS[v2_mode]
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.scale_index = int(scale_index)
        self.Down_K = bool(Down_K)
        self.v2_mode = str(v2_mode)
        self.routing_mode = mode_config.routing
        self.directional_spans = int(mode_config.directional_spans)

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
        self.spatial_group = int(spatial_group)
        self.k_map_count = len(cfg.get("k_list", [])) + (
            1 if cfg.get("lowfreq_att", False) else 0
        )
        if self.k_map_count <= 0:
            raise ValueError("FrequencySelection must expose at least one K map")
        self.freq_weight_conv = nn.Conv2d(
            in_channels,
            self.k_map_count * self.spatial_group,
            kernel_size=3,
            stride=2 if self.Down_K else 1,
            padding=1,
            groups=self.spatial_group,
            bias=False,
        )

        robust = mode_config.descriptor == "robust"
        self.descriptor = AnalyticStructureDescriptorV2(
            scale_index=scale_index,
            robust=robust,
        )
        router_channels = {
            "flat6": 6,
            "factorized_single": 7,
            "flat10": 10,
            "factorized_dual_uniform": 9,
            "factorized_dual": 9,
        }[self.routing_mode]

        # Construct every shared trainable component before the mode-dependent
        # router.  This keeps their initial values identical under the same
        # seed even when router output widths differ across causal comparisons.
        self.spatial_bases = SharedMultiScaleDirectionalDWConv(
            channels=in_channels,
            scale_index=scale_index,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.router = nn.Conv2d(
            AnalyticStructureDescriptorV2.output_channels,
            router_channels,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(self.freq_weight_conv.weight)
        self._initialize_router()

    def _initialize_router(self) -> None:
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        with torch.no_grad():
            if self.routing_mode == "factorized_single":
                self.router.bias[:3].copy_(
                    torch.log(
                        torch.tensor(
                            [1.0, 4.0, 1.0],
                            dtype=self.router.bias.dtype,
                            device=self.router.bias.device,
                        )
                    )
                )
            elif self.routing_mode in (
                "factorized_dual_uniform",
                "factorized_dual",
            ):
                self.router.bias[:3].copy_(
                    torch.log(
                        torch.tensor(
                            [1.0, 8.0, 1.0],
                            dtype=self.router.bias.dtype,
                            device=self.router.bias.device,
                        )
                    )
                )

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

    def _router_logits(self, structure: torch.Tensor) -> torch.Tensor:
        if structure.ndim != 4 or structure.shape[1] != 20:
            raise ValueError("structure must have shape [B,20,H,W]")
        with _fp32_context(structure):
            return F.conv2d(
                structure.detach().float(),
                self.router.weight.float(),
                None if self.router.bias is None else self.router.bias.float(),
                stride=1,
                padding=0,
            )

    def routing_weights(
        self,
        structure: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return normalized routing heads for diagnostics and fusion."""
        logits = self._router_logits(structure)
        if self.routing_mode in ("flat6", "flat10"):
            return {"flat": torch.softmax(logits, dim=1)}

        shape = torch.softmax(logits[:, 0:3], dim=1)
        orientation = torch.softmax(logits[:, 3:7], dim=1)
        heads = {"shape": shape, "orientation": orientation}
        if self.routing_mode in (
            "factorized_dual_uniform",
            "factorized_dual",
        ):
            scale_logits = logits[:, 7:9]
            if self.routing_mode == "factorized_dual_uniform":
                scale_logits = scale_logits * 0.0
            heads["scale"] = torch.softmax(scale_logits, dim=1)
        return heads

    def effective_branch_weights(self, structure: torch.Tensor) -> torch.Tensor:
        """Return gates ordered as SIX_BRANCH_NAMES or TEN_BRANCH_NAMES."""
        heads = self.routing_weights(structure)
        if "flat" in heads:
            return heads["flat"]

        shape = heads["shape"]
        orientation = heads["orientation"]
        effective = [shape[:, 0:1]]
        if self.directional_spans == 1:
            for orientation_index in range(4):
                effective.append(
                    shape[:, 1:2]
                    * orientation[:, orientation_index:orientation_index + 1]
                )
        else:
            scale = heads["scale"]
            for orientation_index in range(4):
                for span_index in range(2):
                    effective.append(
                        shape[:, 1:2]
                        * orientation[
                            :, orientation_index:orientation_index + 1
                        ]
                        * scale[:, span_index:span_index + 1]
                    )
        effective.append(shape[:, 2:3])
        return torch.cat(effective, dim=1)

    def _fuse_spatial_bases(
        self,
        x_spike: torch.Tensor,
        structure: torch.Tensor,
    ) -> torch.Tensor:
        gates = self.effective_branch_weights(structure).to(
            dtype=x_spike.dtype
        )

        fused = gates[:, 0:1] * self.spatial_bases.local(x_spike)
        gate_index = 1
        for orientation in range(4):
            for span in range(self.directional_spans):
                directional = self.spatial_bases.directional(
                    x_spike,
                    orientation=orientation,
                    span=span,
                )
                fused = fused + gates[:, gate_index:gate_index + 1] * directional
                gate_index += 1
        fused = fused + gates[:, gate_index:gate_index + 1] * (
            self.spatial_bases.region(x_spike)
        )
        return fused

    def extra_repr(self) -> str:
        return "scale_index=%d, v2_mode=%r" % (
            self.scale_index,
            self.v2_mode,
        )

    def forward(self, x: torch.Tensor, K=None, return_k: bool = True):
        if x.ndim != 5:
            raise ValueError(
                "DendStructureRoutedConv2dV2 expects [N,B,C,H,W]"
            )
        phases, batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                "Expected %d input channels, got %d"
                % (self.in_channels, channels)
            )

        # This block intentionally preserves the V1 frequency/LIF/K order.
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
        fused = self._fuse_spatial_bases(x_spike, structure)
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
    "AnalyticStructureDescriptorV2",
    "SharedMultiScaleDirectionalDWConv",
    "DendStructureRoutedConv2dV2",
    "V2_MODE_CONFIGS",
    "SIX_BRANCH_NAMES",
    "TEN_BRANCH_NAMES",
]
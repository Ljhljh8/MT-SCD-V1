"""State--decision separated Structure-Routed Convolution V4.

V4 preserves the V1 full spatial response path while removing the legacy
``x_spike -> K_next`` producer from the route operator.  The operator consumes
an optional frequency decision produced by the previous scale and exposes the
parameter-free V1 analytic descriptor as ``structure_state``.  Task-specific
frequency and relation decisions are owned by the surrounding Adapter and
FDPCEncoder respectively.

Public operator interface::

    route_residual, structure_state = route_conv(x, K_freq_in)

where ``x`` is ``[N,B,C,H,W]`` and ``structure_state`` is
``[N,B,6,H,W]``.  No frequency or relation decision head is registered in
the route operator; the inherited spatial router remains part of its response
path.
"""

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
V4_FREQUENCY_BANDS = (2, 4, 8)
V4_STRUCTURE_CHANNELS = 6


def _to_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


@contextmanager
def _fp32_context(value: torch.Tensor):
    """Disable CUDA autocast only for the explicitly frozen FP32 islands."""
    if value.is_cuda:
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


def _consume_legacy_frequency_weight_init_rng(
    in_channels: int,
    map_count: int,
    spatial_group: int,
) -> None:
    """Advance RNG exactly as V1's removed bias-free 3x3 Conv2d would.

    V1 constructs ``freq_weight_conv`` before its router and spatial bases and
    then overwrites that convolution with zeros.  Omitting the layer outright
    would therefore change every later common initialization.  V4 consumes the
    same Kaiming-uniform random sequence on an unregistered temporary tensor;
    it never instantiates or retains a legacy frequency-head module.
    """
    weight = torch.empty(
        int(map_count) * int(spatial_group),
        int(in_channels) // int(spatial_group),
        3,
        3,
    )
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5.0))


class AnalyticStructureDescriptorV4(nn.Module):
    """The parameter-free six-channel analytic descriptor used by V1 full."""

    output_channels = V4_STRUCTURE_CHANNELS

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
            raise ValueError(
                "AnalyticStructureDescriptorV4 expects [N*B,C,H,W], got %r"
                % (tuple(x.shape),)
            )

        with _fp32_context(x):
            value = x.float()
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

            return torch.cat(
                (mu, nu, edge_energy, q1, q2, anisotropy),
                dim=1,
            )


class SharedAxialDirectionalDWConvV4(nn.Module):
    """V1 local, four directional, and region response bases."""

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
            raise ValueError(
                "SharedAxialDirectionalDWConvV4 expects [N*B,C,H,W]"
            )

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


class DendriticFrequencyHead(nn.Module):
    """V4 scale-specific readout from analytic state to three band gains."""

    def __init__(
        self,
        k_list: Sequence[int] = V4_FREQUENCY_BANDS,
        stride: int = 2,
    ):
        super().__init__()
        if tuple(int(k) for k in k_list) != V4_FREQUENCY_BANDS:
            raise ValueError(
                "V4 Frequency Head requires k_list=%r, got %r"
                % (V4_FREQUENCY_BANDS, tuple(k_list))
            )
        if int(stride) != 2:
            raise ValueError("V4 Frequency Head requires stride=2")

        # Conv2d construction consumes RNG even though its weight is then zero.
        # Forking preserves the public construction stream for controlled runs.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(
                V4_STRUCTURE_CHANNELS,
                len(V4_FREQUENCY_BANDS),
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            nn.init.zeros_(self.projection.weight)

    def forward(self, structure_state: torch.Tensor):
        if structure_state.ndim != 5:
            raise ValueError(
                "DendriticFrequencyHead expects [N,B,6,H,W], got %r"
                % (tuple(structure_state.shape),)
            )
        N, B, C, H, W = structure_state.shape
        if C != V4_STRUCTURE_CHANNELS:
            raise ValueError(
                "DendriticFrequencyHead expects six state channels, got %d" % C
            )
        if structure_state.requires_grad or structure_state.grad_fn is not None:
            raise RuntimeError("structure_state must be gradient-isolated")

        state_flat = structure_state.reshape(N * B, C, H, W).contiguous()
        with _fp32_context(state_flat):
            logits = self.projection(state_flat.float())
            gains = 2.0 * torch.sigmoid(logits.float())
        if gains.dtype != torch.float32:
            raise RuntimeError("V4 Frequency Head must return FP32 gains")
        return tuple(gains[:, index:index + 1] for index in range(gains.shape[1]))


class DendStructureRoutedConv2dV4(nn.Module):
    """V1-compatible structured response operator with external decisions."""

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
        v_th: float = 1.0,
        reduction: float = 1.0 / 16.0,
        SN_CLS: bool = False,
        scale_index: int = 1,
    ):
        super().__init__()
        del branch_num, detach_reset, deform_groups, kernel_decompose, use_dct
        del use_zero_dilation, v_th, reduction

        if in_channels != out_channels:
            raise ValueError(
                "DendStructureRoutedConv2dV4 requires equal input/output channels"
            )
        if groups != in_channels:
            raise ValueError(
                "DendStructureRoutedConv2dV4 requires depthwise groups"
            )
        if _to_pair(kernel_size) != (3, 3):
            raise ValueError("DendStructureRoutedConv2dV4 requires kernel_size=3")
        if _to_pair(stride) != (1, 1) or _to_pair(dilation) != (1, 1):
            raise ValueError(
                "DendStructureRoutedConv2dV4 requires stride=dilation=1"
            )
        if _to_pair(padding) != (1, 1) or padding_mode != "repeat":
            raise ValueError(
                "DendStructureRoutedConv2dV4 requires replicate padding=1"
            )
        if bias:
            raise ValueError("DendStructureRoutedConv2dV4 requires bias=False")
        if not pre_fs:
            raise ValueError("V4 keeps FrequencySelection enabled at every stage")
        if scale_index not in INTEGRATION_KERNEL_BY_SCALE:
            raise ValueError("scale_index must be one of 1, 2, 3")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.scale_index = int(scale_index)
        self.diagnostic_sink = None
        self.diagnostic_block_index = None

        cfg = dict(
            k_list=list(V4_FREQUENCY_BANDS),
            lowfreq_att=False,
            lp_type="freq",
            act="sigmoid",
            spatial="conv",
            spatial_group=1,
        )
        if fs_cfg is not None:
            cfg.update(fs_cfg)
        if tuple(cfg.get("k_list", ())) != V4_FREQUENCY_BANDS:
            raise ValueError(
                "V4 requires FrequencySelection k_list=%r" % (V4_FREQUENCY_BANDS,)
            )
        if bool(cfg.get("lowfreq_att", False)):
            raise ValueError("V4 requires an identity low-frequency component")
        if int(cfg.get("spatial_group", 1)) != 1:
            raise ValueError("V4 requires spatial_group=1")
        if str(cfg.get("lp_type", "freq")) != "freq":
            raise ValueError("V4 currently requires lp_type='freq'")
        self.fs_cfg = cfg
        self.frequency_selection = FrequencySelection(in_channels, **cfg)
        if SN_CLS:
            self.lif = Q_IFNode(surrogate_function=Quant())

        # Preserve V1 common initialization and the construction-end RNG state
        # without creating the removed legacy frequency decision module.
        _consume_legacy_frequency_weight_init_rng(
            in_channels=self.in_channels,
            map_count=len(V4_FREQUENCY_BANDS),
            spatial_group=1,
        )

        self.descriptor = AnalyticStructureDescriptorV4(
            INTEGRATION_KERNEL_BY_SCALE[scale_index]
        )
        self.router = nn.Conv2d(
            V4_STRUCTURE_CHANNELS,
            6,
            kernel_size=1,
            bias=True,
        )
        self.spatial_bases = SharedAxialDirectionalDWConvV4(
            in_channels,
            DIRECTION_DILATION_BY_SCALE[scale_index],
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def _validate_frequency_input(
        self,
        K_freq_in,
        flattened_batch: int,
        height: int,
        width: int,
        device: torch.device,
    ):
        if K_freq_in is None:
            return None
        if not isinstance(K_freq_in, (tuple, list)):
            raise TypeError("K_freq_in must be None or a tuple/list of three tensors")
        if len(K_freq_in) != len(V4_FREQUENCY_BANDS):
            raise ValueError(
                "K_freq_in must contain exactly three maps for bands %r, got %d"
                % (V4_FREQUENCY_BANDS, len(K_freq_in))
            )

        validated = []
        expected_shape = (flattened_batch, 1, height, width)
        for band, gain in zip(V4_FREQUENCY_BANDS, K_freq_in):
            if not isinstance(gain, torch.Tensor):
                raise TypeError("K_freq_in band %d must be a torch.Tensor" % band)
            if tuple(gain.shape) != expected_shape:
                raise ValueError(
                    "K_freq_in band %d shape must be %r, got %r"
                    % (band, expected_shape, tuple(gain.shape))
                )
            if gain.device != device:
                raise ValueError(
                    "K_freq_in band %d device %s does not match feature device %s"
                    % (band, gain.device, device)
                )
            if not torch.is_floating_point(gain):
                raise TypeError("K_freq_in band %d must have floating dtype" % band)
            if not torch.isfinite(gain).all():
                raise FloatingPointError(
                    "K_freq_in band %d contains NaN or Inf" % band
                )
            validated.append(gain)
        return tuple(validated)

    def _compute_routing_gates(self, structure_state_flat: torch.Tensor) -> torch.Tensor:
        with _fp32_context(structure_state_flat):
            logits = self.router(structure_state_flat.float())
            gates = torch.softmax(logits.float(), dim=1)
        return gates

    def _emit_frequency_diagnostics(
        self,
        x_flat: torch.Tensor,
        K_consumed,
        modulated_output: torch.Tensor,
    ) -> None:
        sink = self.diagnostic_sink
        if sink is None or K_consumed is None:
            return

        with torch.no_grad():
            neutral_output = self.frequency_selection(x_flat, None)
            value = x_flat.float()
            pre_value = value.clone()
            spectrum = torch.fft.fftshift(torch.fft.fft2(value, norm="ortho"))
            high_bands = []
            height, width = value.shape[-2:]
            for frequency in V4_FREQUENCY_BANDS:
                mask = torch.zeros_like(value[:, 0:1])
                h0 = round(height / 2 - height / (2 * frequency))
                h1 = round(height / 2 + height / (2 * frequency))
                w0 = round(width / 2 - width / (2 * frequency))
                w1 = round(width / 2 + width / (2 * frequency))
                mask[:, :, h0:h1, w0:w1] = 1.0
                low_part = torch.fft.ifft2(
                    torch.fft.ifftshift(spectrum * mask),
                    norm="ortho",
                ).real
                high_bands.append(pre_value - low_part)
                pre_value = low_part

            sink.record_frequency_consumption(
                block_index=self.diagnostic_block_index,
                target_scale=self.scale_index,
                bands=V4_FREQUENCY_BANDS,
                gains=tuple(item.float() for item in K_consumed),
                high_bands=tuple(high_bands),
                modulated_output=modulated_output.float(),
                neutral_output=neutral_output.float(),
            )

    def extra_repr(self) -> str:
        return "scale_index=%d, state_channels=%d" % (
            self.scale_index,
            V4_STRUCTURE_CHANNELS,
        )

    def forward(self, x: torch.Tensor, K_freq_in=None):
        if x.ndim != 5:
            raise ValueError(
                "DendStructureRoutedConv2dV4 expects [N,B,C,H,W], got %r"
                % (tuple(x.shape),)
            )
        phases, batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                "Expected %d input channels, got %d"
                % (self.in_channels, channels)
            )

        x_flat = x.flatten(0, 1).contiguous()
        validated_K = self._validate_frequency_input(
            K_freq_in,
            flattened_batch=phases * batch,
            height=height,
            width=width,
            device=x.device,
        )
        K_consumed = None
        if validated_K is not None:
            # This is the only V4 frequency-consumption cast.  Device was
            # already checked and gradients to the Frequency Head are retained.
            K_consumed = tuple(item.to(dtype=x_flat.dtype) for item in validated_K)

        x_dend_flat = self.frequency_selection(x_flat, K_consumed)
        self._emit_frequency_diagnostics(
            x_flat=x_flat,
            K_consumed=K_consumed,
            modulated_output=x_dend_flat,
        )

        x_dend = x_dend_flat.reshape(
            phases,
            batch,
            channels,
            height,
            width,
        ).contiguous()
        if hasattr(self, "lif"):
            x_dend = self.lif(x_dend)
        x_spike = x_dend.flatten(0, 1).contiguous()

        # The no_grad scope is intentionally limited to analytic extraction.
        with torch.no_grad():
            structure_state_flat = self.descriptor(x_spike.detach().float())
        structure_state_flat = structure_state_flat.detach().float().contiguous()
        if structure_state_flat.requires_grad or structure_state_flat.grad_fn is not None:
            raise RuntimeError("V4 analytic structure_state must be gradient-isolated")

        # Router parameters remain trainable because this is outside no_grad.
        gates = self._compute_routing_gates(structure_state_flat)
        gates_consumed = gates.to(dtype=x_spike.dtype)

        responses = self.spatial_bases(x_spike)
        if len(responses) != 6:
            raise RuntimeError("V4 spatial basis must return exactly six responses")
        fused = gates_consumed[:, 0:1] * responses[0]
        for index in range(1, 6):
            fused = fused + gates_consumed[:, index:index + 1] * responses[index]
        route_residual = self.pointwise(fused)
        route_residual = route_residual.reshape(
            phases,
            batch,
            self.out_channels,
            height,
            width,
        ).contiguous()
        structure_state = structure_state_flat.reshape(
            phases,
            batch,
            V4_STRUCTURE_CHANNELS,
            height,  
            width,
        ).contiguous()
        return route_residual, structure_state


__all__ = [
    "AnalyticStructureDescriptorV4",
    "DendriticFrequencyHead",
    "DendStructureRoutedConv2dV4",
    "V4_FREQUENCY_BANDS",
    "V4_STRUCTURE_CHANNELS",
]
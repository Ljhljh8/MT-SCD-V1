"""
Phase-Deformable Context Attention for physical-phase MTSCD features.

Input and output use [N,B,C,H,W], where N is the remote-sensing phase axis.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant


PairName = Tuple[str, str]
AuxDict = Dict[str, Dict[str, Any]]


def _valid_group_count(channels: int, preferred: int = 32) -> int:
    preferred = max(1, int(preferred))
    candidates = [preferred, 32, 16, 8, 4, 2, 1]
    seen = set()
    for g in candidates:
        if g in seen:
            continue
        seen.add(g)
        if channels % g == 0:
            return g
    return 1


def _make_norm2d(channels: int, norm: str = "gn", num_groups: int = 32) -> nn.Module:
    norm = (norm or "none").lower()
    if norm == "gn":
        return nn.GroupNorm(_valid_group_count(channels, num_groups), channels)
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm in ("none", "identity", "id"):
        return nn.Identity()
    raise ValueError("Unsupported norm type: %s" % norm)


def _ensure_pair_tuple(pair: Sequence[str]) -> PairName:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("Each pair must be a 2-tuple/list, got %r" % (pair,))
    return str(pair[0]), str(pair[1])


class StatelessQIF(nn.Module):
    """
    Stateless integer spike-like activation.
    It performs Quant + normalization but does not keep membrane state.
    """

    def __init__(self, capacity: int = 8, v_threshold: float = 1.0):
        super().__init__()
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        if float(v_threshold) <= 0.0:
            raise ValueError("v_threshold must be positive")
        self.capacity = int(capacity)
        self.v_threshold = float(v_threshold)
        self.quant = Quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(x / self.v_threshold) / float(self.capacity)
# class StatelessQIF(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, i, min_value=0, max_value=8): #1111
#         ctx.min = min_value
#         ctx.max = max_value
#         ctx.save_for_backward(i)
#         return torch.round(torch.clamp(i, min=min_value, max=max_value))
#
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         i, = ctx.saved_tensors
#         grad_input[i < ctx.min] = 0
#         grad_input[i > ctx.max] = 0
#         return grad_input, None, None
class PhaseDeformableContextAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        phase_names: Sequence[str],
        context_pairs: Sequence[PairName],
        num_heads: int = 4,
        num_points: int = 4,
        offset_radius: float = 4.0,
        hidden_channels: Optional[int] = None,
        norm: str = "gn",
        norm_groups: int = 32,
        use_q_if_value: bool = True,
        use_q_if_heads: bool = True,
        residual_init: float = 1e-3,
        detach_offsets: bool = False,
        return_aux_default: bool = False,
        use_stateful_q_if: bool = True,
        use_stateless_integer_activation: bool = True,
        use_null_source: bool = True,

        alpha = 1e-3,
            # Dendritic logit prior.
            # V1:
            #   source          : source-level dendritic logit bias.
            # V2 legacy/debug:
            #   offset_sim      : offset-aligned sim prior.
            #   offset_dual     : legacy sim + diff prior.
            # V2.1 recommended:
            #   offset_residual : source-anchored point residual prior.
            pdca_dend_prior_mode: str = "offset_residual",
            pdca_dend_prior_alpha: float = 1e-3,
            pdca_dend_prior_detach: bool = True,
            pdca_dend_prior_descriptor: str = "mean_std",
            pdca_dend_prior_normalize: str = "zscore",

            # V2.1 source-anchor / point-residual weights.
            pdca_dend_prior_source_weight: float = 1.0,
            pdca_dend_prior_point_weight: float = 0.25,

            # Legacy/debug weights for offset_sim / offset_dual.
            pdca_dend_prior_sim_weight: float = 1.0,
            pdca_dend_prior_diff_weight: float = 0.25,

            pdca_dend_prior_use_conf_gate: bool = True,
            pdca_dend_prior_conf_beta: float = 4.0,
            pdca_dend_prior_conf_tau: float = 0.10,

            # V2.1: soft decay for unreliable large-offset dendritic sampling.
            pdca_dend_prior_use_offset_gate: bool = True,

            # V2.1: point prior is centered across Kp within the same source.
            pdca_dend_prior_center_point: bool = True,

            # Clip prior before multiplying alpha.
            pdca_dend_prior_clip: float = 2.0,

            pdca_dend_prior_affect_null: bool = False,
            pdca_dend_prior_stats: bool = True,

    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.offset_radius = float(offset_radius)
        self.detach_offsets = bool(detach_offsets)
        self.return_aux_default = bool(return_aux_default)
        self.use_null_source = bool(use_null_source)


        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        if self.channels % self.num_heads != 0:
            raise ValueError("channels=%d must be divisible by num_heads=%d" % (self.channels, self.num_heads))

        self.phase_names = tuple(str(name) for name in phase_names)
        if len(self.phase_names) == 0:
            raise ValueError("phase_names must not be empty")
        self.phase_to_index = {name: idx for idx, name in enumerate(self.phase_names)}
        if len(self.phase_to_index) != len(self.phase_names):
            raise ValueError("phase_names must be unique")
        self.act = Q_IFNode(surrogate_function=Quant())
        self.context_pairs = tuple(_ensure_pair_tuple(pair) for pair in context_pairs)
        pair_keys = set()
        for a, b in self.context_pairs:
            if a not in self.phase_to_index or b not in self.phase_to_index:
                raise ValueError("Unknown phase pair %r for phase_names=%r" % ((a, b), self.phase_names))
            if a == b:
                raise ValueError("Self pair is not allowed: %r" % ((a, b),))
            pair_keys.add(frozenset((a, b)))

        self.source_names_by_target = {}
        for target_name in self.phase_names:
            source_names = [
                name
                for name in self.phase_names
                if name != target_name and frozenset((target_name, name)) in pair_keys
            ]
            if self.use_null_source:
                source_names.append("__null__")
            self.source_names_by_target[target_name] = tuple(source_names)

        hidden = int(hidden_channels) if hidden_channels is not None else max(32, self.channels // 2)
        if hidden <= 0:
            raise ValueError("hidden_channels must be positive")

        control_activation = self._make_control_activation(bool(use_stateful_q_if), bool(use_q_if_heads))
        value_activation = self._make_value_activation(
            bool(use_stateful_q_if),
            bool(use_stateless_integer_activation),
            bool(use_q_if_value),
        )
        self.QLIF_offset = Q_IFNode(surrogate_function=Quant())
        self.offset_head = nn.Sequential(
            nn.Conv2d(4 * self.channels, hidden, kernel_size=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            control_activation,
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            self._make_control_activation(bool(use_stateful_q_if), bool(use_q_if_heads)),
            nn.Conv2d(hidden, self.num_heads * self.num_points * 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

        valid_dend_prior_modes = (
            "none",
            "source",
            "offset_sim",
            "offset_dual",
            "offset_residual",
        )
        self.pdca_dend_prior_mode = str(pdca_dend_prior_mode)
        if self.pdca_dend_prior_mode not in valid_dend_prior_modes:
            raise ValueError(
                "pdca_dend_prior_mode must be one of %r, got %r"
                % (valid_dend_prior_modes, self.pdca_dend_prior_mode)
            )

        valid_dend_descriptor_modes = ("mean", "mean_std", "raw")
        self.pdca_dend_prior_descriptor = str(pdca_dend_prior_descriptor)
        if self.pdca_dend_prior_descriptor not in valid_dend_descriptor_modes:
            raise ValueError(
                "pdca_dend_prior_descriptor must be one of %r, got %r"
                % (valid_dend_descriptor_modes, self.pdca_dend_prior_descriptor)
            )

        valid_dend_norm_modes = ("none", "zscore")
        self.pdca_dend_prior_normalize = str(pdca_dend_prior_normalize)
        if self.pdca_dend_prior_normalize not in valid_dend_norm_modes:
            raise ValueError(
                "pdca_dend_prior_normalize must be one of %r, got %r"
                % (valid_dend_norm_modes, self.pdca_dend_prior_normalize)
            )

        self.pdca_dend_prior_detach = bool(pdca_dend_prior_detach)

        self.pdca_dend_prior_source_weight = float(pdca_dend_prior_source_weight)
        self.pdca_dend_prior_point_weight = float(pdca_dend_prior_point_weight)

        self.pdca_dend_prior_sim_weight = float(pdca_dend_prior_sim_weight)
        self.pdca_dend_prior_diff_weight = float(pdca_dend_prior_diff_weight)

        self.pdca_dend_prior_use_conf_gate = bool(pdca_dend_prior_use_conf_gate)
        self.pdca_dend_prior_conf_beta = float(pdca_dend_prior_conf_beta)
        self.pdca_dend_prior_conf_tau = float(pdca_dend_prior_conf_tau)

        self.pdca_dend_prior_use_offset_gate = bool(pdca_dend_prior_use_offset_gate)
        self.pdca_dend_prior_center_point = bool(pdca_dend_prior_center_point)
        self.pdca_dend_prior_clip = float(pdca_dend_prior_clip)

        self.pdca_dend_prior_affect_null = bool(pdca_dend_prior_affect_null)
        self.pdca_dend_prior_stats = bool(pdca_dend_prior_stats)

        if self.pdca_dend_prior_source_weight < 0:
            raise ValueError("pdca_dend_prior_source_weight must be non-negative")
        if self.pdca_dend_prior_point_weight < 0:
            raise ValueError("pdca_dend_prior_point_weight must be non-negative")
        if self.pdca_dend_prior_sim_weight < 0:
            raise ValueError("pdca_dend_prior_sim_weight must be non-negative")
        if self.pdca_dend_prior_diff_weight < 0:
            raise ValueError("pdca_dend_prior_diff_weight must be non-negative")
        if self.pdca_dend_prior_conf_beta <= 0:
            raise ValueError("pdca_dend_prior_conf_beta must be positive")
        if self.pdca_dend_prior_conf_tau < 0:
            raise ValueError("pdca_dend_prior_conf_tau must be non-negative")
        if self.pdca_dend_prior_clip < 0:
            raise ValueError("pdca_dend_prior_clip must be non-negative")

        # Learnable scalar prior strength. It is clamped to non-negative in forward.
        self.alpha = nn.Parameter(torch.tensor(float(pdca_dend_prior_alpha)))
        self.attn_head = nn.Sequential(
            nn.Conv2d(4 * self.channels, hidden, kernel_size=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            self._make_control_activation(bool(use_stateful_q_if), bool(use_q_if_heads)),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            self._make_control_activation(bool(use_stateful_q_if), bool(use_q_if_heads)),
            nn.Conv2d(hidden, self.num_heads * self.num_points, kernel_size=1, bias=True),
        )

        self.null_logit_head = nn.Sequential(
            nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            Q_IFNode(surrogate_function=Quant()),   #nn.GELU(),
            nn.Conv2d(hidden, self.num_heads * self.num_points, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.null_logit_head[-1].weight)
        nn.init.zeros_(self.null_logit_head[-1].bias)

        self.value_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            _make_norm2d(self.channels, norm=norm, num_groups=norm_groups),
            value_activation,
        )
        self.out_proj = nn.Sequential(
            Q_IFNode(surrogate_function=Quant()),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            _make_norm2d(self.channels, norm=norm, num_groups=norm_groups),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    @staticmethod
    def _make_control_activation(use_stateful_q_if: bool, use_q_if_heads: bool) -> nn.Module:
        if use_stateful_q_if and use_q_if_heads:
            # WARNING: stateful Q_IFNode may leak membrane state across directed pairs
            # unless pair-wise reset is applied.
            return Q_IFNode(surrogate_function=Quant())
        return nn.GELU()

    @staticmethod
    def _make_value_activation(
        use_stateful_q_if: bool,
        use_stateless_integer_activation: bool,
        use_q_if_value: bool,
    ) -> nn.Module:
        if not use_q_if_value:
            return nn.GELU()
        if use_stateful_q_if:
            # WARNING: stateful Q_IFNode may leak membrane state across directed pairs
            # unless pair-wise reset is applied.
            return Q_IFNode(surrogate_function=Quant())
        if use_stateless_integer_activation:
            return StatelessQIF()
        return nn.GELU()

    @staticmethod
    def _new_aux() -> AuxDict:
        return {
            "source_weights": {},
        }

    @staticmethod
    def _maybe_detach(x: torch.Tensor, detach: bool) -> torch.Tensor:
        return x.detach() if detach else x

    def _deformable_sample_vectorized(self, value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("value must be [B,C,H,W], got %r" % (tuple(value.shape),))
        if offset.ndim != 6:
            raise ValueError("offset must be [B,G,K,2,H,W], got %r" % (tuple(offset.shape),))

        B, C, H, W = value.shape
        Bo, G, K, two, Ho, Wo = offset.shape
        if B != Bo or H != Ho or W != Wo or two != 2:
            raise ValueError("value/offset shape mismatch: %r vs %r" % (tuple(value.shape), tuple(offset.shape)))
        if G != self.num_heads or K != self.num_points:
            raise ValueError("offset groups/points mismatch: %r" % (tuple(offset.shape),))
        if C % G != 0:
            raise ValueError("C=%d must be divisible by G=%d" % (C, G))

        Cg = C // G
        value_g = value.view(B, G, Cg, H, W).unsqueeze(2).expand(B, G, K, Cg, H, W)
        value_g = value_g.contiguous().view(B * G * K, Cg, H, W)

        # align_corners=True maps pixel offsets with scale 2/(size-1).
        xs = torch.linspace(-1.0, 1.0, W, dtype=offset.dtype, device=offset.device)
        ys = torch.linspace(-1.0, 1.0, H, dtype=offset.dtype, device=offset.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1).view(1, 1, 1, H, W, 2)

        scale_x = 0.0 if W <= 1 else 2.0 / float(W - 1)
        scale_y = 0.0 if H <= 1 else 2.0 / float(H - 1)
        offset_grid = offset.permute(0, 1, 2, 4, 5, 3).contiguous()
        offset_grid_x = offset_grid[..., 0] * scale_x
        offset_grid_y = offset_grid[..., 1] * scale_y
        grid = base_grid + torch.stack((offset_grid_x, offset_grid_y), dim=-1)
        grid = grid.view(B * G * K, H, W, 2)

        sampled = F.grid_sample(
            value_g,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(B, G, K, Cg, H, W)



    # def _apply_dendritic_logit_prior(self, logits, target_idx, source_idx, offset, dendritic_guidance, H, W):
    #
    #     D_t = dendritic_guidance[target_idx]  # [B,1,H,W]
    #     D_s = dendritic_guidance[source_idx]  # [B,1,H,W]
    #
    #     prior = -torch.abs(D_t - D_s)  # [B,1,H,W]
    #     logits = logits + self.alpha * prior.unsqueeze(1)
    #     return logits
    def _prepare_dendritic_descriptor(
        self,
        K_GATE,
        N: int,
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Convert dendritic K maps to a compact phase-wise descriptor.

        Supported input:
            K_GATE: list/tuple of Tensor, each usually [N*B, Ck, Hk, Wk]
            K_GATE: Tensor [N,B,Ck,Hk,Wk]
            K_GATE: Tensor [N*B,Ck,Hk,Wk]

        Return:
            descriptor: [N,B,Cd,H,W]
        """
        if self.pdca_dend_prior_mode == "none":
            return None
        if K_GATE is None:
            return None

        if isinstance(K_GATE, (list, tuple)):
            tensors = [item for item in K_GATE if torch.is_tensor(item)]
            if len(tensors) == 0:
                return None
            kg = torch.cat(tensors, dim=1)
        elif torch.is_tensor(K_GATE):
            kg = K_GATE
        else:
            return None

        kg = kg.to(device=device, dtype=dtype)

        if kg.ndim == 5:
            if kg.shape[0] != N or kg.shape[1] != B:
                raise ValueError(
                    "K_GATE [N,B,C,H,W] shape mismatch: "
                    f"got {tuple(kg.shape)}, expected N={N}, B={B}"
                )
            desc = kg
        elif kg.ndim == 4:
            if kg.shape[0] != N * B:
                raise ValueError(
                    "K_GATE [N*B,C,H,W] shape mismatch: "
                    f"got first dim={kg.shape[0]}, expected {N * B}"
                )
            desc = kg.reshape(N, B, kg.shape[1], kg.shape[2], kg.shape[3]).contiguous()
        else:
            raise ValueError("K_GATE must be Tensor/list with 4D or 5D tensors")

        if desc.shape[-2:] != (H, W):
            desc = desc.flatten(0, 1)
            desc = F.interpolate(desc, size=(H, W), mode="bilinear", align_corners=False)
            desc = desc.reshape(N, B, desc.shape[1], H, W).contiguous()

        if self.pdca_dend_prior_detach:
            desc = desc.detach()

        if self.pdca_dend_prior_descriptor == "mean":
            desc = desc.mean(dim=2, keepdim=True)
        elif self.pdca_dend_prior_descriptor == "mean_std":
            mean = desc.mean(dim=2, keepdim=True)
            var = desc.float().var(dim=2, keepdim=True, unbiased=False).to(dtype=dtype)
            std = torch.sqrt(var.clamp_min(1e-6))
            desc = torch.cat([mean, std], dim=2)
        elif self.pdca_dend_prior_descriptor == "raw":
            pass
        else:
            raise ValueError("Unsupported pdca_dend_prior_descriptor")

        if self.pdca_dend_prior_normalize == "zscore":
            stat_mean = desc.float().mean(dim=(2, 3, 4), keepdim=True).to(dtype=dtype)
            stat_std = desc.float().std(dim=(2, 3, 4), keepdim=True, unbiased=False).to(dtype=dtype)
            desc = (desc - stat_mean) / stat_std.clamp_min(1e-6)
        elif self.pdca_dend_prior_normalize == "none":
            pass
        else:
            raise ValueError("Unsupported pdca_dend_prior_normalize")

        if not torch.isfinite(desc.detach()).all():
            raise FloatingPointError("Dendritic descriptor contains NaN/Inf")

        return desc

    def _sample_dendritic_descriptor(
        self,
        dend_source: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample dendritic descriptor using the same PDCA offsets.

        Args:
            dend_source: [B,Cd,H,W]
            offset:      [B,G,K,2,H,W]

        Return:
            sampled:     [B,G,K,Cd,H,W]
        """
        if dend_source.ndim != 4:
            raise ValueError(
                "dend_source must be [B,Cd,H,W], got %r"
                % (tuple(dend_source.shape),)
            )
        if offset.ndim != 6:
            raise ValueError(
                "offset must be [B,G,K,2,H,W], got %r"
                % (tuple(offset.shape),)
            )

        B, Cd, H, W = dend_source.shape
        Bo, G, K, two, Ho, Wo = offset.shape
        if B != Bo or H != Ho or W != Wo or two != 2:
            raise ValueError(
                "dend_source/offset shape mismatch: %r vs %r"
                % (tuple(dend_source.shape), tuple(offset.shape))
            )

        dend_g = (
            dend_source
            .view(B, 1, 1, Cd, H, W)
            .expand(B, G, K, Cd, H, W)
            .contiguous()
            .view(B * G * K, Cd, H, W)
        )

        xs = torch.linspace(-1.0, 1.0, W, dtype=offset.dtype, device=offset.device)
        ys = torch.linspace(-1.0, 1.0, H, dtype=offset.dtype, device=offset.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1).view(1, 1, 1, H, W, 2)

        scale_x = 0.0 if W <= 1 else 2.0 / float(W - 1)
        scale_y = 0.0 if H <= 1 else 2.0 / float(H - 1)
        offset_grid = offset.permute(0, 1, 2, 4, 5, 3).contiguous()
        offset_grid_x = offset_grid[..., 0] * scale_x
        offset_grid_y = offset_grid[..., 1] * scale_y
        grid = base_grid + torch.stack((offset_grid_x, offset_grid_y), dim=-1)
        grid = grid.view(B * G * K, H, W, 2)

        sampled = F.grid_sample(
            dend_g,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return sampled.view(B, G, K, Cd, H, W)

    def _target_dend_confidence(self, dend_target: torch.Tensor) -> torch.Tensor:
        """
        Estimate target structural confidence from descriptor dispersion.

        Args:
            dend_target: [B,Cd,H,W]

        Return:
            confidence:  [B,1,1,H,W], broadcastable to [B,G,K,H,W]
        """
        if dend_target.shape[1] <= 1:
            raw_conf = dend_target.abs().mean(dim=1, keepdim=True)
        else:
            raw_conf = dend_target.float().std(dim=1, keepdim=True, unbiased=False).to(
                dtype=dend_target.dtype
            )

        conf = torch.sigmoid(
            self.pdca_dend_prior_conf_beta
            * (raw_conf - self.pdca_dend_prior_conf_tau)
        )
        return conf.unsqueeze(1)

    def _apply_dendritic_logit_prior(
        self,
        logits: torch.Tensor,
        target_idx: int,
        source_idx: int,
        offset: torch.Tensor,
        dendritic_descriptor: Optional[torch.Tensor],
    ):
        """
        Apply source-level V1 or offset-aligned V2 dendritic logit prior.

        Args:
            logits:               [B,G,K,H,W]
            offset:               [B,G,K,2,H,W]
            dendritic_descriptor: [N,B,Cd,H,W]

        Returns:
            logits_new: [B,G,K,H,W]
            stats: dict[str, Tensor]
        """
        if self.pdca_dend_prior_mode == "none" or dendritic_descriptor is None:
            return logits, None

        D_t = dendritic_descriptor[target_idx]  # [B,Cd,H,W]
        D_s = dendritic_descriptor[source_idx]  # [B,Cd,H,W]

        stats = {}

        if self.pdca_dend_prior_mode == "source":
            # V1-compatible source-level prior.
            diff = (D_t - D_s).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            prior = -diff.unsqueeze(1)  # [B,1,1,H,W]
            prior = prior.expand_as(logits)
            sim = None

        elif self.pdca_dend_prior_mode in ("offset_sim", "offset_dual"):
            # V2: use the same offsets as PDCA source feature sampling.
            D_s_sampled = self._sample_dendritic_descriptor(D_s, offset)  # [B,G,K,Cd,H,W]
            D_t_expand = D_t.unsqueeze(1).unsqueeze(2)  # [B,1,1,Cd,H,W]

            sim = F.cosine_similarity(
                D_t_expand.float(),
                D_s_sampled.float(),
                dim=3,
                eps=1e-6,
            ).to(dtype=logits.dtype)  # [B,G,K,H,W]

            diff = (D_t_expand - D_s_sampled).abs().mean(dim=3)  # [B,G,K,H,W]
            diff = torch.tanh(diff.float()).to(dtype=logits.dtype)

            if self.pdca_dend_prior_mode == "offset_sim":
                prior = self.pdca_dend_prior_sim_weight * sim
            else:
                # Dual prior:
                # sim rewards structural consistency;
                # diff softly allows structural-change evidence.
                prior = (
                    self.pdca_dend_prior_sim_weight * sim
                    + self.pdca_dend_prior_diff_weight * diff
                )

            if self.pdca_dend_prior_use_conf_gate:
                conf = self._target_dend_confidence(D_t)  # [B,1,1,H,W]
                prior = prior * conf

        else:
            raise ValueError("Unsupported pdca_dend_prior_mode")

        logits_new = logits + self.alpha.to(dtype=logits.dtype) * prior.to(dtype=logits.dtype)

        if self.pdca_dend_prior_stats:
            prior_det = prior.detach().float()
            stats["prior_abs_mean"] = prior_det.abs().mean()
            stats["prior_mean"] = prior_det.mean()
            stats["prior_std"] = prior_det.std(unbiased=False)
            stats["alpha"] = self.alpha.detach().float()
            if sim is not None:
                stats["sim_mean"] = sim.detach().float().mean()
            stats["diff_mean"] = diff.detach().float().mean()

        return logits_new, stats
    def forward(
        self,
        feat: torch.Tensor,
        K_GATE: torch.Tensor,
        return_aux: Optional[bool] = None,
        detach_aux: bool = False,
    ):
        if return_aux is None:
            return_aux = self.return_aux_default
        collect_aux = bool(return_aux)
        pre_feat = feat
        if feat.ndim != 5:
            raise ValueError("PhaseDeformableContextAttention expects [N,B,C,H,W], got %r" % (tuple(feat.shape),))
        N, B, C, H, W = feat.shape
        feat = self.act(feat)
        if N != len(self.phase_names):
            raise ValueError("N=%d does not match phase_names=%r" % (N, self.phase_names))
        if C != self.channels:
            raise ValueError("Expected C=%d, got C=%d" % (self.channels, C))

        aux = self._new_aux() if collect_aux else {}
        residual_by_phase = [torch.zeros_like(feat[idx]) for idx in range(N)]
        Cg = C // self.num_heads
        dendritic_descriptor = self._prepare_dendritic_descriptor(
            K_GATE=K_GATE,
            N=N,
            B=B,
            H=H,
            W=W,
            device=feat.device,
            dtype=feat.dtype,
        )

        for target_idx, target_name in enumerate(self.phase_names):
            target = feat[target_idx]
            logits_by_source = []
            sampled_by_source = []
            source_names = self.source_names_by_target[target_name]

            for src_name in source_names:
                if src_name == "__null__":
                    null_logits = self.null_logit_head(target).view(B, self.num_heads, self.num_points, H, W)
                    null_value = target.new_zeros(B, self.num_heads, self.num_points, Cg, H, W)
                    logits_by_source.append(null_logits)
                    sampled_by_source.append(null_value)
                    continue

                src_idx = self.phase_to_index[src_name]
                source = feat[src_idx]
                evidence = torch.cat(
                    [target, source, torch.abs(target - source), source - target],
                    dim=1,
                )

                offset = self.offset_head(evidence).view(B, self.num_heads, self.num_points, 2, H, W)
                offset = torch.tanh(offset) * self.offset_radius
                if self.detach_offsets:
                    offset = offset.detach()
                # dendritic_guidance = torch.cat(K_GATE, dim=1).reshape(N, B, -1, H, W).mean(dim=2,
                #                                                                                        keepdim=True)
                logits = self.attn_head(evidence).view(B, self.num_heads, self.num_points, H, W)
                logits, _ = self._apply_dendritic_logit_prior(
                    logits=logits,
                    target_idx=target_idx,
                    source_idx=src_idx,
                    offset=offset,
                    dendritic_descriptor=dendritic_descriptor,
                )
                # logits = self._apply_dendritic_logit_prior(
                #     logits=logits,
                #     target_idx=target_idx,
                #     source_idx=src_idx,
                #     offset=offset,
                #     dendritic_guidance=dendritic_guidance,
                #     H=H,
                #     W=W,
                # )
                value = self.value_proj(source)
                sampled = self._deformable_sample_vectorized(value, offset)




                logits_by_source.append(logits)
                sampled_by_source.append(sampled)
            if len(logits_by_source) == 0:
                continue

            logits_stacked = torch.stack(logits_by_source, dim=2)
            Q = logits_stacked.shape[2]
            joint_weights = torch.softmax(
                logits_stacked.view(B, self.num_heads, Q * self.num_points, H, W),
                dim=2,
            ).view(
                B, self.num_heads, Q, self.num_points, H, W
            )

            sampled_stacked = torch.stack(sampled_by_source, dim=2)

            if not torch.isfinite(joint_weights).all():
                raise FloatingPointError("PDCA joint_weights has NaN/Inf before context aggregation")
            if not torch.isfinite(sampled_stacked).all():
                raise FloatingPointError("PDCA sampled_stacked has NaN/Inf before context aggregation")

            context = (joint_weights.unsqueeze(4) * sampled_stacked).sum(dim=(2, 3))

            if not torch.isfinite(context).all():
                raise FloatingPointError("PDCA context has NaN/Inf after aggregation")

            context = context.contiguous().view(B, C, H, W)
            residual_by_phase[target_idx] = residual_by_phase[target_idx] + self.out_proj(context)

            if collect_aux:
                aux["source_weights"][target_name] = self._maybe_detach(joint_weights.sum(dim=3), bool(detach_aux))

        residual = torch.stack(residual_by_phase, dim=0)
        return pre_feat + self.residual_scale * residual, aux
        # return pre_feat + residual, aux

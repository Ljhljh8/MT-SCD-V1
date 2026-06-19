"""
Phase-Deformable Context Attention for physical-phase MTSCD features.

Input and output use [N,B,C,H,W], where N is the remote-sensing phase axis.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant


PairName = Tuple[str, str]
AuxDict = Dict[str, Dict[str, torch.Tensor]]


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
        use_q_if_heads: bool = False,
        residual_init: float = 1e-3,
        detach_offsets: bool = False,
        return_aux_default: bool = False,
        use_stateful_q_if: bool = False,
        use_stateless_integer_activation: bool = True,
        use_null_source: bool = True,
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
            nn.GELU(),
            nn.Conv2d(hidden, self.num_heads * self.num_points, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.null_logit_head[-1].weight)
        nn.init.zeros_(self.null_logit_head[-1].bias)

        self.value_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            _make_norm2d(self.channels, norm=norm, num_groups=norm_groups),
            value_activation,
        )
        self.out_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
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
            "offsets": {},
            "attn_weights": {},
            "source_weights": {},
            "joint_weights": {},
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

    def forward(
        self,
        feat: torch.Tensor,
        return_aux: Optional[bool] = None,
        detach_aux: bool = False,
    ):
        if return_aux is None:
            return_aux = self.return_aux_default
        collect_aux = bool(return_aux)

        if feat.ndim != 5:
            raise ValueError("PhaseDeformableContextAttention expects [N,B,C,H,W], got %r" % (tuple(feat.shape),))
        N, B, C, H, W = feat.shape
        if N != len(self.phase_names):
            raise ValueError("N=%d does not match phase_names=%r" % (N, self.phase_names))
        if C != self.channels:
            raise ValueError("Expected C=%d, got C=%d" % (self.channels, C))

        aux = self._new_aux() if collect_aux else {}
        residual_by_phase = [torch.zeros_like(feat[idx]) for idx in range(N)]
        Cg = C // self.num_heads

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

                logits = self.attn_head(evidence).view(B, self.num_heads, self.num_points, H, W)
                value = self.value_proj(source)
                sampled = self._deformable_sample_vectorized(value, offset)

                logits_by_source.append(logits)
                sampled_by_source.append(sampled)

                if collect_aux:
                    direction_key = "%s<-%s" % (target_name, src_name)
                    aux["offsets"][direction_key] = self._maybe_detach(offset, bool(detach_aux))

            if len(logits_by_source) == 0:
                continue

            logits_stacked = torch.stack(logits_by_source, dim=2)
            Q = logits_stacked.shape[2]
            joint_weights = torch.softmax(logits_stacked.view(B, self.num_heads, Q * self.num_points, H, W), dim=2)
            joint_weights = joint_weights.view(B, self.num_heads, Q, self.num_points, H, W)

            sampled_stacked = torch.stack(sampled_by_source, dim=2)
            context = (joint_weights.unsqueeze(4) * sampled_stacked).sum(dim=(2, 3))
            context = context.contiguous().view(B, C, H, W)
            residual_by_phase[target_idx] = residual_by_phase[target_idx] + self.out_proj(context)

            if collect_aux:
                aux["source_weights"][target_name] = self._maybe_detach(joint_weights.sum(dim=3), bool(detach_aux))
                aux["joint_weights"][target_name] = self._maybe_detach(joint_weights, bool(detach_aux))
                for q_idx, src_name in enumerate(source_names):
                    if src_name == "__null__":
                        continue
                    direction_key = "%s<-%s" % (target_name, src_name)
                    aux["attn_weights"][direction_key] = self._maybe_detach(
                        joint_weights[:, :, q_idx],
                        bool(detach_aux),
                    )

        residual = torch.stack(residual_by_phase, dim=0)
        return feat + self.residual_scale * residual, aux

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
class StatelessIntegerSpikeSTE(nn.Module):
    def __init__(
        self,
        capacity: int = 8,
        threshold: float = 1.0,
        signed: bool = True,
        detach: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.capacity = int(capacity)
        self.threshold = float(threshold)
        self.signed = bool(signed)
        self.detach = bool(detach)
        self.eps = float(eps)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")

    def extra_repr(self) -> str:
        return (
            f"capacity={self.capacity}, threshold={self.threshold}, "
            f"signed={self.signed}, detach={self.detach}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x.detach()).all():
            raise FloatingPointError("StatelessIntegerSpikeSTE input has NaN/Inf")

        x_in = x.detach() if self.detach else x

        # use fp32 around round/clamp under autocast
        with torch.cuda.amp.autocast(enabled=False):
            xf = x_in.float()
            y = xf / max(self.threshold, self.eps)
            if self.signed:
                y_clip = y.clamp(-self.capacity, self.capacity)
            else:
                y_clip = y.clamp(0, self.capacity)

            y_round = torch.round(y_clip)
            # STE: forward round, backward identity inside clamped branch
            y_ste = y_clip + (y_round - y_clip).detach()
            y_out = y_ste / float(self.capacity)

        return y_out.to(dtype=x.dtype)
class TopKRoutingSTE(nn.Module):
    def __init__(self, topk: int = 2, tau: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.topk = int(topk)
        self.tau = float(tau)
        self.eps = float(eps)
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def forward(self, logits_flat: torch.Tensor):
        """
        logits_flat: [B,G,M,H,W], M=Q*Kp
        returns:
            routing_flat: [B,G,M,H,W]
            soft_flat:    [B,G,M,H,W]
            topk_idx:     [B,G,K,H,W]
        """
        if logits_flat.ndim != 5:
            raise ValueError(f"Expected [B,G,M,H,W], got {tuple(logits_flat.shape)}")
        if not torch.isfinite(logits_flat.detach()).all():
            raise FloatingPointError("TopKRoutingSTE logits_flat has NaN/Inf")
        B, G, M, H, W = logits_flat.shape
        k = min(self.topk, M)

        with torch.cuda.amp.autocast(enabled=False):
            soft = torch.softmax(logits_flat.float() / max(self.tau, self.eps), dim=2)
        topk_idx = soft.topk(k=k, dim=2).indices

        hard = torch.zeros_like(soft)
        hard.scatter_(dim=2, index=topk_idx, value=1.0 / float(k))

        routing = hard.detach() - soft.detach() + soft
        return routing.to(dtype=logits_flat.dtype), soft.to(dtype=logits_flat.dtype), topk_idx
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
        use_relation_aux: bool = False,
        relation_aux_pairs: Optional[Sequence[str]] = None,
        relation_aux_hidden_channels: Optional[int] = None,

        pdca_context_spike_mode: str = "none",
        pdca_context_spike_capacity: int = 8,

        pdca_context_spike_threshold: float = 1.0,
        pdca_context_spike_signed: bool = True,
        pdca_context_spike_detach: bool = False,
        pdca_context_spike_stats: bool = True,

        pdca_context_spike_topk: int = 2,
        pdca_context_spike_tau: float = 1.0,
        pdca_context_spike_warmup_epoch: int = 0,  # 训练脚本控制更合适
        alpha = 1e-3
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.offset_radius = float(offset_radius)
        self.detach_offsets = bool(detach_offsets)
        self.return_aux_default = bool(return_aux_default)
        self.use_null_source = bool(use_null_source)
        self.use_relation_aux = bool(use_relation_aux)

        self.pdca_context_spike_mode = str(pdca_context_spike_mode)
        valid_modes = ("none", "weights", "values", "both", "context")
        if self.pdca_context_spike_mode not in valid_modes:
            raise ValueError(f"Unsupported pdca_context_spike_mode={self.pdca_context_spike_mode}")
        self.pdca_context_spike_runtime_mode = self.pdca_context_spike_mode

        self.pdca_context_spike_capacity = int(pdca_context_spike_capacity)
        self.pdca_context_spike_threshold = float(pdca_context_spike_threshold)
        self.pdca_context_spike_signed = bool(pdca_context_spike_signed)
        self.pdca_context_spike_detach = bool(pdca_context_spike_detach)
        self.pdca_context_spike_stats = bool(pdca_context_spike_stats)
        self.pdca_context_spike_topk = int(pdca_context_spike_topk)
        self.pdca_context_spike_tau = float(pdca_context_spike_tau)
        self._pdca_context_spike_valid_modes = valid_modes
        if self.pdca_context_spike_capacity <= 0:
            raise ValueError("pdca_context_spike_capacity must be positive")
        if self.pdca_context_spike_threshold <= 0:
            raise ValueError("pdca_context_spike_threshold must be positive")
        if self.pdca_context_spike_topk <= 0:
            raise ValueError("pdca_context_spike_topk must be positive")
        if self.pdca_context_spike_tau <= 0:
            raise ValueError("pdca_context_spike_tau must be positive")

        if self.pdca_context_spike_mode in ("values", "both", "context"):
            self.context_spike_act = StatelessIntegerSpikeSTE(
                capacity=self.pdca_context_spike_capacity,
                threshold=self.pdca_context_spike_threshold,
                signed=self.pdca_context_spike_signed,
                detach=self.pdca_context_spike_detach,
            )
        else:
            self.context_spike_act = None

        if self.pdca_context_spike_mode in ("weights", "both"):
            self.context_routing_spike = TopKRoutingSTE(
                topk=self.pdca_context_spike_topk,
                tau=self.pdca_context_spike_tau,
            )
        else:
            self.context_routing_spike = None


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

        valid_direction_keys = {
            "%s<-%s" % (target, source)
            for a, b in self.context_pairs
            for target, source in ((a, b), (b, a))
        }
        if relation_aux_pairs is None:
            self.relation_aux_pairs = None
        else:
            parsed_pairs = tuple(str(pair) for pair in relation_aux_pairs)
            if self.use_relation_aux and not parsed_pairs:
                raise ValueError("relation_aux_pairs must not be empty when use_relation_aux=True")
            if len(set(parsed_pairs)) != len(parsed_pairs):
                raise ValueError("relation_aux_pairs must not contain duplicates")
            for direction_key in parsed_pairs:
                if direction_key.count("<-") != 1:
                    raise ValueError("Invalid relation_aux pair: %r" % direction_key)
                target_name, source_name = direction_key.split("<-")
                if target_name == "__null__" or source_name == "__null__":
                    raise ValueError("relation_aux_pairs must not contain __null__: %r" % direction_key)
                if target_name not in self.phase_to_index or source_name not in self.phase_to_index:
                    raise ValueError("Unknown phase in relation_aux pair: %r" % direction_key)
                if target_name == source_name:
                    raise ValueError("Self relation_aux pair is not allowed: %r" % direction_key)
                if direction_key not in valid_direction_keys:
                    raise ValueError("relation_aux pair is not present in context_pairs: %r" % direction_key)
            self.relation_aux_pairs = frozenset(parsed_pairs)

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

        relation_hidden = (
            int(relation_aux_hidden_channels)
            if relation_aux_hidden_channels is not None
            else hidden
        )
        if relation_hidden <= 0:
            raise ValueError("relation_aux_hidden_channels must be positive")

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
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.attn_head = nn.Sequential(
            nn.Conv2d(4 * self.channels, hidden, kernel_size=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            self._make_control_activation(bool(use_stateful_q_if), bool(use_q_if_heads)),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            self._make_control_activation(bool(use_stateful_q_if), bool(use_q_if_heads)),
            nn.Conv2d(hidden, self.num_heads * self.num_points, kernel_size=1, bias=True),
        )

        self.relation_aux_head = None
        if self.use_relation_aux:
            self.relation_aux_head = nn.Sequential(
                nn.Conv2d(4 * self.channels, relation_hidden, kernel_size=1, bias=False),
                _make_norm2d(relation_hidden, norm=norm, num_groups=norm_groups),
                nn.GELU(),
                nn.Conv2d(relation_hidden, relation_hidden, kernel_size=3, padding=1, bias=False),
                _make_norm2d(relation_hidden, norm=norm, num_groups=norm_groups),
                nn.GELU(),
                nn.Conv2d(relation_hidden, 1, kernel_size=1, bias=True),
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
            "offsets": {},
            "attn_weights": {},
            "source_weights": {},
            "joint_weights": {},
            "relation_logits": {},
            "context_spike": {},
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


    def _apply_dendritic_logit_prior(self, logits, target_idx, source_idx, offset, dendritic_guidance, H, W):

        D_t = dendritic_guidance[target_idx]  # [B,1,H,W]
        D_s = dendritic_guidance[source_idx]  # [B,1,H,W]

        prior = -torch.abs(D_t - D_s)  # [B,1,H,W]
        logits = logits + self.alpha * prior.unsqueeze(1)
        return logits
    def forward(
        self,
        feat: torch.Tensor,
        K_GATE: torch.Tensor,
        return_aux: Optional[bool] = None,
        detach_aux: bool = False,
        relation_aux_only: bool = False,
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
        runtime_mode = getattr(
            self,
            "pdca_context_spike_runtime_mode",
            self.pdca_context_spike_mode,
        )
        if runtime_mode not in self._pdca_context_spike_valid_modes:
            raise ValueError(f"Unsupported pdca_context_spike_runtime_mode={runtime_mode}")
        if runtime_mode in ("weights", "both") and self.context_routing_spike is None:
            raise RuntimeError(
                "PDCA context routing was not constructed for runtime mode %s" % runtime_mode
            )
        if runtime_mode in ("values", "both", "context") and self.context_spike_act is None:
            raise RuntimeError(
                "PDCA integer spike activation was not constructed for runtime mode %s" % runtime_mode
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

                direction_key = "%s<-%s" % (target_name, src_name)
                if (
                    collect_aux
                    and self.relation_aux_head is not None
                    and (self.relation_aux_pairs is None or direction_key in self.relation_aux_pairs)
                ):
                    relation_logit = self.relation_aux_head(evidence)
                    aux["relation_logits"][direction_key] = self._maybe_detach(
                        relation_logit,
                        bool(detach_aux),
                    )

                offset = self.offset_head(evidence).view(B, self.num_heads, self.num_points, 2, H, W)
                offset = torch.tanh(offset) * self.offset_radius
                if self.detach_offsets:
                    offset = offset.detach()
                dendritic_guidance = torch.cat(K_GATE, dim=1).reshape(N, B, -1, H, W).mean(dim=2,
                                                                                                       keepdim=True)
                logits = self.attn_head(evidence).view(B, self.num_heads, self.num_points, H, W)
                logits = self._apply_dendritic_logit_prior(
                    logits=logits,
                    target_idx=target_idx,
                    source_idx=src_idx,
                    offset=offset,
                    dendritic_guidance=dendritic_guidance,
                    H=H,
                    W=W,
                )
                value = self.value_proj(source)
                sampled = self._deformable_sample_vectorized(value, offset)




                logits_by_source.append(logits)
                sampled_by_source.append(sampled)

                if collect_aux and not relation_aux_only:
                    aux["offsets"][direction_key] = self._maybe_detach(offset, bool(detach_aux))

            if len(logits_by_source) == 0:
                continue

            logits_stacked = torch.stack(logits_by_source, dim=2)
            Q = logits_stacked.shape[2]
            # joint_weights = torch.softmax(logits_stacked.view(B, self.num_heads, Q * self.num_points, H, W), dim=2)
            # joint_weights = joint_weights.view(B, self.num_heads, Q, self.num_points, H, W)
            #
            # sampled_stacked = torch.stack(sampled_by_source, dim=2)
            # context = (joint_weights.unsqueeze(4) * sampled_stacked).sum(dim=(2, 3))
            # context = context.contiguous().view(B, C, H, W)
            # residual_by_phase[target_idx] = residual_by_phase[target_idx] + self.out_proj(context)

            logits_flat = logits_stacked.view(
                B, self.num_heads, Q * self.num_points, H, W
            )

            soft_flat = torch.softmax(logits_flat, dim=2)
            soft_joint_weights = soft_flat.view(
                B, self.num_heads, Q, self.num_points, H, W
            )

            joint_weights = soft_joint_weights
            routing_topk_idx = None

            if runtime_mode in ("weights", "both"):
                routing_flat, soft_flat_for_stats, routing_topk_idx = self.context_routing_spike(logits_flat)
                joint_weights = routing_flat.view(
                    B, self.num_heads, Q, self.num_points, H, W
                )
            else:
                soft_flat_for_stats = soft_flat

            sampled_stacked = torch.stack(sampled_by_source, dim=2)

            if runtime_mode in ("values", "both"):
                sampled_stacked_before = sampled_stacked
                sampled_stacked = self.context_spike_act(sampled_stacked)
            else:
                sampled_stacked_before = sampled_stacked

            if not torch.isfinite(joint_weights).all():
                raise FloatingPointError("PDCA joint_weights has NaN/Inf before context aggregation")
            if not torch.isfinite(sampled_stacked).all():
                raise FloatingPointError("PDCA sampled_stacked has NaN/Inf before context aggregation")

            context_before_spike = (joint_weights.unsqueeze(4) * sampled_stacked).sum(dim=(2, 3))

            if runtime_mode == "context":
                context = self.context_spike_act(context_before_spike)
            else:
                context = context_before_spike

            if not torch.isfinite(context).all():
                raise FloatingPointError("PDCA context has NaN/Inf after aggregation/spike")

            context = context.contiguous().view(B, C, H, W)
            residual_by_phase[target_idx] = residual_by_phase[target_idx] + self.out_proj(context)

            def _sparsity(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                y = x.detach()
                return (y.abs() <= eps).float().mean()

            def _entropy(prob_flat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                p = prob_flat.detach().float().clamp_min(eps)
                return (-(p * p.log()).sum(dim=2)).mean()
            if collect_aux and not relation_aux_only:
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
            if collect_aux and not relation_aux_only and self.pdca_context_spike_stats:
                stats = {
                    "mode": runtime_mode,
                    "spike_capacity": torch.tensor(float(self.pdca_context_spike_capacity), device=pre_feat.device),
                    "spike_threshold": torch.tensor(float(self.pdca_context_spike_threshold),
                                                    device=pre_feat.device),

                    "joint_weights_entropy_before": _entropy(soft_flat_for_stats),
                    "joint_weights_entropy_after": _entropy(
                        joint_weights.view(B, self.num_heads, Q * self.num_points, H, W)
                        .clamp_min(1e-8)
                    ),
                    "joint_weights_sparsity": _sparsity(joint_weights),

                    "sampled_sparsity": _sparsity(sampled_stacked),
                    "sampled_abs_mean_before": sampled_stacked_before.detach().abs().mean(),
                    "sampled_abs_mean_after": sampled_stacked.detach().abs().mean(),

                    "context_abs_mean_before": context_before_spike.detach().abs().mean(),
                    "context_abs_mean_after": context.detach().abs().mean(),

                    "null_weight_mean": joint_weights[:, :, -1].detach().sum(dim=2).mean()
                    if self.use_null_source else joint_weights.new_zeros(()),
                }

                if routing_topk_idx is not None:
                    stats["routing_topk_idx_mean"] = routing_topk_idx.detach().float().mean()

                aux["context_spike"][target_name] = {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in stats.items()
                }

        residual = torch.stack(residual_by_phase, dim=0)
        return pre_feat + self.residual_scale * residual, aux
        # return pre_feat + residual, aux

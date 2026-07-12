"""
Minimal task-evidence modules for multi-temporal semantic change detection (MTSCD).

This file is intentionally self-contained and PyTorch-only. It is designed as a
small add-on module for the existing MT-SCD codebase rather than a replacement
for the training entrypoint, backbone, FDPC encoder, PDCA block, or decoder.

Default behavior is conservative: new functional paths are disabled by default.
When enabled, the module produces task evidence from physical remote-sensing
phases and uses it as a residual gate. It does not treat the phase axis N as
ordinary SNN simulation time, does not overwrite dendritic K maps, and does not
introduce hidden state that would bypass reset_net.

No performance claim is implied by this implementation. Any accuracy or
stability claim remains NEEDS_EXPERIMENT.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PHASE_NAMES: Tuple[str, ...] = ("t1", "t2", "t3")
DEFAULT_PAIR_NAMES: Tuple[Tuple[str, str], ...] = (("t1", "t2"), ("t2", "t3"), ("t1", "t3"))


def _valid_group_count(channels: int, preferred: int = 32) -> int:
    candidates = (preferred, 32, 16, 8, 4, 2, 1)
    seen = set()
    for g in candidates:
        g = max(1, int(g))
        if g in seen:
            continue
        seen.add(g)
        if channels % g == 0:
            return g
    return 1


def _pair_key(pair: Tuple[str, str]) -> str:
    return f"{pair[0]}_to_{pair[1]}"


def _resolve_pairs(
    phase_names: Sequence[str],
    pair_names: Optional[Sequence[Tuple[str, str]]],
) -> Tuple[Tuple[str, str], ...]:
    phases = tuple(str(p) for p in phase_names)
    if len(phases) < 2:
        raise ValueError("phase_names must contain at least two physical phases")
    phase_set = set(phases)

    if pair_names is None:
        # Explicit all-pair mode for future N-phase datasets, e.g. DynamicEarth-like protocols.
        pairs = tuple((phases[i], phases[j]) for i in range(len(phases)) for j in range(i + 1, len(phases)))
    else:
        pairs = tuple((str(a), str(b)) for a, b in pair_names)

    for a, b in pairs:
        if a == b:
            raise ValueError(f"self pair is invalid: {(a, b)!r}")
        if a not in phase_set or b not in phase_set:
            raise ValueError(f"unknown phase pair {(a, b)!r} for phase_names={phases!r}")
    return pairs


def _standardize_per_sample(e: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # e: [P,B,E,H,W]. Statistics are per pair, per sample, per evidence channel.
    e_float = e.float()
    mean = e_float.mean(dim=(-2, -1), keepdim=True)
    std = e_float.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(float(eps))
    return ((e_float - mean) / std).to(dtype=e.dtype)


def _resize_4d(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class StatelessIntegerSurrogate(nn.Module):
    """Stateless integer-like activation with an explicit STE path.

    Input:
        x: Tensor of any shape, normally a continuous gate in [0, 1].

    Output:
        Tensor with the same shape as ``x``. Values are quantized to
        ``{0/capacity, 1/capacity, ..., capacity/capacity}``.

    Backward behavior:
        If ``use_ste=True``, the forward value is ``round(clamp(x,0,1)*capacity)``
        divided by ``capacity``, while the backward pass is the identity STE:
        ``x + (hard - x).detach()``. Thus round/clamp are not differentiated
        literally. If ``use_ste=False``, the hard quantized value is detached and
        no useful gradient flows through this activation.

    State:
        This module stores no membrane or batch state. It is compatible with
        reset_net-style training because there is no hidden state to reset.
    """

    def __init__(self, capacity: int = 8, use_ste: bool = True, check_finite: bool = True) -> None:
        super().__init__()
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.use_ste = bool(use_ste)
        self.check_finite = bool(check_finite)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.check_finite and not torch.isfinite(x.detach()).all():
            raise FloatingPointError("StatelessIntegerSurrogate received NaN/Inf")
        hard = torch.round(x.clamp(0.0, 1.0) * float(self.capacity)) / float(self.capacity)
        if self.use_ste:
            return x + (hard - x).detach()
        return hard.detach()


class MTSCDTaskEvidenceUnit(nn.Module):
    """Task-evidence residual unit for physical-phase MTSCD features.

    Input:
        x: Tensor with shape ``[N, B, C, H, W]``. ``N`` is the physical
           remote-sensing phase axis, not SNN simulation time.

    Output:
        If ``return_aux=False``: ``y`` with shape ``[N, B, C, H, W]``.
        If ``return_aux=True``: ``(y, aux)`` where ``aux`` contains
        ``pair_evidence`` with shape ``[P, B, E, H, W]`` and ``pair_keys``.

    Evidence semantics:
        For each configured physical pair, the unit computes four dense evidence
        channels: feature absolute disagreement, cosine disagreement, optional
        local high-pass/structure disagreement, and all-phase variance. These
        are task variables for semantic consistency, pseudo-change suppression,
        boundary/structure detail, and multi-phase uncertainty. They are not K
        maps, PDCA source weights, or final change masks.

    Default behavior:
        ``enabled=False`` and ``residual_init=0.0`` keep the old feature path
        unchanged. Enabling this unit is an explicit experimental change.

    State and AMP/DDP:
        The module is stateless apart from learnable parameters. Evidence
        statistics are computed in fp32 and cast back to the input dtype to
        reduce AMP non-finite risk. No CUDA extension or process-local registry
        is used.
    """

    evidence_channels: int = 4

    def __init__(
        self,
        channels: int,
        phase_names: Sequence[str] = DEFAULT_PHASE_NAMES,
        pair_names: Optional[Sequence[Tuple[str, str]]] = DEFAULT_PAIR_NAMES,
        enabled: bool = False,
        feature_residual: bool = True,   # ����
        hidden_channels: Optional[int] = None,
        detach_evidence: bool = True,
        normalize_evidence: bool = True,
        use_highpass_evidence: bool = False,
        use_integer_surrogate: bool = False,
        surrogate_capacity: int = 8,
        residual_init: float = 0.0,
        check_finite: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.phase_names = tuple(str(p) for p in phase_names)
        self.pair_names = _resolve_pairs(self.phase_names, pair_names)
        self.pair_keys = tuple(_pair_key(pair) for pair in self.pair_names)
        self.phase_to_index = {p: i for i, p in enumerate(self.phase_names)}
        self.enabled = bool(enabled)
        self.detach_evidence = bool(detach_evidence)
        self.normalize_evidence = bool(normalize_evidence)
        self.use_highpass_evidence = bool(use_highpass_evidence)
        self.use_integer_surrogate = bool(use_integer_surrogate)
        self.check_finite = bool(check_finite)
        self.eps = float(eps)
        self.feature_residual = bool(feature_residual)

        hidden = int(hidden_channels) if hidden_channels is not None else self.channels
        hidden = max(1, hidden)
        groups = _valid_group_count(self.channels)

        self.local_branch = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False),
            nn.GroupNorm(groups, self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
        )
        self.evidence_to_gate = nn.Sequential(
            nn.Conv2d(self.evidence_channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.channels, kernel_size=1, bias=True),
        )
        self.integer_surrogate = StatelessIntegerSurrogate(
            capacity=surrogate_capacity,
            use_ste=True,
            check_finite=check_finite,
        )
        self.res_scale = nn.Parameter(torch.tensor(float(residual_init)))

        # Zero initialization makes the enabled path start as an identity-like
        # residual gate because residual_scale is also zero by default.
        nn.init.zeros_(self.evidence_to_gate[-1].weight)
        nn.init.zeros_(self.evidence_to_gate[-1].bias)

    def _local_highpass(self, x: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = x.shape
        flat = x.reshape(n * b, c, h, w).float()
        low = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
        high = flat - low
        return high.to(dtype=x.dtype).reshape(n, b, c, h, w).contiguous()

    def _compute_pair_evidence(self, x: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = x.shape
        if n != len(self.phase_names):
            raise ValueError(f"N={n} does not match phase_names={self.phase_names!r}")
        if c != self.channels:
            raise ValueError(f"expected C={self.channels}, got C={c}")

        x_float = x.float()
        phase_var = x_float.var(dim=0, keepdim=False, unbiased=False).mean(dim=1, keepdim=True)
        high = self._local_highpass(x) if self.use_highpass_evidence else None

        evidence: List[torch.Tensor] = []
        for phase_i, phase_j in self.pair_names:
            i = self.phase_to_index[phase_i]
            j = self.phase_to_index[phase_j]
            fi = x_float[i]
            fj = x_float[j]
            abs_disagree = (fj - fi).abs().mean(dim=1, keepdim=True)
            cosine_disagree = (1.0 - F.cosine_similarity(fi, fj, dim=1, eps=self.eps)).clamp(0.0, 2.0).unsqueeze(1)
            if high is None:
                high_disagree = torch.zeros_like(abs_disagree)
            else:
                high_disagree = (high[j].float() - high[i].float()).abs().mean(dim=1, keepdim=True)
            e = torch.cat([abs_disagree, cosine_disagree, high_disagree, phase_var], dim=1)
            evidence.append(e.to(dtype=x.dtype))

        pair_evidence = torch.stack(evidence, dim=0).contiguous()  # [P,B,4,H,W]
        if self.normalize_evidence:
            pair_evidence = _standardize_per_sample(pair_evidence, eps=self.eps)
        if self.check_finite and not torch.isfinite(pair_evidence.detach()).all():
            raise FloatingPointError("MTSCD pair evidence contains NaN/Inf")
        return pair_evidence

    def _pair_to_phase_evidence(self, pair_evidence: torch.Tensor, n: int) -> torch.Tensor:
        p, b, e, h, w = pair_evidence.shape
        out = pair_evidence.new_zeros(n, b, e, h, w)
        count = pair_evidence.new_zeros(n, 1, 1, 1, 1)
        for pair_idx, (phase_i, phase_j) in enumerate(self.pair_names):
            i = self.phase_to_index[phase_i]
            j = self.phase_to_index[phase_j]
            out[i] = out[i] + pair_evidence[pair_idx]
            out[j] = out[j] + pair_evidence[pair_idx]
            count[i] = count[i] + 1.0
            count[j] = count[j] + 1.0
        return out / count.clamp_min(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        detach_aux: bool = True,
    ):
        if x.ndim != 5:
            raise ValueError(f"MTSCDTaskEvidenceUnit expects [N,B,C,H,W], got {tuple(x.shape)}")
        n, b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"expected C={self.channels}, got C={c}")

        if not self.enabled:
            aux = {"pair_evidence": None, "pair_keys": self.pair_keys, "pair_names": self.pair_names}
            return (x, aux) if return_aux else x

        pair_evidence = self._compute_pair_evidence(x)
        if not self.feature_residual:
            y = x
        else:
            phase_evidence = self._pair_to_phase_evidence(pair_evidence, n)
            gate_input = phase_evidence.detach() if self.detach_evidence else phase_evidence
    
            gate_flat = self.evidence_to_gate(gate_input.flatten(0, 1))
            gate_flat = torch.sigmoid(gate_flat)
            if self.use_integer_surrogate:
                gate_flat = self.integer_surrogate(gate_flat)
    
            residual = self.local_branch(x.flatten(0, 1)).reshape(n, b, c, h, w).contiguous()
            gate = gate_flat.reshape(n, b, c, h, w).contiguous()
            y = x + self.res_scale.to(dtype=x.dtype) * gate.to(dtype=x.dtype) * residual.to(dtype=x.dtype)

        if self.check_finite and not torch.isfinite(y.detach()).all():
            raise FloatingPointError("MTSCDTaskEvidenceUnit output contains NaN/Inf")

        if return_aux:
            aux_evidence = pair_evidence.detach() if detach_aux else pair_evidence
            aux: Dict[str, Any] = {
                "pair_evidence": aux_evidence,
                "pair_keys": self.pair_keys,
                "pair_names": self.pair_names,
            }
            return y, aux
        return y


class MTSCDPairDecoderGate(nn.Module):
    """Residual gate for pairwise change-decoder features.

    Input:
        pair_features: Tensor with shape ``[P, B, C, H, W]`` where ``P`` follows
            ``pair_names``. This should be a pairwise feature tensor before the
            binary change head, not final change logits.
        pair_evidence: optional Tensor with shape ``[P, B, E, H_e, W_e]`` or an
            aux dict from ``MTSCDTaskEvidenceUnit`` containing ``pair_evidence``.
        pdca_aux: optional existing MT-SCD aux dict containing
            ``pdca_source_weights`` and ``pdca_source_names_by_target``. If
            ``use_pdca_source_weights=True``, four PDCA relation channels are
            appended: i<-j, j<-i, i<-null, j<-null.

    Output:
        If ``return_aux=False``: gated tensor with shape ``[P, B, C, H, W]``.
        If ``return_aux=True``: ``(gated, aux)`` where ``aux["gate"]`` has shape
        ``[P, B, 1, H, W]``.

    Default behavior:
        ``enabled=False`` returns ``pair_features`` unchanged. The initial gate
        is also identity-biased when enabled because the prediction head is
        zero-initialized and ``raw_gate_scale`` starts negative.

    Detach policy:
        ``detach_evidence`` and ``detach_pdca_guidance`` are explicit constructor
        parameters. This prevents accidental gradient coupling between decoder
        gates, PDCA source weights, and task evidence.
    """

    def __init__(
        self,
        in_channels: int,
        phase_names: Sequence[str] = DEFAULT_PHASE_NAMES,
        pair_names: Optional[Sequence[Tuple[str, str]]] = DEFAULT_PAIR_NAMES,
        evidence_channels: int = MTSCDTaskEvidenceUnit.evidence_channels,
        enabled: bool = False,
        detach_evidence: bool = True,
        use_pdca_source_weights: bool = False,
        detach_pdca_guidance: bool = True,
        pdca_scale_key: Optional[str] = None,
        alpha_max: float = 1.0,
        raw_gate_scale_init: float = -4.0,
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(evidence_channels) <= 0:
            raise ValueError("evidence_channels must be positive")
        self.in_channels = int(in_channels)
        self.phase_names = tuple(str(p) for p in phase_names)
        self.pair_names = _resolve_pairs(self.phase_names, pair_names)
        self.pair_keys = tuple(_pair_key(pair) for pair in self.pair_names)
        self.phase_to_index = {p: i for i, p in enumerate(self.phase_names)}
        self.enabled = bool(enabled)
        self.detach_evidence = bool(detach_evidence)
        self.use_pdca_source_weights = bool(use_pdca_source_weights)
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.pdca_scale_key = None if pdca_scale_key is None else str(pdca_scale_key)
        self.alpha_max = float(alpha_max)
        self.check_finite = bool(check_finite)

        guide_channels = int(evidence_channels) + (4 if self.use_pdca_source_weights else 0)
        self.gate_head = nn.Conv2d(guide_channels, 1, kernel_size=1, bias=True)
        self.raw_gate_scale = nn.Parameter(torch.tensor(float(raw_gate_scale_init)))
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    @staticmethod
    def _scale_dict(pdca_aux: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
        if not pdca_aux:
            return {}
        value = pdca_aux.get(key, {})
        return value if isinstance(value, dict) else {}

    def _choose_pdca_scale_key(self, pdca_aux: Optional[Dict[str, Any]]) -> Optional[str]:
        if self.pdca_scale_key is not None:
            return self.pdca_scale_key
        weights_by_scale = self._scale_dict(pdca_aux, "pdca_source_weights")
        if not weights_by_scale:
            return None
        # Conservative default: use the lexicographically last available scale,
        # matching the common deep-scale guidance use case without hardcoding 3.
        return sorted(str(k) for k in weights_by_scale.keys())[-1]

    def _source_weight(
        self,
        pdca_aux: Optional[Dict[str, Any]],
        scale_key: Optional[str],
        target_name: str,
        source_name: str,
        like: torch.Tensor,
    ) -> torch.Tensor:
        zeros = like.new_zeros(like.shape[0], 1, like.shape[-2], like.shape[-1])
        if scale_key is None or not pdca_aux:
            return zeros
        weights_by_scale = self._scale_dict(pdca_aux, "pdca_source_weights")
        names_by_scale = self._scale_dict(pdca_aux, "pdca_source_names_by_target")
        scale_weights = weights_by_scale.get(scale_key)
        scale_names = names_by_scale.get(scale_key)
        if scale_weights is None or scale_names is None:
            return zeros
        if target_name not in scale_weights or target_name not in scale_names:
            return zeros
        source_names = tuple(scale_names[target_name])
        if source_name not in source_names:
            return zeros
        weights = scale_weights[target_name]
        if weights.ndim != 5:
            raise RuntimeError(
                f"pdca_source_weights[{scale_key}][{target_name}] must be [B,G,Q,H,W], got {tuple(weights.shape)}"
            )
        if self.detach_pdca_guidance:
            weights = weights.detach()
        q_idx = source_names.index(source_name)
        out = weights[:, :, q_idx].mean(dim=1, keepdim=True)
        out = _resize_4d(out.to(device=like.device, dtype=like.dtype), like.shape[-2:])
        return out

    def _pdca_pair_guidance(
        self,
        pdca_aux: Optional[Dict[str, Any]],
        pair_idx: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        phase_i, phase_j = self.pair_names[pair_idx]
        scale_key = self._choose_pdca_scale_key(pdca_aux)
        return torch.cat(
            [
                self._source_weight(pdca_aux, scale_key, phase_i, phase_j, like),
                self._source_weight(pdca_aux, scale_key, phase_j, phase_i, like),
                self._source_weight(pdca_aux, scale_key, phase_i, "__null__", like),
                self._source_weight(pdca_aux, scale_key, phase_j, "__null__", like),
            ],
            dim=1,
        )

    def _extract_pair_evidence(self, pair_evidence: Optional[Any]) -> Optional[torch.Tensor]:
        if pair_evidence is None:
            return None
        if isinstance(pair_evidence, dict):
            pair_evidence = pair_evidence.get("pair_evidence", None)
        if pair_evidence is None:
            return None
        if not torch.is_tensor(pair_evidence):
            raise TypeError("pair_evidence must be a Tensor or aux dict")
        if pair_evidence.ndim != 5:
            raise ValueError(f"pair_evidence must be [P,B,E,H,W], got {tuple(pair_evidence.shape)}")
        return pair_evidence

    def forward(
        self,
        pair_features: torch.Tensor,
        pair_evidence: Optional[Any] = None,
        pdca_aux: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ):
        if pair_features.ndim != 5:
            raise ValueError(f"pair_features must be [P,B,C,H,W], got {tuple(pair_features.shape)}")
        p, b, c, h, w = pair_features.shape
        if p != len(self.pair_names):
            raise ValueError(f"P={p} does not match pair_names={self.pair_names!r}")
        if c != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got C={c}")

        if not self.enabled:
            aux = {"gate": None, "pair_keys": self.pair_keys}
            return (pair_features, aux) if return_aux else pair_features

        evidence = self._extract_pair_evidence(pair_evidence)
        if evidence is None:
            raise ValueError("enabled MTSCDPairDecoderGate requires pair_evidence")
        if evidence.shape[0] != p or evidence.shape[1] != b:
            raise ValueError(
                f"pair_evidence shape {tuple(evidence.shape)} is incompatible with pair_features {tuple(pair_features.shape)}"
            )
        if self.detach_evidence:
            evidence = evidence.detach()

        gates: List[torch.Tensor] = []
        gated_features: List[torch.Tensor] = []
        alpha = self.alpha_max * torch.sigmoid(self.raw_gate_scale).to(dtype=pair_features.dtype)

        for pair_idx in range(p):
            feat = pair_features[pair_idx]
            ev = _resize_4d(evidence[pair_idx].to(device=feat.device, dtype=feat.dtype), (h, w))
            guides = [ev]
            if self.use_pdca_source_weights:
                guides.append(self._pdca_pair_guidance(pdca_aux, pair_idx, feat))
            guide = torch.cat(guides, dim=1)
            gate = torch.sigmoid(self.gate_head(guide))
            gates.append(gate)
            gated_features.append(feat * (1.0 + alpha.view(1, 1, 1, 1) * gate))

        out = torch.stack(gated_features, dim=0).contiguous()
        gate_tensor = torch.stack(gates, dim=0).contiguous()
        if self.check_finite and not torch.isfinite(out.detach()).all():
            raise FloatingPointError("MTSCDPairDecoderGate output contains NaN/Inf")

        if return_aux:
            return out, {"gate": gate_tensor.detach(), "pair_keys": self.pair_keys}
        return out


def _shape_sanity_test() -> None:
    torch.manual_seed(7)
    n, b, c, h, w = 3, 2, 16, 32, 32
    x = torch.randn(n, b, c, h, w)

    unit = MTSCDTaskEvidenceUnit(
        channels=c,
        enabled=True,
        detach_evidence=True,
        use_highpass_evidence=True,
        use_integer_surrogate=True,
        residual_init=0.0,
    )
    y, aux = unit(x, return_aux=True)
    assert y.shape == x.shape, (y.shape, x.shape)
    assert aux["pair_evidence"].shape == (3, b, 4, h, w), aux["pair_evidence"].shape

    pair_channels = 24
    pair_features = torch.randn(3, b, pair_channels, h, w)
    gate = MTSCDPairDecoderGate(
        in_channels=pair_channels,
        enabled=True,
        detach_evidence=True,
        use_pdca_source_weights=False,
    )
    gated, gate_aux = gate(pair_features, pair_evidence=aux, return_aux=True)
    assert gated.shape == pair_features.shape, (gated.shape, pair_features.shape)
    assert gate_aux["gate"].shape == (3, b, 1, h, w), gate_aux["gate"].shape
    print("shape sanity test passed")


if __name__ == "__main__":
    _shape_sanity_test()

"""
FDPC Encoder for MTSCD direct-N setting.

Input protocol:
    feature_xy: List[Tensor]
    feature_xy[s]: [N, B, C_s, H_s, W_s]

Output protocol:
    encoded_xy: List[Tensor]
    encoded_xy[s]: [N, B, C_s, H_s, W_s]

Design constraints:
    - The first axis N is the physical remote-sensing phase index in the
      current direct-N baseline. It is not treated as vanilla SNN time.
    - DendFADCConv2d is used with SN_CLS=False, so this adapter does not run
      a vanilla multi-step LIF accumulation along the phase axis.
    - relation_cue / context_gate / change_risk are intermediate variables
      only. They are not final change maps.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d
from models.Encoders.phase_deformable_context_attention import PhaseDeformableContextAttention
from mmseg.Qtrick_architecture.clock_driven.neuron import MTSCDPRDNIIFNode, Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant, Quant4

PairName = Tuple[str, str]
AuxDict = Dict[str, Dict[str, Dict[str, torch.Tensor]]]


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
    raise ValueError(f"Unsupported norm type: {norm}")


def _resolve_conv_groups(channels: int, conv_groups: Union[str, int]) -> int:
    if isinstance(conv_groups, str):
        key = conv_groups.lower()
        if key in ("dw", "depthwise"):
            return channels
        if key in ("dense", "full", "none"):
            return 1
        raise ValueError(f"Unsupported conv_groups string: {conv_groups}")

    groups = int(conv_groups)
    if groups <= 0:
        raise ValueError("conv_groups must be positive")
    if channels % groups != 0:
        raise ValueError(f"channels={channels} must be divisible by conv_groups={groups}")
    return groups


def _ensure_pair_tuple(pair: Sequence[str]) -> PairName:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(f"Each pair must be a 2-tuple/list, got {pair!r}")
    return str(pair[0]), str(pair[1])


def _normalize_dend_soma_type(dend_soma_type: str) -> str:
    soma_type = str(dend_soma_type or "q_if").lower()
    if soma_type not in ("q_if", "mtscd_prd", "identity", "none"):
        raise ValueError("dend_soma_type must be one of: q_if, mtscd_prd, identity, none")
    return soma_type


def _make_dend_soma(dend_soma_type: str, dend_soma_cfg: Optional[dict]) -> nn.Module:
    soma_type = _normalize_dend_soma_type(dend_soma_type)
    if dend_soma_cfg is not None and not isinstance(dend_soma_cfg, dict):
        raise ValueError("dend_soma_cfg must be a dict or None")
    cfg = dict(dend_soma_cfg or {})
    if soma_type == "q_if":
        return Q_IFNode(surrogate_function=Quant())
    elif soma_type == "mtscd_prd":
        return MTSCDPRDNIIFNode(**cfg)
    elif soma_type in ("identity", "none"):
        return nn.Identity()
    raise ValueError("unreachable dend_soma_type")


class DendriticScaleAdapter(nn.Module):
    """
    Phase-wise local spatial-frequency adapter.

    Input:
        x: [N,B,C,H,W]

    Output:
        y: [N,B,C,H,W]

    Important:
        SN_CLS=False, so DendFADCConv2d is not used as vanilla LIF over N.
    """

    def __init__(
        self,
        channels: int,
        use_dendritic: bool = True,
        kernel_size: int = 3,
        conv_groups: Union[str, int] = "depthwise",
        deform_groups: int = 1,
        norm: str = "gn",
        norm_groups: int = 32,
        residual_init: float = 0.0,
        fs_cfg: Optional[dict] = None,
        kernel_decompose: Optional[str] = "both",
        padding_mode: str = "repeat",
        dend_soma_type: str = "q_if",
        dend_soma_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.use_dendritic = bool(use_dendritic)
        self.dend_soma_type = _normalize_dend_soma_type(dend_soma_type)
        if dend_soma_cfg is not None and not isinstance(dend_soma_cfg, dict):
            raise ValueError("dend_soma_cfg must be a dict or None")
        self.dend_soma_cfg = dict(dend_soma_cfg or {})

        if not self.use_dendritic:
            self.adapter = nn.Identity()
            self.post_norm = nn.Identity()
            self.act = nn.Identity()
            self.res_scale = None
            return

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2
        groups = _resolve_conv_groups(self.channels, conv_groups)

        default_fs_cfg = dict(
            k_list=[2, 4],
            lowfreq_att=False,
            lp_type="freq",
            act="sigmoid",
            spatial="conv",
            spatial_group=1,
        )
        if fs_cfg is not None:
            default_fs_cfg.update(fs_cfg)

        self.adapter = DendFADCConv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=groups,
            bias=False,
            deform_groups=int(deform_groups),
            padding_mode=padding_mode,
            kernel_decompose=kernel_decompose,
            pre_fs=True,
            fs_cfg=default_fs_cfg,
            calculate_next_k=True,
            SN_CLS=True,
        )

        self.post_norm = _make_norm2d(self.channels, norm=norm, num_groups=norm_groups)
        # self.act = nn.GELU()
        self.act = _make_dend_soma(self.dend_soma_type, self.dend_soma_cfg)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, x: torch.Tensor, K=None, return_k: bool = True) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"DendriticScaleAdapter expects [N,B,C,H,W], got {tuple(x.shape)}")

        if not self.use_dendritic:
            return x
        x_pre = x
        x = self.act(x)
        N, B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={C}")

        y, K = self.adapter(x, K=K, return_k=True)

        if y.shape[-2:] != (H, W) or y.shape[2] != C:
            raise RuntimeError(
                "DendriticScaleAdapter must preserve channel and spatial shape, "
                f"but got input={tuple(x.shape)}, output={tuple(y.shape)}"
            )
        y = self.post_norm(y.flatten(0, 1)).reshape(N, B, C, H, W).contiguous()
        # y = self.act(y)

        # return x + self.res_scale * y, K
        return x_pre + self.res_scale * y, K
        # return x_pre + y, K

class PairwiseRelationGate(nn.Module):
    """
    Construct relation cue and context gate for one directed phase pair.

    target: [B,C,H,W]
    source: [B,C,H,W]

    relation input:
        [target, source, abs(target-source), source-target]

    Outputs:
        relation_cue: [B,C_r,H,W]
        context_gate: [B,1,H,W], high means stronger source-to-target injection
        change_risk:  [B,1,H,W], high means higher semantic contamination risk
    """

    def __init__(
        self,
        channels: int,
        relation_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        norm: str = "gn",
        norm_groups: int = 32,
        risk_bias_init: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)

        hidden = int(hidden_channels) if hidden_channels is not None else max(32, self.channels // 2)
        rel_ch = int(relation_channels) if relation_channels is not None else hidden
        self.act = Q_IFNode(surrogate_function=Quant())
        self.phi = nn.Sequential(
            nn.Conv2d(4 * self.channels, hidden, kernel_size=1, bias=False),
            _make_norm2d(hidden, norm=norm, num_groups=norm_groups),
            Q_IFNode(surrogate_function=Quant()),      #nn.GELU(),
            nn.Conv2d(hidden, rel_ch, kernel_size=3, padding=1, bias=False),
            _make_norm2d(rel_ch, norm=norm, num_groups=norm_groups),
            Q_IFNode(surrogate_function=Quant()),      #nn.GELU(),
        )

        self.risk_head = nn.Sequential(
            nn.Conv2d(rel_ch, max(8, rel_ch // 2), kernel_size=3, padding=1, bias=True),
            Q_IFNode(surrogate_function=Quant()),      #nn.GELU(),nn.GELU(),
            nn.Conv2d(max(8, rel_ch // 2), 1, kernel_size=1, bias=True),
        )

        nn.init.constant_(self.risk_head[-1].bias, float(risk_bias_init))

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target.ndim != 4 or source.ndim != 4:
            raise ValueError("PairwiseRelationGate expects target/source [B,C,H,W]")

        if target.shape != source.shape:
            raise ValueError(f"target/source shape mismatch: {tuple(target.shape)} vs {tuple(source.shape)}")

        if target.shape[1] != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={target.shape[1]}")

        diff_abs = torch.abs(target - source)
        diff_signed = source - target

        relation_input = torch.cat(
            [target, source, diff_abs, diff_signed],
            dim=1,
        )

        relation_cue = self.phi(relation_input)
        change_risk = torch.sigmoid(self.risk_head(relation_cue))
        context_gate = 1.0 - change_risk

        return relation_cue, context_gate, change_risk


class FDPCEncoder(nn.Module):
    """
    Frequency-Dendritic Phase Context Encoder, minimal v1.

    This version contains:
        1) per-scale dendritic local enhancement;
        2) high-level pairwise relation gate;
        3) gated residual cross-phase context injection.

    It intentionally does not implement deformable sampling yet.
    The current relation gate is CDPA-lite:
        relation cue -> context gate -> residual source feature injection.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        phase_names: Sequence[str] = ("t1", "t2", "t3"),
        context_pairs: Sequence[PairName] = (("t1", "t2"), ("t2", "t3"), ("t1", "t3")),
        dendritic_scales: Iterable[int] = (1, 2, 3, 4),
        relation_scales: Iterable[int] = (3, 4),
        conv_groups: Union[str, int] = "depthwise",
        deform_groups: int = 1,
        dend_kernel_size: int = 3,
        fs_cfg: Optional[dict] = None,
        kernel_decompose: Optional[str] = "both",
        norm: str = "gn",
        norm_groups: int = 32,
        dend_residual_init: float = 0.0,
        dend_soma_type: str = "q_if",
        dend_soma_cfg: Optional[dict] = None,
        context_residual_init: float = 0.0,
        relation_channels: Optional[int] = None,
        relation_hidden_channels: Optional[int] = None,
        detach_context_gate: bool = False,
        return_aux_default: bool = False,
        relation_mode: str = "prg",
        pdca_cfg: Optional[dict] = None,
    ):
        super().__init__()

        if len(in_channels) == 0:
            raise ValueError("in_channels must not be empty")
        if relation_mode not in ("prg", "pdca", "none"):
            raise ValueError("relation_mode must be one of: prg, pdca, none")

        self.in_channels = [int(c) for c in in_channels]
        self.num_scales = len(self.in_channels)

        self.phase_names = tuple(str(name) for name in phase_names)
        self.phase_to_index = {name: idx for idx, name in enumerate(self.phase_names)}

        self.context_pairs = tuple(_ensure_pair_tuple(pair) for pair in context_pairs)
        self.dendritic_scales = set(int(s) for s in dendritic_scales)
        self.relation_scales = set(int(s) for s in relation_scales)

        self.detach_context_gate = bool(detach_context_gate)
        self.return_aux_default = bool(return_aux_default)
        self.relation_mode = relation_mode
        self.pdca_cfg = dict(pdca_cfg or {})

        for s in self.dendritic_scales | self.relation_scales:
            if s < 0 or s >= self.num_scales:
                raise ValueError(f"scale index {s} is out of range for {self.num_scales} scales")

        for a, b in self.context_pairs:
            if a not in self.phase_to_index or b not in self.phase_to_index:
                raise ValueError(f"Unknown phase pair {(a, b)} for phase_names={self.phase_names}")
            if a == b:
                raise ValueError(f"Self pair is not allowed: {(a, b)}")

        self.scale_adapters = nn.ModuleList()
        for s, channels in enumerate(self.in_channels):
            self.scale_adapters.append(
                DendriticScaleAdapter(
                    channels=channels,
                    use_dendritic=s in self.dendritic_scales,
                    kernel_size=dend_kernel_size,
                    conv_groups=conv_groups,
                    deform_groups=deform_groups,
                    norm=norm,
                    norm_groups=norm_groups,
                    residual_init=dend_residual_init,
                    fs_cfg=fs_cfg,
                    kernel_decompose=kernel_decompose,
                    dend_soma_type=dend_soma_type,
                    dend_soma_cfg=dend_soma_cfg,
                )
            )

        self.relation_gates = nn.ModuleDict()
        self.value_projs = nn.ModuleDict()
        self.context_scales = nn.ParameterDict()
        self.pdca_blocks = nn.ModuleDict()

        for s in sorted(self.relation_scales):
            channels = self.in_channels[s]
            key = str(s)

            if self.relation_mode == "prg":
                self.relation_gates[key] = PairwiseRelationGate(
                    channels=channels,
                    relation_channels=relation_channels,
                    hidden_channels=relation_hidden_channels,
                    norm=norm,
                    norm_groups=norm_groups,
                )

                self.value_projs[key] = nn.Sequential(
                    Q_IFNode(surrogate_function=Quant()),  # nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False, )
                )

                self.context_scales[key] = nn.Parameter(
                    torch.tensor(float(context_residual_init))
                )
            elif self.relation_mode == "pdca":
                self.pdca_blocks[key] = PhaseDeformableContextAttention(
                    channels=channels,
                    phase_names=self.phase_names,
                    context_pairs=self.context_pairs,
                    **self._resolve_pdca_cfg_for_scale(key)
                )

    @staticmethod
    def _new_aux() -> AuxDict:
        return {
            "relation_cues": {},
            "context_gates": {},
            "change_risks": {},
            "pdca_offsets": {},
            "pdca_attn_weights": {},
            "pdca_source_weights": {},
            "pdca_joint_weights": {},
        }

    def _resolve_pdca_cfg_for_scale(self, scale_key: str) -> dict:
        cfg = dict(self.pdca_cfg)
        per_scale = cfg.pop("per_scale", {})
        if scale_key in per_scale:
            cfg.update(per_scale[scale_key])
        elif int(scale_key) in per_scale:
            cfg.update(per_scale[int(scale_key)])
        return cfg

    def _store_aux(
        self,
        aux: AuxDict,
        scale_key: str,
        direction_key: str,
        relation_cue: torch.Tensor,
        context_gate: torch.Tensor,
        change_risk: torch.Tensor,
        detach_aux: bool,
    ) -> None:
        aux["relation_cues"].setdefault(scale_key, {})[direction_key] = (
            relation_cue.detach() if detach_aux else relation_cue
        )
        aux["context_gates"].setdefault(scale_key, {})[direction_key] = (
            context_gate.detach() if detach_aux else context_gate
        )
        aux["change_risks"].setdefault(scale_key, {})[direction_key] = (
            change_risk.detach() if detach_aux else change_risk
        )

    def _apply_relation_scale(
        self,
        scale_index: int,
        feat: torch.Tensor,
        aux: AuxDict,
        collect_aux: bool,
        detach_aux: bool,
    ) -> torch.Tensor:
        if feat.ndim != 5:
            raise ValueError(f"Expected feature [N,B,C,H,W], got {tuple(feat.shape)}")

        N, B, C, H, W = feat.shape

        if N != len(self.phase_names):
            raise ValueError(
                f"Feature phase dimension N={N} does not match phase_names={self.phase_names}"
            )

        scale_key = str(scale_index)
        gate_module = self.relation_gates[scale_key]
        value_proj = self.value_projs[scale_key]

        residual_by_phase = [
            torch.zeros_like(feat[phase_idx])
            for phase_idx in range(N)
        ]

        for phase_a, phase_b in self.context_pairs:
            ia = self.phase_to_index[phase_a]
            ib = self.phase_to_index[phase_b]

            # Compute two directed injections separately.
            # This is necessary because signed difference changes direction.
            directed_pairs = ((ia, ib), (ib, ia))

            for dst_idx, src_idx in directed_pairs:
                dst_name = self.phase_names[dst_idx]
                src_name = self.phase_names[src_idx]
                direction_key = f"{dst_name}<-{src_name}"

                relation_cue, context_gate, change_risk = gate_module(
                    feat[dst_idx],
                    feat[src_idx],
                )

                if collect_aux:
                    self._store_aux(
                        aux=aux,
                        scale_key=scale_key,
                        direction_key=direction_key,
                        relation_cue=relation_cue,
                        context_gate=context_gate,
                        change_risk=change_risk,
                        detach_aux=detach_aux,
                    )

                gate = context_gate.detach() if self.detach_context_gate else context_gate

                src_value = value_proj(feat[src_idx])

                residual_by_phase[dst_idx] = residual_by_phase[dst_idx] + gate * src_value

        residual = torch.stack(residual_by_phase, dim=0)

        return feat + self.context_scales[scale_key] * residual

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        return_aux: Optional[bool] = None,
        detach_aux: bool = False,
    ) -> Tuple[List[torch.Tensor], AuxDict]:
        if return_aux is None:
            return_aux = self.return_aux_default

        if len(feature_xy) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature scales, got {len(feature_xy)}"
            )

        encoded: List[torch.Tensor] = []
        K = None
        for s, (feat, adapter) in enumerate(zip(feature_xy, self.scale_adapters)):
            if feat.ndim != 5:
                raise ValueError(
                    f"feature_xy[{s}] must be [N,B,C,H,W], got {tuple(feat.shape)}"
                )

            if feat.shape[2] != self.in_channels[s]:
                raise ValueError(
                    f"feature_xy[{s}] channel mismatch: "
                    f"expected {self.in_channels[s]}, got {feat.shape[2]}"
                )
            if not isinstance(adapter.adapter, nn.Identity):
                feat_i, K = adapter(feat, K=K)
            else:
                feat_i = adapter(feat)
            encoded.append(feat_i)

        aux = self._new_aux()

        for s in sorted(self.relation_scales):
            if self.relation_mode == "prg":
                encoded[s] = self._apply_relation_scale(
                    scale_index=s,
                    feat=encoded[s],
                    aux=aux,
                    collect_aux=bool(return_aux),
                    detach_aux=bool(detach_aux),
                )
            elif self.relation_mode == "pdca":
                scale_key = str(s)
                encoded[s], pdca_aux = self.pdca_blocks[scale_key](
                    encoded[s],
                    return_aux=bool(return_aux),
                    detach_aux=bool(detach_aux),
                )
                if return_aux:
                    aux["pdca_offsets"][scale_key] = pdca_aux.get("offsets", {})
                    aux["pdca_attn_weights"][scale_key] = pdca_aux.get("attn_weights", {})
                    aux["pdca_source_weights"][scale_key] = pdca_aux.get("source_weights", {})
                    aux["pdca_joint_weights"][scale_key] = pdca_aux.get("joint_weights", {})
            elif self.relation_mode == "none":
                pass

        return encoded, aux if return_aux else {}

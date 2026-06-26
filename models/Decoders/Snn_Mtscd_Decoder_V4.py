import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant, Quant4
from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d
PHASE_NAMES = ("t1", "t2", "t3")
TRANSITION_NAMES = ("t1_to_t2", "t2_to_t3", "t1_to_t3")


def _to_sorted_unique_int_list(indices: Sequence[int]) -> List[int]:
    if not isinstance(indices, (list, tuple)):
        raise TypeError(f"Expected list/tuple of indices, but got {type(indices)}.")
    out = [int(i) for i in indices]
    if len(out) == 0:
        raise ValueError("Window indices must not be empty.")
    if len(set(out)) != len(out):
        raise ValueError(f"Window indices must be unique, but got {out}.")
    return sorted(out)


def build_default_phase_and_transition_windows(
    T: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Optional[List[int]]]]:
    """
    All indices are 0-based.

    Defaults:
    - T=12 -> K=4, R=0
    - T=16 -> K=4, R=2
    - Else if T % 3 == 0 -> equally split three contiguous phase windows
    """
    if T == 12:
        return (
            {
                "t1": [0, 1, 2, 3],
                "t2": [4, 5, 6, 7],
                "t3": [8, 9, 10, 11],
            },
            {
                "t1_to_t2": None,
                "t2_to_t3": None,
                "t1_to_t3": None,
            },
        )

    if T == 16:
        return (
            {
                "t1": [0, 1, 2, 3],
                "t2": [6, 7, 8, 9],
                "t3": [12, 13, 14, 15],
            },
            {
                "t1_to_t2": [3, 4, 5, 6],
                "t2_to_t3": [9, 10, 11, 12],
                "t1_to_t3": None,
            },
        )

    if T % 3 == 0:
        k = T // 3
        return (
            {
                "t1": list(range(0, k)),
                "t2": list(range(k, 2 * k)),
                "t3": list(range(2 * k, 3 * k)),
            },
            {
                "t1_to_t2": None,
                "t2_to_t3": None,
                "t1_to_t3": None,
            },
        )

    raise ValueError(
        f"Cannot infer default windows from T={T}. "
        "Please pass explicit phase_windows."
    )


def normalize_windows(
    T: int,
    phase_windows: Optional[Dict[str, Sequence[int]]] = None,
    transition_windows: Optional[Dict[str, Optional[Sequence[int]]]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, Optional[List[int]]]]:
    if phase_windows is None:
        auto_phase, auto_transition = build_default_phase_and_transition_windows(T)
        phase_windows = auto_phase
        if transition_windows is None:
            transition_windows = auto_transition

    if not isinstance(phase_windows, dict):
        raise TypeError(f"phase_windows must be dict or None, but got {type(phase_windows)}.")

    if transition_windows is None:
        transition_windows = {k: None for k in TRANSITION_NAMES}
    if not isinstance(transition_windows, dict):
        raise TypeError(f"transition_windows must be dict or None, but got {type(transition_windows)}.")

    phase_out: Dict[str, List[int]] = {}
    for key in PHASE_NAMES:
        if key not in phase_windows:
            raise ValueError(f"phase_windows must contain key '{key}'.")
        idx = _to_sorted_unique_int_list(phase_windows[key])
        if min(idx) < 0 or max(idx) >= T:
            raise ValueError(f"phase window {key}={idx} exceeds T={T}.")
        phase_out[key] = idx

    trans_out: Dict[str, Optional[List[int]]] = {}
    for key in TRANSITION_NAMES:
        value = transition_windows.get(key, None)
        if value is None:
            trans_out[key] = None
        else:
            idx = _to_sorted_unique_int_list(value)
            if min(idx) < 0 or max(idx) >= T:
                raise ValueError(f"transition window {key}={idx} exceeds T={T}.")
            trans_out[key] = idx

    return phase_out, trans_out


class DendBlock(nn.Module):
    """Input: [B, Cin, H, W] -> Output: [B, Cout, H, W]"""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        # self.block = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.GELU(),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.GELU(),
        # )
        kernel_size = 3
        padding = kernel_size // 2
        default_fs_cfg = dict(
            k_list=[2, 4],
            lowfreq_att=False,
            lp_type="freq",
            act="sigmoid",
            spatial="conv",
            spatial_group=1,
        )
        self.block = DendFADCConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=groups,
            bias=False,
            deform_groups=1,
            padding_mode="repeat",
            kernel_decompose="both",
            pre_fs=True,
            fs_cfg=default_fs_cfg,
            calculate_next_k=True,
            SN_CLS=True,
            Down_K=False
        )
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, k):
        x, k = self.block(x.unsqueeze(0), k)
        x = self.BN(x.flatten(0, 1))
        return x, k

class ConvBlock(nn.Module):
    """Input: [B, Cin, H, W] -> Output: [B, Cout, H, W]"""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Q_IFNode(surrogate_function=Quant()),   #nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Q_IFNode(surrogate_function=Quant()),   #nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x)

        return x
class UpFuseBlock(nn.Module):
    """
    x:    [B, Cx, H_low, W_low]
    skip: [B, Cs, H_high, W_high]
    out:  [B, Cout, H_high, W_high]
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_DendSize: int=128):
        super().__init__()

        self.fuse1 = DendBlock(in_channels + skip_channels, out_channels)
        self.fuse2 = ConvBlock(in_channels + skip_channels, out_channels)
        self.use_DendSize = use_DendSize
    def forward(self, x: torch.Tensor, skip: torch.Tensor, k=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if k != None and k[0].shape[-2:] != skip.shape[-2:]:
            K_tensor = torch.cat(k, dim=1)  # [B, 2, H, W]

            K_up_tensor = F.interpolate(
                K_tensor,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )  # [B, 2, H1, W1]

            k = list(torch.chunk(K_up_tensor, chunks=2, dim=1))
        x = torch.cat([x, skip], dim=1)
        if skip.shape[-1:][0] >= self.use_DendSize:
            x = self.fuse2(x)
        else:
            x, k = self.fuse1(x, k)
        # x, k = self.fuse2(x, k)
        return x, k


class TemporalReadout(nn.Module):
    """
    seq: [T, B, C, H, W]
    pooled: [B, C, H, W]

    mode='mean':
        mean over selected SNN-time window

    mode='attention':
        lightweight global temporal attention:
        score_t is predicted from globally pooled spatial context and then
        broadcast to all pixels in that step.
    """

    def __init__(self, channels: int, mode: str = "attention", anchor_bias: float = 1.5):
        super().__init__()
        self.channels = int(channels)
        self.mode = mode
        self.anchor_bias = float(anchor_bias)

        if mode not in ("mean", "attention"):
            raise ValueError(f"Unsupported temporal readout mode: {mode}")

        if self.mode == "attention":
            hidden = max(8, channels // 2)
            self.score_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            )
        else:
            self.score_net = None

    def forward(
        self,
        seq: torch.Tensor,
        time_window: Sequence[int],
        anchor_index: Optional[int] = None,
        analog_seq: Optional[torch.Tensor] = None,
        analog_weight: float = 0.0,
    ) -> torch.Tensor:
        if seq.ndim != 5:
            raise ValueError(f"Expected seq shape [T,B,C,H,W], but got {tuple(seq.shape)}.")
        T, B, C, H, W = seq.shape
        if C != self.channels:
            raise ValueError(f"Channel mismatch: expected {self.channels}, got {C}.")

        idx = _to_sorted_unique_int_list(time_window)
        if min(idx) < 0 or max(idx) >= T:
            raise ValueError(f"time_window={idx} exceeds T={T}.")

        # x: [Tw, B, C, H, W]
        x = seq[idx]

        # Future extension point for membrane/analog readout.
        if analog_seq is not None:
            if analog_seq.shape != seq.shape:
                raise ValueError(
                    f"analog_seq must match seq shape, got {tuple(analog_seq.shape)} vs {tuple(seq.shape)}."
                )
            x = (1.0 - float(analog_weight)) * x + float(analog_weight) * analog_seq[idx]

        if self.mode == "mean":
            return x.mean(dim=0)

        tw = x.shape[0]
        score = self.score_net(x.flatten(0, 1))     # [(Tw*B), 1, 1, 1]
        score = score.view(tw, B, 1, 1, 1)          # [Tw, B, 1, 1, 1]

        if anchor_index is not None:
            if anchor_index not in idx:
                raise ValueError(f"anchor_index={anchor_index} is not inside time_window={idx}.")
            rel_anchor = idx.index(anchor_index)
            score[rel_anchor] = score[rel_anchor] + self.anchor_bias

        attn = torch.softmax(score, dim=0)          # [Tw, B, 1, 1, 1]
        pooled = (attn * x).sum(dim=0)              # [B, C, H, W]
        return pooled


class SemanticDecoder(nn.Module):
    """
    feats_high_to_low[s]: [B, C_s, H_s, W_s]
    return:              [B, D, H_0, W_0]
    """

    def __init__(self, in_channels: Sequence[int], decoder_channels: int):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)

        self.laterals = nn.ModuleList([
            nn.Conv2d(c, self.decoder_channels, kernel_size=1, bias=False)
            for c in self.in_channels
        ])
        # self.deep_block = ConvBlock(self.decoder_channels, self.decoder_channels)
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels,use_DendSize=16)
            for _ in range(len(self.in_channels) - 1)
        ])

    def forward(self, feats_high_to_low: Sequence[torch.Tensor], k=None) -> torch.Tensor:
        if len(feats_high_to_low) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} scales, got {len(feats_high_to_low)}."
            )

        proj: List[torch.Tensor] = []
        for s, (feat, c_exp) in enumerate(zip(feats_high_to_low, self.in_channels)):
            if feat.ndim != 4:
                raise ValueError(f"Scale {s} must be 4D [B,C,H,W], but got {tuple(feat.shape)}.")
            if feat.shape[1] != c_exp:
                raise ValueError(f"Scale {s} expects C={c_exp}, got C={feat.shape[1]}.")
            proj.append(self.laterals[s](feat))     # [B, D, H_s, W_s]

        x, k = self.deep_block(proj[-1], k)               # deepest scale
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)

        return x                                    # [B, D, H_0, W_0]


class ChangeDecoder(nn.Module):
    """
    t1_feats[s], t3_feats[s]: [B, C_s, H_s, W_s]
    transition_feats[s]:      [B, C_s, H_s, W_s] or None

    Supported diff_mode:
    - abs:        |F_t3 - F_t1|
    - abs_signed: [|F_t3 - F_t1|, relu(F_t3-F_t1), relu(F_t1-F_t3)]
    - concat:     [F_t1, F_t3, |F_t3 - F_t1|]
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        diff_mode: str = "abs_signed",
        use_transition_fusion: bool = True,
    ):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)
        self.diff_mode = diff_mode
        self.use_transition_fusion = bool(use_transition_fusion)

        if self.diff_mode not in ("abs", "abs_signed", "concat"):
            raise ValueError(f"Unsupported diff_mode={diff_mode}")

        proj_in_channels = []
        for c in self.in_channels:
            base = c if self.diff_mode == "abs" else 3 * c
            if self.use_transition_fusion:
                base += c
            proj_in_channels.append(base)

        self.laterals = nn.ModuleList([
            nn.Conv2d(c, self.decoder_channels, kernel_size=1, bias=False)
            for c in proj_in_channels
        ])
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, use_DendSize=16)
            for _ in range(len(self.in_channels) - 1)
        ])

    def _make_diff(self, f1: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(f3 - f1)
        if self.diff_mode == "abs":
            return abs_diff
        if self.diff_mode == "abs_signed":
            return torch.cat([abs_diff, F.relu(f3 - f1), F.relu(f1 - f3)], dim=1)
        if self.diff_mode == "concat":
            return torch.cat([f1, f3, abs_diff], dim=1)
        raise RuntimeError("Unreachable diff_mode branch.")

    def forward(
        self,
        t1_feats_high_to_low: Sequence[torch.Tensor],
        t3_feats_high_to_low: Sequence[torch.Tensor],
        transition_feats_high_to_low: Optional[Sequence[Optional[torch.Tensor]]] = None, k=None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        n = len(self.in_channels)
        if len(t1_feats_high_to_low) != n or len(t3_feats_high_to_low) != n:
            raise ValueError("ChangeDecoder got inconsistent number of scales.")
        if transition_feats_high_to_low is None:
            transition_feats_high_to_low = [None] * n
        if len(transition_feats_high_to_low) != n:
            raise ValueError("transition_feats_high_to_low has wrong number of scales.")

        multi_scale_change_inputs: List[torch.Tensor] = []
        proj: List[torch.Tensor] = []

        for s in range(n):
            f1 = t1_feats_high_to_low[s]
            f3 = t3_feats_high_to_low[s]
            c_exp = self.in_channels[s]

            if f1.ndim != 4 or f3.ndim != 4:
                raise ValueError("ChangeDecoder expects [B,C,H,W] features.")
            if f1.shape != f3.shape:
                raise ValueError(f"Shape mismatch at scale {s}: {tuple(f1.shape)} vs {tuple(f3.shape)}.")
            if f1.shape[1] != c_exp:
                raise ValueError(f"Scale {s} expects C={c_exp}, got C={f1.shape[1]}.")

            diff = self._make_diff(f1, f3)          # [B, Cdiff, H, W]

            if self.use_transition_fusion:
                trans = transition_feats_high_to_low[s]
                trans = torch.zeros_like(f1) if trans is None else trans
                if trans.shape != f1.shape:
                    raise ValueError(
                        f"Transition feature shape mismatch at scale {s}: {tuple(trans.shape)} vs {tuple(f1.shape)}."
                    )
                x_in = torch.cat([diff, trans], dim=1)
            else:
                x_in = diff

            multi_scale_change_inputs.append(x_in)
            proj.append(self.laterals[s](x_in))     # [B, D, H_s, W_s]

        x, k = self.deep_block(proj[-1], k)               # deepest scale
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)

        return x, multi_scale_change_inputs         # x: [B, D, H_0, W_0]


class PDCASpatialGate(nn.Module):
    """PDCA guidance [B,4,H,W] -> spatial gate [B,1,H,W]."""

    def __init__(self, r_channels: int = 4, init_bias: float = 0.0):
        super().__init__()
        self.r_channels = int(r_channels)
        self.conv = nn.Conv2d(self.r_channels, 1, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, float(init_bias))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.ndim != 4 or r.shape[1] != self.r_channels:
            raise ValueError(
                "PDCASpatialGate expects [B,%d,H,W], got %r"
                % (self.r_channels, tuple(r.shape))
            )
        return torch.sigmoid(self.conv(r))


class PDCAGuidedPairwiseChangeDecoder(nn.Module):
    """
    phase_feats[phase][s]: [B,C_s,H_s,W_s]
    pdca guidance R_ij_s: [B,4,H_s,W_s]
    output logits: fixed pairs [B,1,H,W] for t1_to_t2/t2_to_t3/t1_to_t3
    """

    FIXED_PAIR_KEYS = ("t1_to_t2", "t2_to_t3", "t1_to_t3")

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        num_change_classes: int = 1,
        pair_names: Sequence[Tuple[str, str]] = (
            ("t1", "t2"),
            ("t2", "t3"),
            ("t1", "t3"),
        ),
        phase_names: Sequence[str] = ("t1", "t2", "t3"),
        detach_pdca_guidance: bool = True,
        use_pdca_guidance: bool = True,
        alpha_max: float = 1.0,
    ):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)
        self.num_change_classes = int(num_change_classes)
        self.pair_names = tuple((str(a), str(b)) for a, b in pair_names)
        self.phase_names = tuple(str(name) for name in phase_names)
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.use_pdca_guidance = bool(use_pdca_guidance)
        self.alpha_max = float(alpha_max)
        self.pair_keys = tuple("%s_to_%s" % pair for pair in self.pair_names)

        if self.pair_keys != self.FIXED_PAIR_KEYS:
            raise ValueError("pair order must be %r, got %r" % (self.FIXED_PAIR_KEYS, self.pair_keys))
        if self.alpha_max < 0.0:
            raise ValueError("alpha_max must be non-negative")

        self.diff_proj = nn.ModuleList([
            nn.Conv2d(3 * c, self.decoder_channels, kernel_size=1, bias=False)
            for c in self.in_channels
        ])
        self.spatial_gates = nn.ModuleList([
            PDCASpatialGate(r_channels=4) for _ in self.in_channels
        ])
        self.raw_gate_scales = nn.ParameterList([
            nn.Parameter(torch.full((1,), -4.0)) for _ in self.in_channels
        ])
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, use_DendSize=16)
            for _ in range(len(self.in_channels) - 1)
        ])
        self.change_head = ChangeHead(self.decoder_channels, self.num_change_classes)

    @staticmethod
    def _make_diff(fi: torch.Tensor, fj: torch.Tensor) -> torch.Tensor:
        delta = fj - fi
        return torch.cat([torch.abs(delta), F.relu(delta), F.relu(-delta)], dim=1)

    @staticmethod
    def _scale_dict(pdca_aux: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
        if not pdca_aux:
            return {}
        value = pdca_aux.get(key, {})
        return value if isinstance(value, dict) else {}

    def _has_any_guidance(self, pdca_aux: Optional[Dict[str, Any]]) -> bool:
        weights_by_scale = self._scale_dict(pdca_aux, "pdca_source_weights")
        names_by_scale = self._scale_dict(pdca_aux, "pdca_source_names_by_target")
        for scale_key, scale_weights in weights_by_scale.items():
            scale_names = names_by_scale.get(scale_key)
            if scale_weights and scale_names:
                return True
        return False

    def _source_weight(
        self,
        pdca_aux: Dict[str, Any],
        scale_key: str,
        target_name: str,
        source_name: str,
        like: torch.Tensor,
    ) -> torch.Tensor:
        weights_by_scale = self._scale_dict(pdca_aux, "pdca_source_weights")
        names_by_scale = self._scale_dict(pdca_aux, "pdca_source_names_by_target")
        scale_weights = weights_by_scale.get(scale_key)
        scale_names = names_by_scale.get(scale_key)
        if scale_weights is None or scale_names is None:
            return like.new_zeros(like.shape[0], 1, like.shape[-2], like.shape[-1])
        if target_name not in scale_weights or target_name not in scale_names:
            raise RuntimeError("PDCA guidance missing target '%s' at scale %s" % (target_name, scale_key))
        source_names = tuple(scale_names[target_name])
        if source_name not in source_names:
            raise RuntimeError(
                "PDCA guidance missing source '%s' for target '%s' at scale %s"
                % (source_name, target_name, scale_key)
            )
        weights = scale_weights[target_name]
        if weights.ndim != 5:
            raise RuntimeError(
                "pdca_source_weights[%s][%s] must be [B,G,Q,H,W], got %r"
                % (scale_key, target_name, tuple(weights.shape))
            )
        if self.detach_pdca_guidance:
            weights = weights.detach()
        q_idx = source_names.index(source_name)
        out = weights[:, :, q_idx].mean(dim=1, keepdim=True)
        if out.shape[-2:] != like.shape[-2:]:
            out = F.interpolate(out, size=like.shape[-2:], mode="bilinear", align_corners=False)
        return out.to(device=like.device, dtype=like.dtype)

    def _make_guidance(
        self,
        pdca_aux: Optional[Dict[str, Any]],
        scale_key: str,
        phase_i: str,
        phase_j: str,
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        if not self.use_pdca_guidance:
            return like.new_zeros(like.shape[0], 4, like.shape[-2], like.shape[-1]), False
        if not pdca_aux:
            return like.new_zeros(like.shape[0], 4, like.shape[-2], like.shape[-1]), False

        weights_by_scale = self._scale_dict(pdca_aux, "pdca_source_weights")
        names_by_scale = self._scale_dict(pdca_aux, "pdca_source_names_by_target")
        has_weights = scale_key in weights_by_scale
        has_names = scale_key in names_by_scale
        if not has_weights and not has_names:
            return like.new_zeros(like.shape[0], 4, like.shape[-2], like.shape[-1]), False
        if has_weights != has_names:
            raise RuntimeError(
                "PDCA guidance scale %s requires both pdca_source_weights and "
                "pdca_source_names_by_target" % scale_key
            )

        r = torch.cat(
            [
                self._source_weight(pdca_aux, scale_key, phase_i, phase_j, like),
                self._source_weight(pdca_aux, scale_key, phase_j, phase_i, like),
                self._source_weight(pdca_aux, scale_key, phase_i, "__null__", like),
                self._source_weight(pdca_aux, scale_key, phase_j, "__null__", like),
            ],
            dim=1,
        )
        return r, True

    def forward(
        self,
        phase_feats: Dict[str, List[torch.Tensor]],
        output_size: Tuple[int, int],
        pdca_aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if self.use_pdca_guidance and not self._has_any_guidance(pdca_aux):
            raise RuntimeError(
                "PDCA-guided pair decoder requires pdca_source_weights and "
                "pdca_source_names_by_target on at least one relation scale"
            )

        num_scales = len(self.in_channels)
        for phase_name in self.phase_names:
            if phase_name not in phase_feats or len(phase_feats[phase_name]) != num_scales:
                raise ValueError("phase_feats[%s] must contain %d scales" % (phase_name, num_scales))

        proj_all: List[torch.Tensor] = []
        pair_debug: Dict[str, Any] = {
            "gate": {key: [] for key in self.pair_keys},
            "has_pdca_guidance": {key: [] for key in self.pair_keys},
            "alpha": [],
        }

        for s, c_exp in enumerate(self.in_channels):
            x_by_pair = []
            gates_by_pair = []
            has_by_pair = []
            scale_key = str(s)
            for pair_key, (phase_i, phase_j) in zip(self.pair_keys, self.pair_names):
                fi = phase_feats[phase_i][s]
                fj = phase_feats[phase_j][s]
                if fi.shape != fj.shape or fi.ndim != 4:
                    raise ValueError("Pair %s scale %d expects matching [B,C,H,W] features" % (pair_key, s))
                if fi.shape[1] != c_exp:
                    raise ValueError("Scale %d expects C=%d, got C=%d" % (s, c_exp, fi.shape[1]))
                diff = self._make_diff(fi, fj)
                r, has_guidance = self._make_guidance(pdca_aux, scale_key, phase_i, phase_j, fi)
                gate = self.spatial_gates[s](r)
                alpha = self.alpha_max * torch.sigmoid(self.raw_gate_scales[s]).view(1, 1, 1, 1)
                x_by_pair.append(self.diff_proj[s](diff) * (1.0 + alpha * gate))
                gates_by_pair.append(gate)
                has_by_pair.append(has_guidance)

            proj_all.append(torch.cat(x_by_pair, dim=0))  # [3B,D,H_s,W_s]
            pair_debug["alpha"].append((self.alpha_max * torch.sigmoid(self.raw_gate_scales[s])).detach())
            for pair_key, gate, has_guidance in zip(self.pair_keys, gates_by_pair, has_by_pair):
                pair_debug["gate"][pair_key].append(gate)
                pair_debug["has_pdca_guidance"][pair_key].append(bool(has_guidance))

        # Decode all fixed pairs in one shared pass. k is intentionally fresh per forward.
        x, k = self.deep_block(proj_all[-1], None)
        for block, skip in zip(self.up_blocks, reversed(proj_all[:-1])):
            x, k = block(x, skip, k)

        logits_all = self.change_head(x)
        logits_all = F.interpolate(logits_all, size=output_size, mode="bilinear", align_corners=False)
        batch_size = phase_feats[self.pair_names[0][0]][0].shape[0]
        logits_split = torch.split(logits_all, batch_size, dim=0)
        change_logits_dict = {
            pair_key: logits for pair_key, logits in zip(self.pair_keys, logits_split)
        }
        return change_logits_dict, pair_debug


class PhaseAffine(nn.Module):
    """Optional phase-specific affine modulation."""

    def __init__(self, num_phases: int, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_phases, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_phases, channels, 1, 1))

    def forward(self, x: torch.Tensor, phase_idx: int) -> torch.Tensor:
        return x * self.gamma[phase_idx] + self.beta[phase_idx]


class SegmentationHead(nn.Module):
    """x: [B, D, H, W] -> logits: [B, num_sem_classes, H, W]"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_phases: int = 3,
        dropout: float = 0.0,
        use_phase_classifier_bias: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.use_phase_classifier_bias = bool(use_phase_classifier_bias)
        if self.use_phase_classifier_bias:
            self.phase_bias = nn.Parameter(torch.zeros(num_phases, num_classes, 1, 1))
        else:
            self.register_parameter("phase_bias", None)

    def forward(self, x: torch.Tensor, phase_idx: Optional[int] = None) -> torch.Tensor:
        logits = self.classifier(self.dropout(x))
        if self.use_phase_classifier_bias:
            if phase_idx is None:
                raise ValueError("phase_idx must be provided when use_phase_classifier_bias=True.")
            logits = logits + self.phase_bias[phase_idx]
        return logits


class ChangeHead(nn.Module):
    """x: [B, D, H, W] -> logits: [B, num_change_classes, H, W]"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class MTSCDDecoderNet(nn.Module):
    """
    Decoder/readout-only network for:
        feature_xy[s]: [T, B, C_s, H_s, W_s]

    Output:
        sem_logits: [B, 3, num_sem_classes, H_in, W_in]
        chg_logits: [B, num_change_classes, H_in, W_in]

    Important:
    - T is SNN neural time, NOT physical phase index.
    - feature_xy outer index is scale index, NOT physical phase index.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        num_sem_classes: int,
        num_change_classes: int,
        input_size: Optional[Tuple[int, int]] = None,
        phase_windows: Optional[Dict[str, Sequence[int]]] = None,
        transition_windows: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        temporal_readout: str = "attention",
        diff_mode: str = "abs_signed",
        share_semantic_decoder: bool = True,
        feature_order: str = "high_to_low",
        phase_anchor_bias: float = 1.5,
        use_phase_affine: bool = False,
        use_phase_classifier_bias: bool = False,
        use_transition_fusion: bool = True,
        sem_head_dropout: float = 0.0,
        chg_head_dropout: float = 0.0,
        return_intermediates_default: bool = False,
        use_pdca_guided_pair_decoder: bool = False,
        detach_pdca_guidance: bool = True,
        use_pdca_guidance: bool = True,
    ):
        super().__init__()

        if not isinstance(in_channels, (list, tuple)) or len(in_channels) == 0:
            raise ValueError("in_channels must be a non-empty list/tuple.")
        if feature_order not in ("high_to_low", "low_to_high"):
            raise ValueError("feature_order must be 'high_to_low' or 'low_to_high'.")

        # Normalize channels to internal high_to_low order.
        self.feature_order = feature_order
        self.in_channels = list(map(int, in_channels))
        if self.feature_order == "low_to_high":
            self.in_channels = list(reversed(self.in_channels))

        self.num_scales = len(self.in_channels)
        self.decoder_channels = int(decoder_channels)
        self.num_sem_classes = int(num_sem_classes)
        self.num_change_classes = int(num_change_classes)
        self.input_size = tuple(input_size) if input_size is not None else None
        self.phase_windows = phase_windows
        self.transition_windows = transition_windows
        self.temporal_readout = temporal_readout
        self.diff_mode = diff_mode
        self.share_semantic_decoder = bool(share_semantic_decoder)
        self.use_phase_affine = bool(use_phase_affine)
        self.use_phase_classifier_bias = bool(use_phase_classifier_bias)
        self.use_transition_fusion = bool(use_transition_fusion)
        self.return_intermediates_default = bool(return_intermediates_default)
        self.use_pdca_guided_pair_decoder = bool(use_pdca_guided_pair_decoder)
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.use_pdca_guidance = bool(use_pdca_guidance)

        # One temporal readout per scale for semantic branch.
        self.phase_readouts = nn.ModuleList([
            TemporalReadout(c, mode=self.temporal_readout, anchor_bias=phase_anchor_bias)
            for c in self.in_channels
        ])

        # Transition readout: same pooling mode, but no anchor bias.
        self.transition_readouts = nn.ModuleList([
            TemporalReadout(c, mode=self.temporal_readout, anchor_bias=0.0)
            for c in self.in_channels
        ])

        # Semantic decoder(s)
        if self.share_semantic_decoder:
            self.semantic_decoder = SemanticDecoder(self.in_channels, self.decoder_channels)
            self.semantic_head = SegmentationHead(
                self.decoder_channels,
                self.num_sem_classes,
                num_phases=3,
                dropout=sem_head_dropout,
                use_phase_classifier_bias=self.use_phase_classifier_bias,
            )
            self.phase_affine = PhaseAffine(3, self.decoder_channels) if self.use_phase_affine else None
        else:
            self.semantic_decoder = nn.ModuleDict({
                p: SemanticDecoder(self.in_channels, self.decoder_channels) for p in PHASE_NAMES
            })
            self.semantic_head = nn.ModuleDict({
                p: SegmentationHead(
                    self.decoder_channels,
                    self.num_sem_classes,
                    num_phases=1,
                    dropout=sem_head_dropout,
                    use_phase_classifier_bias=self.use_phase_classifier_bias,
                )
                for p in PHASE_NAMES
            })
            self.phase_affine = (
                nn.ModuleDict({p: PhaseAffine(1, self.decoder_channels) for p in PHASE_NAMES})
                if self.use_phase_affine else None
            )

        if self.use_pdca_guided_pair_decoder:
            self.pair_change_decoder = PDCAGuidedPairwiseChangeDecoder(
                in_channels=self.in_channels,
                decoder_channels=self.decoder_channels,
                num_change_classes=self.num_change_classes,
                detach_pdca_guidance=self.detach_pdca_guidance,
                use_pdca_guidance=self.use_pdca_guidance,
            )
            self.change_decoder = None
            self.change_head = None
        else:
            self.pair_change_decoder = None
            self.change_decoder = ChangeDecoder(
                in_channels=self.in_channels,
                decoder_channels=self.decoder_channels,
                diff_mode=self.diff_mode,
                use_transition_fusion=self.use_transition_fusion,
            )
            self.change_head = ChangeHead(self.decoder_channels, self.num_change_classes, dropout=chg_head_dropout)

    def _normalize_feature_order(self, feature_xy: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feature_xy = list(feature_xy)
        return feature_xy if self.feature_order == "high_to_low" else list(reversed(feature_xy))

    def _validate_feature_xy(self, feature_xy: Sequence[torch.Tensor]) -> Tuple[int, int]:
        if not isinstance(feature_xy, (list, tuple)):
            raise TypeError(f"feature_xy must be list/tuple, but got {type(feature_xy)}.")
        if len(feature_xy) != self.num_scales:
            raise ValueError(
                f"len(feature_xy)={len(feature_xy)} does not match len(in_channels)={self.num_scales}."
            )

        feat_list = self._normalize_feature_order(feature_xy)
        T = None
        B = None

        for s, (feat, c_exp) in enumerate(zip(feat_list, self.in_channels)):
            if not torch.is_tensor(feat):
                raise TypeError(f"feature_xy[{s}] must be Tensor, but got {type(feat)}.")
            if feat.ndim != 5:
                raise ValueError(
                    f"feature_xy[{s}] must be 5D [T,B,C,H,W], but got {tuple(feat.shape)}."
                )

            t, b, c, h, w = feat.shape
            if c != c_exp:
                raise ValueError(f"feature_xy[{s}] expects C={c_exp}, got C={c}.")
            if h <= 0 or w <= 0:
                raise ValueError(f"feature_xy[{s}] has invalid spatial size {(h, w)}.")

            if T is None:
                T, B = t, b
            else:
                if t != T:
                    raise ValueError(f"All scales must share the same T, got {T} and {t}.")
                if b != B:
                    raise ValueError(f"All scales must share the same B, got {B} and {b}.")

        return int(T), int(B)

    def _resolve_output_size(
        self,
        input_size: Optional[Tuple[int, int]],
        highest_res_feature: torch.Tensor,
    ) -> Tuple[int, int]:
        if input_size is not None:
            out_size = tuple(int(v) for v in input_size)
        elif self.input_size is not None:
            out_size = tuple(int(v) for v in self.input_size)
        else:
            out_size = tuple(int(v) for v in highest_res_feature.shape[-2:])
        if len(out_size) != 2:
            raise ValueError(f"input_size must be (H, W), but got {out_size}.")
        return out_size

    def _read_phase_and_transition_features(
        self,
        feature_xy_high_to_low: Sequence[torch.Tensor],
        phase_windows: Dict[str, List[int]],
        transition_windows: Dict[str, Optional[List[int]]],
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[Optional[torch.Tensor]]]]:
        phase_feats: Dict[str, List[torch.Tensor]] = {k: [] for k in PHASE_NAMES}
        transition_feats: Dict[str, List[Optional[torch.Tensor]]] = {k: [] for k in TRANSITION_NAMES}

        for s, seq in enumerate(feature_xy_high_to_low):
            # seq: [T, B, C_s, H_s, W_s]
            for phase_name in PHASE_NAMES:
                window = phase_windows[phase_name]
                anchor_idx = window[0]  # default anchor = first element of the window
                pooled = self.phase_readouts[s](
                    seq, time_window=window, anchor_index=anchor_idx
                )
                phase_feats[phase_name].append(pooled)  # [B, C_s, H_s, W_s]

            for trans_name in TRANSITION_NAMES:
                window = transition_windows.get(trans_name, None)
                if window is None:
                    transition_feats[trans_name].append(None)
                else:
                    pooled = self.transition_readouts[s](
                        seq, time_window=window, anchor_index=None
                    )
                    transition_feats[trans_name].append(pooled)  # [B, C_s, H_s, W_s]

        return phase_feats, transition_feats

    def _decode_semantic(
        self,
        phase_feats: Dict[str, List[torch.Tensor]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sem_logits_dict: Dict[str, torch.Tensor] = {}
        sem_decoded_dict: Dict[str, torch.Tensor] = {}

        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            feats = phase_feats[phase_name]  # list of [B, C_s, H_s, W_s], high_to_low

            if self.share_semantic_decoder:
                decoded = self.semantic_decoder(feats)  # [B, D, H0, W0]
                if self.phase_affine is not None:
                    decoded = self.phase_affine(decoded, phase_idx)
                logits = self.semantic_head(decoded, phase_idx=phase_idx)
            else:
                decoded = self.semantic_decoder[phase_name](feats)
                if self.phase_affine is not None:
                    decoded = self.phase_affine[phase_name](decoded, phase_idx=0)
                logits = self.semantic_head[phase_name](
                    decoded,
                    phase_idx=0 if self.use_phase_classifier_bias else None,
                )

            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            sem_logits_dict[phase_name] = logits
            sem_decoded_dict[phase_name] = decoded

        sem_logits = torch.stack(
            [sem_logits_dict["t1"], sem_logits_dict["t2"], sem_logits_dict["t3"]],
            dim=1,
        )  # [B, 3, num_sem_classes, H, W]

        return sem_logits, sem_logits_dict, sem_decoded_dict

    def _fuse_transition_for_t1_to_t3(
        self,
        transition_feats: Dict[str, List[Optional[torch.Tensor]]],
    ) -> List[Optional[torch.Tensor]]:
        # Priority 1: user/auto provided t1_to_t3 transition window
        if "t1_to_t3" in transition_feats and any(x is not None for x in transition_feats["t1_to_t3"]):
            return transition_feats["t1_to_t3"]

        # Priority 2: average t1_to_t2 and t2_to_t3 if both exist
        t12 = transition_feats.get("t1_to_t2", [None] * self.num_scales)
        t23 = transition_feats.get("t2_to_t3", [None] * self.num_scales)

        fused: List[Optional[torch.Tensor]] = []
        for s in range(self.num_scales):
            a = t12[s]
            b = t23[s]
            if a is None and b is None:
                fused.append(None)
            elif a is None:
                fused.append(b)
            elif b is None:
                fused.append(a)
            else:
                if a.shape[-2:] != b.shape[-2:]:
                    b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
                fused.append(0.5 * (a + b))
        return fused

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        input_size: Optional[Tuple[int, int]] = None,
        return_intermediates: Optional[bool] = None,
        pdca_aux: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[Optional[torch.Tensor]]], Dict[str, List[int]], Dict[str, Optional[List[int]]]]]:
        # Validate feature structure
        T, _ = self._validate_feature_xy(feature_xy)
        feature_xy_high_to_low = self._normalize_feature_order(feature_xy)

        # Resolve windows
        phase_windows, transition_windows = normalize_windows(
            T=T,
            phase_windows=self.phase_windows,
            transition_windows=self.transition_windows,
        )

        # Resolve output size
        # feature_xy_high_to_low[0][0]: [B, C0, H0, W0]
        output_size = self._resolve_output_size(input_size, feature_xy_high_to_low[0][0])

        if return_intermediates is None:
            return_intermediates = self.return_intermediates_default

        # -----------------------------
        # Goal 1: multi-scale phase readout
        # phase_feats["t1"][s]: [B, C_s, H_s, W_s]
        # -----------------------------
        phase_feats, transition_feats = self._read_phase_and_transition_features(
            feature_xy_high_to_low, phase_windows, transition_windows
        )

        # -----------------------------
        # Goal 2: semantic branch
        # sem_logits: [B, 3, num_sem_classes, H, W]
        # -----------------------------
        sem_logits, sem_logits_dict, sem_decoded_feats = self._decode_semantic(
            phase_feats, output_size
        )

        # -----------------------------
        # Goal 3: change branch
        # chg_logits: [B, num_change_classes, H, W]
        # -----------------------------
        pair_debug = None
        change_decoded = None
        multi_scale_change_inputs = None
        transition_t13 = None
        if self.use_pdca_guided_pair_decoder:
            change_logits_dict, pair_debug = self.pair_change_decoder(
                phase_feats=phase_feats,
                output_size=output_size,
                pdca_aux=pdca_aux,
            )
            chg_logits = change_logits_dict["t1_to_t3"]
        else:
            transition_t13 = self._fuse_transition_for_t1_to_t3(transition_feats)
            change_decoded, multi_scale_change_inputs = self.change_decoder(
                phase_feats["t1"], phase_feats["t3"], transition_t13
            )
            chg_logits = self.change_head(change_decoded)
            chg_logits = F.interpolate(chg_logits, size=output_size, mode="bilinear", align_corners=False)
            change_logits_dict = {"t1_to_t3": chg_logits}

        outputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[Optional[torch.Tensor]]], Dict[str, List[int]], Dict[str, Optional[List[int]]]]] = {
            "sem_logits": sem_logits,
            "sem_logits_dict": sem_logits_dict,
            "chg_logits": chg_logits,
            "change_logits_dict": change_logits_dict,
            "phase_windows": phase_windows,
            "transition_windows": transition_windows,
        }
        if pair_debug is not None:
            outputs["pair_gate_debug"] = pair_debug

        if return_intermediates:
            outputs["phase_features"] = phase_feats
            outputs["transition_features"] = transition_feats
            outputs["semantic_decoded_features"] = sem_decoded_feats
            if self.use_pdca_guided_pair_decoder:
                outputs["change_features"] = {
                    "pair_gate_debug": pair_debug,
                }
            else:
                outputs["change_features"] = {
                    "multi_scale_change_inputs": multi_scale_change_inputs,
                    "decoded_change_feature": change_decoded,
                    "fused_transition_t1_to_t3": transition_t13,
                }

        return outputs
import torch

if __name__ == "__main__":
    T, B = 8, 2
    H, W = 256, 256
    in_channels = [64, 128, 256, 512, 768]
    num_sem_classes = 13
    num_change_classes = 1

    # feature_xy[s]: [T, B, C_s, H_s, W_s]
    feature_xy = [
        torch.randn(T, B, 64, 128, 128),
        torch.randn(T, B, 128, 64, 64),
        torch.randn(T, B, 256, 32, 32),
        torch.randn(T, B, 512, 16, 16),
        torch.randn(T, B, 768, 8, 8),
    ]

    model = MTSCDDecoderNet(
        in_channels=in_channels,
        decoder_channels=256,
        num_sem_classes=num_sem_classes,
        num_change_classes=num_change_classes,
        input_size=(H, W),
        phase_windows={"t1": [0, 1], "t2": [3, 4], "t3": [6, 7]},          # T=12 -> 默认 [0:4], [4:8], [8:12]
        transition_windows={"t1_to_t2": [2], "t2_to_t3": [5], "t1_to_t3": None},     # T=12 -> 默认无 transition windows
        temporal_readout="attention",
        diff_mode="abs_signed",
        share_semantic_decoder=True,
        feature_order="high_to_low",
        use_phase_affine=False,
        use_phase_classifier_bias=False,
        use_transition_fusion=True,
        return_intermediates_default=True,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(feature_xy, input_size=(H, W))

    print("sem_logits shape:", outputs["sem_logits"].shape)
    print("chg_logits shape:", outputs["chg_logits"].shape)
    print("sem_logits_dict[t1] shape:", outputs["sem_logits_dict"]["t1"].shape)
    print("resolved phase_windows:", outputs["phase_windows"])
    print("resolved transition_windows:", outputs["transition_windows"])

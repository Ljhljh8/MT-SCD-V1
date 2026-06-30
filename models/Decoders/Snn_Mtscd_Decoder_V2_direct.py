from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant
from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d

PHASE_NAMES = ("t1", "t2", "t3")


def _as_2tuple(size: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if len(size) != 2:
        raise ValueError(f"input_size must be (H, W), got {size!r}")
    return int(size[0]), int(size[1])


def _interpolate_logits(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DendBlock(nn.Module):
    """
    4D decoder block.

    Input:
        x: [B*, Cin, H, W], where B* may be B or N*B.
    Output:
        y: [B*, Cout, H, W]

    DendFADCConv2d internally receives [1, B*, C, H, W]. Therefore flattening
    the physical phase axis into B* does not require any extra phase readout.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
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
            Down_K=False,
        )
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, k=None):
        if x.ndim != 4:
            raise ValueError(f"DendBlock expects [B*,C,H,W], got {tuple(x.shape)}")

        y, k = self.block(x.unsqueeze(0), k)
        y = self.BN(y.flatten(0, 1))
        return y, k


class ConvBlock(nn.Module):
    """Input: [B*, Cin, H, W] -> Output: [B*, Cout, H, W]."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            Q_IFNode(surrogate_function=Quant()),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Q_IFNode(surrogate_function=Quant()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ConvBlock expects [B*,C,H,W], got {tuple(x.shape)}")
        return self.block(x)


class UpFuseBlock(nn.Module):
    """
    x:    [B*, Cx, H_low, W_low]
    skip: [B*, Cs, H_high, W_high]
    out:  [B*, Cout, H_high, W_high]
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_DendSize: int = 128):
        super().__init__()
        self.fuse1 = DendBlock(in_channels + skip_channels, out_channels)
        self.fuse2 = ConvBlock(in_channels + skip_channels, out_channels)
        self.use_DendSize = int(use_DendSize)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, k=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        if k is not None and len(k) > 0 and k[0].shape[-2:] != skip.shape[-2:]:
            # Current DendFADC setting uses fs_cfg k_list=[2,4], hence two K maps.
            k_tensor = torch.cat(k, dim=1)
            k_tensor = F.interpolate(k_tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            k = list(torch.chunk(k_tensor, chunks=len(k), dim=1))

        x = torch.cat([x, skip], dim=1)
        if skip.shape[-1] >= self.use_DendSize:
            return self.fuse2(x), k
        return self.fuse1(x, k)


class DirectSemanticDecoder(nn.Module):
    """
    Direct N*B semantic decoder.

    Input:
        feats_high_to_low[s]: [N,B,C_s,H_s,W_s]
    Output:
        decoded_all: [N*B,D,H_0,W_0]
    """

    def __init__(self, in_channels: Sequence[int], decoder_channels: int):
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.decoder_channels = int(decoder_channels)

        self.laterals = nn.ModuleList(
            [nn.Sequential(
                Q_IFNode(surrogate_function=Quant()),
                nn.Conv2d(c, self.decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.decoder_channels),
            )
                for c in self.in_channels]
        )
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList(
            [
                UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, use_DendSize=16)
                for _ in range(len(self.in_channels) - 1)
            ]
        )

    def forward(self, feats_high_to_low: Sequence[torch.Tensor], k=None) -> torch.Tensor:
        if len(feats_high_to_low) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} scales, got {len(feats_high_to_low)}")

        proj: List[torch.Tensor] = []
        for s, (feat, c_exp) in enumerate(zip(feats_high_to_low, self.in_channels)):
            if feat.ndim != 5:
                raise ValueError(f"Scale {s} must be [N,B,C,H,W], got {tuple(feat.shape)}")
            if feat.shape[2] != c_exp:
                raise ValueError(f"Scale {s} expects C={c_exp}, got C={feat.shape[2]}")
            feat_flat = feat.flatten(0, 1).contiguous()  # [N*B,C,H,W]
            proj.append(self.laterals[s](feat_flat))

        x, k = self.deep_block(proj[-1], k)
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)
        return x


class SegmentationHead(nn.Module):
    """x: [B*, D, H, W] -> logits: [B*, num_classes, H, W]."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class ChangeHead(nn.Module):
    """x: [B, D, H, W] -> logits: [B, num_change_classes, H, W]."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class DirectChangeDecoder(nn.Module):
    """
    Direct t1->t3 change decoder without phase feature dictionaries.

    Input:
        feature_xy[s]: [N,B,C_s,H_s,W_s]

    It only indexes the requested pair at each scale:
        f_i = feature_xy[s][pair_i]
        f_j = feature_xy[s][pair_j]

    diff_mode is kept compatible with the previous ChangeDecoder:
        abs        -> |F_j - F_i|
        abs_signed -> [|F_j-F_i|, relu(F_j-F_i), relu(F_i-F_j)]
        concat     -> [F_i, F_j, |F_j-F_i|]
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        diff_mode: str = "abs_signed",
        pair_indices: Tuple[int, int] = (0, 2),
    ):
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.decoder_channels = int(decoder_channels)
        self.diff_mode = str(diff_mode)
        self.pair_indices = (int(pair_indices[0]), int(pair_indices[1]))

        if self.diff_mode not in ("abs", "abs_signed", "concat"):
            raise ValueError(f"Unsupported diff_mode={diff_mode}")

        proj_in_channels = [c if self.diff_mode == "abs" else 3 * c for c in self.in_channels]
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, self.decoder_channels, kernel_size=1, bias=False) for c in proj_in_channels]
        )
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList(
            [
                UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, use_DendSize=16)
                for _ in range(len(self.in_channels) - 1)
            ]
        )

    def _make_pair_input(self, f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(f_j - f_i)
        if self.diff_mode == "abs":
            return abs_diff
        if self.diff_mode == "abs_signed":
            return torch.cat([abs_diff, F.relu(f_j - f_i), F.relu(f_i - f_j)], dim=1)
        if self.diff_mode == "concat":
            return torch.cat([f_i, f_j, abs_diff], dim=1)
        raise RuntimeError("Unreachable diff_mode branch")

    def forward(self, feature_xy: Sequence[torch.Tensor], k=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if len(feature_xy) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} scales, got {len(feature_xy)}")

        pair_i, pair_j = self.pair_indices
        multi_scale_change_inputs: List[torch.Tensor] = []
        proj: List[torch.Tensor] = []

        for s, (feat, c_exp) in enumerate(zip(feature_xy, self.in_channels)):
            if feat.ndim != 5:
                raise ValueError(f"Scale {s} must be [N,B,C,H,W], got {tuple(feat.shape)}")
            if feat.shape[0] <= max(pair_i, pair_j):
                raise ValueError(
                    f"Scale {s} has N={feat.shape[0]}, but change_pair_indices={self.pair_indices} was requested"
                )
            if feat.shape[2] != c_exp:
                raise ValueError(f"Scale {s} expects C={c_exp}, got C={feat.shape[2]}")

            f_i = feat[pair_i].contiguous()  # [B,C,H,W]
            f_j = feat[pair_j].contiguous()  # [B,C,H,W]
            x_in = self._make_pair_input(f_i, f_j)
            multi_scale_change_inputs.append(x_in)
            proj.append(self.laterals[s](x_in))

        x, k = self.deep_block(proj[-1], k)
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)
        return x, multi_scale_change_inputs


class DirectPDCAFeatureGuidedPairChangeDecoder(DirectChangeDecoder):
    """
    Minimal compatibility path for use_pdca_guided_pair_decoder=True.

    The uploaded decoder does not contain the original PDCA-guided pair decoder.
    This class preserves the constructor/optimizer split and allows a guidance map
    to modulate per-scale pair inputs when the current branch supplies it.

    Expected pdca_guidance formats:
        None: no modulation, unless guidance_required=True.
        list/tuple: guidance[s] is [B,1,H_s,W_s] or [B,H_s,W_s].
        dict: key may be int scale index or str(scale index).

    This is a compatibility hook, not evidence that PDCA guidance improves metrics.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        diff_mode: str = "abs_signed",
        pair_indices: Tuple[int, int] = (0, 2),
        guidance_strength: float = 1.0,
        guidance_required: bool = False,
    ):
        super().__init__(in_channels, decoder_channels, diff_mode=diff_mode, pair_indices=pair_indices)
        self.guidance_strength = float(guidance_strength)
        self.guidance_required = bool(guidance_required)

    def _get_guidance_at_scale(self, pdca_guidance: Any, scale_idx: int) -> Optional[torch.Tensor]:
        if pdca_guidance is None:
            return None
        if isinstance(pdca_guidance, (list, tuple)):
            if scale_idx >= len(pdca_guidance):
                return None
            return pdca_guidance[scale_idx]
        if isinstance(pdca_guidance, dict):
            if scale_idx in pdca_guidance:
                return pdca_guidance[scale_idx]
            key = str(scale_idx)
            if key in pdca_guidance:
                return pdca_guidance[key]
            return None
        raise TypeError(f"Unsupported pdca_guidance type: {type(pdca_guidance)}")

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        pdca_guidance: Any = None,
        k=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.guidance_required and pdca_guidance is None:
            raise RuntimeError(
                "use_pdca_guided_pair_decoder=True requires pdca_guidance, "
                "but MTSCDDecoderNet.forward(..., pdca_guidance=None) was called."
            )

        pair_i, pair_j = self.pair_indices
        multi_scale_change_inputs: List[torch.Tensor] = []
        proj: List[torch.Tensor] = []

        for s, (feat, c_exp) in enumerate(zip(feature_xy, self.in_channels)):
            if feat.ndim != 5:
                raise ValueError(f"Scale {s} must be [N,B,C,H,W], got {tuple(feat.shape)}")
            if feat.shape[0] <= max(pair_i, pair_j):
                raise ValueError(
                    f"Scale {s} has N={feat.shape[0]}, but change_pair_indices={self.pair_indices} was requested"
                )
            if feat.shape[2] != c_exp:
                raise ValueError(f"Scale {s} expects C={c_exp}, got C={feat.shape[2]}")

            f_i = feat[pair_i].contiguous()
            f_j = feat[pair_j].contiguous()
            x_in = self._make_pair_input(f_i, f_j)

            guidance = self._get_guidance_at_scale(pdca_guidance, s)
            if guidance is not None:
                if guidance.ndim == 3:
                    guidance = guidance.unsqueeze(1)
                if guidance.ndim != 4 or guidance.shape[0] != x_in.shape[0]:
                    raise ValueError(
                        f"Guidance at scale {s} must be [B,1,H,W] or [B,H,W], got {tuple(guidance.shape)}"
                    )
                guidance = guidance.float()
                if guidance.shape[-2:] != x_in.shape[-2:]:
                    guidance = F.interpolate(guidance, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
                x_in = x_in * (1.0 + self.guidance_strength * guidance)

            multi_scale_change_inputs.append(x_in)
            proj.append(self.laterals[s](x_in))

        x, k = self.deep_block(proj[-1], k)
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)
        return x, multi_scale_change_inputs


class MTSCDDecoderNet(nn.Module):
    """
    Direct 5D / N*B flatten decoder for MTSCD.

    Input:
        feature_xy[s]: [N,B,C_s,H_s,W_s]

    Outputs:
        sem_logits:      [B,3,num_sem_classes,H,W]
        sem_logits_seq:  [N,B,num_sem_classes,H,W]
        sem_logits_dict: at least keys t1/t2/t3 when N>=3
        chg_logits:      [B,num_change_classes,H,W]

    Removed from the forward path:
        TemporalReadout, phase_windows, transition_windows, transition_feats,
        transition fusion, and phase feature dictionaries.
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
        use_transition_fusion: bool = False,
        sem_head_dropout: float = 0.0,
        chg_head_dropout: float = 0.0,
        return_intermediates_default: bool = False,
        change_pair_indices: Tuple[int, int] = (0, 2),
        use_pdca_guided_pair_decoder: bool = False,
        pdca_guidance_strength: float = 1.0,
        pdca_guidance_required: bool = False,
        **unused_kwargs,
    ):
        super().__init__()

        if not isinstance(in_channels, (list, tuple)) or len(in_channels) == 0:
            raise ValueError("in_channels must be a non-empty list/tuple")
        if feature_order not in ("high_to_low", "low_to_high"):
            raise ValueError("feature_order must be 'high_to_low' or 'low_to_high'")
        if not share_semantic_decoder:
            raise ValueError("Direct N*B semantic path only supports share_semantic_decoder=True")
        if use_transition_fusion:
            raise ValueError("Direct decoder removes transition fusion; set use_transition_fusion=False")
        if use_phase_affine:
            raise ValueError("Direct decoder currently removes phase_affine; set use_phase_affine=False")
        if use_phase_classifier_bias:
            raise ValueError("Direct decoder currently removes phase classifier bias; set use_phase_classifier_bias=False")

        self.feature_order = feature_order
        self.in_channels = [int(c) for c in in_channels]
        if self.feature_order == "low_to_high":
            self.in_channels = list(reversed(self.in_channels))

        self.num_scales = len(self.in_channels)
        self.decoder_channels = int(decoder_channels)
        self.num_sem_classes = int(num_sem_classes)
        self.num_change_classes = int(num_change_classes)
        self.input_size = _as_2tuple(input_size)
        self.diff_mode = str(diff_mode)
        self.return_intermediates_default = bool(return_intermediates_default)
        self.change_pair_indices = (int(change_pair_indices[0]), int(change_pair_indices[1]))
        self.use_pdca_guided_pair_decoder = bool(use_pdca_guided_pair_decoder)

        # Accepted only for backward-compatible construction. They are not used.
        self.phase_windows = phase_windows
        self.transition_windows = transition_windows
        self.temporal_readout = temporal_readout
        self.phase_anchor_bias = phase_anchor_bias

        self.semantic_decoder = DirectSemanticDecoder(self.in_channels, self.decoder_channels)
        self.semantic_head = SegmentationHead(
            self.decoder_channels,
            self.num_sem_classes,
            dropout=sem_head_dropout,
        )

        if self.use_pdca_guided_pair_decoder:
            self.pair_change_decoder = DirectPDCAFeatureGuidedPairChangeDecoder(
                in_channels=self.in_channels,
                decoder_channels=self.decoder_channels,
                diff_mode=self.diff_mode,
                pair_indices=self.change_pair_indices,
                guidance_strength=pdca_guidance_strength,
                guidance_required=pdca_guidance_required,
            )
            self.pair_change_head = ChangeHead(
                self.decoder_channels,
                self.num_change_classes,
                dropout=chg_head_dropout,
            )
            self.change_decoder = None
            self.change_head = None
        else:
            self.change_decoder = DirectChangeDecoder(
                in_channels=self.in_channels,
                decoder_channels=self.decoder_channels,
                diff_mode=self.diff_mode,
                pair_indices=self.change_pair_indices,
            )
            self.change_head = ChangeHead(
                self.decoder_channels,
                self.num_change_classes,
                dropout=chg_head_dropout,
            )
            self.pair_change_decoder = None
            self.pair_change_head = None

    def _normalize_feature_order(self, feature_xy: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feature_xy = list(feature_xy)
        return feature_xy if self.feature_order == "high_to_low" else list(reversed(feature_xy))

    def _infer_nb_and_output_size(
        self,
        feature_xy_high_to_low: Sequence[torch.Tensor],
        input_size: Optional[Tuple[int, int]],
    ) -> Tuple[int, int, Tuple[int, int]]:
        if not isinstance(feature_xy_high_to_low, (list, tuple)) or len(feature_xy_high_to_low) != self.num_scales:
            raise ValueError(f"feature_xy must contain {self.num_scales} scales")
        first = feature_xy_high_to_low[0]
        if not torch.is_tensor(first) or first.ndim != 5:
            raise ValueError(f"feature_xy[0] must be [N,B,C,H,W], got {type(first)} / {getattr(first, 'shape', None)}")
        n, b = int(first.shape[0]), int(first.shape[1])



        # Minimal structural check: same N/B and expected channels.
        for s, (feat, c_exp) in enumerate(zip(feature_xy_high_to_low, self.in_channels)):
            if not torch.is_tensor(feat) or feat.ndim != 5:
                raise ValueError(f"feature_xy[{s}] must be [N,B,C,H,W], got {type(feat)} / {getattr(feat, 'shape', None)}")
            if int(feat.shape[0]) != n or int(feat.shape[1]) != b:
                raise ValueError(f"All scales must share N/B. Scale 0 has {(n,b)}, scale {s} has {tuple(feat.shape[:2])}")
            if int(feat.shape[2]) != int(c_exp):
                raise ValueError(f"Scale {s} expects C={c_exp}, got C={feat.shape[2]}")

        out_size = _as_2tuple(input_size) or self.input_size or tuple(int(v) for v in first.shape[-2:])
        return n, b, out_size

    def _build_semantic_outputs(
        self,
        sem_logits_all: torch.Tensor,
        n: int,
        b: int,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        sem_logits_all = _interpolate_logits(sem_logits_all, output_size)
        sem_logits_seq = sem_logits_all.view(n, b, self.num_sem_classes, output_size[0], output_size[1]).contiguous()

        sem_logits_dict: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(PHASE_NAMES):
            sem_logits_dict[name] = sem_logits_seq[idx]

        sem_logits = torch.stack([sem_logits_dict["t1"], sem_logits_dict["t2"], sem_logits_dict["t3"]], dim=1)
        return sem_logits, sem_logits_seq, sem_logits_dict

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        input_size: Optional[Tuple[int, int]] = None,
        return_intermediates: Optional[bool] = None,
        pdca_guidance: Any = None,
        pdca_aux: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]]:
        feature_xy_high_to_low = self._normalize_feature_order(feature_xy)
        n, b, output_size = self._infer_nb_and_output_size(feature_xy_high_to_low, input_size)

        if return_intermediates is None:
            return_intermediates = self.return_intermediates_default

        # Semantic path: one decoder pass for all physical phases.
        semantic_decoded_all = self.semantic_decoder(feature_xy_high_to_low)  # [N*B,D,H0,W0]
        sem_logits_all = self.semantic_head(semantic_decoded_all)             # [N*B,K,H0,W0]
        sem_logits, sem_logits_seq, sem_logits_dict = self._build_semantic_outputs(
            sem_logits_all, n=n, b=b, output_size=output_size
        )

        # Change path: direct pair indexing at each scale. No phase dict, no transition dict.
        if self.use_pdca_guided_pair_decoder:
            change_decoded, multi_scale_change_inputs = self.pair_change_decoder(
                feature_xy_high_to_low,
                pdca_guidance=pdca_guidance,
            )
            chg_logits = self.pair_change_head(change_decoded)
            change_mode = "pdca_guided_pair"
        else:
            change_decoded, multi_scale_change_inputs = self.change_decoder(feature_xy_high_to_low)
            chg_logits = self.change_head(change_decoded)
            change_mode = "direct_pair"

        chg_logits = _interpolate_logits(chg_logits, output_size)

        outputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]] = {
            "sem_logits": sem_logits,
            "sem_logits_seq": sem_logits_seq,
            "sem_logits_dict": sem_logits_dict,
            "chg_logits": chg_logits,
        }

        if return_intermediates:
            outputs["semantic_decoded_seq"] = semantic_decoded_all.view(
                n,
                b,
                self.decoder_channels,
                semantic_decoded_all.shape[-2],
                semantic_decoded_all.shape[-1],
            ).contiguous()
            outputs["change_features"] = {
                "mode": change_mode,
                "change_pair_indices": self.change_pair_indices,
                "multi_scale_change_inputs": multi_scale_change_inputs,
                "decoded_change_feature": change_decoded,
            }
            if self.use_pdca_guided_pair_decoder:
                outputs["change_features"]["pdca_guidance_provided"] = pdca_guidance is not None

        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, B = 3, 2
    H, W = 512, 512
    in_channels = [32, 64, 128, 360]
    feature_xy = [
        torch.randn(N, B, 32, 128, 128, device=device),
        torch.randn(N, B, 64, 64, 64, device=device),
        torch.randn(N, B, 128, 32, 32, device=device),
        torch.randn(N, B, 360, 16, 16, device=device),
    ]
    model = MTSCDDecoderNet(
        in_channels=in_channels,
        decoder_channels=256,
        num_sem_classes=13,
        num_change_classes=1,
        input_size=(H, W),
        diff_mode="abs_signed",
        use_transition_fusion=False,
        return_intermediates_default=True,
    ).to(device)
    outputs = model(feature_xy, input_size=(H, W))
    print(outputs["sem_logits"].shape)
    print(outputs["sem_logits_dict"]["t1"].shape)
    print(outputs["chg_logits"].shape)

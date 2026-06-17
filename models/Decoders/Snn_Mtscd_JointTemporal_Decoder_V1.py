import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union

from models.Decoders.Snn_Mtscd_Decoder_V2 import (
    PHASE_NAMES,
    ChangeDecoder,
    ChangeHead,
    SegmentationHead,
    SemanticDecoder,
)


class MTSCDJointTemporalDecoderNet(nn.Module):
    """
    Joint temporal decoder for direct-N MTSCD features.

    Input:
        feature_xy[s]: [N, B, C_s, H_s, W_s]

    Important:
        N is the physical remote-sensing phase index ordered as t1/t2/t3.
        It is not SNN micro-time, so this decoder does not apply temporal
        readout windows and does not call _read_phase_and_transition_features.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        num_sem_classes: int,
        num_change_classes: int,
        input_size: Optional[Tuple[int, int]] = None,
        diff_mode: str = "abs_signed",
        share_semantic_decoder: bool = True,
        feature_order: str = "high_to_low",
        use_transition_fusion: bool = False,
        sem_head_dropout: float = 0.0,
        chg_head_dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(in_channels, (list, tuple)) or len(in_channels) == 0:
            raise ValueError("in_channels must be a non-empty list/tuple.")
        if feature_order not in ("high_to_low", "low_to_high"):
            raise ValueError("feature_order must be 'high_to_low' or 'low_to_high'.")

        self.feature_order = feature_order
        self.in_channels = list(map(int, in_channels))
        if self.feature_order == "low_to_high":
            self.in_channels = list(reversed(self.in_channels))

        self.num_scales = len(self.in_channels)
        self.decoder_channels = int(decoder_channels)
        self.num_sem_classes = int(num_sem_classes)
        self.num_change_classes = int(num_change_classes)
        self.input_size = tuple(input_size) if input_size is not None else None
        self.diff_mode = diff_mode
        self.share_semantic_decoder = bool(share_semantic_decoder)
        self.use_transition_fusion = bool(use_transition_fusion)

        if self.share_semantic_decoder:
            self.semantic_decoder = SemanticDecoder(self.in_channels, self.decoder_channels)
            self.semantic_head = SegmentationHead(
                self.decoder_channels,
                self.num_sem_classes,
                num_phases=len(PHASE_NAMES),
                dropout=sem_head_dropout,
            )
        else:
            self.semantic_decoder = nn.ModuleDict({
                phase_name: SemanticDecoder(self.in_channels, self.decoder_channels)
                for phase_name in PHASE_NAMES
            })
            self.semantic_head = nn.ModuleDict({
                phase_name: SegmentationHead(
                    self.decoder_channels,
                    self.num_sem_classes,
                    num_phases=1,
                    dropout=sem_head_dropout,
                )
                for phase_name in PHASE_NAMES
            })

        self.change_decoder = ChangeDecoder(
            in_channels=self.in_channels,
            decoder_channels=self.decoder_channels,
            diff_mode=self.diff_mode,
            use_transition_fusion=self.use_transition_fusion,
        )
        self.change_head = ChangeHead(
            self.decoder_channels,
            self.num_change_classes,
            dropout=chg_head_dropout,
        )

    def _normalize_feature_order(self, feature_xy: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feature_xy = list(feature_xy)
        return feature_xy if self.feature_order == "high_to_low" else list(reversed(feature_xy))

    def _validate_feature_xy(self, feature_xy: Sequence[torch.Tensor]) -> Tuple[int, int]:
        if not isinstance(feature_xy, (list, tuple)):
            raise TypeError("feature_xy must be list/tuple, but got {}.".format(type(feature_xy)))
        if len(feature_xy) != self.num_scales:
            raise ValueError(
                "len(feature_xy)={} does not match len(in_channels)={}.".format(
                    len(feature_xy), self.num_scales
                )
            )

        feat_list = self._normalize_feature_order(feature_xy)
        expected_n = len(PHASE_NAMES)
        batch_size = None

        for scale_idx, (feat, expected_channels) in enumerate(zip(feat_list, self.in_channels)):
            if not torch.is_tensor(feat):
                raise TypeError(
                    "feature_xy[{}] must be Tensor, but got {}.".format(scale_idx, type(feat))
                )
            if feat.ndim != 5:
                raise ValueError(
                    "feature_xy[{}] must be 5D [N,B,C,H,W], got {}.".format(
                        scale_idx, tuple(feat.shape)
                    )
                )

            n, b, c, h, w = feat.shape
            if n != expected_n:
                raise ValueError(
                    "feature_xy[{}] phase dimension N must be {}, got {}.".format(
                        scale_idx, expected_n, n
                    )
                )
            if c != expected_channels:
                raise ValueError(
                    "feature_xy[{}] expects C={}, got C={}.".format(
                        scale_idx, expected_channels, c
                    )
                )
            if h <= 0 or w <= 0:
                raise ValueError(
                    "feature_xy[{}] has invalid spatial size {}.".format(scale_idx, (h, w))
                )

            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError(
                    "All scales must share the same B, got {} and {}.".format(batch_size, b)
                )

        return expected_n, int(batch_size)

    def _resolve_output_size(
        self,
        input_size: Optional[Tuple[int, int]],
        highest_res_feature: torch.Tensor,
    ) -> Tuple[int, int]:
        if input_size is not None:
            output_size = tuple(int(v) for v in input_size)
        elif self.input_size is not None:
            output_size = self.input_size
        else:
            output_size = tuple(int(v) for v in highest_res_feature.shape[-2:])

        if len(output_size) != 2 or output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("input_size must be positive (H, W), but got {}.".format(output_size))
        return output_size

    def _build_phase_features(
        self,
        feature_xy_high_to_low: Sequence[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        phase_feats: Dict[str, List[torch.Tensor]] = {phase_name: [] for phase_name in PHASE_NAMES}

        for seq in feature_xy_high_to_low:
            # N is physical phase index, not SNN micro-time.
            phase_feats["t1"].append(seq[0])
            phase_feats["t2"].append(seq[1])
            phase_feats["t3"].append(seq[2])

        return phase_feats

    def _decode_semantic(
        self,
        phase_feats: Dict[str, List[torch.Tensor]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sem_logits_dict: Dict[str, torch.Tensor] = {}

        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            feats = phase_feats[phase_name]
            if self.share_semantic_decoder:
                decoded = self.semantic_decoder(feats)
                logits = self.semantic_head(decoded, phase_idx=phase_idx)
            else:
                decoded = self.semantic_decoder[phase_name](feats)
                logits = self.semantic_head[phase_name](decoded, phase_idx=None)

            sem_logits_dict[phase_name] = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        sem_logits = torch.stack(
            [sem_logits_dict["t1"], sem_logits_dict["t2"], sem_logits_dict["t3"]],
            dim=1,
        )
        return sem_logits, sem_logits_dict

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        self._validate_feature_xy(feature_xy)
        feature_xy_high_to_low = self._normalize_feature_order(feature_xy)
        output_size = self._resolve_output_size(input_size, feature_xy_high_to_low[0][0])

        phase_feats = self._build_phase_features(feature_xy_high_to_low)
        sem_logits, sem_logits_dict = self._decode_semantic(phase_feats, output_size)

        change_decoded, _ = self.change_decoder(
            phase_feats["t1"],
            phase_feats["t3"],
            transition_feats_high_to_low=None,
        )
        chg_logits = self.change_head(change_decoded)
        chg_logits = F.interpolate(
            chg_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "sem_logits": sem_logits,
            "sem_logits_dict": sem_logits_dict,
            "chg_logits": chg_logits,
        }

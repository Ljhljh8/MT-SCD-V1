import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmseg.Qtrick_architecture.clock_driven.neuron import MTSCDPRDNIIFNode


PHASE_NAMES = ("t1", "t2", "t3")
PAIR_LONG = "long"
PAIR_ALL_WUSU = "all_wusu"


def _interpolate_5d(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if x.dim() != 5:
        raise ValueError("Expected 5D tensor [T,B,C,H,W], got {}.".format(tuple(x.shape)))
    T, B, C, _, _ = x.shape
    y = F.interpolate(
        x.flatten(0, 1),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return y.reshape(T, B, C, size[0], size[1]).contiguous()


class PreSpikeConv2d5D(nn.Module):
    """Pre-spike 5D wrapper: spike [T,B,C,H,W], flatten [T*B], then Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = False,
        norm: bool = False,
        neuron_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spike = MTSCDPRDNIIFNode(**(neuron_kwargs or {}))
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(self.out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("PreSpikeConv2d5D expects [T,B,C,H,W], got {}.".format(tuple(x.shape)))
        T, B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError("Expected C={}, got {}.".format(self.in_channels, C))

        x = self.spike(x)
        y = self.conv(x.flatten(0, 1).contiguous())
        y = self.norm(y)
        _, _, H_out, W_out = y.shape
        return y.reshape(T, B, self.out_channels, H_out, W_out).contiguous()


class PreSpikeConvBlock5D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, neuron_kwargs: Optional[dict] = None):
        super().__init__()
        self.conv1 = PreSpikeConv2d5D(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm=True,
            neuron_kwargs=neuron_kwargs,
        )
        self.conv2 = PreSpikeConv2d5D(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm=True,
            neuron_kwargs=neuron_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UpFuseBlock5D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        neuron_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.block = PreSpikeConvBlock5D(
            in_channels + skip_channels,
            out_channels,
            neuron_kwargs=neuron_kwargs,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or skip.dim() != 5:
            raise ValueError("UpFuseBlock5D expects 5D x/skip tensors.")
        if x.shape[:2] != skip.shape[:2]:
            raise ValueError("x and skip must share first two dims, got {} and {}.".format(x.shape[:2], skip.shape[:2]))

        x = _interpolate_5d(x, tuple(skip.shape[-2:]))
        x = torch.cat([x, skip], dim=2)
        return self.block(x)


class PreSpikeHead5D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, neuron_kwargs: Optional[dict] = None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spike = MTSCDPRDNIIFNode(**(neuron_kwargs or {}))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("PreSpikeHead5D expects [T,B,C,H,W], got {}.".format(tuple(x.shape)))
        T, B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError("Expected C={}, got {}.".format(self.in_channels, C))

        x = self.spike(x).flatten(0, 1).contiguous()
        y = self.classifier(self.dropout(x))
        return y.reshape(T, B, self.out_channels, H, W).contiguous()


class SharedDecoder5D(nn.Module):
    def __init__(self, in_channels: Sequence[int], decoder_channels: int, neuron_kwargs: Optional[dict] = None):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)

        self.laterals = nn.ModuleList([
            PreSpikeConv2d5D(
                channels,
                self.decoder_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                norm=False,
                neuron_kwargs=neuron_kwargs,
            )
            for channels in self.in_channels
        ])
        self.deep_block = PreSpikeConvBlock5D(
            self.decoder_channels,
            self.decoder_channels,
            neuron_kwargs=neuron_kwargs,
        )
        self.up_blocks = nn.ModuleList([
            UpFuseBlock5D(
                self.decoder_channels,
                self.decoder_channels,
                self.decoder_channels,
                neuron_kwargs=neuron_kwargs,
            )
            for _ in range(len(self.in_channels) - 1)
        ])

    def forward(self, feats_high_to_low: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats_high_to_low) != len(self.in_channels):
            raise ValueError(
                "Expected {} scales, got {}.".format(len(self.in_channels), len(feats_high_to_low))
            )

        proj = []
        for scale_idx, (feat, expected_channels, lateral) in enumerate(
            zip(feats_high_to_low, self.in_channels, self.laterals)
        ):
            if feat.dim() != 5:
                raise ValueError("Scale {} must be [T,B,C,H,W], got {}.".format(scale_idx, tuple(feat.shape)))
            if feat.shape[2] != expected_channels:
                raise ValueError(
                    "Scale {} expects C={}, got {}.".format(scale_idx, expected_channels, feat.shape[2])
                )
            proj.append(lateral(feat))

        x = self.deep_block(proj[-1])
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x = block(x, skip)
        return x


class MTSCDDecoderNetV3(nn.Module):
    """
    Direct-N 5D decoder for MTSCD features.

    feature_xy[s]: [N, B, C_s, H_s, W_s], where N is physical phase index.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        num_sem_classes: int,
        num_change_classes: int,
        input_size: Optional[Tuple[int, int]] = None,
        feature_order: str = "high_to_low",
        phase_names: Sequence[str] = PHASE_NAMES,
        pairs: Union[str, Sequence[Tuple[int, int, str]]] = PAIR_LONG,
        sem_head_dropout: float = 0.0,
        chg_head_dropout: float = 0.0,
        return_intermediates_default: bool = False,
        neuron_kwargs: Optional[dict] = None,
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
        self.phase_names = tuple(str(name) for name in phase_names)
        self.pairs = pairs
        self.return_intermediates_default = bool(return_intermediates_default)
        self.neuron_kwargs = dict(neuron_kwargs or {})

        self.semantic_decoder = SharedDecoder5D(
            self.in_channels,
            self.decoder_channels,
            neuron_kwargs=self.neuron_kwargs,
        )
        self.semantic_head = PreSpikeHead5D(
            self.decoder_channels,
            self.num_sem_classes,
            dropout=sem_head_dropout,
            neuron_kwargs=self.neuron_kwargs,
        )
        self.change_decoder = SharedDecoder5D(
            [3 * channels for channels in self.in_channels],
            self.decoder_channels,
            neuron_kwargs=self.neuron_kwargs,
        )
        self.change_head = PreSpikeHead5D(
            self.decoder_channels,
            self.num_change_classes,
            dropout=chg_head_dropout,
            neuron_kwargs=self.neuron_kwargs,
        )

    def _normalize_feature_order(self, feature_xy: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feature_xy = list(feature_xy)
        return feature_xy if self.feature_order == "high_to_low" else list(reversed(feature_xy))

    def _validate_feature_xy(self, feature_xy: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], int, int]:
        if not isinstance(feature_xy, (list, tuple)):
            raise TypeError("feature_xy must be list/tuple, got {}.".format(type(feature_xy)))
        if len(feature_xy) != self.num_scales:
            raise ValueError(
                "len(feature_xy)={} does not match len(in_channels)={}.".format(
                    len(feature_xy), self.num_scales
                )
            )

        feat_list = self._normalize_feature_order(feature_xy)
        num_phases = None
        batch_size = None
        for scale_idx, (feat, expected_channels) in enumerate(zip(feat_list, self.in_channels)):
            if not torch.is_tensor(feat):
                raise TypeError("feature_xy[{}] must be Tensor, got {}.".format(scale_idx, type(feat)))
            if feat.dim() != 5:
                raise ValueError(
                    "feature_xy[{}] must be [N,B,C,H,W], got {}.".format(scale_idx, tuple(feat.shape))
                )
            n, b, c, h, w = feat.shape
            if c != expected_channels:
                raise ValueError("feature_xy[{}] expects C={}, got {}.".format(scale_idx, expected_channels, c))
            if h <= 0 or w <= 0:
                raise ValueError("feature_xy[{}] has invalid spatial size {}.".format(scale_idx, (h, w)))
            if num_phases is None:
                num_phases = int(n)
                batch_size = int(b)
            elif n != num_phases or b != batch_size:
                raise ValueError("All scales must share N and B.")
        return feat_list, int(num_phases), int(batch_size)

    def _resolve_output_size(self, input_size: Optional[Tuple[int, int]], highest_res_feature: torch.Tensor) -> Tuple[int, int]:
        if input_size is not None:
            out_size = tuple(int(v) for v in input_size)
        elif self.input_size is not None:
            out_size = tuple(int(v) for v in self.input_size)
        else:
            out_size = tuple(int(v) for v in highest_res_feature.shape[-2:])
        if len(out_size) != 2:
            raise ValueError("input_size must be (H, W), got {}.".format(out_size))
        return out_size

    def _resolve_pair_specs(self, num_phases: int) -> List[Tuple[int, int, str]]:
        if isinstance(self.pairs, str):
            if self.pairs == PAIR_LONG:
                pair_specs = [(0, 2, "t1_to_t3")]
            elif self.pairs == PAIR_ALL_WUSU:
                pair_specs = [(0, 1, "t1_to_t2"), (1, 2, "t2_to_t3"), (0, 2, "t1_to_t3")]
            else:
                raise ValueError("Unsupported pairs='{}'.".format(self.pairs))
        else:
            pair_specs = [(int(i), int(j), str(name)) for i, j, name in self.pairs]

        seen_names = set()
        for i, j, name in pair_specs:
            if i == j:
                raise ValueError("Pair '{}' uses identical phase indices {}.".format(name, i))
            if i < 0 or j < 0 or i >= num_phases or j >= num_phases:
                raise ValueError("Pair '{}' indices ({}, {}) exceed N={}.".format(name, i, j, num_phases))
            if not name:
                raise ValueError("Pair name must be non-empty.")
            if name in seen_names:
                raise ValueError("Duplicate pair name '{}'.".format(name))
            seen_names.add(name)
        if "t1_to_t3" not in seen_names:
            raise ValueError("pairs must include 't1_to_t3' so chg_logits can alias it.")
        return pair_specs

    @staticmethod
    def _build_pair_features(
        feats_high_to_low: Sequence[torch.Tensor],
        pair_specs: Sequence[Tuple[int, int, str]],
    ) -> List[torch.Tensor]:
        pair_features = []
        for feat in feats_high_to_low:
            per_pair = []
            for i, j, _ in pair_specs:
                fi = feat[i]
                fj = feat[j]
                delta = fj - fi
                per_pair.append(torch.cat([torch.abs(delta), F.relu(delta), F.relu(-delta)], dim=1))
            pair_features.append(torch.stack(per_pair, dim=0).contiguous())
        return pair_features

    @staticmethod
    def _make_sem_logits_dict(sem_logits: torch.Tensor, phase_names: Sequence[str]) -> Dict[str, torch.Tensor]:
        if sem_logits.shape[0] != len(phase_names):
            return {}
        return {str(name): sem_logits[idx] for idx, name in enumerate(phase_names)}

    @staticmethod
    def _make_change_logits_dict(
        chg_logits: torch.Tensor,
        pair_specs: Sequence[Tuple[int, int, str]],
    ) -> Dict[str, torch.Tensor]:
        return {name: chg_logits[idx] for idx, (_, _, name) in enumerate(pair_specs)}

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        input_size: Optional[Tuple[int, int]] = None,
        return_intermediates: Optional[bool] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]]:
        feature_xy_high_to_low, num_phases, _ = self._validate_feature_xy(feature_xy)
        pair_specs = self._resolve_pair_specs(num_phases)
        output_size = self._resolve_output_size(input_size, feature_xy_high_to_low[0])
        if return_intermediates is None:
            return_intermediates = self.return_intermediates_default

        sem_decoded = self.semantic_decoder(feature_xy_high_to_low)
        sem_logits = self.semantic_head(sem_decoded)
        sem_logits = _interpolate_5d(sem_logits, output_size)
        sem_logits_dict = self._make_sem_logits_dict(sem_logits, self.phase_names)

        pair_features = self._build_pair_features(feature_xy_high_to_low, pair_specs)
        change_decoded = self.change_decoder(pair_features)
        chg_logits_per_pair = self.change_head(change_decoded)
        chg_logits_per_pair = _interpolate_5d(chg_logits_per_pair, output_size)
        chg_logits_dict = self._make_change_logits_dict(chg_logits_per_pair, pair_specs)

        outputs = {
            "sem_logits": sem_logits,
            "sem_logits_dict": sem_logits_dict,
            "chg_logits_dict": chg_logits_dict,
            "chg_logits": chg_logits_dict["t1_to_t3"],
        }

        if return_intermediates:
            outputs["intermediates"] = {
                "semantic_decoded": sem_decoded,
                "pair_features": pair_features,
                "change_decoded": change_decoded,
                "pair_specs": pair_specs,
            }

        return outputs


__all__ = [
    "MTSCDDecoderNetV3",
    "PreSpikeConv2d5D",
    "PreSpikeConvBlock5D",
    "UpFuseBlock5D",
    "PreSpikeHead5D",
]

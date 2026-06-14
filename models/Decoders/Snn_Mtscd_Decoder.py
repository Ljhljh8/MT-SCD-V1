from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic blocks
# ============================================================

def _as_list(x: Union[int, Sequence[int]], n: int) -> List[int]:
    if isinstance(x, (list, tuple)):
        assert len(x) == n, f"Expected length {n}, but got {len(x)}"
        return list(x)
    return [int(x) for _ in range(n)]


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not norm,
        )
        self.bn = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two ConvBNAct blocks for U-Net-style refinement."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None) -> None:
        super().__init__()
        mid_ch = out_ch if mid_ch is None else mid_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, kernel_size=3),
            ConvBNAct(mid_ch, out_ch, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpProject(nn.Module):
    """
    Upsample a feature map to a target spatial size.

    mode='bilinear': F.interpolate + 1x1 projection.
    mode='deconv'  : ConvTranspose2d + optional bilinear correction.
    """

    def __init__(self, in_ch: int, out_ch: int, mode: str = "bilinear") -> None:
        super().__init__()
        assert mode in {"bilinear", "deconv"}
        self.mode = mode
        if mode == "bilinear":
            self.op = ConvBNAct(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.op = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if self.mode == "bilinear":
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return self.op(x)

        x = self.op(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class FuseBlock(nn.Module):
    """
    Fuse top-down feature and lateral skip feature.

    fuse_mode='concat': cat([skip, up]) -> DoubleConv.
    fuse_mode='add'   : project both to out_ch -> add -> DoubleConv.
    """

    def __init__(self, skip_ch: int, up_ch: int, out_ch: int, fuse_mode: str = "concat") -> None:
        super().__init__()
        assert fuse_mode in {"concat", "add"}
        self.fuse_mode = fuse_mode
        if fuse_mode == "concat":
            self.fuse = DoubleConv(skip_ch + up_ch, out_ch)
        else:
            self.skip_proj = ConvBNAct(skip_ch, out_ch, kernel_size=1, padding=0)
            self.up_proj = ConvBNAct(up_ch, out_ch, kernel_size=1, padding=0)
            self.fuse = DoubleConv(out_ch, out_ch)

    def forward(self, skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.fuse_mode == "concat":
            x = torch.cat([skip, up], dim=1)
        else:
            x = self.skip_proj(skip) + self.up_proj(up)
        return self.fuse(x)


# ============================================================
# Temporal/SNN-step to physical-phase aggregation
# ============================================================

class WindowAttentionAggregator(nn.Module):
    """
    Aggregate SNN neural steps inside each physical phase window.

    Input at one scale:  feat_s = [T_snn, B, C, H, W]
    Output for one window: [B, C, H, W]

    agg_mode:
        - 'mean'  : average all steps in the window.
        - 'last'  : use the last step in the window.
        - 'first' : use the first step in the window.
        - 'attn'  : learn pixel-wise temporal attention over steps.

    This module is intentionally used only to aggregate steps belonging
    to the same physical phase. Transition-only steps should be excluded
    from phase_windows so they do not contaminate semantic readout.
    """

    def __init__(self, channels: int, agg_mode: str = "attn") -> None:
        super().__init__()
        assert agg_mode in {"mean", "last", "first", "attn"}
        self.agg_mode = agg_mode
        self.score = nn.Conv2d(channels, 1, kernel_size=1) if agg_mode == "attn" else None

    def forward(self, feat: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        # feat: [T, B, C, H, W]
        assert feat.dim() == 5, f"Expected [T,B,C,H,W], got {tuple(feat.shape)}"
        T = feat.shape[0]
        idx = [int(i) for i in indices]
        assert len(idx) > 0, "A phase window cannot be empty."
        assert min(idx) >= 0 and max(idx) < T, f"Invalid window indices {idx} for T={T}."

        x = feat[idx]  # [K, B, C, H, W]
        if self.agg_mode == "mean":
            return x.mean(dim=0)
        if self.agg_mode == "first":
            return x[0]
        if self.agg_mode == "last":
            return x[-1]
        elif self.agg_mode == "attn":
            K, B, C, H, W = x.shape
            scores = self.score(x.flatten(0, 1)).view(K, B, 1, H, W)
            alpha = torch.softmax(scores, dim=0)
            return (alpha * x).sum(dim=0)


class PhasePyramidBuilder(nn.Module):
    """
    Convert feature_xy from SNN-step features to physical-phase pyramids.

    feature_xy:
        List of 5 tensors, each [T_snn, B, C_i, H_i, W_i]

    phase_windows:
        List[List[int]], length=num_phases.
        Example for T=12, K=4, no transition steps:
            [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
        Example for T=16, K=4, R=2, with transition steps excluded:
            [[0,1,2,3], [6,7,8,9], [12,13,14,15]]

    Output:
        phase_pyramids[p][s] = [B, C_s, H_s, W_s]
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        num_phases: int = 3,
        agg_mode: str = "attn",
        feature_order: str = "shallow_to_deep",
    ) -> None:
        super().__init__()
        assert feature_order in {"shallow_to_deep", "deep_to_shallow"}
        self.encoder_channels = list(encoder_channels)
        self.num_phases = num_phases
        self.feature_order = feature_order
        self.aggregators = nn.ModuleList([
            WindowAttentionAggregator(c, agg_mode=agg_mode) for c in self.encoder_channels
        ])

    @staticmethod
    def default_windows(T_snn: int, num_phases: int = 3) -> List[List[int]]:
        """
        Conservative default:
        - If T_snn == num_phases: each step is treated as one physical phase.
        - If T_snn is divisible by num_phases: split consecutive windows equally.
        - Otherwise, require explicit phase_windows.
        """
        if T_snn == num_phases:
            return [[i] for i in range(num_phases)]
        if T_snn % num_phases == 0:
            k = T_snn // num_phases
            return [list(range(i * k, (i + 1) * k)) for i in range(num_phases)]
        raise ValueError(
            f"Cannot infer phase_windows from T_snn={T_snn}, num_phases={num_phases}. "
            "Please provide explicit phase_windows, e.g. "
            "[[0,1,2,3], [6,7,8,9], [12,13,14,15]] for T=16."
        )

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        phase_windows: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[List[torch.Tensor]]:
        assert len(feature_xy) == len(self.encoder_channels), (
            f"Expected {len(self.encoder_channels)} scales, got {len(feature_xy)}."
        )

        feats = list(feature_xy)
        if self.feature_order == "deep_to_shallow":
            feats = list(reversed(feats))

        T_snn = feats[0].shape[0]
        if phase_windows is None:
            phase_windows = self.default_windows(T_snn, self.num_phases)
        phase_windows = [list(w) for w in phase_windows]
        assert len(phase_windows) == self.num_phases, (
            f"Expected {self.num_phases} phase windows, got {len(phase_windows)}."
        )

        phase_pyramids: List[List[torch.Tensor]] = [[] for _ in range(self.num_phases)]

        for s, (feat_s, aggregator_s, c_expected) in enumerate(zip(feats, self.aggregators, self.encoder_channels)):
            assert feat_s.dim() == 5, f"feature_xy[{s}] must be [T,B,C,H,W], got {tuple(feat_s.shape)}"
            T, B, C, H, W = feat_s.shape
            assert T == T_snn, "All feature scales must share the same T_snn."
            assert C == c_expected, f"Scale {s}: expected C={c_expected}, got C={C}."
            for p in range(self.num_phases):
                phase_pyramids[p].append(aggregator_s(feat_s, phase_windows[p]))

        return phase_pyramids


# ============================================================
# U-Net pyramid decoder
# ============================================================

class UNetPyramidDecoder(nn.Module):
    """
    Generic U-Net/FPN-style decoder.

    Input features must be ordered shallow -> deep:
        [B,C0,H0,W0], [B,C1,H1,W1], ..., [B,C4,H4,W4]
    Output:
        [B, decoder_channels[0], H0, W0]
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Union[int, Sequence[int]] = 256,
        upsample_mode: str = "bilinear",
        fuse_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        n = len(self.encoder_channels)
        self.decoder_channels = _as_list(decoder_channels, n)

        self.lateral = nn.ModuleList([
            ConvBNAct(c_in, c_out, kernel_size=1, padding=0)
            for c_in, c_out in zip(self.encoder_channels, self.decoder_channels)
        ])

        self.up_blocks = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList()
        for i in range(n - 1, 0, -1):
            self.up_blocks.append(UpProject(self.decoder_channels[i], self.decoder_channels[i - 1], mode=upsample_mode))
            self.fuse_blocks.append(
                FuseBlock(
                    skip_ch=self.decoder_channels[i - 1],
                    up_ch=self.decoder_channels[i - 1],
                    out_ch=self.decoder_channels[i - 1],
                    fuse_mode=fuse_mode,
                )
            )

        self.out_refine = DoubleConv(self.decoder_channels[0], self.decoder_channels[0])

    def forward(
        self,
        features: Sequence[torch.Tensor],
        return_pyramid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert len(features) == len(self.encoder_channels), (
            f"Expected {len(self.encoder_channels)} feature scales, got {len(features)}."
        )
        feats = [proj(f) for proj, f in zip(self.lateral, features)]
        dec_feats: List[Optional[torch.Tensor]] = [None] * len(feats)

        x = feats[-1]
        dec_feats[-1] = x
        for block_idx, i in enumerate(range(len(feats) - 1, 0, -1)):
            x = self.up_blocks[block_idx](x, size=feats[i - 1].shape[-2:])
            x = self.fuse_blocks[block_idx](skip=feats[i - 1], up=x)
            dec_feats[i - 1] = x

        x = self.out_refine(x)
        dec_feats[0] = x
        if return_pyramid:
            return x, [d for d in dec_feats if d is not None]
        return x


# ============================================================
# Semantic decoder and change decoder
# ============================================================

class MultiPhaseSemanticDecoder(nn.Module):
    """
    Decode each physical phase independently.

    share_decoder=True:
        Use one shared U-Net decoder for all phases. Recommended by default,
        because all phases share the same semantic class space.
    share_decoder=False:
        Use one decoder per phase. This is closer to Seg_Decoder1/2/3 style.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Union[int, Sequence[int]] = 256,
        num_phases: int = 3,
        upsample_mode: str = "bilinear",
        fuse_mode: str = "concat",
        share_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_phases = num_phases
        self.share_decoder = share_decoder
        if share_decoder:
            self.decoder = UNetPyramidDecoder(encoder_channels, decoder_channels, upsample_mode, fuse_mode)
        else:
            self.decoders = nn.ModuleList([
                UNetPyramidDecoder(encoder_channels, decoder_channels, upsample_mode, fuse_mode)
                for _ in range(num_phases)
            ])

    def forward(
        self,
        phase_pyramids: Sequence[Sequence[torch.Tensor]],
        return_pyramids: bool = False,
    ):
        assert len(phase_pyramids) == self.num_phases
        semantic_feats: List[torch.Tensor] = []
        semantic_pyramids: List[List[torch.Tensor]] = []
        for p in range(self.num_phases):
            if self.share_decoder:
                out = self.decoder(phase_pyramids[p], return_pyramid=return_pyramids)
            else:
                out = self.decoders[p](phase_pyramids[p], return_pyramid=return_pyramids)
            if return_pyramids:
                feat_p, pyr_p = out
                semantic_feats.append(feat_p)
                semantic_pyramids.append(pyr_p)
            else:
                semantic_feats.append(out)
        if return_pyramids:
            return semantic_feats, semantic_pyramids
        return semantic_feats


class MultiScaleDifferenceBuilder(nn.Module):
    """
    Build multi-scale difference features between physical phase_a and phase_b.

    diff_mode:
        - 'abs_sub'   : |F_b - F_a|, direction-invariant and stable.
        - 'sub'       : F_b - F_a, direction-sensitive.
        - 'concat_abs': [F_a, F_b, |F_b-F_a|] -> 1x1 adapter.
    """

    def __init__(self, encoder_channels: Sequence[int], diff_mode: str = "abs_sub") -> None:
        super().__init__()
        assert diff_mode in {"abs_sub", "sub", "concat_abs"}
        self.diff_mode = diff_mode
        self.encoder_channels = list(encoder_channels)
        if diff_mode == "concat_abs":
            self.adapters = nn.ModuleList([
                ConvBNAct(c * 3, c, kernel_size=1, padding=0) for c in self.encoder_channels
            ])
        else:
            self.adapters = nn.ModuleList([
                ConvBNAct(c, c, kernel_size=3) for c in self.encoder_channels
            ])

    def forward(
        self,
        phase_pyramids: Sequence[Sequence[torch.Tensor]],
        phase_a: int = 0,
        phase_b: int = 2,
    ) -> List[torch.Tensor]:
        feats_a = phase_pyramids[phase_a]
        feats_b = phase_pyramids[phase_b]
        diff_pyramid: List[torch.Tensor] = []
        for fa, fb, adapter in zip(feats_a, feats_b, self.adapters):
            if self.diff_mode == "abs_sub":
                d = torch.abs(fb - fa)
            elif self.diff_mode == "sub":
                d = fb - fa
            else:
                d = torch.cat([fa, fb, torch.abs(fb - fa)], dim=1)
            diff_pyramid.append(adapter(d))
        return diff_pyramid


class ChangeDecoder(nn.Module):
    """U-Net-style decoder over multi-scale phase-difference features."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Union[int, Sequence[int]] = 256,
        upsample_mode: str = "bilinear",
        fuse_mode: str = "concat",
        diff_mode: str = "abs_sub",
        phase_pair: Tuple[int, int] = (0, 2),
    ) -> None:
        super().__init__()
        self.phase_pair = phase_pair
        self.diff_builder = MultiScaleDifferenceBuilder(encoder_channels, diff_mode=diff_mode)
        self.decoder = UNetPyramidDecoder(encoder_channels, decoder_channels, upsample_mode, fuse_mode)

    def forward(
        self,
        phase_pyramids: Sequence[Sequence[torch.Tensor]],
        return_diff_pyramid: bool = False,
        return_decoder_pyramid: bool = False,
    ):
        diff_pyramid = self.diff_builder(phase_pyramids, self.phase_pair[0], self.phase_pair[1])
        decoded = self.decoder(diff_pyramid, return_pyramid=return_decoder_pyramid)
        if return_decoder_pyramid:
            change_feat, change_decoder_pyramid = decoded
        else:
            change_feat, change_decoder_pyramid = decoded, None

        if return_diff_pyramid and return_decoder_pyramid:
            return change_feat, diff_pyramid, change_decoder_pyramid
        if return_diff_pyramid:
            return change_feat, diff_pyramid
        if return_decoder_pyramid:
            return change_feat, change_decoder_pyramid
        return change_feat


# ============================================================
# Output heads
# ============================================================

class SegmentationHead(nn.Module):
    """Semantic segmentation head. Returns raw logits, not softmax."""

    def __init__(self, in_ch: int, num_classes: int, mid_ch: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, kernel_size=3),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_ch, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class ChangeHead(nn.Module):
    """
    Change detection head. Returns raw logits, not sigmoid.

    For binary change detection, use num_change_classes=1 and train with BCEWithLogitsLoss.
    For multi-class change detection, set num_change_classes>1 and train with CrossEntropyLoss.
    """

    def __init__(self, in_ch: int, num_change_classes: int = 1, mid_ch: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, kernel_size=3),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_ch, num_change_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ============================================================
# Main decoder head
# ============================================================

class SNNMTSCDDecoderHead(nn.Module):
    """
    Decoder head for SNN-based multi-temporal semantic change detection.

    Input:
        feature_xy: List of 5 feature maps.
        feature_xy[s]: [T_snn, B, C_s, H_s, W_s]

    Important:
        T_snn is neural simulation time, not necessarily the number of physical phases.
        Use phase_windows to specify which SNN steps belong to each physical phase.

    Output dict:
        semantic_logits: List of num_phases tensors, each [B, K_sem, H, W]
        change_logits  : [B, K_chg, H, W]
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        num_semantic_classes: int,
        num_change_classes: int = 1,
        num_phases: int = 3,
        decoder_channels: Union[int, Sequence[int]] = 256,
        feature_order: str = "shallow_to_deep",
        phase_agg_mode: str = "attn",
        upsample_mode: str = "bilinear",
        fuse_mode: str = "concat",
        share_semantic_decoder: bool = True,
        diff_mode: str = "abs_sub",
        phase_pair: Tuple[int, int] = (0, 2),
        sem_head_mid_ch: int = 128,
        chg_head_mid_ch: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        self.num_phases = num_phases
        self.feature_order = feature_order

        decoder_channels_list = _as_list(decoder_channels, len(self.encoder_channels))
        out_ch = decoder_channels_list[0]

        self.phase_builder = PhasePyramidBuilder(
            encoder_channels=self.encoder_channels,
            num_phases=num_phases,
            agg_mode=phase_agg_mode,
            feature_order=feature_order,
        )

        self.semantic_decoder = MultiPhaseSemanticDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels_list,
            num_phases=num_phases,
            upsample_mode=upsample_mode,
            fuse_mode=fuse_mode,
            share_decoder=share_semantic_decoder,
        )

        self.change_decoder = ChangeDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels_list,
            upsample_mode=upsample_mode,
            fuse_mode=fuse_mode,
            diff_mode=diff_mode,
            phase_pair=phase_pair,
        )

        self.semantic_heads = nn.ModuleList([
            SegmentationHead(out_ch, num_semantic_classes, mid_ch=sem_head_mid_ch, dropout=dropout)
            for _ in range(num_phases)
        ])
        self.change_head = ChangeHead(out_ch, num_change_classes, mid_ch=chg_head_mid_ch, dropout=dropout)

    def forward(
        self,
        feature_xy: Sequence[torch.Tensor],
        out_size: Tuple[int, int],
        phase_windows: Optional[Sequence[Sequence[int]]] = None,
        return_features: bool = False,
        return_aux_pyramids: bool = False,
        return_dict: bool = True,
    ) -> Union[Dict[str, Any], Tuple[List[torch.Tensor], torch.Tensor]]:
        # 1) [T_snn,B,C,H,W] -> physical phase pyramids.
        #    phase_pyramids[p][s] = [B,C_s,H_s,W_s]
        phase_pyramids = self.phase_builder(feature_xy, phase_windows=phase_windows)

        # 2) Semantic branch: decode each physical phase independently.
        sem_out = self.semantic_decoder(phase_pyramids, return_pyramids=return_aux_pyramids)
        if return_aux_pyramids:
            semantic_feats, semantic_decoder_pyramids = sem_out
        else:
            semantic_feats, semantic_decoder_pyramids = sem_out, None

        # 3) Change branch: decode multi-scale difference between phase_pair, default t1 vs t3.
        chg_out = self.change_decoder(
            phase_pyramids,
            return_diff_pyramid=return_aux_pyramids,
            return_decoder_pyramid=return_aux_pyramids,
        )
        if return_aux_pyramids:
            change_feat, diff_pyramid, change_decoder_pyramid = chg_out
        else:
            change_feat, diff_pyramid, change_decoder_pyramid = chg_out, None, None

        # 4) Output heads. Return raw logits.
        semantic_logits = [
            head(feat, out_size=out_size) for head, feat in zip(self.semantic_heads, semantic_feats)
        ]
        change_logits = self.change_head(change_feat, out_size=out_size)

        if not return_dict and not return_features and not return_aux_pyramids:
            return semantic_logits, change_logits

        outputs: Dict[str, Any] = {
            "semantic_logits": semantic_logits,
            "change_logits": change_logits,
            "phase_pyramids": phase_pyramids,
        }
        if return_features:
            outputs["semantic_feats"] = semantic_feats
            outputs["change_feat"] = change_feat
        if return_aux_pyramids:
            outputs["semantic_decoder_pyramids"] = semantic_decoder_pyramids
            outputs["diff_pyramid"] = diff_pyramid
            outputs["change_decoder_pyramid"] = change_decoder_pyramid
        return outputs


class SNNMTSCDNet(nn.Module):
    """
    Full wrapper:
        x -> SpikeMixBlock/backbone -> feature_xy -> SNNMTSCDDecoderHead -> predictions

    Required backbone output:
        feature_xy = [feat_s0, feat_s1, feat_s2, feat_s3, feat_s4]
        feat_si.shape = [T_snn, B, C_i, H_i, W_i]
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder_channels: Sequence[int],
        num_semantic_classes: int,
        num_change_classes: int = 1,
        num_phases: int = 3,
        decoder_channels: Union[int, Sequence[int]] = 256,
        feature_order: str = "shallow_to_deep",
        phase_agg_mode: str = "attn",
        upsample_mode: str = "bilinear",
        fuse_mode: str = "concat",
        share_semantic_decoder: bool = True,
        diff_mode: str = "abs_sub",
        phase_pair: Tuple[int, int] = (0, 2),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder_head = SNNMTSCDDecoderHead(
            encoder_channels=encoder_channels,
            num_semantic_classes=num_semantic_classes,
            num_change_classes=num_change_classes,
            num_phases=num_phases,
            decoder_channels=decoder_channels,
            feature_order=feature_order,
            phase_agg_mode=phase_agg_mode,
            upsample_mode=upsample_mode,
            fuse_mode=fuse_mode,
            share_semantic_decoder=share_semantic_decoder,
            diff_mode=diff_mode,
            phase_pair=phase_pair,
        )

    def forward(
        self,
        x: torch.Tensor,
        phase_windows: Optional[Sequence[Sequence[int]]] = None,
        return_features: bool = False,
        return_aux_pyramids: bool = False,
        return_dict: bool = True,
    ):
        feature_xy = self.backbone(x)
        out_size = x.shape[-2:]
        outputs = self.decoder_head(
            feature_xy=feature_xy,
            out_size=out_size,
            phase_windows=phase_windows,
            return_features=return_features,
            return_aux_pyramids=return_aux_pyramids,
            return_dict=return_dict,
        )
        if isinstance(outputs, dict):
            outputs["feature_xy"] = feature_xy
        return outputs


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    B, H, W = 2, 256, 256
    encoder_channels = [64, 128, 256, 512, 512]

    # Example 1: T_snn=12, K=4, no transition steps.
    feature_xy_12 = [
        torch.randn(12, B, 64,  H // 4,  W // 4),
        torch.randn(12, B, 128, H // 8,  W // 8),
        torch.randn(12, B, 256, H // 16, W // 16),
        torch.randn(12, B, 512, H // 32, W // 32),
        torch.randn(12, B, 512, H // 32, W // 32),
    ]
    phase_windows_12 = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    decoder = SNNMTSCDDecoderHead(
        encoder_channels=encoder_channels,
        num_semantic_classes=13,
        num_change_classes=1,
        num_phases=3,
        decoder_channels=[256, 256, 256, 256, 256],
        phase_agg_mode="attn",
        share_semantic_decoder=True,
        diff_mode="abs_sub",
        phase_pair=(0, 2),
    )
    outs = decoder(feature_xy_12, out_size=(H, W), phase_windows=phase_windows_12)
    print([x.shape for x in outs["semantic_logits"]], outs["change_logits"].shape)

    # Example 2: T_snn=16, K=4, R=2, transition steps excluded from semantic readout.
    # zero-based windows from the report:
    # t1: 0,1,2,3; t1->t2 transition: 4,5;
    # t2: 6,7,8,9; t2->t3 transition: 10,11;
    # t3: 12,13,14,15.
    phase_windows_16 = [[0, 1, 2, 3], [6, 7, 8, 9], [12, 13, 14, 15]]

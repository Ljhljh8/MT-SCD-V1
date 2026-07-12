"""
Direct 5D / N x B flatten decoder for MTSCD / GSTM-SCD.

Assumption:
    feature_xy is a high-to-low list of 5D tensors [N, B, C, H, W].
    The first axis N is the physical remote-sensing phase index.

Main changes from the old V4 decoder:
    - No forward-time normalize_windows().
    - No _read_phase_and_transition_features() main path.
    - No transition feature fusion path.
    - Semantic branch decodes all physical phases in one [N*B, C, H, W] pass.
    - Change branch indexes phase features directly from the 5D list only where needed.
    - PDCA-guided pair decoder is retained and adapted to direct 5D feature access.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.Qtrick_architecture.clock_driven.neuron import Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant
from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d
from models.mtscd_task_evidence_unit import MTSCDPairDecoderGate

PHASE_NAMES = ("t1", "t2", "t3")
PAIR_NAMES = (("t1", "t2"), ("t2", "t3"), ("t1", "t3"))
PAIR_KEYS = ("t1_to_t2", "t2_to_t3", "t1_to_t3")


class DendBlock(nn.Module):
    """Input: [B, Cin, H, W] -> Output: [B, Cout, H, W].

    Direct-NB path passes B = N * original_B. Internally this block still
    creates a one-step 5D tensor for DendFADCConv2d, preserving the original
    decoder convention.
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
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=int(groups),
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
        self.BN = nn.BatchNorm2d(int(out_channels))

    def forward(self, x: torch.Tensor, k=None) -> Tuple[torch.Tensor, Any]:
        N, B, _, H, W = x.shape
        x, k = self.block(x, k)
        x = self.BN(x.flatten(0, 1)).view(N, B, -1, H, W)
        return x, k


class ConvBlock(nn.Module):
    """Input: [B, Cin, H, W] -> Output: [B, Cout, H, W]."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        self.activation = Q_IFNode(surrogate_function=Quant())
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(int(out_channels)),
        )

        self.activation1 = Q_IFNode(surrogate_function=Quant())
        self.block1 = nn.Sequential(
            nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False, groups=int(groups)),
            nn.BatchNorm2d(int(out_channels)),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, B, C, H, W = x.shape
        x = self.activation(x).flatten(0, 1)
        x = self.block(x).view(N, B, -1, H, W)
        x = self.activation1(x).flatten(0, 1)
        x = self.block1(x).view(N, B, -1 ,H, W)
        return x


class UpFuseBlock(nn.Module):
    """Upsample x to skip resolution and fuse.

    x:    [B, Cx, H_low, W_low]
    skip: [B, Cs, H_high, W_high]
    out:  [B, Cout, H_high, W_high]
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_DendSize: int = 128, groups: int = 1):
        super().__init__()
        self.fuse1 = DendBlock(int(in_channels) + int(skip_channels), int(out_channels),groups=int(out_channels))
        self.fuse2 = ConvBlock(int(in_channels) + int(skip_channels), int(out_channels),groups=int(out_channels))
        self.use_DendSize = int(use_DendSize)

    def _resize_k(self, k, size: Tuple[int, int]):
        if k is None or not isinstance(k, (list, tuple)) or len(k) == 0:
            return k
        if not torch.is_tensor(k[0]) or k[0].shape[-2:] == size:
            return k
        k_tensor = torch.cat(list(k), dim=1)
        k_up = F.interpolate(k_tensor, size=size, mode="bilinear", align_corners=False)
        return list(torch.chunk(k_up, chunks=len(k), dim=1))

    def forward(self, x: torch.Tensor, skip: torch.Tensor, k=None) -> Tuple[torch.Tensor, Any]:
        N, B, C, H, W = skip.shape
        x = F.interpolate(x.flatten(0,1), size=skip.shape[-2:], mode="bilinear", align_corners=False).view(N, B, C ,H ,W)
        k = self._resize_k(k, skip.shape[-2:])
        x = torch.cat([x, skip], dim=2)
        if skip.shape[-1] >= self.use_DendSize:
            x = self.fuse2(x)
        else:
            x, k = self.fuse1(x, k)
        return x, k


class SemanticDecoder(nn.Module):
    """4D multi-scale decoder.

    feats_high_to_low[s]: [B, C_s, H_s, W_s]
    return:              [B, D, H_0, W_0]
    """

    def __init__(self, in_channels: Sequence[int], decoder_channels: int):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)
        # self.act =
        self.laterals = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, self.decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels),
        )
            for c in self.in_channels
        ])
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels, groups=self.decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, groups=self.decoder_channels, use_DendSize=16)
            for _ in range(len(self.in_channels) - 1)
        ])

    def forward(self, feats_high_to_low: Sequence[torch.Tensor], k=None) -> torch.Tensor:
        proj: List[torch.Tensor] = []
        for feat, lateral in zip(feats_high_to_low, self.laterals):
            N, B, _, H, W = feat.shape

            proj.append(lateral(feat.flatten(0, 1)).view(N, B, -1, H, W).contiguous())
        x, k = self.deep_block(proj[-1], k)
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)
        return x


class Direct5DSemanticDecoder(nn.Module):
    """Decode all physical phases in one shared [N*B] semantic pass."""

    def __init__(self, in_channels: Sequence[int], decoder_channels: int):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder = SemanticDecoder(self.in_channels, int(decoder_channels))

    def forward(self, feature_xy_high_to_low: Sequence[torch.Tensor]) -> torch.Tensor:
        # feature_xy[s]: [N,B,C,H,W] -> [N*B,C,H,W]
        N, B = feature_xy_high_to_low[0].shape[:2]
        # flat_feats = [feat.flatten(0, 1) for feat in feature_xy_high_to_low]
        decoded_flat = self.decoder(feature_xy_high_to_low)  # [N*B,D,H0,W0]
        # D, H0, W0 = decoded_flat.shape[2:]
        # return decoded_flat.reshape(N, B, D, H0, W0).contiguous()
        return decoded_flat

class DirectPairChangeDecoder(nn.Module):
    """Direct 5D multi-scale pair change decoder.

    It preserves the old multi-scale diff decoder idea but no longer requires a
    prebuilt phase_feats dictionary.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        diff_mode: str = "abs_signed",
        phase_pair: Tuple[int, int] = (0, 2),
    ):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)
        self.diff_mode = str(diff_mode)
        self.phase_pair = tuple(int(i) for i in phase_pair)
        if self.diff_mode not in ("abs", "abs_signed", "concat"):
            raise ValueError(f"Unsupported diff_mode={self.diff_mode}")

        proj_in_channels = []
        for c in self.in_channels:
            proj_in_channels.append(c if self.diff_mode == "abs" else 3 * c)
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, self.decoder_channels, kernel_size=1, bias=False)
            for c in proj_in_channels
        ])
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, use_DendSize=16)
            for _ in range(len(self.in_channels) - 1)
        ])

    def _make_diff(self, f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
        delta = f_j - f_i
        abs_diff = delta.abs()
        if self.diff_mode == "abs":
            return abs_diff
        if self.diff_mode == "abs_signed":
            return torch.cat([abs_diff, F.relu(delta), F.relu(-delta)], dim=1)
        if self.diff_mode == "concat":
            return torch.cat([f_i, f_j, abs_diff], dim=1)
        raise RuntimeError("unreachable diff_mode")

    def forward(self, feature_xy_high_to_low: Sequence[torch.Tensor]) -> torch.Tensor:
        idx_i, idx_j = self.phase_pair
        proj: List[torch.Tensor] = []
        for feat, lateral in zip(feature_xy_high_to_low, self.laterals):
            f_i = feat[idx_i]
            f_j = feat[idx_j]
            x_in = self._make_diff(f_i, f_j)
            proj.append(lateral(x_in).unsqueeze(0))              # 不同尺度之间映射到相同通道维度的diff特征
        x, k = self.deep_block(proj[-1], None)
        for block, skip in zip(self.up_blocks, reversed(proj[:-1])):
            x, k = block(x, skip, k)
        return x.squeeze(0)


class PDCASpatialGate(nn.Module):
    """PDCA guidance [B,4,H,W] -> spatial gate [B,1,H,W]."""

    def __init__(self, r_channels: int = 4, init_bias: float = 0.0):
        super().__init__()
        self.r_channels = int(r_channels)
        self.conv = nn.Conv2d(self.r_channels, 1, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, float(init_bias))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(r))


class PDCAGuidedPairwiseChangeDecoder(nn.Module):
    """PDCA-guided direct 5D pairwise change decoder.

    Input:
        feature_xy_high_to_low[s]: [N,B,C_s,H_s,W_s]
        pdca_aux["pdca_source_weights"][scale][target]: [B,G,Q,H,W]
        pdca_aux["pdca_source_names_by_target"][scale][target]: tuple/list of source names

    Output:
        change_logits_dict[pair_key]: [B,num_change_classes,H,W]
    """

    FIXED_PAIR_KEYS = PAIR_KEYS

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: int,
        num_change_classes: int = 1,
        pair_names: Sequence[Tuple[str, str]] = PAIR_NAMES,
        phase_names: Sequence[str] = PHASE_NAMES,
        detach_pdca_guidance: bool = True,
        use_pdca_guidance: bool = True,
        alpha_max: float = 1.0,
        decoder_task_gate_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.decoder_channels = int(decoder_channels)
        self.num_change_classes = int(num_change_classes)
        self.pair_names = tuple((str(a), str(b)) for a, b in pair_names)
        self.phase_names = tuple(str(name) for name in phase_names)
        self.phase_to_index = {name: idx for idx, name in enumerate(self.phase_names)}
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.use_pdca_guidance = bool(use_pdca_guidance)
        self.alpha_max = float(alpha_max)
        self.pair_keys = tuple("%s_to_%s" % pair for pair in self.pair_names)
        if self.pair_keys != self.FIXED_PAIR_KEYS:
            raise ValueError("pair order must be %r, got %r" % (self.FIXED_PAIR_KEYS, self.pair_keys))

        self.diff_proj = nn.ModuleList([
            nn.Conv2d(3 * c, self.decoder_channels, kernel_size=1, bias=False)
            for c in self.in_channels
        ])
        self.spatial_gates = nn.ModuleList([PDCASpatialGate(r_channels=4) for _ in self.in_channels])
        self.raw_gate_scales = nn.ParameterList([
            nn.Parameter(torch.full((1,), -4.0)) for _ in self.in_channels
        ])
        self.deep_block = DendBlock(self.decoder_channels, self.decoder_channels)
        self.up_blocks = nn.ModuleList([
            UpFuseBlock(self.decoder_channels, self.decoder_channels, self.decoder_channels, use_DendSize=16)
            for _ in range(len(self.in_channels) - 1)
        ])
        self.change_head = ChangeHead(self.decoder_channels, self.num_change_classes)
        self.decoder_task_gate_cfg = dict(decoder_task_gate_cfg or {})
        self.use_decoder_task_gate = bool(self.decoder_task_gate_cfg.get("enabled", False))
        self.decoder_task_gate_use_pdca_source_weights = bool(
            self.decoder_task_gate_cfg.get("use_pdca_source_weights", False)
        )
        
        gate_cfg = dict(self.decoder_task_gate_cfg)
        gate_cfg.pop("enabled", None)
        
        self.task_decoder_gates = nn.ModuleList([
            MTSCDPairDecoderGate(
                in_channels=self.decoder_channels,
                phase_names=self.phase_names,
                pair_names=self.pair_names,
                enabled=self.use_decoder_task_gate,
                **gate_cfg,
            )
            for _ in self.in_channels
        ])
    @staticmethod
    def _task_evidence_for_scale(task_evidence_aux: Optional[Dict[str, Any]], scale_key: str):
        if not task_evidence_aux:
            return None
    
        if "pair_evidence" in task_evidence_aux:
            return task_evidence_aux
    
        if scale_key in task_evidence_aux:
            return task_evidence_aux[scale_key]
    
        keys = sorted(str(k) for k in task_evidence_aux.keys())
        if not keys:
            return None
        return task_evidence_aux[keys[-1]]
    @staticmethod
    def _make_diff(f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
        delta = f_j - f_i
        return torch.cat([delta.abs(), F.relu(delta), F.relu(-delta)], dim=1)

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
            if scale_weights and names_by_scale.get(scale_key):
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
                "PDCA guidance scale %s requires both pdca_source_weights and pdca_source_names_by_target"
                % scale_key
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
        feature_xy_high_to_low: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
        pdca_aux: Optional[Dict[str, Any]] = None,
        task_evidence_aux: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_pdca_guidance and not self._has_any_guidance(pdca_aux):
            raise RuntimeError(
                "PDCA-guided pair decoder requires pdca_source_weights and "
                "pdca_source_names_by_target on at least one relation scale"
            )
        
        if self.use_decoder_task_gate and task_evidence_aux is None:
            raise RuntimeError("enabled decoder task gate requires task_evidence_aux")
        
        if self.use_decoder_task_gate and self.decoder_task_gate_use_pdca_source_weights:
            if not self._has_any_guidance(pdca_aux):
                raise RuntimeError(
                    "decoder task gate with PDCA source weights requires pdca_aux"
                )
        if self.use_pdca_guidance and not self._has_any_guidance(pdca_aux):
            raise RuntimeError(
                "PDCA-guided pair decoder requires pdca_source_weights and "
                "pdca_source_names_by_target on at least one relation scale"
            )

        proj_all: List[torch.Tensor] = []

        for s, (feat, diff_proj) in enumerate(zip(feature_xy_high_to_low, self.diff_proj)):
            # feat: [N,B,C,H,W]; pair indexing is local and on demand.
            x_by_pair: List[torch.Tensor] = []
            scale_key = str(s)
            for pair_key, (phase_i, phase_j) in zip(self.pair_keys, self.pair_names):
                fi = feat[self.phase_to_index[phase_i]]
                fj = feat[self.phase_to_index[phase_j]]
                diff = self._make_diff(fi, fj)
                pair_feat = diff_proj(diff)

                r, has_guidance = self._make_guidance(pdca_aux, scale_key, phase_i, phase_j, fi)
                if self.use_pdca_guidance and has_guidance:
                    gate = self.spatial_gates[s](r)
                    alpha = self.alpha_max * torch.sigmoid(self.raw_gate_scales[s]).view(1, 1, 1, 1)
                    pair_feat = pair_feat * (1.0 + alpha * gate)
                
                x_by_pair.append(pair_feat)

            pair_features = torch.stack(x_by_pair, dim=0)  # [P,B,D,H,W]

            if self.use_decoder_task_gate:
                evidence = self._task_evidence_for_scale(task_evidence_aux, scale_key)
                pair_features = self.task_decoder_gates[s](
                    pair_features,
                    pair_evidence=evidence,
                    pdca_aux=pdca_aux,
                    return_aux=False,
                )
            
            proj_all.append(pair_features.flatten(0, 1).unsqueeze(0))  # [1,P*B,D,H,W]

        x, k = self.deep_block(proj_all[-1], None)
        for block, skip in zip(self.up_blocks, reversed(proj_all[:-1])):
            x, k = block(x, skip, k)

        logits_all = self.change_head(x.squeeze(0))
        logits_all = F.interpolate(logits_all, size=output_size, mode="bilinear", align_corners=False)
        batch_size = feature_xy_high_to_low[0].shape[1]
        logits_split = torch.split(logits_all, batch_size, dim=0)
        change_logits_dict = {pair_key: logits for pair_key, logits in zip(self.pair_keys, logits_split)}
        return change_logits_dict


class PhaseAffine(nn.Module):
    """Optional phase-specific affine modulation for decoded sequence."""

    def __init__(self, num_phases: int, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(int(num_phases), int(channels), 1, 1))
        self.beta = nn.Parameter(torch.zeros(int(num_phases), int(channels), 1, 1))

    def forward(self, x: torch.Tensor, phase_idx: int) -> torch.Tensor:
        return x * self.gamma[int(phase_idx)] + self.beta[int(phase_idx)]

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [N,B,C,H,W]. Bias is defined for the first configured phases.
        n_apply = min(x_seq.shape[0], self.gamma.shape[0])
        if n_apply <= 0:
            return x_seq
        out = x_seq.clone()
        out[:n_apply] = out[:n_apply] * self.gamma[:n_apply, None] + self.beta[:n_apply, None]
        return out


class SegmentationHead(nn.Module):
    """x: [B, D, H, W] -> logits: [B, num_sem_classes, H, W]."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_phases: int = 3,
        dropout: float = 0.0,
        use_phase_classifier_bias: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(int(in_channels), int(num_classes), kernel_size=1)

        self.use_phase_classifier_bias = bool(use_phase_classifier_bias)
        if self.use_phase_classifier_bias:
            self.phase_bias = nn.Parameter(torch.zeros(int(num_phases), int(num_classes), 1, 1))
        else:
            self.register_parameter("phase_bias", None)

    def forward(self, x: torch.Tensor, phase_idx: Optional[int] = None) -> torch.Tensor:
        logits = self.classifier(self.dropout(x))
        if self.use_phase_classifier_bias:
            if phase_idx is None:
                raise ValueError("phase_idx must be provided when use_phase_classifier_bias=True.")
            logits = logits + self.phase_bias[int(phase_idx)]
        return logits
    def forward_seq(self, x_seq: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        # x_seq: [N,B,D,H,W] -> [N,B,K,Hout,Wout]
        N, B = x_seq.shape[:2]
        logits_flat = self.classifier(self.dropout(x_seq.flatten(0, 1)))
        logits_flat = F.interpolate(logits_flat, size=output_size, mode="bilinear", align_corners=False)
        K, H, W = logits_flat.shape[1:]
        logits_seq = logits_flat.reshape(N, B, K, H, W).contiguous()
        if self.use_phase_classifier_bias:
            n_apply = min(N, self.phase_bias.shape[0])
            logits_seq[:n_apply] = logits_seq[:n_apply] + self.phase_bias[:n_apply, None]
        return logits_seq


class ChangeHead(nn.Module):
    """x: [B, D, H, W] -> logits: [B, num_change_classes, H, W]."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(int(in_channels), int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class MTSCDDecoderNet(nn.Module):
    """Direct 5D decoder.

    Input:
        feature_xy[s]: [N,B,C_s,H_s,W_s]

    Output:
        sem_logits: [B,N,num_sem_classes,H,W]
        sem_logits_dict["t1"/"t2"/"t3"]: [B,num_sem_classes,H,W]
        chg_logits: [B,num_change_classes,H,W]
        change_logits_dict["t1_to_t3"] and, in guided mode, all fixed pair logits.
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
        use_pdca_guided_pair_decoder: bool = False,
        detach_pdca_guidance: bool = True,
        use_pdca_guidance: bool = True,
        decoder_task_gate_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if feature_order not in ("high_to_low", "low_to_high"):
            raise ValueError("feature_order must be 'high_to_low' or 'low_to_high'.")
        self.feature_order = str(feature_order)
        self.in_channels = list(map(int, in_channels))
        if self.feature_order == "low_to_high":
            self.in_channels = list(reversed(self.in_channels))
        self.num_scales = len(self.in_channels)
        self.decoder_channels = int(decoder_channels)
        self.num_sem_classes = int(num_sem_classes)
        self.num_change_classes = int(num_change_classes)
        self.input_size = tuple(input_size) if input_size is not None else None
        self.diff_mode = str(diff_mode)
        self.share_semantic_decoder = bool(share_semantic_decoder)
        self.use_phase_affine = bool(use_phase_affine)
        self.use_phase_classifier_bias = bool(use_phase_classifier_bias)
        self.use_pdca_guided_pair_decoder = bool(use_pdca_guided_pair_decoder)
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.use_pdca_guidance = bool(use_pdca_guidance)
        self.pair_change_decoder = PDCAGuidedPairwiseChangeDecoder(
            in_channels=self.in_channels,
            decoder_channels=self.decoder_channels,
            num_change_classes=self.num_change_classes,
            detach_pdca_guidance=self.detach_pdca_guidance,
            use_pdca_guidance=self.use_pdca_guidance,
            decoder_task_gate_cfg=decoder_task_gate_cfg,
        )
        # Deprecated constructor compatibility only. These are not used by forward.
        self.phase_windows = phase_windows
        self.transition_windows = transition_windows
        self.temporal_readout = temporal_readout
        self.use_transition_fusion = False
        if use_transition_fusion:
            # Kept intentionally non-fatal for old config compatibility.
            self.use_transition_fusion = False
        del phase_anchor_bias

        if self.share_semantic_decoder:
            self.semantic_decoder = Direct5DSemanticDecoder(self.in_channels, self.decoder_channels)
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
            self.change_decoder = DirectPairChangeDecoder(
                in_channels=self.in_channels,
                decoder_channels=self.decoder_channels,
                diff_mode=self.diff_mode,
                phase_pair=(0, 2),
            )
            self.change_head = ChangeHead(self.decoder_channels, self.num_change_classes, dropout=chg_head_dropout)

    def _normalize_feature_order(self, feature_xy: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feature_xy = list(feature_xy)
        return feature_xy if self.feature_order == "high_to_low" else list(reversed(feature_xy))

    def _resolve_output_size(self, input_size: Optional[Tuple[int, int]], highest_res_feature: torch.Tensor) -> Tuple[int, int]:
        if input_size is not None:
            return tuple(int(v) for v in input_size)
        if self.input_size is not None:
            return tuple(int(v) for v in self.input_size)
        return tuple(int(v) for v in highest_res_feature.shape[-2:])

    def _decode_semantic_shared(
        self,
        feature_xy_high_to_low: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        decoded_seq = self.semantic_decoder(feature_xy_high_to_low)  # [N,B,D,H0,W0]
        if self.phase_affine is not None:
            decoded_seq = self.phase_affine.forward_seq(decoded_seq)
        sem_logits_seq = self.semantic_head.forward_seq(decoded_seq, output_size=output_size)
        sem_logits = sem_logits_seq.permute(1, 0, 2, 3, 4).contiguous()  # [B,N,K,H,W]
        sem_logits_dict = {
            name: sem_logits_seq[idx]
            for idx, name in enumerate(PHASE_NAMES)
            if idx < sem_logits_seq.shape[0]
        }
        return sem_logits, sem_logits_dict, decoded_seq, sem_logits_seq

    def _decode_semantic_phasewise(
        self,
        feature_xy_high_to_low: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        sem_logits_dict: Dict[str, torch.Tensor] = {}
        sem_decoded_dict: Dict[str, torch.Tensor] = {}
        logits_seq: List[torch.Tensor] = []
        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            feats = [feat[phase_idx] for feat in feature_xy_high_to_low]
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
            logits_seq.append(logits)
        sem_logits_seq = torch.stack(logits_seq, dim=0)
        sem_logits = sem_logits_seq.permute(1, 0, 2, 3, 4).contiguous()
        return sem_logits, sem_logits_dict, sem_decoded_dict, sem_logits_seq

    def forward(
        self,
        feature_xy_high_to_low: Sequence[torch.Tensor],
        input_size: Optional[Tuple[int, int]] = None,
        pdca_aux: Optional[Dict[str, Any]] = None,
        task_evidence_aux: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Minimal assumptions only. feature_xy is a high-to-low list of [N,B,C,H,W].
        assert len(feature_xy_high_to_low) == self.num_scales, "feature_xy scale count mismatch"
        assert feature_xy_high_to_low[0].ndim == 5, "feature_xy[0] must be [N,B,C,H,W]"
        N, B, _, _, _ = feature_xy_high_to_low[0].shape
        assert N >= 3, "WUSU-compatible decoder requires at least three physical phases"

        feature_xy_high_to_low = self._normalize_feature_order(feature_xy_high_to_low)
        output_size = self._resolve_output_size(input_size, feature_xy_high_to_low[0])
        # sem_logits：(B, N, Class, H, W)  sem_logits_dict: {'T1':(B,Class, H, W), 'T2':(B,Class, H, W), 'T3':(B,Class, H, W)}
        # sem_decoded：(N, B, D, H, W)  sem_logits_seq: (N, B, Class, H, W)
        if self.share_semantic_decoder:
            sem_logits, sem_logits_dict, sem_decoded, sem_logits_seq = self._decode_semantic_shared(
                feature_xy_high_to_low, output_size
            )
        else:
            sem_logits, sem_logits_dict, sem_decoded, sem_logits_seq = self._decode_semantic_phasewise(
                feature_xy_high_to_low, output_size
            )
        if self.use_pdca_guided_pair_decoder:
            change_logits_dict = self.pair_change_decoder(
                                  feature_xy_high_to_low=feature_xy_high_to_low,
                                  output_size=output_size,
                                  pdca_aux=pdca_aux,
                                  task_evidence_aux=task_evidence_aux,
                              )
            chg_logits = change_logits_dict["t1_to_t3"]
        else:
            change_decoded = self.change_decoder(feature_xy_high_to_low)
            chg_logits = self.change_head(change_decoded)
            chg_logits = F.interpolate(chg_logits, size=output_size, mode="bilinear", align_corners=False)
            change_logits_dict = {"t1_to_t3": chg_logits}

        outputs: Dict[str, Any] = {
            "sem_logits": sem_logits,
            "sem_logits_dict": sem_logits_dict,
            "chg_logits": chg_logits,
            "change_logits_dict": change_logits_dict,
        }

        return outputs

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        att_weight = self.activaton(y)

        return att_weight

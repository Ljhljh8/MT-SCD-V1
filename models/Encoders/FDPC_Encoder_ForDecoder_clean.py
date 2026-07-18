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
    - Clean relation guidance is limited to PDCA source weights needed by the
      guided pair decoder.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d
from models.dend_structure_routed_conv_v1_ablation import DendStructureRoutedConv2d
from models.dend_structure_routed_conv_v2 import DendStructureRoutedConv2dV2
from models.dend_structure_routed_conv_v3 import DendStructureRoutedConv2dV3
from models.Encoders.phase_deformable_context_attention_fordecoder_clean_v22 import PhaseDeformableContextAttention
from mmseg.Qtrick_architecture.clock_driven.neuron import MTSCDPRDNIIFNode, Q_IFNode
from mmseg.Qtrick_architecture.clock_driven.surrogate import Quant, Quant4

PairName = Tuple[str, str]
AuxDict = Dict[str, Dict[str, Any]]


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
        Down_K: bool = False,
        dend_spatial_conv_type: str = "fadc",
        scale_index: Optional[int] = None,

        routeconv_ablation_mode: str = "full",
        routeconv_v2_mode: str = "v2_6",
        routeconv_v3_mode: str = "v3_6",
    ):
        super().__init__()
        self.channels = int(channels)
        self.use_dendritic = bool(use_dendritic)
        self.dend_soma_type = _normalize_dend_soma_type(dend_soma_type)
        self.dend_spatial_conv_type = str(dend_spatial_conv_type).lower()

        self.routeconv_ablation_mode = str(routeconv_ablation_mode).lower()
        self.routeconv_v2_mode = str(routeconv_v2_mode).lower()
        self.routeconv_v3_mode = str(routeconv_v3_mode).lower()
        if (
            self.dend_spatial_conv_type != "structure_routed_v1"
            and self.routeconv_ablation_mode != "full"
        ):
            raise ValueError(
                "routeconv_ablation_mode requires "
                "dend_spatial_conv_type='structure_routed_v1'"
            )
        if (
            self.dend_spatial_conv_type != "structure_routed_v2"
            and self.routeconv_v2_mode != "v2_6"
        ):
            raise ValueError(
                "routeconv_v2_mode requires "
                "dend_spatial_conv_type='structure_routed_v2'"
            )
        if self.routeconv_v2_mode not in (
            "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v2_6"
        ):
            raise ValueError("routeconv_v2_mode must be one of: v2_1, v2_2, v2_3, v2_4, v2_5, v2_6")
        if (
            self.dend_spatial_conv_type != "structure_routed_v3"
            and self.routeconv_v3_mode != "v3_6"
        ):
            raise ValueError(
                "routeconv_v3_mode requires "
                "dend_spatial_conv_type='structure_routed_v3'"
            )
        if self.routeconv_v3_mode not in (
            "v3_1", "v3_2", "v3_3", "v3_4", "v3_5", "v3_6", "v3_7", "v3_8"
        ):
            raise ValueError(
                "routeconv_v3_mode must be one of: "
                "v3_1, v3_2, v3_3, v3_4, v3_5, v3_6", "v3_7", "v3_8"
            )
        if self.dend_spatial_conv_type not in (
            "fadc",
            "structure_routed_v1",
            "structure_routed_v2",
            "structure_routed_v3",
        ):
            raise ValueError(
                "dend_spatial_conv_type must be one of: "
                "fadc, structure_routed_v1, structure_routed_v2, "
                "structure_routed_v3"
            )
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

        adapter_cls = {
            "fadc": DendFADCConv2d,
            "structure_routed_v1": DendStructureRoutedConv2d,
            "structure_routed_v2": DendStructureRoutedConv2dV2,
            "structure_routed_v3": DendStructureRoutedConv2dV3,
        }[self.dend_spatial_conv_type]

        adapter_kwargs = dict(
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
            Down_K=Down_K,
        )
        if adapter_cls is DendStructureRoutedConv2d:
            adapter_kwargs["scale_index"] = scale_index
            adapter_kwargs["ablation_mode"] = self.routeconv_ablation_mode
        elif adapter_cls is DendStructureRoutedConv2dV2:
            adapter_kwargs["scale_index"] = scale_index
            adapter_kwargs["v2_mode"] = self.routeconv_v2_mode
        elif adapter_cls is DendStructureRoutedConv2dV3:
            adapter_kwargs["scale_index"] = scale_index
            adapter_kwargs["v3_mode"] = self.routeconv_v3_mode

        self.adapter = adapter_cls(**adapter_kwargs)

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
        # x = self.act(x)
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

class FDPCEncoder(nn.Module):
    """
    Frequency-Dendritic Phase Context Encoder, minimal v1.

    This clean version contains per-scale dendritic local enhancement and the
    PDCA blocks required by the guided pair decoder. It does not construct PRG
    or expose relation/debug tensors in aux.
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
        return_aux_default: bool = False,
        relation_mode: str = "pdca",
        pdca_cfg: Optional[dict] = None,
        pdca_dend_prior_source_weight=1.0,
        pdca_dend_prior_point_weight=0.25,
        pdca_dend_prior_use_offset_gate=True,
        pdca_dend_prior_center_point=True,
        pdca_dend_prior_clip=2.0,
        dend_spatial_conv_type: str = "fadc",
        routeconv_ablation_mode: str = "full",
        routeconv_v2_mode: str = "v2_6",
        routeconv_v3_mode: str = "v3_6",
    ):
        super().__init__()

        if len(in_channels) == 0:
            raise ValueError("in_channels must not be empty")
        if relation_mode not in ("pdca", "none"):
            raise ValueError("clean FDPC relation_mode must be one of: pdca, none")

        self.in_channels = [int(c) for c in in_channels]
        self.num_scales = len(self.in_channels)

        self.phase_names = tuple(str(name) for name in phase_names)
        self.phase_to_index = {name: idx for idx, name in enumerate(self.phase_names)}

        self.context_pairs = tuple(_ensure_pair_tuple(pair) for pair in context_pairs)
        self.dendritic_scales = set(int(s) for s in dendritic_scales)
        self.relation_scales = set(int(s) for s in relation_scales)

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
        Down_K = [True, True, True, False]
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
                    Down_K=Down_K[s],
                    dend_spatial_conv_type=dend_spatial_conv_type,
                    scale_index=s,
                    routeconv_ablation_mode=routeconv_ablation_mode,
                    routeconv_v2_mode=routeconv_v2_mode,
                    routeconv_v3_mode=routeconv_v3_mode,
                )
            )

        self.pdca_blocks = nn.ModuleDict()

        for s in sorted(self.relation_scales):
            channels = self.in_channels[s]
            key = str(s)

            if self.relation_mode == "pdca":
                self.pdca_blocks[key] = PhaseDeformableContextAttention(
                    channels=channels,
                    phase_names=self.phase_names,
                    context_pairs=self.context_pairs,
                    **self._resolve_pdca_cfg_for_scale(key)
                )


    @staticmethod
    def _new_aux():
        return {
            "pdca_source_weights": {},
            "pdca_source_names_by_target": {},
        }

    def _resolve_pdca_cfg_for_scale(self, scale_key: str) -> dict:
        cfg = dict(self.pdca_cfg)
        per_scale = cfg.pop("per_scale", {})
        if scale_key in per_scale:
            cfg.update(per_scale[scale_key])
        elif int(scale_key) in per_scale:
            cfg.update(per_scale[int(scale_key)])
        return cfg

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
        K_GATE: List[torch.Tensor] = []
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
            # if s in self.relation_scales:
            K_GATE.append(K)
            encoded.append(feat_i)

        aux = self._new_aux()

        for s in sorted(self.relation_scales):
            if self.relation_mode == "pdca":
                scale_key = str(s)
                encoded[s], pdca_aux = self.pdca_blocks[scale_key](
                    encoded[s], K_GATE[s],
                    return_aux=bool(return_aux),
                    detach_aux=bool(detach_aux),
                )
                if return_aux:
                    aux["pdca_source_weights"][scale_key] = pdca_aux.get("source_weights", {})
                    aux["pdca_source_names_by_target"][scale_key] = dict(
                        self.pdca_blocks[scale_key].source_names_by_target
                    )
            elif self.relation_mode == "none":
                pass

        return encoded, aux if return_aux else {}
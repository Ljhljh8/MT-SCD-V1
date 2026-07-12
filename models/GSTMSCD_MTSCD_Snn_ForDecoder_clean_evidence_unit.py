import os

import torch
from torch import nn

from mmseg.models.backbones import Spiking_vit_MetaFormer as SDTV2Backbone
from models.Decoders.Snn_Mtscd_Decoder_V4_DIRECTTRAIN_clean_evidence_unit import MTSCDDecoderNet
from models.Encoders.FDPC_Encoder_ForDecoder_clean_evidence_unit import FDPCEncoder


norm_cfg = dict(type="SyncBN", requires_grad=True)


class GSTMSCD_WUSU(nn.Module):
    def __init__(
        self,
        backbone,
        pretrained,
        nclass,
        relation_mode="pdca",
        use_pdca_guided_pair_decoder=False,
        detach_pdca_guidance=True,
        use_pdca_guidance=True,
        pdca_dend_prior_mode="offset_dual",
        pdca_dend_prior_alpha=1e-3,
        pdca_dend_prior_detach=True,
        pdca_dend_prior_descriptor="mean_std",
        pdca_dend_prior_normalize="zscore",
        pdca_dend_prior_sim_weight=1.0,
        pdca_dend_prior_diff_weight=0.25,
        pdca_dend_prior_use_conf_gate=True,
        pdca_dend_prior_conf_beta=4.0,
        pdca_dend_prior_conf_tau=0.10,
        pdca_dend_prior_affect_null=False,
        pdca_dend_prior_stats=False,
        k_mode='none',

        pdca_dend_prior_source_weight=1.0,
        pdca_dend_prior_point_weight=0.25,
        pdca_dend_prior_use_offset_gate=True,
        pdca_dend_prior_center_point=True,
        pdca_dend_prior_clip=2.0,

        task_calibrated: bool = False,
        stc_detach_context: bool = True,
        stc_detach_k_gate: bool = True,
        stc_update_k_from_prev: bool = False,
        stc_modulate_k: bool = True,
        stc_residual_init: float = 0.0,
        stc_k_scale_init: float = 0.0,
        stc_gate_kernel_size: int = 3,
        stc_gate_temperature: float = 1.0,
        stc_use_noise_suppression: bool = True,
        reset_before_forward: bool = False,
        
        collect_pdca_aux=None,
        task_evidence_cfg=None,
        decoder_task_gate_cfg=None,
        
    ):
        super().__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.relation_mode = relation_mode
        self.use_pdca_guided_pair_decoder = bool(use_pdca_guided_pair_decoder)
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.use_pdca_guidance = bool(use_pdca_guidance)

        if (
            self.use_pdca_guided_pair_decoder
            and self.use_pdca_guidance
            and self.relation_mode != "pdca"
        ):
            raise RuntimeError(
                "PDCA-guided pair decoder requires relation_mode='pdca' unless PDCA guidance is disabled."
            )

        self.channel_nums = [32, 64, 128, 360]
        if backbone != "sdtv2":
            raise ValueError("clean ForDecoder path supports backbone='sdtv2', got %r" % backbone)

        self.backbone = SDTV2Backbone(
            img_size_h=512,
            img_size_w=512,
            patch_size=16,
            in_channels=4,
            embed_dim=[64, 128, 256, 360],
            num_heads=8,
            mlp_ratios=4,
            num_classes=13,
            qkv_bias=False,
            depths=8,
            sr_ratios=1,
            T=1,
            norm_eval=True,
            norm_cfg=norm_cfg,
            decode_mode="Qsnn",
            init_cfg=None,
        )
        if pretrained:
            self._load_internal_pretrain()

        encoder_blocks = []
        num_encoder_blocks = 4
        self.task_evidence_cfg = dict(task_evidence_cfg or {})
        self.decoder_task_gate_cfg = dict(decoder_task_gate_cfg or {})
        self.collect_task_evidence_aux = bool(self.decoder_task_gate_cfg.get("enabled", False))
        for block_idx in range(num_encoder_blocks):
            task_cfg = dict(self.task_evidence_cfg)
            task_scale = int(task_cfg.pop("scale", 3))
        
            enabled = bool(task_cfg.get("enabled", False))
            if block_idx != num_encoder_blocks - 1:
                task_cfg["enabled"] = False
            else:
                task_cfg["enabled"] = enabled
        
            encoder_blocks.append(
                FDPCEncoder(
                    in_channels=self.channel_nums,
                    phase_names=("t1", "t2", "t3"),
                    context_pairs=(("t1", "t2"), ("t2", "t3"), ("t1", "t3")),
                    dendritic_scales=(1, 2, 3),
                    relation_scales=(3,),
                    conv_groups="depthwise",
                    deform_groups=1,
                    dend_kernel_size=3,
                    fs_cfg=dict(
                        k_list=[2, 4, 8],
                        lowfreq_att=False,
                        lp_type="freq",
                        act="sigmoid",
                        spatial="conv",
                        spatial_group=1,
                    ),
                    kernel_decompose="both",
                    norm="gn",
                    dend_residual_init=0.0,
                    relation_mode=relation_mode,
                    pdca_cfg=dict(
                        num_heads=4,
                        num_points=24,
                        offset_radius=4.0,
                        use_null_source=True,
                        residual_init=1e-3,
                        pdca_dend_prior_mode=pdca_dend_prior_mode,
                        pdca_dend_prior_alpha=pdca_dend_prior_alpha,
                        pdca_dend_prior_detach=pdca_dend_prior_detach,
                        pdca_dend_prior_descriptor=pdca_dend_prior_descriptor,
                        pdca_dend_prior_normalize=pdca_dend_prior_normalize,
                        pdca_dend_prior_sim_weight=pdca_dend_prior_sim_weight,
                        pdca_dend_prior_diff_weight=pdca_dend_prior_diff_weight,
                        pdca_dend_prior_use_conf_gate=pdca_dend_prior_use_conf_gate,
                        pdca_dend_prior_conf_beta=pdca_dend_prior_conf_beta,
                        pdca_dend_prior_conf_tau=pdca_dend_prior_conf_tau,
                        pdca_dend_prior_affect_null=pdca_dend_prior_affect_null,
                        pdca_dend_prior_stats=pdca_dend_prior_stats,
                        per_scale={
                            "2": {"offset_radius": 64.0},
                            "3": {"offset_radius": 32.0},
                        },
                    ),
                    return_aux_default=False,
                    task_evidence_cfg=task_cfg,
                    task_evidence_scales=(task_scale,),


                    pdca_dend_prior_source_weight=pdca_dend_prior_source_weight,
                    pdca_dend_prior_point_weight=pdca_dend_prior_point_weight,
                    pdca_dend_prior_use_offset_gate=pdca_dend_prior_use_offset_gate,
                    pdca_dend_prior_center_point=pdca_dend_prior_center_point,
                    pdca_dend_prior_clip=pdca_dend_prior_clip,
                )
            )
        
        self.encoder = nn.ModuleList(encoder_blocks)
        
        
        self.collect_pdca_aux = (
            self.use_pdca_guided_pair_decoder and self.use_pdca_guidance
            if collect_pdca_aux is None
            else bool(collect_pdca_aux)
        )
        

        
        self.decoder = nn.ModuleList(
            [
                MTSCDDecoderNet(
                    in_channels=self.channel_nums,
                    decoder_channels=256,
                    num_sem_classes=13,
                    num_change_classes=1,
                    input_size=(512, 512),
                    phase_windows={"t1": [0], "t2": [1], "t3": [2]},
                    transition_windows={"t1_to_t2": None, "t2_to_t3": None, "t1_to_t3": None},
                    temporal_readout="attention",
                    diff_mode="abs_signed",
                    share_semantic_decoder=True,
                    feature_order="high_to_low",
                    use_phase_affine=False,
                    use_phase_classifier_bias=False,
                    use_transition_fusion=False,
                    use_pdca_guided_pair_decoder=self.use_pdca_guided_pair_decoder,
                    detach_pdca_guidance=self.detach_pdca_guidance,
                    use_pdca_guidance=self.use_pdca_guidance,
                    decoder_task_gate_cfg=self.decoder_task_gate_cfg,
                )
            ]
        )

    def _load_internal_pretrain(self):
        pretrain_path = "./GSTM-SCD_Pretraining-weights/Meta-Spikeformer-15M.pth"
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(
                "Internal pretrained checkpoint is missing: %s. "
                "Provide this file, pass --pretrained false, or pass --pretrain-from "
                "to the clean training entrypoint for an explicit non-strict warm start."
                % pretrain_path
            )
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        source_state = checkpoint["model"]
        target_state = self.backbone.state_dict()
        updated = {}
        for key, value in source_state.items():
            if key not in target_state:
                continue
            if key == "downsample1_1.encode_conv.weight":
                new_value = torch.zeros_like(target_state[key])
                new_value[:, :3, :, :] = value
                updated[key] = new_value
            elif key.startswith(("downsample1_1.", "levels.")):
                updated[key] = value
            else:
                updated[key] = value
        self.backbone.load_state_dict(updated, strict=True)
        print("Successfully loaded pre-training weights!")

    def forward(self, x, return_change_logits_dict: bool = False):
        _, _, _, h, w = x.shape
        feature_xy = self.backbone(x)

        pdca_aux = None
        task_evidence_aux = None
        
        for index, block in enumerate(self.encoder):
            is_last_block = index == len(self.encoder) - 1
            collect_aux = is_last_block and (
                self.collect_pdca_aux or self.collect_task_evidence_aux
            )
        
            if collect_aux:
                feature_xy, aux = block(
                    feature_xy,
                    return_aux=True,
                    detach_aux=self.detach_pdca_guidance,
                )
                if self.collect_pdca_aux:
                    pdca_aux = aux
                if self.collect_task_evidence_aux:
                    task_evidence_aux = aux.get("task_evidence", {})
            else:
                feature_xy, _ = block(feature_xy, return_aux=False)
       
        for block in self.decoder:
            outs = block(
                        feature_xy,
                        input_size=(h, w),
                        pdca_aux=pdca_aux,
                        task_evidence_aux=task_evidence_aux,
                    )

        seg1 = outs["sem_logits_dict"]["t1"]
        seg2 = outs["sem_logits_dict"]["t2"]
        seg3 = outs["sem_logits_dict"]["t3"]
        change13 = outs["chg_logits"].squeeze(1)
        if return_change_logits_dict:
            return seg1, seg2, seg3, change13, outs["change_logits_dict"]
        return seg1, seg2, seg3, change13

import os

import torch
from torch import nn

from mmseg.models.backbones import Spiking_vit_MetaFormer as SDTV2Backbone
from models.Decoders.Snn_Mtscd_Decoder_V4_DIRECTTRAIN_clean_VDend1 import MTSCDDecoderNet
from models.Encoders.FDPC_Encoder_ForDecoder_clean_VDend1 import FDPCEncoder


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

        self.encoder = nn.ModuleList(
            [
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
                        # NEW SWITCH
                        task_calibrated=task_calibrated,
                        stc_detach_context=stc_detach_context,
                        stc_detach_k_gate=stc_detach_k_gate,
                        stc_update_k_from_prev=stc_update_k_from_prev,
                        stc_modulate_k=stc_modulate_k,
                        stc_residual_init=stc_residual_init,
                        stc_k_scale_init=stc_k_scale_init,
                        stc_gate_kernel_size=stc_gate_kernel_size,
                        stc_gate_temperature=stc_gate_temperature,
                        stc_use_noise_suppression=stc_use_noise_suppression,
                        reset_before_forward=reset_before_forward,


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
                    k_mode = k_mode
                )
                for _ in range(4)
            ]
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

        guidance = None
        for index, block in enumerate(self.encoder):
            collect_guidance = (
                self.use_pdca_guided_pair_decoder
                and self.use_pdca_guidance
                and index == len(self.encoder) - 1   # ONLY FOR THE LAST
            )
            if collect_guidance:
                feature_xy, guidance = block(
                    feature_xy,
                    return_aux=True,
                    detach_aux=self.detach_pdca_guidance,
                )
            else:
                feature_xy, _ = block(feature_xy, return_aux=False)

        outs = None
        for block in self.decoder:
            outs = block(
                feature_xy,
                input_size=(h, w),
                pdca_aux=guidance,
            )

        seg1 = outs["sem_logits_dict"]["t1"]
        seg2 = outs["sem_logits_dict"]["t2"]
        seg3 = outs["sem_logits_dict"]["t3"]
        change13 = outs["chg_logits"].squeeze(1)
        if return_change_logits_dict:
            return seg1, seg2, seg3, change13, outs["change_logits_dict"]
        return seg1, seg2, seg3, change13

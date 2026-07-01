from models.Decoders.Snn_Mtscd_Decoder_V4_DIRECTTRAIN import MTSCDDecoderNet
import torch
from torch import nn
# from models.Backbones.sdtv2 import Spiking_vit_MetaFormer as SDTV2Backbone
# from models.Backbones.sdtv3 import Spiking_vit_MetaFormerv2  as SDTV3Backbone
from mmseg.models.backbones import Spiking_vit_MetaFormer as SDTV2Backbone
from models.Encoders.FDPC_Encoder_ForDecoder import FDPCEncoder
# from models.SNN_Models_DendFADC import Spiking_vit_MetaFormer
from functools import partial

#WUSU最优模型
norm_cfg = dict(type='SyncBN', requires_grad=True)
class GSTMSCD_WUSU(nn.Module):
    def __init__(
        self,
        backbone,
        pretrained,
        nclass,
        # lightweight,
        # M,
        # Lambda,
        relation_mode="pdca",
        use_pdca_relation_aux=False,
        use_pairrel_aux=False,
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
        pdca_dend_prior_stats=True,
    ):
        super(GSTMSCD_WUSU, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        # self.lightweight = lightweight

        self.use_pdca_relation_aux = bool(use_pdca_relation_aux)
        self.use_pairrel_aux = bool(use_pairrel_aux)
        self.relation_mode = relation_mode
        self.use_pdca_guided_pair_decoder = bool(use_pdca_guided_pair_decoder)
        self.detach_pdca_guidance = bool(detach_pdca_guidance)
        self.use_pdca_guidance = bool(use_pdca_guidance)
        self.pdca_dend_prior_mode = str(pdca_dend_prior_mode)
        self.pdca_dend_prior_alpha = float(pdca_dend_prior_alpha)
        self.pdca_dend_prior_detach = bool(pdca_dend_prior_detach)
        self.pdca_dend_prior_descriptor = str(pdca_dend_prior_descriptor)
        self.pdca_dend_prior_normalize = str(pdca_dend_prior_normalize)
        self.pdca_dend_prior_sim_weight = float(pdca_dend_prior_sim_weight)
        self.pdca_dend_prior_diff_weight = float(pdca_dend_prior_diff_weight)
        self.pdca_dend_prior_use_conf_gate = bool(pdca_dend_prior_use_conf_gate)
        self.pdca_dend_prior_conf_beta = float(pdca_dend_prior_conf_beta)
        self.pdca_dend_prior_conf_tau = float(pdca_dend_prior_conf_tau)
        self.pdca_dend_prior_affect_null = bool(pdca_dend_prior_affect_null)
        self.pdca_dend_prior_stats = bool(pdca_dend_prior_stats)

        if (
            self.use_pdca_guided_pair_decoder
            and self.use_pdca_guidance
            and self.relation_mode != "pdca"
        ):
            raise RuntimeError(
                "PDCA-guided pair decoder requires relation_mode='pdca' unless PDCA guidance is disabled."
            )

        # self.channel_nums = [64, 128, 256, 256, 256]
        self.channel_nums = [32, 64, 128, 360]

        if backbone == "sdtv2":
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
        elif backbone == "sdtv3":
            self.backbone = SDTV3Backbone(
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
                decode_mode="QTrick",
                init_cfg=None,
            )
        updated_weights = {}
        pretrained_weights = torch.load('./GSTM-SCD_Pretraining-weights/Meta-Spikeformer-15M.pth')
        new_dict = pretrained_weights['model']
        # 防止权重不匹配
        for key, value in new_dict.items():
            if key.startswith(('downsample1_1.', 'levels.')):
                if key == 'downsample1_1.encode_conv.weight':
                    # 检查当前模型的 conv1.weight 形状
                    current_conv1_weight = self.backbone.state_dict()[key]
                    # 创建一个新的权重，形状与当前模型一致
                    new_conv1_weight = torch.zeros_like(current_conv1_weight)
                    # 将预训练权重的前3通道复制到新权重的前3通道
                    new_conv1_weight[:, :3, :, :] = value
                    # 将新权重添加到 updated_weights
                    updated_weights[key] = new_conv1_weight
                else:
                    if key in self.backbone.state_dict():
                        updated_weights[key] = value
            else:
                if key in self.backbone.state_dict():
                    updated_weights[key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        # self.encoder = Spiking_vit_MetaFormer(
        #     detach_reset=True,
        #     img_size_h=512,
        #     img_size_w=512,
        #     patch_size=16,
        #     embed_dim=[64, 128, 256, 256, 256],
        #     num_heads=8,
        #     mlp_ratios=4,
        #     in_channels=32,
        #     num_classes=13,
        #     qkv_bias=False,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     depths=8,
        #     sr_ratios=1,
        # )

        # self.decoder = SNNMTSCDDecoderHead(
        #     encoder_channels=self.channel_nums,
        #     num_semantic_classes=13,
        #     num_change_classes=1,
        #     num_phases=3,
        #     decoder_channels=[256, 256, 256, 256, 256],
        #     phase_agg_mode="attn",
        #     share_semantic_decoder=True,
        #     diff_mode="abs_sub",
        #     phase_pair=(0, 2),
        # )

        T, B = 8, 2
        H, W = 512, 512
        num_sem_classes = 13
        num_change_classes = 1
        self.encoder = nn.ModuleList(
            [
                FDPCEncoder(
                    in_channels=self.channel_nums,  # [64,128,256,256,256]    [32, 64, 128, 360]
                    phase_names=("t1", "t2", "t3"),
                    context_pairs=(("t1", "t2"), ("t2", "t3"), ("t1", "t3")),
                    dendritic_scales=(1, 2, 3),  # f1 不加树突模块         (1, 2, 3, 4)    (1, 2, 3)
                    relation_scales=(3,),  # 只在高层做 relation gate      (3, 4),    (2, 3, ),
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
                    context_residual_init=0.0,
                    detach_context_gate=False,
                    relation_mode=relation_mode,
                    # relation_mode="pdca",
                    pdca_cfg=dict(
                        num_heads=4,
                        num_points=24,
                        offset_radius=4.0,
                        use_null_source=True,
                        residual_init=1e-3,
                        use_relation_aux=self.use_pdca_relation_aux and j == 3,
                        relation_aux_pairs=("t1<-t3", "t3<-t1"),

                        pdca_dend_prior_mode=self.pdca_dend_prior_mode,
                        pdca_dend_prior_alpha=self.pdca_dend_prior_alpha,
                        pdca_dend_prior_detach=self.pdca_dend_prior_detach,
                        pdca_dend_prior_descriptor=self.pdca_dend_prior_descriptor,
                        pdca_dend_prior_normalize=self.pdca_dend_prior_normalize,
                        pdca_dend_prior_sim_weight=self.pdca_dend_prior_sim_weight,
                        pdca_dend_prior_diff_weight=self.pdca_dend_prior_diff_weight,
                        pdca_dend_prior_use_conf_gate=self.pdca_dend_prior_use_conf_gate,
                        pdca_dend_prior_conf_beta=self.pdca_dend_prior_conf_beta,
                        pdca_dend_prior_conf_tau=self.pdca_dend_prior_conf_tau,
                        pdca_dend_prior_affect_null=self.pdca_dend_prior_affect_null,
                        pdca_dend_prior_stats=self.pdca_dend_prior_stats,

                        per_scale={
                            "2": {"offset_radius": 64.0},
                            "3": {"offset_radius": 32.0},
                        },
                    ),
                    return_aux_default=False,
                )
                for j in range(4)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                MTSCDDecoderNet(
                    in_channels=self.channel_nums,
                    decoder_channels=256,
                    num_sem_classes=num_sem_classes,
                    num_change_classes=num_change_classes,
                    input_size=(H, W),
                    phase_windows={"t1": [0], "t2": [1], "t3": [2]},
                    # {"t1": [0, 1], "t2": [3, 4], "t3": [6, 7]}   {"t1": [0], "t2": [1], "t3": [2]},
                    transition_windows={"t1_to_t2": None, "t2_to_t3": None, "t1_to_t3": None},
                    # T=12 -> 默认无 transition windows
                    temporal_readout="attention",
                    diff_mode="abs_signed",
                    share_semantic_decoder=True,
                    feature_order="high_to_low",
                    use_phase_affine=False,
                    use_phase_classifier_bias=False,
                    use_transition_fusion=False,
                    return_intermediates_default=True,
                    use_pdca_guided_pair_decoder=self.use_pdca_guided_pair_decoder,
                    detach_pdca_guidance=self.detach_pdca_guidance,
                    use_pdca_guidance=self.use_pdca_guidance,
                )
                for j in range(1)
            ]
        )

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, return_aux: bool = False):
        t, b, c, h, w = x.shape
        feature_xy = self.backbone(x)

        encoder_aux_list = []
        pdca_aux = None
        for index, blk in enumerate(self.encoder):
            collect_pair_guidance = (
                self.use_pdca_guided_pair_decoder
                and self.use_pdca_guidance
                and index == len(self.encoder) - 1  # 仅最后一层
            )
            collect_relation_aux = (
                return_aux
                and self.use_pdca_relation_aux
                and index == len(self.encoder) - 1
            )
            if collect_pair_guidance or collect_relation_aux:
                feature_xy, encoder_aux = blk(
                    feature_xy,
                    return_aux=True,
                    detach_aux=False,
                    relation_aux_only=bool(self.use_pdca_relation_aux),
                )
                if collect_pair_guidance:
                    pdca_aux = encoder_aux
                if collect_relation_aux:
                    encoder_aux_list.append(encoder_aux)
            else:
                feature_xy, _ = blk(feature_xy, return_aux=False)
        # upsampled_xy = [4, 8, 16, 32, 32]
        # for idx, feature in enumerate(feature_xy):
        #     feature_xy[idx] = feature.reshape(b, -1, h // upsampled_xy[idx], w // upsampled_xy[idx], t).permute(4, 0, 1, 2, 3)
        # outs = self.decoder(feature_xy, out_size=(h, w), phase_windows=phase_windows_8_K2_R1)
        for blk in self.decoder:
            outs = blk(
                feature_xy,
                input_size=(h, w),
                return_intermediates=False,
                pdca_aux=pdca_aux,
            )



        seg1, seg2, seg3 = outs["sem_logits_dict"]["t1"], outs["sem_logits_dict"]["t2"], outs["sem_logits_dict"]["t3"]
        change13 = outs["chg_logits"]

        if return_aux:
            aux = {}
            if self.use_pdca_guided_pair_decoder:
                aux["change_logits_dict"] = outs.get("change_logits_dict", {})
                aux["pair_gate_debug"] = outs.get("pair_gate_debug", {})
                aux["encoder_aux"] = pdca_aux
            if self.use_pdca_relation_aux:
                aux["encoder_aux"] = encoder_aux_list
            if self.use_pairrel_aux:
                aux["encoder_features"] = {2: feature_xy[2], 3: feature_xy[3]}
            return seg1, seg2, seg3, change13.squeeze(1), aux
        return seg1, seg2, seg3, change13.squeeze(1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MTGrootV3D_SV3(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # model = ST_VSSM_Siam().to(device)
    model = GSTMSCD_WUSU(backbone='sdtv2', pretrained=False, nclass=13, lightweight=True, M=6, Lambda=0.00005).to(device)
    print(model)
    image1 = torch.randn(2, 4, 512, 512).to(device)
    image2 = torch.randn(2, 4, 512, 512).to(device)
    image3 = torch.randn(2, 4, 512, 512).to(device)
    image4 = torch.randn(2, 4, 512, 512).to(device)
    image5 = torch.randn(2, 4, 512, 512).to(device)
    image6 = torch.randn(2, 4, 512, 512).to(device)
    # seg1, seg2, seg3, change = model(image1, image2, image3)
    x = torch.stack([image1, image2, image3], dim=0)
    fs = model(x)
    # print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, inputs=(x,))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

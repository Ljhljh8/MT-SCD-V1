import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from unittest.mock import patch


repo_root = Path(__file__).resolve().parents[1]
mmseg = types.ModuleType("mmseg")
mmseg.__path__ = [str(repo_root / "mmseg")]
sys.modules.setdefault("mmseg", mmseg)

from models.Encoders.phase_deformable_context_attention import (
    PhaseDeformableContextAttention,
)
from models.Encoders.FDPC_Encoder import FDPCEncoder


decoder_module = types.ModuleType("models.Decoders.Snn_Mtscd_Decoder_V2")
backbones_module = types.ModuleType("mmseg.models.backbones")
mmseg_models_module = types.ModuleType("mmseg.models")
pae_module = types.ModuleType("utils.PAE_NET")


class StubBackbone(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, x):
        return [x, x, x, x]


class StubDecoder(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.return_intermediates_args = []

    def forward(self, feature_xy, input_size=None, return_intermediates=None):
        self.return_intermediates_args.append(return_intermediates)
        b = feature_xy[0].shape[1]
        h, w = input_size
        sem = {
            name: feature_xy[0].new_zeros(b, 13, h, w)
            for name in PHASE_NAMES
        }
        return {"sem_logits_dict": sem, "chg_logits": feature_xy[0].new_zeros(b, 1, h, w)}


class StubPAENTE(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()


class StubEncoder(nn.Module):
    def forward(
        self,
        feature_xy,
        return_aux=False,
        detach_aux=False,
        relation_aux_only=False,
    ):
        del detach_aux
        aux = {"pdca_relation_logits": {"3": {}}} if return_aux and relation_aux_only else {}
        return feature_xy, aux


decoder_module.MTSCDDecoderNet = StubDecoder
backbones_module.Spiking_vit_MetaFormer = StubBackbone
pae_module.PAENTE = StubPAENTE
sys.modules["models.Decoders.Snn_Mtscd_Decoder_V2"] = decoder_module
sys.modules["mmseg.models"] = mmseg_models_module
sys.modules["mmseg.models.backbones"] = backbones_module
sys.modules["utils.PAE_NET"] = pae_module

from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU
from utils.pdca_aux_loss import pdca_relation_aux_loss


PHASE_NAMES = ("t1", "t2", "t3")
CONTEXT_PAIRS = (("t1", "t2"), ("t2", "t3"), ("t1", "t3"))
RAS_PAIRS = ("t1<-t3", "t3<-t1")


class PDCARASV1Test(unittest.TestCase):
    def make_pdca(self, **overrides):
        kwargs = dict(
            channels=360,
            phase_names=PHASE_NAMES,
            context_pairs=CONTEXT_PAIRS,
            num_heads=4,
            num_points=24,
            use_null_source=True,
            use_relation_aux=True,
            relation_aux_pairs=RAS_PAIRS,
            relation_aux_hidden_channels=32,
        )
        kwargs.update(overrides)
        return PhaseDeformableContextAttention(**kwargs)

    def test_relation_logits_shape_and_stateless_head(self):
        feat = torch.randn(3, 2, 360, 16, 16)
        pdca = self.make_pdca()

        out, aux = pdca(feat, return_aux=True, relation_aux_only=True)

        self.assertEqual(out.shape, feat.shape)
        self.assertEqual(aux["relation_logits"]["t1<-t3"].shape, (2, 1, 16, 16))
        self.assertEqual(aux["relation_logits"]["t3<-t1"].shape, (2, 1, 16, 16))
        self.assertTrue(any(isinstance(module, nn.GELU) for module in pdca.relation_aux_head.modules()))
        self.assertFalse(
            any(module.__class__.__name__ == "Q_IFNode" for module in pdca.relation_aux_head.modules())
        )

    def test_relation_aux_pairs_are_validated(self):
        invalid_pairs = (
            ("t1-t3",),
            ("t1<-t4",),
            ("t1<-__null__",),
            ("t1<-t1",),
            ("t1<-t2", "t1<-t2"),
        )
        for pairs in invalid_pairs:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.make_pdca(relation_aux_pairs=pairs)

        with self.assertRaises(ValueError):
            self.make_pdca(
                context_pairs=(("t1", "t3"),),
                relation_aux_pairs=("t1<-t2",),
            )

    def test_relation_aux_only_keeps_full_pdca_main_path(self):
        pdca = self.make_pdca(channels=64, num_points=2, relation_aux_hidden_channels=16)
        feat = torch.randn(3, 1, 64, 8, 8)
        calls = {"offset": 0, "attn": 0, "value": 0, "sample": 0, "out": 0}

        hooks = []
        for name, module in (
            ("offset", pdca.offset_head),
            ("attn", pdca.attn_head),
            ("value", pdca.value_proj),
            ("out", pdca.out_proj),
        ):
            hooks.append(module.register_forward_hook(lambda _m, _i, _o, key=name: calls.__setitem__(key, calls[key] + 1)))

        original_sample = pdca._deformable_sample_vectorized

        def counted_sample(value, offset):
            calls["sample"] += 1
            return original_sample(value, offset)

        pdca._deformable_sample_vectorized = counted_sample
        try:
            out, aux = pdca(feat, return_aux=True, relation_aux_only=True)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(out.shape, feat.shape)
        self.assertEqual(calls, {"offset": 6, "attn": 6, "value": 6, "sample": 6, "out": 3})
        self.assertEqual(set(aux["relation_logits"]), set(RAS_PAIRS))
        for key in ("offsets", "attn_weights", "source_weights", "joint_weights"):
            self.assertEqual(aux[key], {})

    def test_fdpc_propagates_relation_logits_only(self):
        encoder = FDPCEncoder(
            in_channels=(16, 16, 16, 16),
            phase_names=PHASE_NAMES,
            context_pairs=CONTEXT_PAIRS,
            dendritic_scales=(),
            relation_scales=(3,),
            relation_mode="pdca",
            pdca_cfg=dict(
                num_heads=4,
                num_points=2,
                use_relation_aux=True,
                relation_aux_pairs=RAS_PAIRS,
                relation_aux_hidden_channels=8,
            ),
        )
        features = [torch.randn(3, 1, 16, 8, 8) for _ in range(4)]

        outputs, aux = encoder(features, return_aux=True, relation_aux_only=True)

        self.assertEqual([out.shape for out in outputs], [feat.shape for feat in features])
        self.assertEqual(set(aux["pdca_relation_logits"]["3"]), set(RAS_PAIRS))
        for key in (
            "pdca_offsets",
            "pdca_attn_weights",
            "pdca_source_weights",
            "pdca_joint_weights",
        ):
            self.assertEqual(aux[key]["3"], {})

        loss, _ = pdca_relation_aux_loss([aux], torch.randint(0, 2, (1, 8, 8)).float())
        loss.backward()
        relation_head = encoder.pdca_blocks["3"].relation_aux_head
        self.assertTrue(all(param.grad is not None for param in relation_head.parameters()))

    def test_model_default_and_pdca_aux_payload(self):
        with patch("torch.load", return_value={"model": {}}):
            model = GSTMSCD_WUSU(
                backbone="sdtv2",
                pretrained=False,
                nclass=13,
                lightweight=True,
                M=6,
                Lambda=0.00005,
                relation_mode="pdca",
                use_pdca_relation_aux=True,
            )

        relation_head_keys = [key for key in model.state_dict() if ".relation_aux_head." in key]
        self.assertTrue(relation_head_keys)
        self.assertTrue(all(key.startswith("encoder.3.") for key in relation_head_keys))
        model.encoder = nn.ModuleList([StubEncoder() for _ in range(4)])

        x = torch.randn(3, 1, 4, 8, 8)
        default_outputs = model(x)
        aux_outputs = model(x, return_aux=True)

        self.assertEqual(len(default_outputs), 4)
        self.assertEqual(len(aux_outputs), 5)
        self.assertEqual(set(aux_outputs[4]), {"encoder_aux"})
        self.assertEqual(len(aux_outputs[4]["encoder_aux"]), 1)
        self.assertEqual(model.decoder[0].return_intermediates_args, [False, False])

        with patch("torch.load", return_value={"model": {}}):
            baseline = GSTMSCD_WUSU(
                backbone="sdtv2",
                pretrained=False,
                nclass=13,
                lightweight=True,
                M=6,
                Lambda=0.00005,
                relation_mode="pdca",
                use_pdca_relation_aux=False,
            )
        self.assertFalse(any(".relation_aux_head." in key for key in baseline.state_dict()))


if __name__ == "__main__":
    unittest.main()

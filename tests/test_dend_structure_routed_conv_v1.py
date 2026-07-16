import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


repo_root = Path(__file__).resolve().parents[1]
mmseg = types.ModuleType("mmseg")
mmseg.__path__ = [str(repo_root / "mmseg")]
sys.modules.setdefault("mmseg", mmseg)

from models.dend_structure_routed_conv_v1 import DendStructureRoutedConv2d
from models.dendsn_lifFADC_Snn_v2 import DendFADCConv2d
from models.Encoders.FDPC_Encoder_ForDecoder_clean import FDPCEncoder


class CleanStructureRoutedWiringTest(unittest.TestCase):
    def make_encoder(self, dend_spatial_conv_type="fadc"):
        return FDPCEncoder(
            in_channels=[4, 4, 4, 4],
            dendritic_scales=(1, 2, 3),
            relation_scales=(),
            relation_mode="none",
            fs_cfg=dict(k_list=[2, 4], lowfreq_att=False, lp_type="avgpool"),
            dend_spatial_conv_type=dend_spatial_conv_type,
        )

    def test_clean_encoder_keeps_fadc_default_and_builds_independent_v1_stages(self):
        baseline = self.make_encoder()
        self.assertIsInstance(baseline.scale_adapters[1].adapter, DendFADCConv2d)

        encoders = [self.make_encoder("structure_routed_v1") for _ in range(4)]
        canonicals = []
        for encoder in encoders:
            self.assertIsInstance(encoder.scale_adapters[0].adapter, nn.Identity)
            for scale_index in (1, 2, 3):
                adapter = encoder.scale_adapters[scale_index]
                self.assertIsInstance(adapter.adapter, DendStructureRoutedConv2d)
                self.assertEqual(adapter.adapter.scale_index, scale_index)
                canonicals.append(adapter.adapter.spatial_bases.canonical)
        self.assertEqual(len({id(parameter) for parameter in canonicals}), 12)

        features = [
            torch.randn(3, 1, 4, 16, 16),
            torch.randn(3, 1, 4, 8, 8),
            torch.randn(3, 1, 4, 4, 4),
            torch.randn(3, 1, 4, 2, 2),
        ]
        encoded, aux = encoders[0](features)
        self.assertEqual([item.shape for item in encoded], [item.shape for item in features])
        self.assertEqual(aux, {})

    def test_clean_model_and_train_entry_are_explicitly_wired(self):
        model_source = (
            repo_root / "models" / "GSTMSCD_MTSCD_Snn_ForDecoder_clean_V4.py"
        ).read_text(encoding="utf-8")
        train_source = (repo_root / "train_WUSU_main_clean_pairbcd.py").read_text(
            encoding="utf-8"
        )
        build_model_block = train_source.split("def build_model", 1)[1].split("\ndef ", 1)[0]

        self.assertIn('dend_spatial_conv_type="structure_routed_v1"', model_source)
        self.assertIn("dend_residual_init=0.01", model_source)
        self.assertIn("incompatible.missing_keys", build_model_block)
        self.assertIn("incompatible.unexpected_keys", build_model_block)


if __name__ == "__main__":
    unittest.main()

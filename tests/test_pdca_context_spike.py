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

from models.Encoders.FDPC_Encoder import FDPCEncoder
from models.Encoders.phase_deformable_context_attention import PhaseDeformableContextAttention
import train_WUSU_ddp_accum_v9_ForDecoder as train


def phase_names(n):
    return tuple("t%d" % (idx + 1) for idx in range(n))


def adjacent_pairs(names):
    return tuple((names[idx], names[idx + 1]) for idx in range(len(names) - 1))


def assert_finite_stats(testcase, stats_by_target):
    testcase.assertTrue(stats_by_target)
    for stats in stats_by_target.values():
        testcase.assertIn("mode", stats)
        for key, value in stats.items():
            if torch.is_tensor(value):
                testcase.assertFalse(value.requires_grad, key)
                testcase.assertTrue(torch.isfinite(value).all(), key)


class PDCAContextSpikeTest(unittest.TestCase):
    def run_pdca_case(self, mode, n, use_null_source):
        names = phase_names(n)
        x = torch.randn(n, 1, 32, 6, 5, requires_grad=True)
        pdca = PhaseDeformableContextAttention(
            channels=32,
            phase_names=names,
            context_pairs=adjacent_pairs(names),
            num_heads=4,
            num_points=2,
            hidden_channels=16,
            use_null_source=use_null_source,
            pdca_context_spike_mode=mode,
        )

        self.assertEqual(pdca.pdca_context_spike_runtime_mode, mode)
        if mode == "none":
            self.assertIsNone(pdca.context_spike_act)
            self.assertIsNone(pdca.context_routing_spike)

        out, aux = pdca(x, return_aux=True, detach_aux=True)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertIn("context_spike", aux)
        assert_finite_stats(self, aux["context_spike"])
        out.float().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_modes_n3_and_n6_with_and_without_null_source(self):
        for mode in ("none", "weights", "values", "both", "context"):
            for n in (3, 6):
                for use_null_source in (True, False):
                    with self.subTest(mode=mode, n=n, use_null_source=use_null_source):
                        self.run_pdca_case(mode, n, use_null_source)

    def test_runtime_mode_can_disable_constructed_routing(self):
        names = phase_names(3)
        x = torch.randn(3, 1, 32, 5, 5)
        pdca = PhaseDeformableContextAttention(
            channels=32,
            phase_names=names,
            context_pairs=adjacent_pairs(names),
            num_heads=4,
            num_points=2,
            hidden_channels=16,
            pdca_context_spike_mode="weights",
        )

        pdca.pdca_context_spike_runtime_mode = "none"
        out, aux = pdca(x, return_aux=True, detach_aux=True)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(pdca.pdca_context_spike_mode, "weights")
        self.assertEqual(pdca.pdca_context_spike_runtime_mode, "none")
        self.assertTrue(all(stats["mode"] == "none" for stats in aux["context_spike"].values()))

    def test_fdpc_propagates_context_spike_aux(self):
        features = [
            torch.randn(3, 1, 32, 8, 8),
            torch.randn(3, 1, 32, 8, 8),
            torch.randn(3, 1, 32, 8, 8),
            torch.randn(3, 1, 32, 8, 8),
        ]
        encoder = FDPCEncoder(
            in_channels=(32, 32, 32, 32),
            phase_names=("t1", "t2", "t3"),
            context_pairs=(("t1", "t2"), ("t2", "t3")),
            dendritic_scales=(),
            relation_scales=(3,),
            relation_mode="pdca",
            pdca_cfg=dict(
                num_heads=4,
                num_points=2,
                hidden_channels=16,
                use_null_source=True,
                pdca_context_spike_mode="weights",
            ),
        )

        outputs, aux = encoder(features, return_aux=True, detach_aux=True)

        self.assertEqual([out.shape for out in outputs], [feat.shape for feat in features])
        self.assertIn("pdca_context_spike", aux)
        self.assertIn("3", aux["pdca_context_spike"])
        assert_finite_stats(self, aux["pdca_context_spike"]["3"])

    def test_train_parser_defaults_and_warmup_helper(self):
        args = train.build_parser().parse_args([])
        self.assertEqual(args.pdca_context_spike_mode, "none")
        self.assertTrue(args.pdca_context_spike_signed)

        class DummyPDCA(nn.Module):
            def __init__(self):
                super().__init__()
                self.pdca_context_spike_mode = "weights"
                self.pdca_context_spike_runtime_mode = "weights"

        model = nn.Sequential(DummyPDCA())
        train.set_pdca_context_spike_mode(model, "none")

        pdca = model[0]
        self.assertEqual(pdca.pdca_context_spike_mode, "weights")
        self.assertEqual(pdca.pdca_context_spike_runtime_mode, "none")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_amp_forward_backward(self):
        names = phase_names(3)
        x = torch.randn(3, 1, 32, 6, 5, device="cuda", requires_grad=True)
        pdca = PhaseDeformableContextAttention(
            channels=32,
            phase_names=names,
            context_pairs=adjacent_pairs(names),
            num_heads=4,
            num_points=2,
            hidden_channels=16,
            pdca_context_spike_mode="both",
        ).cuda()

        with torch.cuda.amp.autocast(True):
            out, aux = pdca(x, return_aux=True, detach_aux=True)
        self.assertEqual(out.shape, x.shape)
        assert_finite_stats(self, aux["context_spike"])
        out.float().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()

import inspect
import sys
import types
import unittest
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
mmseg = types.ModuleType("mmseg")
mmseg.__path__ = [str(repo_root / "mmseg")]
sys.modules.setdefault("mmseg", mmseg)

from models.Encoders.phase_deformable_context_attention import PhaseDeformableContextAttention
from models.Encoders.FDPC_Encoder import FDPCEncoder


PHASE_NAMES = ("t1", "t2", "t3")
CONTEXT_PAIRS = (("t1", "t2"), ("t2", "t3"), ("t1", "t3"))


class PDCAShapeTest(unittest.TestCase):
    def test_pdca_shape_aux_and_source_weights(self):
        torch.manual_seed(1)
        n, b, c, h, w = 3, 2, 128, 32, 32
        heads, points = 4, 4
        x = torch.randn(n, b, c, h, w)
        pdca = PhaseDeformableContextAttention(
            channels=c,
            phase_names=PHASE_NAMES,
            context_pairs=CONTEXT_PAIRS,
            num_heads=heads,
            num_points=points,
        )

        out, aux = pdca(x, return_aux=True, detach_aux=True)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(set(aux), {"offsets", "attn_weights", "source_weights", "joint_weights"})
        self.assertEqual(
            pdca.source_names_by_target,
            {
                "t1": ("t2", "t3", "__null__"),
                "t2": ("t1", "t3", "__null__"),
                "t3": ("t1", "t2", "__null__"),
            },
        )

        for target_name, source_names in pdca.source_names_by_target.items():
            joint = aux["joint_weights"][target_name]
            source = aux["source_weights"][target_name]
            self.assertEqual(joint.shape, (b, heads, len(source_names), points, h, w))
            self.assertEqual(source.shape, (b, heads, len(source_names), h, w))
            self.assertTrue(torch.allclose(joint.sum(dim=(2, 3)), torch.ones(b, heads, h, w), atol=1e-5))
            self.assertTrue(torch.allclose(source, joint.sum(dim=3), atol=1e-6))
            self.assertFalse(joint.requires_grad)
            self.assertFalse(source.requires_grad)

            for q_idx, src_name in enumerate(source_names):
                if src_name == "__null__":
                    continue
                direction_key = "%s<-%s" % (target_name, src_name)
                self.assertEqual(aux["offsets"][direction_key].shape, (b, heads, points, 2, h, w))
                self.assertEqual(aux["attn_weights"][direction_key].shape, (b, heads, points, h, w))
                self.assertTrue(torch.allclose(aux["attn_weights"][direction_key], joint[:, :, q_idx], atol=1e-6))
                self.assertFalse(aux["offsets"][direction_key].requires_grad)
                self.assertFalse(aux["attn_weights"][direction_key].requires_grad)

    def test_pdca_no_pair_order_leakage_with_stateless_activation(self):
        torch.manual_seed(2)
        x = torch.randn(3, 1, 64, 16, 16)
        pdca_a = PhaseDeformableContextAttention(
            channels=64,
            phase_names=PHASE_NAMES,
            context_pairs=CONTEXT_PAIRS,
            num_heads=4,
            num_points=3,
        )
        pdca_b = PhaseDeformableContextAttention(
            channels=64,
            phase_names=PHASE_NAMES,
            context_pairs=tuple(reversed(CONTEXT_PAIRS)),
            num_heads=4,
            num_points=3,
        )
        pdca_b.load_state_dict(pdca_a.state_dict())
        pdca_a.eval()
        pdca_b.eval()

        out_a, _ = pdca_a(x, return_aux=False)
        out_b, _ = pdca_b(x, return_aux=False)

        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-5))

    def test_grid_sample_vectorized_design(self):
        source = inspect.getsource(PhaseDeformableContextAttention)
        self.assertIn("_deformable_sample_vectorized", source)
        self.assertIn("F.grid_sample", source)
        self.assertNotIn("range(self.num_heads)", source)
        self.assertNotIn("range(self.num_points)", source)

    def test_fdpc_encoder_relation_modes_shape_and_backward(self):
        features = [
            torch.randn(3, 2, 32, 64, 64, requires_grad=True),
            torch.randn(3, 2, 64, 32, 32, requires_grad=True),
            torch.randn(3, 2, 128, 16, 16, requires_grad=True),
            torch.randn(3, 2, 360, 8, 8, requires_grad=True),
        ]

        for mode in ("none", "prg", "pdca"):
            encoder = FDPCEncoder(
                in_channels=[32, 64, 128, 360],
                phase_names=PHASE_NAMES,
                context_pairs=CONTEXT_PAIRS,
                dendritic_scales=(),
                relation_scales=(2, 3),
                relation_mode=mode,
                pdca_cfg=dict(num_heads=4, num_points=2, use_null_source=True),
                return_aux_default=False,
            )
            inputs = [feat.detach().clone().requires_grad_(True) for feat in features]
            outputs, aux = encoder(inputs, return_aux=(mode == "pdca"), detach_aux=True)

            self.assertEqual([tuple(out.shape) for out in outputs], [tuple(feat.shape) for feat in inputs])
            if mode == "pdca":
                self.assertIn("pdca_offsets", aux)
                self.assertIn("pdca_attn_weights", aux)
                self.assertIn("pdca_source_weights", aux)
                self.assertIn("pdca_joint_weights", aux)
            else:
                self.assertEqual(aux, {})

            loss = sum(out.mean() for out in outputs)
            loss.backward()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_pdca_cuda_and_amp_smoke(self):
        torch.cuda.reset_peak_memory_stats()
        encoder = FDPCEncoder(
            in_channels=[32, 64, 128, 360],
            phase_names=PHASE_NAMES,
            context_pairs=CONTEXT_PAIRS,
            dendritic_scales=(),
            relation_scales=(2, 3),
            relation_mode="pdca",
            pdca_cfg=dict(num_heads=4, num_points=2, use_null_source=True),
        ).cuda()
        features = [
            torch.randn(3, 1, 32, 16, 16, device="cuda", requires_grad=True),
            torch.randn(3, 1, 64, 16, 16, device="cuda", requires_grad=True),
            torch.randn(3, 1, 128, 64, 64, device="cuda", requires_grad=True),
            torch.randn(3, 1, 360, 32, 32, device="cuda", requires_grad=True),
        ]

        outputs, _ = encoder(features, return_aux=False)
        sum(out.mean() for out in outputs).backward()
        memory = torch.cuda.max_memory_allocated()
        print("pdca_cuda_max_memory_allocated=%d" % memory)

        pdca = PhaseDeformableContextAttention(
            channels=128,
            phase_names=PHASE_NAMES,
            context_pairs=CONTEXT_PAIRS,
            num_heads=4,
            num_points=2,
        ).cuda()
        x = torch.randn(3, 1, 128, 32, 32, device="cuda")
        with torch.cuda.amp.autocast(True):
            out, aux = pdca(x, return_aux=False)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(aux, {})


if __name__ == "__main__":
    unittest.main()

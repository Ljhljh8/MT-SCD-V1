import unittest
import sys
import types
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
mmseg = types.ModuleType("mmseg")
mmseg.__path__ = [str(repo_root / "mmseg")]
sys.modules.setdefault("mmseg", mmseg)

from mmseg.Qtrick_architecture.clock_driven.neuron import MTSCDPRDNIIFNode


class FDPCDendSomaTest(unittest.TestCase):
    def test_mtscd_round_ste_masks_gradients_outside_capacity(self):
        x = torch.tensor([-1.0, 0.5, 9.0], requires_grad=True)

        y = MTSCDPRDNIIFNode._round_ste(x, capacity=8)
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([0.0, 1.0, 0.0])))

    def test_mtscd_prd_uses_configurable_gate_source_and_detached_aux(self):
        x = torch.zeros(2, 1, 1, 1, 1, dtype=torch.float64)
        x[0] = 2.0

        post_node = MTSCDPRDNIIFNode(
            gate_mode="rho_only",
            state_source="post_output",
            carry_scale_init=1.0,
            alpha_rho=0.0,
            alpha_gamma=0.0,
        )
        raw_node = MTSCDPRDNIIFNode(
            gate_mode="rho_only",
            state_source="raw_u",
            carry_scale_init=1.0,
            alpha_rho=0.0,
            alpha_gamma=0.0,
        )

        post = post_node(x)
        raw = raw_node(x)

        self.assertEqual(post.shape, x.shape)
        self.assertEqual(raw.shape, x.shape)
        self.assertAlmostEqual(post[1].item(), 0.0)
        self.assertAlmostEqual(raw[1].item(), 0.125)
        self.assertEqual(set(post_node.last_aux), {"risk", "gate", "carry_abs_mean"})
        self.assertEqual(post_node.last_aux["risk"].shape, (2, 1, 1, 1, 1))
        self.assertEqual(post_node.last_aux["gate"].shape, (2, 1, 1, 1, 1))
        self.assertEqual(post_node.last_aux["carry_abs_mean"].shape, (2,))
        self.assertEqual(post_node.last_aux["gate"].dtype, torch.float64)
        self.assertTrue(all(not value.requires_grad for value in post_node.last_aux.values()))

    def test_mtscd_prd_gate_modes_and_force_fp32_aux_dtype(self):
        x = torch.ones(2, 1, 1, 1, 1, dtype=torch.float64)

        self.assertEqual(
            MTSCDPRDNIIFNode(gate_mode="rho_only", alpha_rho=0.0, alpha_gamma=0.0)(x).shape,
            x.shape,
        )
        gamma_node = MTSCDPRDNIIFNode(gate_mode="gamma_only", alpha_rho=0.0, alpha_gamma=0.0)
        dual_node = MTSCDPRDNIIFNode(gate_mode="dual", alpha_rho=0.0, alpha_gamma=0.0)
        fp32_node = MTSCDPRDNIIFNode(force_fp32=True, alpha_rho=0.0, alpha_gamma=0.0)

        gamma_node(x)
        dual_node(x)
        fp32_node(x)

        self.assertAlmostEqual(gamma_node.last_aux["gate"][1].item(), 0.5)
        self.assertAlmostEqual(dual_node.last_aux["gate"][1].item(), 0.25)
        self.assertEqual(fp32_node.last_aux["gate"].dtype, torch.float32)

    def test_mtscd_prd_rejects_bad_gate_and_state_source(self):
        with self.assertRaises(ValueError):
            MTSCDPRDNIIFNode(gate_mode="bad")
        with self.assertRaises(ValueError):
            MTSCDPRDNIIFNode(state_source="bad")

    def test_fdpc_encoder_wires_only_dendritic_soma_selection(self):
        source = (repo_root / "models" / "Encoders" / "FDPC_Encoder.py").read_text(encoding="utf-8")
        adapter_block = source.split("class DendriticScaleAdapter", 1)[1].split("class PairwiseRelationGate", 1)[0]

        self.assertIn("MTSCDPRDNIIFNode", source)
        self.assertIn('dend_soma_type: str = "q_if"', source)
        self.assertIn("dend_soma_cfg: Optional[dict] = None", source)
        self.assertIn('if soma_type == "q_if"', source)
        self.assertIn('elif soma_type == "mtscd_prd"', source)
        self.assertIn('elif soma_type in ("identity", "none")', source)
        self.assertIn("dend_soma_type=dend_soma_type", source)
        self.assertNotIn("PairwiseRelationGate(", adapter_block)


if __name__ == "__main__":
    unittest.main()

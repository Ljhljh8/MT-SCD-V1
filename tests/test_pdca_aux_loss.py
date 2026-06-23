import unittest

import torch

from utils.pdca_aux_loss import (
    build_ratio_weight,
    downsample_change_ratio,
    pdca_relation_aux_loss,
)


class PDCAAuxLossTest(unittest.TestCase):
    def test_downsample_change_ratio_shape_and_range(self):
        mask_bn = torch.randint(0, 2, (2, 512, 512)).float()
        ratio = downsample_change_ratio(mask_bn, (16, 16))

        self.assertEqual(ratio.shape, (2, 1, 16, 16))
        self.assertGreaterEqual(float(ratio.min()), 0.0)
        self.assertLessEqual(float(ratio.max()), 1.0)

    def test_build_ratio_weight_only_downweights_open_ambiguous_interval(self):
        ratio = torch.tensor([[[[0.00, 0.05, 0.10, 0.20, 1.00]]]])
        weight = build_ratio_weight(
            ratio,
            tau_neg=0.05,
            tau_pos=0.20,
            ambiguous_weight=0.25,
        )

        expected = torch.tensor([[[[1.00, 1.00, 0.25, 1.00, 1.00]]]])
        self.assertTrue(torch.equal(weight, expected))

    def test_relation_aux_loss_backpropagates_and_detaches_all_stats(self):
        logits = {
            "t1<-t3": torch.zeros(2, 1, 4, 4, requires_grad=True),
            "t3<-t1": torch.ones(2, 1, 4, 4, requires_grad=True),
        }
        encoder_aux_list = [{"pdca_relation_logits": {"3": logits}}]
        mask_bn = torch.cat(
            [torch.zeros(1, 8, 8), torch.ones(1, 8, 8)],
            dim=0,
        )

        loss, stats = pdca_relation_aux_loss(encoder_aux_list, mask_bn)

        expected_keys = {
            "pdca_aux_loss",
            "pdca_aux_target_mean",
            "pdca_aux_logit_mean",
            "pdca_aux_positive_ratio",
            "pdca_aux_prob_mean",
            "pdca_aux_changed_loss",
            "pdca_aux_unchanged_loss",
        }
        self.assertEqual(set(stats), expected_keys)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(stats["pdca_aux_target_mean"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["pdca_aux_positive_ratio"]), 0.5, places=6)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))

        loss.backward()
        self.assertTrue(all(logit.grad is not None for logit in logits.values()))

    def test_relation_aux_loss_rejects_missing_logits(self):
        mask_bn = torch.zeros(1, 8, 8)
        invalid_aux_lists = (
            [],
            [{}],
            [{"pdca_relation_logits": {"3": {"t1<-t3": torch.zeros(1, 1, 4, 4)}}}],
        )
        for encoder_aux_list in invalid_aux_lists:
            with self.subTest(aux=encoder_aux_list), self.assertRaises(RuntimeError):
                pdca_relation_aux_loss(encoder_aux_list, mask_bn)


if __name__ == "__main__":
    unittest.main()

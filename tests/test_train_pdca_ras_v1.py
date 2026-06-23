import unittest
from types import SimpleNamespace

import train_WUSU_ddp_accum_v9 as train


class TrainPDCARASV1Test(unittest.TestCase):
    def test_pdca_aux_parser_defaults(self):
        args = train.build_parser().parse_args([])

        self.assertFalse(args.enable_pairrel_aux)
        self.assertFalse(args.pdca_aux)
        self.assertEqual(args.pdca_aux_weight, 0.05)
        self.assertEqual(args.pdca_aux_warmup_epochs, 5.0)
        self.assertEqual(args.pdca_aux_scale_key, "3")
        self.assertEqual(args.pdca_aux_tau_neg, 0.05)
        self.assertEqual(args.pdca_aux_tau_pos, 0.20)
        self.assertEqual(args.pdca_aux_ambiguous_weight, 0.25)

    def test_pdca_aux_and_pairrel_are_rejected_together(self):
        args = train.build_parser().parse_args(["--pdca_aux", "--enable-pairrel-aux"])

        with self.assertRaisesRegex(
            ValueError,
            "This experiment tests PDCA-RAS only. Please disable PairRelAux.",
        ):
            train.validate_args(args)

    def test_pretrain_incompatible_keys_only_allow_relation_head_missing(self):
        relation_only = SimpleNamespace(
            missing_keys=["encoder.3.pdca_blocks.3.relation_aux_head.0.weight"],
            unexpected_keys=[],
        )
        train.validate_pdca_pretrain_incompatible(relation_only, pdca_aux=True)

        core_missing = SimpleNamespace(
            missing_keys=["backbone.levels.0.weight"],
            unexpected_keys=[],
        )
        with self.assertRaises(RuntimeError):
            train.validate_pdca_pretrain_incompatible(core_missing, pdca_aux=True)

        unexpected = SimpleNamespace(
            missing_keys=[],
            unexpected_keys=["unexpected.weight"],
        )
        with self.assertRaises(RuntimeError):
            train.validate_pdca_pretrain_incompatible(unexpected, pdca_aux=True)

    def test_resume_requires_relation_head_state_when_pdca_aux_is_enabled(self):
        model_keys = {
            "backbone.weight",
            "encoder.3.pdca_blocks.3.relation_aux_head.0.weight",
        }
        with self.assertRaises(RuntimeError):
            train.validate_pdca_resume_state(
                {"backbone.weight": object()},
                model_keys,
                pdca_aux=True,
            )

        train.validate_pdca_resume_state(
            {
                "backbone.weight": object(),
                "encoder.3.pdca_blocks.3.relation_aux_head.0.weight": object(),
            },
            model_keys,
            pdca_aux=True,
        )


if __name__ == "__main__":
    unittest.main()

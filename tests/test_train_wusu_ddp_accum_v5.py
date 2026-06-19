import importlib
import inspect

import pytest


def test_v5_parser_keeps_core_flags_and_drops_legacy_flags():
    train_v5 = importlib.import_module("train_WUSU_ddp_accum_v5")

    args = train_v5.build_parser().parse_args(
        [
            "--batch-size",
            "2",
            "--accum-steps",
            "4",
            "--amp",
            "--sync-bn",
            "--opt",
            "adamp",
            "--reference-batch-size",
            "4",
            "--reference-accum-steps",
            "1",
            "--grad-clip-norm",
            "1.0",
            "--amp-debug-nonfinite",
        ]
    )

    assert args.batch_size == 2
    assert args.accum_steps == 4
    assert args.amp is True
    assert args.sync_bn is True
    assert args.opt == "adamp"
    assert args.reference_batch_size == 4
    assert args.reference_accum_steps == 1
    assert args.grad_clip_norm == pytest.approx(1.0)
    assert args.amp_debug_nonfinite is True
    assert not hasattr(args, "save_mask")
    assert not hasattr(args, "tta")
    assert not hasattr(args, "use_pseudo_label")
    assert not hasattr(args, "change_output_api")


def test_v5_binary_dice_uses_probability_input_without_internal_sigmoid():
    train_v5 = importlib.import_module("train_WUSU_ddp_accum_v5")

    criteria_source = inspect.getsource(train_v5.build_binary_criteria)
    loss_source = inspect.getsource(train_v5.compute_binary_change_loss)

    assert 'DiceLoss(activation="none")' in criteria_source
    assert "torch.sigmoid(logits)" in loss_source

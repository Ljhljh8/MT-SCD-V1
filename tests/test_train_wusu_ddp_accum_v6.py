import importlib
import inspect

import pytest


def test_v6_parser_defaults_are_precision_first_and_no_legacy_flags():
    train_v6 = importlib.import_module("train_WUSU_ddp_accum_v6")

    args = train_v6.build_parser().parse_args([])

    assert args.lr == pytest.approx(0.0005)
    assert args.reference_batch_size == 2
    assert args.reference_accum_steps == 1
    assert args.amp is False
    assert args.sync_bn is False
    assert args.grad_clip_norm == pytest.approx(0.0)
    assert not hasattr(args, "change_output_api")
    assert not hasattr(args, "save_mask")
    assert not hasattr(args, "tta")
    assert not hasattr(args, "use_pseudo_label")


def test_v6_binary_loss_is_single_sigmoid_logits_api():
    train_v6 = importlib.import_module("train_WUSU_ddp_accum_v6")

    criteria_source = inspect.getsource(train_v6.build_loss_functions)
    binary_source = inspect.getsource(train_v6.compute_binary_change_loss)

    assert 'DiceLoss(activation="none")' in criteria_source
    assert binary_source.count("torch.sigmoid") == 1
    assert "BCEWithLogitsLoss" in criteria_source


def test_v6_reference_updates_default_to_single_gpu_baseline():
    train_v6 = importlib.import_module("train_WUSU_ddp_accum_v6")
    args = train_v6.build_parser().parse_args(["--epochs", "100"])

    total_updates, warmup_updates = train_v6.resolve_reference_update_counts(
        args,
        dataset_len=976,
        actual_total_updates=7000,
    )

    assert total_updates == 48800
    assert warmup_updates == pytest.approx(9760.0)


def test_v6_amp_step_result_does_not_advance_when_nonfinite():
    train_v6 = importlib.import_module("train_WUSU_ddp_accum_v6")

    assert train_v6.should_advance_after_amp_step(grad_finite=False, previous_scale=1024.0, current_scale=512.0) is False
    assert train_v6.should_advance_after_amp_step(grad_finite=True, previous_scale=1024.0, current_scale=1024.0) is True

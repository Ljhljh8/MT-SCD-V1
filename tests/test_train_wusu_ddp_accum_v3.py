import argparse
import importlib

import pytest


def test_parser_accepts_required_ddp_accum_amp_arguments():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    args = train_v3.build_parser().parse_args(
        [
            "--batch-size",
            "2",
            "--accum-steps",
            "4",
            "--amp",
            "--sync-bn",
            "--find-unused-parameters",
            "--workers",
            "3",
            "--change-output-api",
            "logits",
            "--reference-batch-size",
            "2",
            "--reference-accum-steps",
            "1",
            "--reference-total-updates",
            "5000",
            "--reference-warmup-updates",
            "1000",
            "--grad-clip-norm",
            "1.5",
        ]
    )

    assert args.batch_size == 2
    assert args.accum_steps == 4
    assert args.amp is True
    assert args.sync_bn is True
    assert args.find_unused_parameters is True
    assert args.workers == 3
    assert args.change_output_api == "logits"
    assert args.reference_batch_size == 2
    assert args.reference_accum_steps == 1
    assert args.reference_total_updates == 5000
    assert args.reference_warmup_updates == 1000
    assert args.grad_clip_norm == pytest.approx(1.5)


def test_should_update_on_accum_boundary_and_last_microbatch():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    assert train_v3.should_update_optimizer(0, 5, 2) is False
    assert train_v3.should_update_optimizer(1, 5, 2) is True
    assert train_v3.should_update_optimizer(4, 5, 2) is True


def test_strip_module_prefix_handles_ddp_and_plain_state_dicts():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    ddp_state = {"module.backbone.weight": 1, "module.decoder.bias": 2}
    plain_state = {"backbone.weight": 1}

    assert train_v3.strip_module_prefix(ddp_state) == {
        "backbone.weight": 1,
        "decoder.bias": 2,
    }
    assert train_v3.strip_module_prefix(plain_state) == plain_state


def test_compute_poly_warmup_lr_advances_by_update_step_not_microbatch():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    args = argparse.Namespace(lr=0.0005, warmup=True)

    assert train_v3.compute_poly_warmup_lr(args, update_step=1, total_updates=10, warmup_updates=2) == pytest.approx(0.00025)
    assert train_v3.compute_poly_warmup_lr(args, update_step=2, total_updates=10, warmup_updates=2) == pytest.approx(0.0005 * (1.0 - 2.0 / 10.0) ** 1.5)
    assert train_v3.compute_poly_warmup_lr(args, update_step=10, total_updates=10, warmup_updates=2) == pytest.approx(0.0)


def test_lr_lambda_matches_legacy_poly_warmup_by_optimizer_update_index():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    lr_lambda = train_v3.build_poly_warmup_lr_lambda(
        warmup_enabled=True,
        total_updates=10,
        warmup_updates=2,
    )

    assert lr_lambda(0) == pytest.approx(0.5)
    assert lr_lambda(1) == pytest.approx((1.0 - 2.0 / 10.0) ** 1.5)
    assert lr_lambda(9) == pytest.approx(0.0)


def test_reference_batch_size_resolves_single_gpu_reference_update_counts():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    args = argparse.Namespace(
        epochs=100,
        warmup=True,
        reference_batch_size=2,
        reference_accum_steps=1,
        reference_total_updates=None,
        reference_warmup_updates=None,
    )

    total_updates, warmup_updates = train_v3.resolve_reference_update_counts(
        args,
        dataset_len=103,
        actual_total_updates=1000,
    )

    assert total_updates == 5100
    assert warmup_updates == pytest.approx(1020.0)


def test_explicit_reference_updates_override_batch_size_resolution():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    args = argparse.Namespace(
        epochs=100,
        warmup=True,
        reference_batch_size=2,
        reference_accum_steps=1,
        reference_total_updates=777,
        reference_warmup_updates=55,
    )

    total_updates, warmup_updates = train_v3.resolve_reference_update_counts(
        args,
        dataset_len=103,
        actual_total_updates=1000,
    )

    assert total_updates == 777
    assert warmup_updates == 55


def test_change_probability_explicitly_converts_logits_only_at_metric_time():
    torch = pytest.importorskip("torch")
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    logits = torch.tensor([-20.0, 0.0, 20.0])
    probs = train_v3.change_probability(logits, output_api="logits")

    assert probs[0].item() == pytest.approx(0.0, abs=1e-6)
    assert probs[1].item() == pytest.approx(0.5)
    assert probs[2].item() == pytest.approx(1.0, abs=1e-6)

    already_probs = torch.tensor([0.2, 0.8])
    assert torch.equal(
        train_v3.change_probability(already_probs, output_api="probability"),
        already_probs,
    )


def test_dataset_random_flip_policy_disables_validation_augmentation():
    train_v3 = importlib.import_module("train_WUSU_ddp_accum_v3")

    assert train_v3.dataset_random_flip("train") is True
    assert train_v3.dataset_random_flip("val") is False

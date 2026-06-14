import argparse
import importlib

import pytest


def test_parser_accepts_timm_optimizer_scheduler_arguments_without_duplicate_clip_knobs():
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")

    args = train_v4.build_parser().parse_args(
        [
            "--opt",
            "adamp",
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.01",
            "--opt-betas",
            "0.9",
            "0.999",
            "--sched",
            "poly",
            "--sched-on-updates",
            "--reference-total-updates",
            "5100",
            "--reference-warmup-updates",
            "1020",
            "--clip-grad",
            "1.0",
        ]
    )

    assert args.opt == "adamp"
    assert args.lr == pytest.approx(0.0005)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.opt_betas == pytest.approx((0.9, 0.999))
    assert args.sched == "poly"
    assert args.sched_on_updates is True
    assert args.reference_total_updates == 5100
    assert args.reference_warmup_updates == 1020
    assert args.grad_clip_norm == pytest.approx(1.0)
    assert not hasattr(args, "clip_grad")


def test_build_timm_optimizer_uses_unwrapped_model_and_single_factory(monkeypatch):
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")
    calls = {}

    def fake_optimizer_kwargs(cfg):
        return {
            "opt": cfg.opt,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "momentum": cfg.momentum,
        }

    def fake_create_optimizer_v2(model, **kwargs):
        calls["model"] = model
        calls["kwargs"] = kwargs
        return "optimizer"

    class Wrapped:
        module = "inner-model"

    args = argparse.Namespace(
        opt="adamp",
        lr=0.0005,
        weight_decay=0.01,
        momentum=0.9,
        filter_bias_and_bn=False,
    )
    monkeypatch.setattr(train_v4, "optimizer_kwargs", fake_optimizer_kwargs)
    monkeypatch.setattr(train_v4, "create_optimizer_v2", fake_create_optimizer_v2)

    optimizer = train_v4.build_timm_optimizer(Wrapped(), args)

    assert optimizer == "optimizer"
    assert calls["model"] == "inner-model"
    assert calls["kwargs"] == {
        "opt": "adamp",
        "lr": 0.0005,
        "weight_decay": 0.01,
        "momentum": 0.9,
        "filter_bias_and_bn": False,
    }


def test_timm_scheduler_update_config_uses_reference_updates_as_scheduler_time():
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")
    args = train_v4.build_parser().parse_args(
        [
            "--sched",
            "poly",
            "--epochs",
            "100",
            "--decay-epochs",
            "30",
            "--decay-milestones",
            "30",
            "60",
            "--sched-on-updates",
        ]
    )

    scheduler_args, updates_per_epoch = train_v4.build_timm_scheduler_config(
        args,
        total_updates=5100,
        warmup_updates=1020,
    )

    assert updates_per_epoch == 1
    assert scheduler_args.sched_on_updates is True
    assert scheduler_args.epochs == 5100
    assert scheduler_args.warmup_epochs == pytest.approx(1020.0)
    assert scheduler_args.decay_epochs == pytest.approx(30 * 51)
    assert scheduler_args.decay_milestones == [30 * 51, 60 * 51]


def test_build_timm_scheduler_passes_update_config_and_normalizes_tuple(monkeypatch):
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")
    calls = {}
    fake_scheduler = object()

    def fake_create_scheduler(args, optimizer, updates_per_epoch=0):
        calls["args"] = args
        calls["optimizer"] = optimizer
        calls["updates_per_epoch"] = updates_per_epoch
        return fake_scheduler, 5100

    args = train_v4.build_parser().parse_args(["--sched", "poly", "--sched-on-updates"])
    monkeypatch.setattr(train_v4, "create_scheduler", fake_create_scheduler)

    scheduler, num_epochs, scheduler_args, updates_per_epoch = train_v4.build_timm_scheduler(
        args,
        optimizer="optimizer",
        total_updates=5100,
        warmup_updates=1020,
    )

    assert scheduler is fake_scheduler
    assert num_epochs == 5100
    assert scheduler_args.epochs == 5100
    assert updates_per_epoch == 1
    assert calls["optimizer"] == "optimizer"
    assert calls["updates_per_epoch"] == 1


def test_scheduler_step_helpers_separate_update_and_epoch_schedulers():
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")
    calls = []

    class UpdateScheduler:
        def step_update(self, num_updates):
            calls.append(("update", num_updates))

        def step(self, epoch, metric=None):
            calls.append(("epoch", epoch, metric))

    class EpochScheduler:
        def step(self, epoch, metric=None):
            calls.append(("epoch", epoch, metric))

    train_v4.step_timm_scheduler_update(UpdateScheduler(), 17)
    train_v4.step_timm_scheduler_epoch(EpochScheduler(), 3, metric=0.42)

    assert calls == [("update", 17), ("epoch", 3, 0.42)]


def test_validate_args_rejects_plateau_scheduler_on_update_steps():
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")
    args = train_v4.build_parser().parse_args(["--sched", "plateau", "--sched-on-updates"])

    with pytest.raises(ValueError, match="plateau"):
        train_v4.validate_args(args)


def test_validate_args_fills_scheduler_specific_decay_rate_defaults():
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")

    poly_args = train_v4.build_parser().parse_args(["--sched", "poly"])
    train_v4.validate_args(poly_args)
    assert poly_args.decay_rate == pytest.approx(1.5)

    step_args = train_v4.build_parser().parse_args(["--sched", "step"])
    train_v4.validate_args(step_args)
    assert step_args.decay_rate == pytest.approx(0.1)


def test_no_warmup_zeroes_timm_scheduler_warmup_for_epoch_scheduler():
    train_v4 = importlib.import_module("train_WUSU_ddp_accum_v4")
    args = train_v4.build_parser().parse_args(["--sched-on-epochs", "--no-warmup"])

    scheduler_args, updates_per_epoch = train_v4.build_timm_scheduler_config(
        args,
        total_updates=100,
        warmup_updates=20,
    )

    assert updates_per_epoch == 0
    assert scheduler_args.warmup_epochs == pytest.approx(0.0)

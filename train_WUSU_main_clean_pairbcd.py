#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield

try:
    from timm.optim import create_optimizer_v2, optimizer_kwargs
    from timm.scheduler import create_scheduler
except ModuleNotFoundError:
    create_optimizer_v2 = None
    optimizer_kwargs = None
    create_scheduler = None


PAIR_NAMES = (("t1", "t2"), ("t2", "t3"), ("t1", "t3"))
PAIR_KEYS = ("t1_to_t2", "t2_to_t3", "t1_to_t3")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("invalid boolean value: %r" % value)


class FloatTupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


def build_parser():
    parser = argparse.ArgumentParser("Clean WUSU GSTMSCD PairBCD DDP AMP Training")

    parser.add_argument("--data_name", "--data-name", dest="data_name", default="WUSU")
    parser.add_argument("--Net_name", "--net-name", dest="Net_name", default="GSTMSCD_clean_pairbcd")
    parser.add_argument("--backbone", default="sdtv2")
    parser.add_argument("--data_root", "--data-root", dest="data_root", default=None)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--pretrain_from", "--pretrain-from", dest="pretrain_from", default=None)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")
    parser.add_argument("--val-mode", choices=["t1_t3", "all_pairs"], default="t1_t3")

    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="checkpoints_clean_pairbcd")
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", default=None)

    parser.add_argument("--relation-mode", choices=["pdca"], default="pdca")
    parser.add_argument("--use-pdca-guided-pair-decoder", action="store_true")
    parser.add_argument("--pdca-dend-prior-mode", default="offset_residual", choices=["none", "source","source_gain", "offset_sim", "offset_dual", "offset_residual", "offset_improve", "offset_gate"])
    parser.add_argument("--pdca-dend-prior-alpha", type=float, default=1e-3)
    parser.add_argument("--pdca-dend-prior-detach", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-descriptor", default="mean_std", choices=["mean", "mean_std", "raw", "delta", "gain"])
    parser.add_argument("--pdca-dend-prior-normalize", default="zscore", choices=["none", "zscore"])

    ## pdca v21 add parm
    parser.add_argument("--pdca-dend-prior-source-weight", type=float, default=1.0)
    parser.add_argument("--pdca-dend-prior-point-weight", type=float, default=0.25)
    ## pdca v21 add parm
    parser.add_argument("--pdca-dend-prior-sim-weight", type=float, default=1.0)
    parser.add_argument("--pdca-dend-prior-diff-weight", type=float, default=0.25)
    parser.add_argument("--pdca-dend-prior-use-conf-gate", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-conf-beta", type=float, default=4.0)
    parser.add_argument("--pdca-dend-prior-conf-tau", type=float, default=0.10)

    ## pdca v21 add parm
    parser.add_argument("--pdca-dend-prior-use-offset-gate", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-center-point", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-clip", type=float, default=2.0)
    ## pdca v21 add parm
    parser.add_argument("--pdca-dend-prior-affect-null", type=str2bool, default=False)
    parser.add_argument("--pdca-dend-prior-stats", type=str2bool, default=False)

    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-bcd-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-similarity-weight", type=float, default=1.0)
    parser.add_argument("--pair-bcd-lambda-adj", type=float, default=1.0)
    parser.add_argument("--pair-bcd-lambda-13", type=float, default=1.0)
    parser.add_argument("--pair-bcd-dice-weight", type=float, default=1.0)

    parser.add_argument("--dend-spatial-conv-type",
                        choices=["fadc", "structure_routed_v1", "structure_routed_v2",
                                 "structure_routed_v3"],
                        default="fadc", )
    parser.add_argument("--routeconv-ablation-mode",
                        choices=["full", "uniform_route", "global_route", "no_axis_descriptor",
                                 "isotropic_direction_pool", ], default="full", )
    parser.add_argument("--routeconv-v2-mode",
                        choices=["v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v2_6"],
                        default="v2_6", )
    parser.add_argument("--routeconv-v3-mode",
                        choices=["v3_1", "v3_2", "v3_3", "v3_4", "v3_5", "v3_6"],
                        default="v3_6", )
    parser.add_argument("--dend-residual-init", type=float, default=0.0)

    parser.add_argument("--opt", default="adamp")
    parser.add_argument("--opt-eps", default=None, type=float)
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", action=FloatTupleAction)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--filter-bias-and-bn", dest="filter_bias_and_bn", action="store_true")
    parser.add_argument("--no-filter-bias-and-bn", dest="filter_bias_and_bn", action="store_false")
    parser.set_defaults(filter_bias_and_bn=False)

    parser.add_argument("--sched", choices=["poly"], default="poly")
    parser.add_argument("--sched-on-updates", dest="sched_on_updates", action="store_true")
    parser.add_argument("--sched-on-epochs", dest="sched_on_updates", action="store_false")
    parser.set_defaults(sched_on_updates=True)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)     # 0.0
    parser.add_argument("--min-lr", type=float, default=1e-6)        # 0.0
    parser.add_argument("--decay-epochs", type=float, default=30)
    parser.add_argument("--decay-milestones", type=int, nargs="+", default=[30, 60])
    parser.add_argument("--decay-rate", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=5)
    parser.add_argument("--warmup-prefix", action="store_true", default=False)
    parser.add_argument("--cooldown-epochs", type=int, default=5)
    parser.add_argument("--patience-epochs", type=int, default=10)
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--reference_batch_size", "--reference-batch-size", dest="reference_batch_size", type=int, default=None)
    parser.add_argument("--reference_accum_steps", "--reference-accum-steps", dest="reference_accum_steps", type=int, default=1)
    parser.add_argument("--reference_total_updates", "--reference-total-updates", dest="reference_total_updates", type=int, default=None)
    parser.add_argument("--reference_warmup_updates", "--reference-warmup-updates", dest="reference_warmup_updates", type=float, default=None)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum_steps", "--accum-steps", dest="accum_steps", type=int, default=1)
    parser.add_argument("--sync_bn", "--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument("--grad_clip_norm", "--grad-clip-norm", "--clip-grad", dest="grad_clip_norm", type=float, default=0, metavar="NORM")
    parser.add_argument("--find_unused_parameters", "--find-unused-parameters", dest="find_unused_parameters", action="store_true")
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.set_defaults(find_unused_parameters=True)

    parser.add_argument("--dist_url", "--dist-url", dest="dist_url", default="env://")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", "--world-size", dest="world_size", type=int, default=1)
    return parser


def validate_args(args):
    if not args.use_pdca_guided_pair_decoder:
        raise ValueError("clean pair_bcd training requires --use-pdca-guided-pair-decoder")
    if (
        args.routeconv_ablation_mode != "full"
        and args.dend_spatial_conv_type != "structure_routed_v1"
    ):
        raise ValueError("--routeconv-ablation-mode requires --dend-spatial-conv-type structure_routed_v1")
    if (
        args.routeconv_v2_mode != "v2_6"
        and args.dend_spatial_conv_type != "structure_routed_v2"
    ):
        raise ValueError("non-default --routeconv-v2-mode requires --dend-spatial-conv-type structure_routed_v2")
    if (
        args.routeconv_v3_mode != "v3_6"
        and args.dend_spatial_conv_type != "structure_routed_v3"
    ):
        raise ValueError(
            "non-default --routeconv-v3-mode requires "
            "--dend-spatial-conv-type structure_routed_v3"
        )
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.accum_steps < 1 or args.reference_accum_steps < 1:
        raise ValueError("accumulation steps must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr < 0 or args.weight_decay < 0 or args.grad_clip_norm < 0:
        raise ValueError("lr, weight decay and grad clip must be non-negative")
    if args.opt_betas is not None and len(args.opt_betas) not in (2, 3):
        raise ValueError("--opt-betas must contain two values for Adam/AdamP or three values for Adan")
    for name in (
        "seg_loss_weight",
        "pair_bcd_loss_weight",
        "pair_similarity_weight",
        "pair_bcd_lambda_adj",
        "pair_bcd_lambda_13",
        "pair_bcd_dice_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError("%s must be non-negative" % name)
    if args.reference_total_updates is not None and args.reference_total_updates < 1:
        raise ValueError("--reference-total-updates must be >= 1")
    if args.reference_warmup_updates is not None and args.reference_warmup_updates < 0:
        raise ValueError("--reference-warmup-updates must be >= 0")
    if args.decay_rate is None:
        args.decay_rate = 1.5
    if args.decay_rate <= 0 or args.decay_epochs <= 0:
        raise ValueError("decay settings must be positive")


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process(args):
    return int(getattr(args, "rank", 0)) == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    args.distributed = args.world_size > 1
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA/NCCL")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        dist.barrier()


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def seed_everything(seed, rank=0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reset_snn_state(model, functional_module):
    functional_module.reset_net(unwrap_model(model))


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def strip_module_prefix(state_dict):
    if not state_dict or not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_model_weights(model, path, strict=False):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    return unwrap_model(model).load_state_dict(state_dict, strict=strict)


def build_dataloaders(args):
    import datasets.MultiSiamese_RS_ST_TL as RS

    if args.data_root is not None:
        RS.root = args.data_root

    trainset = RS.Data(mode="train", random_flip=True)
    valset = RS.Data(mode="val", random_flip=False) if is_main_process(args) else None
    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=True,
    )
    valloader = None
    if is_main_process(args):
        valloader = DataLoader(
            valset,
            batch_size=args.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers,
            drop_last=False,
        )
    # valloader = DataLoader(
    #     valset,
    #     batch_size=args.val_batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=args.workers,
    #     drop_last=False,
    # )
    return RS, trainset, trainloader, train_sampler, valloader


def internal_pretrain_path():
    return "./GSTM-SCD_Pretraining-weights/Meta-Spikeformer-15M.pth"


def build_model(args, RS, device):
    from models.GSTMSCD_MTSCD_Snn_ForDecoder_clean import GSTMSCD_WUSU as Net

    use_internal_pretrain = bool(args.pretrained)
    if use_internal_pretrain and not os.path.exists(internal_pretrain_path()):
        if not args.pretrain_from:
            raise FileNotFoundError(
                "Internal pretrained checkpoint is missing: %s. "
                "Provide it, pass --pretrained false, or pass --pretrain-from."
                % internal_pretrain_path()
            )
        use_internal_pretrain = False
        if is_main_process(args):
            print(
                "Internal pretrained checkpoint missing; using explicit --pretrain-from warm start.",
                flush=True,
            )

    model = Net(
        args.backbone,
        use_internal_pretrain,
        len(RS.ST_CLASSES),
        relation_mode=args.relation_mode,
        use_pdca_guided_pair_decoder=args.use_pdca_guided_pair_decoder,
        detach_pdca_guidance=True,
        use_pdca_guidance=True,
        pdca_dend_prior_mode=args.pdca_dend_prior_mode,
        pdca_dend_prior_alpha=args.pdca_dend_prior_alpha,
        pdca_dend_prior_detach=args.pdca_dend_prior_detach,
        pdca_dend_prior_descriptor=args.pdca_dend_prior_descriptor,
        pdca_dend_prior_normalize=args.pdca_dend_prior_normalize,
        pdca_dend_prior_sim_weight=args.pdca_dend_prior_sim_weight,
        pdca_dend_prior_diff_weight=args.pdca_dend_prior_diff_weight,
        pdca_dend_prior_use_conf_gate=args.pdca_dend_prior_use_conf_gate,
        pdca_dend_prior_conf_beta=args.pdca_dend_prior_conf_beta,
        pdca_dend_prior_conf_tau=args.pdca_dend_prior_conf_tau,
        pdca_dend_prior_affect_null=args.pdca_dend_prior_affect_null,
        pdca_dend_prior_stats=args.pdca_dend_prior_stats,

        pdca_dend_prior_source_weight=args.pdca_dend_prior_source_weight,
        pdca_dend_prior_point_weight=args.pdca_dend_prior_point_weight,
        pdca_dend_prior_use_offset_gate=args.pdca_dend_prior_use_offset_gate,
        pdca_dend_prior_center_point=args.pdca_dend_prior_center_point,
        pdca_dend_prior_clip=args.pdca_dend_prior_clip,

        dend_spatial_conv_type=args.dend_spatial_conv_type,
        routeconv_ablation_mode=args.routeconv_ablation_mode,
        routeconv_v2_mode=args.routeconv_v2_mode,
        routeconv_v3_mode=args.routeconv_v3_mode,
        dend_residual_init=args.dend_residual_init,

    )
    if args.pretrain_from:
        incompatible = load_model_weights(model, args.pretrain_from, strict=False)
        if is_main_process(args):
            print(
                "Loaded non-strict pretrain weights from %s: missing keys=%r, unexpected keys=%r"
                % (
                    args.pretrain_from,
                    list(incompatible.missing_keys),
                    list(incompatible.unexpected_keys),
                ),
                flush=True,
            )

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    if args.sync_bn and args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model,
        #     device_ids=[args.gpu],
        #     find_unused_parameters=True
        # )
    return model


def require_timm_optimizer():
    if create_optimizer_v2 is None or optimizer_kwargs is None:
        raise ImportError("timm.optim is required for clean optimizer construction")


def require_timm_scheduler():
    if create_scheduler is None:
        raise ImportError("timm.scheduler is required for v9-compatible clean scheduler construction")


def build_timm_optimizer(model, args):
    require_timm_optimizer()
    kwargs = optimizer_kwargs(cfg=args)
    kwargs["filter_bias_and_bn"] = bool(getattr(args, "filter_bias_and_bn", False))
    return create_optimizer_v2(unwrap_model(model), **kwargs)


def reference_updates_per_epoch(total_updates, epochs):
    return max(1, int(math.ceil(float(total_updates) / float(max(1, int(epochs))))))


def resolve_reference_update_counts(args, dataset_len, actual_total_updates):
    if args.reference_total_updates is not None:
        total_updates = int(args.reference_total_updates)
    elif args.reference_batch_size is not None:
        reference_micro_batches = int(dataset_len) // int(args.reference_batch_size)
        reference_updates = max(1, math.ceil(reference_micro_batches / int(args.reference_accum_steps)))
        total_updates = reference_updates * int(args.epochs)
    else:
        total_updates = int(actual_total_updates)
    total_updates = max(1, total_updates)
    if args.reference_warmup_updates is not None:
        warmup_updates = float(args.reference_warmup_updates)
    else:
        warmup_updates = total_updates // int(args.epochs) * int(args.warmup_epochs) if args.warmup else 0.0
    return max(1, int(total_updates)), warmup_updates


def _with_timm_scheduler_defaults(args):
    scheduler_args = argparse.Namespace(**vars(args))
    defaults = {
        "lr_noise": None,
        "lr_noise_pct": 0.67,
        "lr_noise_std": 1.0,
        "lr_cycle_mul": 1.0,
        "lr_cycle_decay": 0.1,
        "lr_cycle_limit": 1,
        "lr_k_decay": 1.0,
        "eval_metric": "score",
    }
    for name, value in defaults.items():
        if not hasattr(scheduler_args, name):
            setattr(scheduler_args, name, value)
    return scheduler_args


def build_timm_scheduler_config(args, total_updates, warmup_updates):
    scheduler_args = _with_timm_scheduler_defaults(args)
    if not getattr(args, "warmup", True):
        warmup_updates = 0.0
        scheduler_args.warmup_epochs = 0.0
    if getattr(args, "sched_on_updates", True):
        updates_per_reference_epoch = reference_updates_per_epoch(total_updates, args.epochs)
        scheduler_args.epochs = int(total_updates)
        scheduler_args.warmup_epochs = float(warmup_updates or 0.0)
        scheduler_args.decay_epochs = float(args.decay_epochs) * updates_per_reference_epoch
        scheduler_args.decay_milestones = [int(m * updates_per_reference_epoch) for m in args.decay_milestones]
        return scheduler_args, 1
    return scheduler_args, 0


def normalize_timm_scheduler_result(result):
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("timm create_scheduler returned an unexpected tuple")
        return result
    return result, None


def build_timm_scheduler(args, optimizer, total_updates, warmup_updates):
    require_timm_scheduler()
    scheduler_args, updates_per_epoch = build_timm_scheduler_config(args, total_updates, warmup_updates)
    try:
        result = create_scheduler(scheduler_args, optimizer, updates_per_epoch=updates_per_epoch)
    except TypeError as exc:
        if scheduler_args.sched_on_updates:
            raise RuntimeError("This timm version does not support update-based scheduling") from exc
        result = create_scheduler(scheduler_args, optimizer)
    scheduler, num_epochs = normalize_timm_scheduler_result(result)
    return scheduler, num_epochs, scheduler_args, updates_per_epoch


def scheduler_state_dict(scheduler):
    if scheduler is None or not hasattr(scheduler, "state_dict"):
        return None
    return scheduler.state_dict()


def load_scheduler_state(scheduler, state_dict):
    if scheduler is None or state_dict is None or not hasattr(scheduler, "load_state_dict"):
        return
    scheduler.load_state_dict(state_dict)


def step_timm_scheduler_update(scheduler, update_step):
    if scheduler is None:
        return
    if hasattr(scheduler, "step_update"):
        scheduler.step_update(update_step)
    else:
        scheduler.step()


def step_timm_scheduler_epoch(scheduler, epoch, metric=None):
    if scheduler is None:
        return
    if metric is None:
        scheduler.step(epoch)
        return
    try:
        scheduler.step(epoch, metric)
    except TypeError:
        scheduler.step(epoch)


def sync_scheduler_to_resume_position(scheduler, global_update_step, start_epoch, step_on_updates):
    if scheduler is None:
        return
    if step_on_updates:
        step_timm_scheduler_update(scheduler, global_update_step)
    else:
        step_timm_scheduler_epoch(scheduler, start_epoch)


def step_scheduler_epoch(ctx, epoch, metric):
    if ctx.scheduler_step_on_updates:
        return
    scheduler_metric = metric if ctx.args.sched == "plateau" else None
    step_timm_scheduler_epoch(ctx.scheduler, epoch + 1, scheduler_metric)


def build_loss_functions(args):
    from torch.nn import CrossEntropyLoss
    from utils.loss import ChangeSimilarity, PairwiseBinaryChangeLoss

    return {
        "seg": CrossEntropyLoss(ignore_index=-1),
        "pair_bcd": PairwiseBinaryChangeLoss(
            lambda_adj=args.pair_bcd_lambda_adj,
            lambda_13=args.pair_bcd_lambda_13,
            dice_weight=args.pair_bcd_dice_weight,
        ),
        "similarity": ChangeSimilarity(),
    }


def should_update_optimizer(step, num_steps, accum_steps):
    return ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)


def compute_losses(out1, out2, out3, change_logits_dict, masks, criteria, args):
    from utils.loss import make_pairwise_change_targets

    mask1, mask2, mask3 = masks
    sem_targets = {
        "t1": mask1 - 1,
        "t2": mask2 - 1,
        "t3": mask3 - 1,
    }
    pair_targets = make_pairwise_change_targets(sem_targets, ignore_index=99)

    loss_seg = (
        criteria["seg"](out1.float(), sem_targets["t1"])
        + criteria["seg"](out2.float(), sem_targets["t2"])
        + criteria["seg"](out3.float(), sem_targets["t3"])
    ) / 3.0
    loss_pair_bcd, _ = criteria["pair_bcd"](change_logits_dict, pair_targets)
    loss_similarity = (
        criteria["similarity"](out1.float(), out2.float(), pair_targets["t1_to_t2"]["target"])
        + criteria["similarity"](out2.float(), out3.float(), pair_targets["t2_to_t3"]["target"])
        + criteria["similarity"](out1.float(), out3.float(), pair_targets["t1_to_t3"]["target"])
    ) / 3.0
    loss = (
        args.seg_loss_weight * loss_seg
        + args.pair_bcd_loss_weight * loss_pair_bcd
        + args.pair_similarity_weight * loss_similarity
    )
    return loss, {
        "loss": float(loss.detach()),
        "seg": float(loss_seg.detach()),
        "pair_bcd": float(loss_pair_bcd.detach()),
        "similarity": float(loss_similarity.detach()),
    }


def train_one_epoch(ctx, epoch):
    args = ctx.args
    model = ctx.model
    if ctx.train_sampler is not None:
        ctx.train_sampler.set_epoch(epoch)
    model.train()
    ctx.optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "seg": 0.0, "pair_bcd": 0.0, "similarity": 0.0}
    total_steps = len(ctx.trainloader)

    for step, batch in enumerate(ctx.trainloader):
        img1, img2, img3, mask1, mask2, mask3, _mask_bn, _sample_id = batch
        img1 = img1.float().to(ctx.device, non_blocking=True)
        img2 = img2.float().to(ctx.device, non_blocking=True)
        img3 = img3.float().to(ctx.device, non_blocking=True)
        mask1 = mask1.long().to(ctx.device, non_blocking=True)
        mask2 = mask2.long().to(ctx.device, non_blocking=True)
        mask3 = mask3.long().to(ctx.device, non_blocking=True)
        x = torch.stack([img1, img2, img3], dim=0)
        update_now = should_update_optimizer(step, total_steps, args.accum_steps)
        sync_context = model.no_sync() if args.distributed and not update_now else nullcontext()

        with sync_context:
            with torch.cuda.amp.autocast(enabled=args.amp):
                out1, out2, out3, _change13, change_logits_dict = model(
                    x,
                    return_change_logits_dict=True,
                )
            with torch.cuda.amp.autocast(enabled=False):
                loss, stats = compute_losses(
                    out1,
                    out2,
                    out3,
                    change_logits_dict,
                    (mask1, mask2, mask3),
                    ctx.criteria,
                    args,
                )
                if not torch.isfinite(loss.detach()).all():
                    raise FloatingPointError("non-finite loss at epoch=%d step=%d" % (epoch, step))
                loss_to_backward = loss / float(args.accum_steps)
            if args.amp:
                ctx.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        if update_now:
            if args.amp:
                ctx.scaler.unscale_(ctx.optimizer)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip_norm)

            advanced = True
            if args.amp:
                previous_scale = float(ctx.scaler.get_scale())
                ctx.scaler.step(ctx.optimizer)
                ctx.scaler.update()
                advanced = float(ctx.scaler.get_scale()) >= previous_scale
            else:
                ctx.optimizer.step()

            if advanced:
                ctx.global_update_step += 1
                if ctx.scheduler_step_on_updates:
                    step_timm_scheduler_update(ctx.scheduler, ctx.global_update_step)
            ctx.optimizer.zero_grad(set_to_none=True)

        reset_snn_state(model, ctx.functional)
        for key, value in stats.items():
            totals[key] += value

    denom = max(1, total_steps)
    out = {key: value / denom for key, value in totals.items()}
    out["lr"] = float(ctx.optimizer.param_groups[0]["lr"])
    return out


def _metric_dict(metric):
    _, score, miou, sek, fscd, oa, sc_precision, sc_recall = metric.evaluate_SECOND()
    return {
        "score": float(score),
        "miou": float(miou),
        "sek": float(sek),
        "Fscd": float(fscd),
        "OA": float(oa),
        "SC_Precision": float(sc_precision),
        "SC_Recall": float(sc_recall),
    }


@torch.no_grad()
def validate_t1_t3(ctx, epoch):
    args = ctx.args
    if not is_main_process(args):
        if args.distributed:
            dist.barrier()
        return {}
    model = unwrap_model(ctx.model)
    model.eval()
    metric = ctx.IOUandSek(num_classes=len(ctx.RS.ST_CLASSES))

    for batch in ctx.valloader:
        img1, img2, img3, mask1, _mask2, mask3, mask_bn, _sample_id = batch
        img1 = img1.float().to(ctx.device, non_blocking=True)
        img2 = img2.float().to(ctx.device, non_blocking=True)
        img3 = img3.float().to(ctx.device, non_blocking=True)
        x = torch.stack([img1, img2, img3], dim=0)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out1, _out2, out3, change13 = model(x)

        pred1 = torch.argmax(out1.float(), dim=1).cpu().numpy() + 1
        pred3 = torch.argmax(out3.float(), dim=1).cpu().numpy() + 1
        pred_change = (torch.sigmoid(change13.float()) > 0.5).cpu().numpy().astype(np.uint8)
        gt1 = mask1.clone()
        gt3 = mask3.clone()
        gt1[mask_bn == 0] = 0
        gt3[mask_bn == 0] = 0
        pred1[pred_change == 0] = 0
        pred3[pred_change == 0] = 0
        metric.add_batch(pred1, gt1.numpy())
        metric.add_batch(pred3, gt3.numpy())
        reset_snn_state(model, ctx.functional)

    stats = _metric_dict(metric)
    if args.distributed:
        dist.barrier()
    return stats


@torch.no_grad()
def validate_all_pairs(ctx, epoch):
    args = ctx.args
    if not is_main_process(args):
        if args.distributed:
            dist.barrier()
        return {}
    model = unwrap_model(ctx.model)
    model.eval()
    metric = ctx.IOUandSek(num_classes=len(ctx.RS.ST_CLASSES))

    for batch in ctx.valloader:
        img1, img2, img3, mask1, mask2, mask3, _mask_bn, _sample_id = batch
        img1 = img1.float().to(ctx.device, non_blocking=True)
        img2 = img2.float().to(ctx.device, non_blocking=True)
        img3 = img3.float().to(ctx.device, non_blocking=True)
        x = torch.stack([img1, img2, img3], dim=0)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out1, out2, out3, _change13, change_logits_dict = model(
                x,
                return_change_logits_dict=True,
            )

        preds = {
            "t1": torch.argmax(out1.float(), dim=1).cpu() + 1,
            "t2": torch.argmax(out2.float(), dim=1).cpu() + 1,
            "t3": torch.argmax(out3.float(), dim=1).cpu() + 1,
        }
        labels = {"t1": mask1.long(), "t2": mask2.long(), "t3": mask3.long()}

        for pair_key, (phase_i, phase_j) in zip(PAIR_KEYS, PAIR_NAMES):
            label_i = labels[phase_i]
            label_j = labels[phase_j]
            valid = (label_i > 0) & (label_j > 0)
            gt_change = (label_i != label_j) & valid
            pred_change = (
                torch.sigmoid(change_logits_dict[pair_key].detach().float().cpu().squeeze(1)) > 0.5
            )

            pred_i = preds[phase_i].clone()
            pred_j = preds[phase_j].clone()
            gt_i = label_i.clone()
            gt_j = label_j.clone()

            pred_i[~pred_change] = 0
            pred_j[~pred_change] = 0
            gt_i[~gt_change] = 0
            gt_j[~gt_change] = 0
            pred_i[~valid] = 0
            pred_j[~valid] = 0
            gt_i[~valid] = 0
            gt_j[~valid] = 0

            metric.add_batch(pred_i.numpy(), gt_i.numpy())
            metric.add_batch(pred_j.numpy(), gt_j.numpy())

        reset_snn_state(model, ctx.functional)

    stats = _metric_dict(metric)
    if args.distributed:
        dist.barrier()
    return stats


def checkpoint_dir(args):
    return args.output_dir


def checkpoint_payload(ctx, epoch):
    return {
        "epoch": epoch,
        "model": unwrap_model(ctx.model).state_dict(),
        "optimizer": ctx.optimizer.state_dict(),
        "scaler": ctx.scaler.state_dict() if ctx.args.amp else None,
        "scheduler": scheduler_state_dict(ctx.scheduler),
        "best_metric": ctx.best_metric,
        "global_update_step": ctx.global_update_step,
        "args": vars(ctx.args),
    }


def save_checkpoint(ctx, epoch, val_stats):
    args = ctx.args
    if not is_main_process(args):
        return
    os.makedirs(checkpoint_dir(args), exist_ok=True)
    score = float(val_stats["score"])
    is_best = score >= float(ctx.best_metric)
    if is_best:
        ctx.best_metric = score
    torch.save(checkpoint_payload(ctx, epoch), os.path.join(checkpoint_dir(args), "latest.pth"))
    if is_best:
        best_name = "epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" % (
            epoch,
            val_stats["score"] * 100,
            val_stats["miou"] * 100,
            val_stats["sek"] * 100,
            val_stats["Fscd"] * 100,
            val_stats["OA"] * 100,
        )
        torch.save(checkpoint_payload(ctx, epoch), os.path.join(checkpoint_dir(args), best_name))


def load_checkpoint(ctx, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    unwrap_model(ctx.model).load_state_dict(state_dict, strict=True)
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            ctx.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None and ctx.args.amp:
            ctx.scaler.load_state_dict(checkpoint["scaler"])
        ctx.best_metric = float(checkpoint.get("best_metric", 0.0))
        ctx.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        ctx.global_update_step = int(checkpoint.get("global_update_step", ctx.start_epoch * ctx.updates_per_epoch))
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            load_scheduler_state(ctx.scheduler, checkpoint["scheduler"])
        elif ctx.global_update_step > 0:
            sync_scheduler_to_resume_position(
                ctx.scheduler,
                ctx.global_update_step,
                ctx.start_epoch,
                ctx.scheduler_step_on_updates,
            )
    if is_main_process(ctx.args):
        print("Resumed checkpoint from %s" % checkpoint_path, flush=True)


def make_log_path(args):
    if args.log_dir is None:
        args.log_dir = args.output_dir
    os.makedirs(args.log_dir, exist_ok=True)
    return os.path.join(args.log_dir, "log.jsonl")


def write_epoch_log(log_path, payload):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    init_distributed_mode(args)
    validate_args(args)
    seed_everything(args.seed, args.rank)

    from spikingjelly.clock_driven import functional
    from utils.metric import IOUandSek

    try:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:%d" % args.local_rank if torch.cuda.is_available() else "cpu")
        RS, trainset, trainloader, train_sampler, valloader = build_dataloaders(args)
        model = build_model(args, RS, device)
        optimizer = build_timm_optimizer(model, args)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        criteria = build_loss_functions(args)

        updates_per_epoch = max(1, math.ceil(len(trainloader) / args.accum_steps))
        actual_total_updates = updates_per_epoch * args.epochs
        total_updates, warmup_updates = resolve_reference_update_counts(args, len(trainset), actual_total_updates)
        scheduler, _num_epochs, scheduler_args, _scheduler_updates_per_epoch = build_timm_scheduler(
            args,
            optimizer,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )

        ctx = SimpleNamespace(
            args=args,
            RS=RS,
            IOUandSek=IOUandSek,
            functional=functional,
            device=device,
            model=model,
            criteria=criteria,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_on_updates=bool(getattr(scheduler_args, "sched_on_updates", False)),
            trainloader=trainloader,
            train_sampler=train_sampler,
            valloader=valloader,
            updates_per_epoch=updates_per_epoch,
            global_update_step=0,
            start_epoch=0,
            best_metric=0.0,
        )

        if is_main_process(args):
            n_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
            print(
                "params_M=%.2f, batch_per_gpu=%d, world_size=%d, accum_steps=%d, updates_per_epoch=%d, total_updates=%d, warmup_updates=%.1f"
                % (
                    n_params / 1e6,
                    args.batch_size,
                    args.world_size,
                    args.accum_steps,
                    updates_per_epoch,
                    total_updates,
                    warmup_updates,
                ),
                flush=True,
            )

        if args.resume:
            load_checkpoint(ctx, args.resume)

        log_path = make_log_path(args) if is_main_process(args) else None
        if args.eval_only:
            val_stats = validate_t1_t3(ctx, ctx.start_epoch)
            payload = {"epoch": ctx.start_epoch, "val": val_stats, "best_score": float(ctx.best_metric)}
            if args.val_mode == "all_pairs":
                payload["val_all_pairs_diagnostic"] = validate_all_pairs(ctx, ctx.start_epoch)
            if is_main_process(args):
                print("EPOCH_SUMMARY " + json.dumps(payload, ensure_ascii=False), flush=True)
            return

        start_time = datetime.now()
        for epoch in range(ctx.start_epoch, args.epochs):
            train_stats = train_one_epoch(ctx, epoch)
            val_stats = validate_t1_t3(ctx, epoch)
            diagnostic_stats = validate_all_pairs(ctx, epoch) if args.val_mode == "all_pairs" else None
            step_scheduler_epoch(ctx, epoch, val_stats["score"] if val_stats else None)
            save_checkpoint(ctx, epoch, val_stats)

            if is_main_process(args):
                payload = {
                    "epoch": epoch,
                    "train": train_stats,
                    "val": val_stats,
                    "best_score": float(ctx.best_metric),
                }
                if diagnostic_stats is not None:
                    payload["val_all_pairs_diagnostic"] = diagnostic_stats
                print("EPOCH_SUMMARY " + json.dumps(payload, ensure_ascii=False), flush=True)
                write_epoch_log(log_path, payload)

        if is_main_process(args):
            print("Training time: %s" % (datetime.now() - start_time), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
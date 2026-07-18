#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from types import MappingProxyType
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
EVALUATION_SCHEMA_VERSION = 2

MECHANISM_CONFIG_SCHEMA_VERSION = 1


def _mechanism_profile(**overrides):
    config = dict(
        dend_spatial_conv_type="structure_routed_v1",
        routeconv_ablation_mode="full",
        routeconv_v2_mode="v2_6",
        routeconv_v3_mode="v3_6",
        dend_residual_init=0.01,
        pdca_dend_prior_mode="offset_residual",
        pdca_dend_prior_alpha=1e-3,
        pdca_dend_prior_detach=True,
        pdca_dend_prior_descriptor="mean_std",
        pdca_dend_prior_normalize="zscore",
        pdca_dend_prior_source_weight=1.0,
        pdca_dend_prior_point_weight=0.25,
        pdca_dend_prior_sim_weight=1.0,
        pdca_dend_prior_diff_weight=0.25,
        pdca_dend_prior_use_conf_gate=True,
        pdca_dend_prior_conf_beta=4.0,
        pdca_dend_prior_conf_tau=0.10,
        pdca_dend_prior_use_offset_gate=True,
        pdca_dend_prior_center_point=True,
        pdca_dend_prior_clip=2.0,
        pdca_dend_prior_affect_null=False,
        pdca_dend_prior_stats=False,
        v4_frequency_enabled=False,
        v4_relation_enabled=False,
        legacy_dendritic_prior_enabled=True,
    )
    config.update(overrides)
    return MappingProxyType(config)


# Insertion order is part of the schema-v1 experiment contract.
MECHANISM_PROFILE_REGISTRY = MappingProxyType(
    {
        "l0_v1_full_legacy_offset_residual": _mechanism_profile(),
        "l1_v1_no_prior": _mechanism_profile(
            pdca_dend_prior_mode="none",
        ),
        "v4_route_only": _mechanism_profile(
            dend_spatial_conv_type="structure_routed_v4",
            pdca_dend_prior_mode="none",
            legacy_dendritic_prior_enabled=False,
        ),
        "v4_freq_only": _mechanism_profile(
            dend_spatial_conv_type="structure_routed_v4",
            pdca_dend_prior_mode="none",
            v4_frequency_enabled=True,
            legacy_dendritic_prior_enabled=False,
        ),
        "v4_relation_only": _mechanism_profile(
            dend_spatial_conv_type="structure_routed_v4",
            pdca_dend_prior_mode="none",
            v4_relation_enabled=True,
            legacy_dendritic_prior_enabled=False,
        ),
        "v4_full": _mechanism_profile(
            dend_spatial_conv_type="structure_routed_v4",
            pdca_dend_prior_mode="none",
            v4_frequency_enabled=True,
            v4_relation_enabled=True,
            legacy_dendritic_prior_enabled=False,
        ),
    }
)

PROTECTED_MECHANISM_OPTIONS = frozenset(
    {
        "--dend-spatial-conv-type",
        "--routeconv-ablation-mode",
        "--routeconv-v2-mode",
        "--routeconv-v3-mode",
        "--dend-residual-init",
        "--pdca-dend-prior-mode",
        "--pdca-dend-prior-alpha",
        "--pdca-dend-prior-detach",
        "--pdca-dend-prior-descriptor",
        "--pdca-dend-prior-normalize",
        "--pdca-dend-prior-source-weight",
        "--pdca-dend-prior-point-weight",
        "--pdca-dend-prior-sim-weight",
        "--pdca-dend-prior-diff-weight",
        "--pdca-dend-prior-use-conf-gate",
        "--pdca-dend-prior-conf-beta",
        "--pdca-dend-prior-conf-tau",
        "--pdca-dend-prior-use-offset-gate",
        "--pdca-dend-prior-center-point",
        "--pdca-dend-prior-clip",
        "--pdca-dend-prior-affect-null",
        "--pdca-dend-prior-stats",
    }
)


def _canonical_config_hash(config):
    payload = json.dumps(
        config,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _explicit_long_options(raw_argv):
    options = set()
    for token in raw_argv:
        if isinstance(token, str) and token.startswith("--"):
            options.add(token.split("=", 1)[0])
    return options


def resolve_mechanism_config(args, raw_argv):
    args.raw_argv = list(raw_argv)
    profile_name = args.dend_mechanism_config
    args.mechanism_profile = profile_name

    if profile_name is None:
        args.mechanism_config_schema_version = None
        args.resolved_mechanism_config = None
        args.resolved_mechanism_config_hash = None
        args.v4_frequency_enabled = False
        args.v4_relation_enabled = False
        args.legacy_dendritic_prior_enabled = True
        if args.dend_spatial_conv_type == "structure_routed_v4":
            raise ValueError(
                "structure_routed_v4 must be enabled through "
                "--dend-mechanism-config, not low-level flags"
            )
        return args

    explicit_options = _explicit_long_options(raw_argv)
    conflicts = sorted(explicit_options & PROTECTED_MECHANISM_OPTIONS)
    if conflicts:
        raise ValueError(
            "--dend-mechanism-config is the sole mechanism source; remove "
            "explicit protected options: %s" % ", ".join(conflicts)
        )

    if profile_name not in MECHANISM_PROFILE_REGISTRY:
        raise ValueError("unknown mechanism profile %r" % profile_name)
    resolved = dict(MECHANISM_PROFILE_REGISTRY[profile_name])
    for name, value in resolved.items():
        setattr(args, name, value)
    args.mechanism_config_schema_version = MECHANISM_CONFIG_SCHEMA_VERSION
    args.resolved_mechanism_config = resolved
    args.resolved_mechanism_config_hash = _canonical_config_hash(resolved)
    return args


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
    parser.add_argument(
        "--dend-mechanism-config",
        choices=tuple(MECHANISM_PROFILE_REGISTRY.keys()),
        default=None,
        help=(
            "Versioned sole-source mechanism profile. When set, all protected "
            "low-level dendritic/PDCA-prior options are forbidden."
        ),
    )
    parser.add_argument(
        "--v4-mechanism-diagnostics",
        action="store_true",
        help="Run parameter-free V4 A+ diagnostics in eval-only mode.",
    )

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
                                 "structure_routed_v3", "structure_routed_v4"],
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
    if args.dend_spatial_conv_type == "structure_routed_v4":
        if args.mechanism_profile not in (
            "v4_route_only",
            "v4_freq_only",
            "v4_relation_only",
            "v4_full",
        ):
            raise ValueError("structure_routed_v4 requires a V4 mechanism profile")
        if args.legacy_dendritic_prior_enabled:
            raise ValueError("V4 requires legacy_dendritic_prior_enabled=False")
        if args.pdca_dend_prior_mode != "none":
            raise ValueError("V4 requires pdca_dend_prior_mode='none'")
    elif (
        bool(args.v4_frequency_enabled)
        or bool(args.v4_relation_enabled)
        or not bool(args.legacy_dendritic_prior_enabled)
    ):
        raise ValueError("legacy profiles cannot enable V4-only model flags")
    if args.mechanism_profile is not None and args.seed != 42:
        raise ValueError(
            "schema-v1 core mechanism profiles are frozen to seed=42; "
            "multi-seed validation is DEFERRED"
        )
    if args.mechanism_profile is not None and args.pretrain_from:
        raise ValueError(
            "schema-v1 core mechanism profiles forbid non-strict "
            "--pretrain-from warm starts; use the same raw internal pretrained "
            "checkpoint for all six runs"
        )
    if args.mechanism_profile is not None and not args.pretrained:
        raise ValueError(
            "schema-v1 core mechanism profiles require the shared original "
            "internal pretrained checkpoint"
        )
    if args.v4_mechanism_diagnostics:
        if not args.eval_only:
            raise ValueError("--v4-mechanism-diagnostics requires --eval-only")
        if args.mechanism_profile not in (
            "v4_freq_only",
            "v4_relation_only",
            "v4_full",
        ):
            raise ValueError(
                "V4 diagnostics require v4_freq_only, v4_relation_only, or v4_full"
            )
        if not args.resume:
            raise ValueError("V4 diagnostics require --resume with the selected checkpoint")
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pretrained_provenance(args):
    if args.pretrained:
        path = internal_pretrain_path()
    elif args.pretrain_from:
        path = args.pretrain_from
    else:
        path = None
    args.pretrained_checkpoint_path = path
    args.pretrained_checkpoint_sha256 = (
        _sha256_file(path) if path and os.path.isfile(path) else None
    )


def resolve_git_provenance(args):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        args.git_commit = commit
        args.git_dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        args.git_commit = None
        args.git_dirty = None


def build_model(args, RS, device):
    from models.GSTMSCD_MTSCD_Snn_ForDecoder_clean_V4 import GSTMSCD_WUSU as Net

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
        v4_frequency_enabled=args.v4_frequency_enabled,
        v4_relation_enabled=args.v4_relation_enabled,
        legacy_dendritic_prior_enabled=args.legacy_dendritic_prior_enabled,

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
    assert_mechanism_structure(model, args)

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


def assert_mechanism_structure(model, args):
    from models.dend_structure_routed_conv_v4 import DendriticFrequencyHead
    from models.Encoders.FDPC_Encoder_ForDecoder_clean_V4 import DirectedRelationHead
    from models.Encoders.phase_deformable_context_attention_fordecoder_clean_v22_V4 import (
        PhaseDeformableContextAttention,
    )

    base_model = unwrap_model(model)
    modules = tuple(base_model.modules())
    frequency_head_count = sum(
        isinstance(module, DendriticFrequencyHead) for module in modules
    )
    relation_head_count = sum(
        isinstance(module, DirectedRelationHead) for module in modules
    )
    legacy_alpha_count = sum(
        isinstance(module, PhaseDeformableContextAttention)
        and module.alpha is not None
        for module in modules
    )
    for module in modules:
        if isinstance(module, (DendriticFrequencyHead, DirectedRelationHead)):
            for parameter in module.parameters():
                if parameter.dtype != torch.float32:
                    raise RuntimeError("V4 Head parameters must initialize as FP32")
                if torch.count_nonzero(parameter.detach()).item() != 0:
                    raise RuntimeError("V4 Head parameters must initialize exactly to zero")

    expected_by_profile = {
        "l0_v1_full_legacy_offset_residual": (0, 0, 4),
        "l1_v1_no_prior": (0, 0, 4),
        "v4_route_only": (0, 0, 0),
        "v4_freq_only": (8, 0, 0),
        "v4_relation_only": (0, 4, 0),
        "v4_full": (8, 4, 0),
    }
    if args.mechanism_profile is not None:
        actual = (frequency_head_count, relation_head_count, legacy_alpha_count)
        expected = expected_by_profile[args.mechanism_profile]
        if actual != expected:
            raise RuntimeError(
                "mechanism structure mismatch for %s: expected "
                "(Frequency Heads, Relation Heads, legacy alpha)=%r, got %r"
                % (args.mechanism_profile, expected, actual)
            )

    total_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )
    if is_main_process(args):
        print(
            "mechanism=%r schema=%r freq_heads=%d relation_heads=%d "
            "legacy_alpha=%d total_params=%d trainable_params=%d"
            % (
                args.mechanism_profile,
                args.mechanism_config_schema_version,
                frequency_head_count,
                relation_head_count,
                legacy_alpha_count,
                total_parameters,
                trainable_parameters,
            ),
            flush=True,
        )


def assert_head_optimizer_registration(model, optimizer, args):
    from models.dend_structure_routed_conv_v4 import DendriticFrequencyHead
    from models.Encoders.FDPC_Encoder_ForDecoder_clean_V4 import DirectedRelationHead

    head_parameters = []
    for module in unwrap_model(model).modules():
        if isinstance(module, (DendriticFrequencyHead, DirectedRelationHead)):
            for parameter in module.parameters():
                if parameter.dtype != torch.float32:
                    raise RuntimeError("V4 Head parameters must remain FP32")
                head_parameters.append(parameter)

    optimizer_counts = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer_counts[id(parameter)] = optimizer_counts.get(id(parameter), 0) + 1
    for parameter in head_parameters:
        count = optimizer_counts.get(id(parameter), 0)
        if count != 1:
            raise RuntimeError(
                "each active V4 Head parameter must occur once in optimizer; got %d"
                % count
            )


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


def _assert_metric_dict_exact(reference, candidate, name):
    if set(reference) != set(candidate):
        raise RuntimeError(
            "%s metric keys differ: reference=%r candidate=%r"
            % (name, sorted(reference), sorted(candidate))
        )
    mismatches = {}
    for metric_name in reference:
        reference_value = float(reference[metric_name])
        candidate_value = float(candidate[metric_name])
        if not math.isfinite(reference_value):
            raise FloatingPointError(
                "%s reference metric %s is non-finite" % (name, metric_name)
            )
        if not math.isfinite(candidate_value):
            raise FloatingPointError(
                "%s candidate metric %s is non-finite" % (name, metric_name)
            )
        if reference_value != candidate_value:
            mismatches[metric_name] = {
                "reference": reference_value,
                "candidate": candidate_value,
            }
    if mismatches:
        raise RuntimeError(
            "%s metrics differ: %s"
            % (name, json.dumps(mismatches, ensure_ascii=False, sort_keys=True))
        )


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
    pooled_metric = ctx.IOUandSek(num_classes=len(ctx.RS.ST_CLASSES))
    metric_by_pair = {
        pair_key: ctx.IOUandSek(num_classes=len(ctx.RS.ST_CLASSES))
        for pair_key in PAIR_KEYS
    }
    logits_t1_t3_consistent = True
    labels_t1_t3_consistent = True

    for batch in ctx.valloader:
        img1, img2, img3, mask1, mask2, mask3, mask_bn, _sample_id = batch
        img1 = img1.float().to(ctx.device, non_blocking=True)
        img2 = img2.float().to(ctx.device, non_blocking=True)
        img3 = img3.float().to(ctx.device, non_blocking=True)
        x = torch.stack([img1, img2, img3], dim=0)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out1, out2, out3, _change13, change_logits_dict = model(
                x,
                return_change_logits_dict=True,
            )
        logits_t1_t3_consistent = logits_t1_t3_consistent and torch.equal(
            change_logits_dict["t1_to_t3"].detach().float().squeeze(1),
            _change13.detach().float(),
        )
        label_change_13 = mask1 != mask3
        batch_labels_consistent = torch.equal(mask_bn != 0, label_change_13)
        labels_t1_t3_consistent = (
            labels_t1_t3_consistent and batch_labels_consistent
        )
        if not batch_labels_consistent:
            mismatch_count = int(
                torch.count_nonzero((mask_bn != 0) != label_change_13).item()
            )
            raise RuntimeError(
                "mask_bn differs from (mask1 != mask3) on %d pixels"
                % mismatch_count
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
            # WUSU authoritative definition: every label transition is change.
            # Ground truth must never suppress a model prediction.
            gt_change = label_i != label_j
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
            metric_by_pair[pair_key].add_batch(pred_i.numpy(), gt_i.numpy())
            metric_by_pair[pair_key].add_batch(pred_j.numpy(), gt_j.numpy())
            pooled_metric.add_batch(pred_i.numpy(), gt_i.numpy())
            pooled_metric.add_batch(pred_j.numpy(), gt_j.numpy())

        reset_snn_state(model, ctx.functional)

    per_pair = {
        pair_key: _metric_dict(metric_by_pair[pair_key])
        for pair_key in PAIR_KEYS
    }
    metric_names = tuple(next(iter(per_pair.values())).keys())
    macro = {
        name: sum(float(per_pair[pair_key][name]) for pair_key in PAIR_KEYS)
        / float(len(PAIR_KEYS))
        for name in metric_names
    }
    stats = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "per_pair": per_pair,
        "macro": macro,
        "pooled": _metric_dict(pooled_metric),
        "definition_checks": {
            "change_logits_t1_to_t3_equal_change13": bool(
                logits_t1_t3_consistent
            ),
            "label_change_t1_to_t3_equal_mask_bn_full_image": bool(
                labels_t1_t3_consistent
            ),
        },
    }
    if args.distributed:
        dist.barrier()
    return stats


class _BoundedTensorStats:
    """CPU-FP64 streaming moments plus a fixed bounded histogram."""

    def __init__(self, lower, upper, center, bins=4096):
        self.lower = float(lower)
        self.upper = float(upper)
        self.center = float(center)
        self.bins = int(bins)
        self.count = 0
        self.total = torch.zeros((), dtype=torch.float64)
        self.total_square = torch.zeros((), dtype=torch.float64)
        self.total_abs_center = torch.zeros((), dtype=torch.float64)
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.near_lower_count = 0
        self.near_upper_count = 0
        self.histogram = torch.zeros(self.bins, dtype=torch.float64)

    def update(self, value):
        value = value.detach().float().cpu().to(dtype=torch.float64).reshape(-1)
        if value.numel() == 0:
            return
        if not torch.isfinite(value).all():
            raise FloatingPointError("diagnostic tensor contains NaN or Inf")
        self.count += int(value.numel())
        self.total += value.sum()
        self.total_square += value.square().sum()
        self.total_abs_center += (value - self.center).abs().sum()
        self.minimum = min(self.minimum, float(value.min()))
        self.maximum = max(self.maximum, float(value.max()))
        near_width = 0.01 * (self.upper - self.lower)
        self.near_lower_count += int((value <= self.lower + near_width).sum())
        self.near_upper_count += int((value >= self.upper - near_width).sum())
        clipped = value.clamp(self.lower, self.upper).float()
        self.histogram += torch.histc(
            clipped,
            bins=self.bins,
            min=self.lower,
            max=self.upper,
        ).to(dtype=torch.float64)

    def _quantile(self, probability):
        if self.count == 0:
            return None
        target = float(probability) * float(max(0, self.count - 1))
        index = int(
            torch.searchsorted(
                self.histogram.cumsum(dim=0),
                torch.tensor(target, dtype=torch.float64),
                right=True,
            ).clamp(max=self.bins - 1)
        )
        width = (self.upper - self.lower) / float(self.bins)
        return self.lower + (float(index) + 0.5) * width

    def finalize(self):
        if self.count == 0:
            return {"count": 0}
        mean = self.total / float(self.count)
        variance = (
            self.total_square / float(self.count) - mean.square()
        ).clamp_min(0.0)
        return {
            "count": self.count,
            "mean": float(mean),
            "population_std": float(torch.sqrt(variance)),
            "mean_abs_from_neutral": float(
                self.total_abs_center / float(self.count)
            ),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "q05_hist4096": self._quantile(0.05),
            "q50_hist4096": self._quantile(0.50),
            "q95_hist4096": self._quantile(0.95),
            "near_lower_1pct_ratio": self.near_lower_count / float(self.count),
            "near_upper_1pct_ratio": self.near_upper_count / float(self.count),
        }


class V4MechanismDiagnosticAccumulator:
    """Non-module A+ accumulator; it registers no parameter or buffer."""

    activity_tol = 1e-6
    conditional_point_tol = 1e-6

    def __init__(self):
        self.frequency_head = {}
        self.frequency_consumed = {}
        self.frequency_energy = {}
        self.frequency_output = {}
        self.relation_head = {}
        self.relation_consumed = {}
        self.relation_effect = {}
        self.relation_direction_pending = {}
        self.relation_direction_difference = {}
        self.conditional_point_max_abs_error = 0.0

    @staticmethod
    def _key(*parts):
        return "/".join(str(part) for part in parts)

    @staticmethod
    def _add_pair(accumulator, numerator, denominator):
        numerator = numerator.detach().float().cpu().to(torch.float64)
        denominator = denominator.detach().float().cpu().to(torch.float64)
        accumulator[0] += numerator.sum()
        accumulator[1] += denominator.sum()

    def record_frequency_head(self, block_index, source_scale, bands, gains):
        link = "%d->%d" % (source_scale, source_scale + 1)
        for band, gain in zip(bands, gains):
            key = self._key("block", block_index, "link", link, "band", band)
            stats = self.frequency_head.setdefault(
                key,
                _BoundedTensorStats(0.0, 2.0, 1.0),
            )
            stats.update(gain)

    def record_frequency_consumption(
        self,
        block_index,
        target_scale,
        bands,
        gains,
        high_bands,
        modulated_output,
        neutral_output,
    ):
        link = "%d->%d" % (target_scale - 1, target_scale)
        output_key = self._key("block", block_index, "link", link)
        output_pair = self.frequency_output.setdefault(
            output_key,
            [torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)],
        )
        self._add_pair(
            output_pair,
            (modulated_output - neutral_output).abs(),
            neutral_output.abs(),
        )

        for band, gain, high_band in zip(bands, gains, high_bands):
            key = self._key("block", block_index, "link", link, "band", band)
            consumed = self.frequency_consumed.setdefault(
                key,
                _BoundedTensorStats(0.0, 2.0, 1.0),
            )
            consumed.update(gain)
            energy = self.frequency_energy.setdefault(
                key,
                [
                    torch.zeros((), dtype=torch.float64),
                    torch.zeros((), dtype=torch.float64),
                    0,
                ],
            )
            absolute_modulation = ((gain.float() - 1.0) * high_band.float()).abs()
            self._add_pair(energy, absolute_modulation, high_band.float().abs())
            energy[2] += int(absolute_modulation.numel())

    def record_relation_head(
        self,
        block_index,
        target_name,
        source_name,
        prior,
    ):
        key = self._key(
            "block",
            block_index,
            "edge",
            "%s<-%s" % (target_name, source_name),
        )
        stats = self.relation_head.setdefault(
            key,
            _BoundedTensorStats(-0.25, 0.25, 0.0),
        )
        stats.update(prior)

        unordered_pair = tuple(sorted((target_name, source_name)))
        pending_key = (block_index, unordered_pair)
        pending = self.relation_direction_pending.pop(pending_key, None)
        if pending is None:
            self.relation_direction_pending[pending_key] = (
                target_name,
                source_name,
                prior.detach(),
            )
        else:
            _previous_target, _previous_source, previous_prior = pending
            if tuple(previous_prior.shape) != tuple(prior.shape):
                raise ValueError("reverse relation-prior shapes must match")
            direction_key = self._key(
                "block",
                block_index,
                "pair",
                "%s<->%s" % unordered_pair,
            )
            pair = self.relation_direction_difference.setdefault(
                direction_key,
                [torch.zeros((), dtype=torch.float64), 0],
            )
            difference = (
                prior.detach().float().cpu()
                - previous_prior.float().cpu()
            ).abs().to(torch.float64)
            pair[0] += difference.sum()
            pair[1] += int(difference.numel())

    def record_relation_consumption(
        self,
        block_index,
        scale_index,
        target_name,
        source_name,
        prior_consumed,
    ):
        key = self._key(
            "block",
            block_index,
            "scale",
            scale_index,
            "edge",
            "%s<-%s" % (target_name, source_name),
        )
        stats = self.relation_consumed.setdefault(
            key,
            _BoundedTensorStats(-0.25, 0.25, 0.0),
        )
        stats.update(prior_consumed)

    def record_relation_effect(
        self,
        block_index,
        scale_index,
        target_name,
        source_names,
        source_marginal_base,
        source_marginal_final,
        conditional_point_max_abs_error,
    ):
        delta = (source_marginal_final - source_marginal_base).abs()
        for source_index, source_name in enumerate(source_names):
            key = self._key(
                "block",
                block_index,
                "scale",
                scale_index,
                "edge",
                "%s<-%s" % (target_name, source_name),
            )
            pair = self.relation_effect.setdefault(
                key,
                [torch.zeros((), dtype=torch.float64), 0],
            )
            value = delta[:, :, source_index].detach().float().cpu().to(torch.float64)
            pair[0] += value.sum()
            pair[1] += int(value.numel())
        self.conditional_point_max_abs_error = max(
            self.conditional_point_max_abs_error,
            float(conditional_point_max_abs_error.detach().float().cpu()),
        )

    @staticmethod
    def _global_center_abs(stats_by_key):
        total = torch.zeros((), dtype=torch.float64)
        count = 0
        for stats in stats_by_key.values():
            total += stats.total_abs_center
            count += stats.count
        return float(total / float(count)) if count else 0.0

    @staticmethod
    def _global_ratio(pairs_by_key):
        numerator = torch.zeros((), dtype=torch.float64)
        denominator = torch.zeros((), dtype=torch.float64)
        for pair in pairs_by_key.values():
            numerator += pair[0]
            denominator += pair[1]
        return float(numerator / (denominator + 1e-12))

    def finalize(self):
        if self.relation_direction_pending:
            raise RuntimeError(
                "unpaired directed relation diagnostics remain at finalize"
            )
        frequency_mean_abs = self._global_center_abs(self.frequency_head)
        frequency_normalized_modulation = self._global_ratio(
            self.frequency_energy
        )
        relation_mean_abs = self._global_center_abs(self.relation_head)
        relation_delta_total = torch.zeros((), dtype=torch.float64)
        relation_delta_count = 0
        for total, count in self.relation_effect.values():
            relation_delta_total += total
            relation_delta_count += int(count)
        relation_marginal_change = (
            float(relation_delta_total / float(relation_delta_count))
            if relation_delta_count
            else 0.0
        )

        return {
            "quantile_method": "fixed 4096-bin bounded histogram",
            "activity_tol": self.activity_tol,
            "frequency": {
                "head_fp32": {
                    key: value.finalize()
                    for key, value in sorted(self.frequency_head.items())
                },
                "consumed": {
                    key: value.finalize()
                    for key, value in sorted(self.frequency_consumed.items())
                },
                "per_band_modulation": {
                    key: {
                        "absolute_modulation_sum": float(value[0]),
                        "high_band_abs_sum": float(value[1]),
                        "absolute_modulation_mean": (
                            float(value[0] / float(value[2]))
                            if value[2]
                            else 0.0
                        ),
                        "high_band_abs_mean": (
                            float(value[1] / float(value[2]))
                            if value[2]
                            else 0.0
                        ),
                        "relative_modulation": float(
                            value[0] / (value[1] + 1e-12)
                        ),
                    }
                    for key, value in sorted(self.frequency_energy.items())
                },
                "output_change": {
                    key: float(value[0] / (value[1] + 1e-12))
                    for key, value in sorted(self.frequency_output.items())
                },
                "global_mean_abs_K_minus_1": frequency_mean_abs,
                "global_normalized_modulation_energy": (
                    frequency_normalized_modulation
                ),
                "activity_pass": bool(
                    frequency_mean_abs > self.activity_tol
                    and frequency_normalized_modulation > self.activity_tol
                ) if self.frequency_head else None,
            },
            "relation": {
                "head_fp32": {
                    key: value.finalize()
                    for key, value in sorted(self.relation_head.items())
                },
                "consumed": {
                    key: value.finalize()
                    for key, value in sorted(self.relation_consumed.items())
                },
                "source_marginal_mean_abs_change": {
                    key: float(total / float(count)) if count else 0.0
                    for key, (total, count) in sorted(self.relation_effect.items())
                },
                "global_mean_abs_P_rel": relation_mean_abs,
                "global_source_marginal_mean_abs_change": (
                    relation_marginal_change
                ),
                "directed_reverse_mean_abs_difference": {
                    key: float(total / float(count)) if count else 0.0
                    for key, (total, count) in sorted(
                        self.relation_direction_difference.items()
                    )
                },
                "conditional_point_max_abs_error": (
                    self.conditional_point_max_abs_error
                ),
                "conditional_point_invariant_pass": bool(
                    self.conditional_point_max_abs_error
                    <= self.conditional_point_tol
                ) if self.relation_head else None,
                "activity_pass": bool(
                    relation_mean_abs > self.activity_tol
                    and relation_marginal_change > self.activity_tol
                ) if self.relation_head else None,
            },
        }


def _outputs_torch_equal(first, second):
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(
            _outputs_torch_equal(left, right)
            for left, right in zip(first, second)
        )
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _outputs_torch_equal(first[key], second[key]) for key in first
        )
    return first == second


@torch.no_grad()
def run_v4_mechanism_diagnostics(ctx):
    args = ctx.args
    if not is_main_process(args):
        if args.distributed:
            dist.barrier()
        return {}

    model = unwrap_model(ctx.model)
    if not hasattr(model, "set_v4_diagnostic_sink"):
        raise RuntimeError("model does not expose the V4 diagnostic sink")
    model.eval()
    accumulator = V4MechanismDiagnosticAccumulator()

    for batch in ctx.valloader:
        img1, img2, img3 = batch[0], batch[1], batch[2]
        x = torch.stack(
            [
                img1.float().to(ctx.device, non_blocking=True),
                img2.float().to(ctx.device, non_blocking=True),
                img3.float().to(ctx.device, non_blocking=True),
            ],
            dim=0,
        )

        model.set_v4_diagnostic_sink(None)
        reset_snn_state(model, ctx.functional)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_without_diagnostics = model(x)

        reset_snn_state(model, ctx.functional)
        model.set_v4_diagnostic_sink(accumulator)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_with_diagnostics = model(x)

        if not _outputs_torch_equal(
            output_without_diagnostics,
            output_with_diagnostics,
        ):
            model.set_v4_diagnostic_sink(None)
            raise RuntimeError(
                "V4 diagnostic on/off torch.equal check failed; performance "
                "results from this implementation are invalid"
            )
        model.set_v4_diagnostic_sink(None)
        reset_snn_state(model, ctx.functional)

    report = accumulator.finalize()
    relation_check = report["relation"]["conditional_point_invariant_pass"]
    if relation_check is False:
        raise RuntimeError(
            "PDCA conditional-point invariant exceeded 1e-6; implementation invalid"
        )
    report_path = os.path.join(args.output_dir, "v4_mechanism_diagnostics.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    if args.distributed:
        dist.barrier()
    return report


def checkpoint_dir(args):
    return args.output_dir


def mechanism_manifest_payload(args):
    return {
        "mechanism_profile": args.mechanism_profile,
        "mechanism_config_schema_version": args.mechanism_config_schema_version,
        "resolved_mechanism_config": args.resolved_mechanism_config,
        "resolved_mechanism_config_hash": args.resolved_mechanism_config_hash,
    }


def ensure_mechanism_manifest(args):
    if args.mechanism_profile is None:
        return
    manifest_path = os.path.join(args.output_dir, "mechanism_manifest.json")
    expected = mechanism_manifest_payload(args)
    if is_main_process(args):
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != expected:
                raise RuntimeError(
                    "output directory mechanism manifest mismatch: %s"
                    % manifest_path
                )
        else:
            if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
                raise RuntimeError(
                    "profile-based run refuses non-empty output directory without "
                    "a matching mechanism_manifest.json: %s" % args.output_dir
                )
            os.makedirs(args.output_dir, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(expected, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
    if args.distributed:
        dist.barrier()


def checkpoint_payload(ctx, epoch):
    payload = {
        "epoch": epoch,
        "model": unwrap_model(ctx.model).state_dict(),
        "optimizer": ctx.optimizer.state_dict(),
        "scaler": ctx.scaler.state_dict() if ctx.args.amp else None,
        "scheduler": scheduler_state_dict(ctx.scheduler),
        "best_metric": ctx.best_metric,
        "best_epoch": ctx.best_epoch,
        "global_update_step": ctx.global_update_step,
        "args": vars(ctx.args),
        "raw_argv": list(ctx.args.raw_argv),
        "mechanism_profile": ctx.args.mechanism_profile,
        "mechanism_config_schema_version": (
            ctx.args.mechanism_config_schema_version
        ),
        "resolved_mechanism_config": ctx.args.resolved_mechanism_config,
        "resolved_mechanism_config_hash": (
            ctx.args.resolved_mechanism_config_hash
        ),
        "full_run_config": dict(vars(ctx.args)),
        "pretrained_checkpoint_path": ctx.args.pretrained_checkpoint_path,
        "pretrained_checkpoint_sha256": (
            ctx.args.pretrained_checkpoint_sha256
        ),
        "git_commit": ctx.args.git_commit,
        "git_dirty": ctx.args.git_dirty,
    }
    return payload


def save_checkpoint(ctx, epoch, val_stats):
    args = ctx.args
    if not is_main_process(args):
        return
    os.makedirs(checkpoint_dir(args), exist_ok=True)
    score = float(val_stats["score"])
    # Strict comparison preserves the earliest epoch on an exact Score tie.
    is_best = score > float(ctx.best_metric)
    if is_best:
        ctx.best_metric = score
        ctx.best_epoch = int(epoch)
    torch.save(checkpoint_payload(ctx, epoch), os.path.join(checkpoint_dir(args), "latest.pth"))
    if is_best:
        torch.save(
            checkpoint_payload(ctx, epoch),
            os.path.join(checkpoint_dir(args), "best.pth"),
        )


def _validate_resume_mechanism_metadata(checkpoint, args):
    if args.mechanism_profile is None:
        return
    if not isinstance(checkpoint, dict):
        raise RuntimeError("profile-based resume requires a metadata checkpoint")
    required = (
        "mechanism_profile",
        "mechanism_config_schema_version",
        "resolved_mechanism_config",
        "resolved_mechanism_config_hash",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise RuntimeError(
            "profile-based resume checkpoint lacks mechanism metadata: %s"
            % ", ".join(missing)
        )
    expected = mechanism_manifest_payload(args)
    actual = {name: checkpoint[name] for name in required}
    if actual != expected:
        raise RuntimeError(
            "resume mechanism mismatch: expected %s, got %s"
            % (
                json.dumps(expected, sort_keys=True),
                json.dumps(actual, sort_keys=True),
            )
        )


def load_checkpoint(ctx, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_resume_mechanism_metadata(checkpoint, ctx.args)
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    unwrap_model(ctx.model).load_state_dict(state_dict, strict=True)
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            ctx.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None and ctx.args.amp:
            ctx.scaler.load_state_dict(checkpoint["scaler"])
        ctx.best_metric = float(checkpoint.get("best_metric", 0.0))
        ctx.best_epoch = int(checkpoint.get("best_epoch", -1))
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


def load_selected_best_model(ctx):
    best_path = os.path.join(checkpoint_dir(ctx.args), "best.pth")
    if ctx.args.distributed:
        dist.barrier()
    if not os.path.exists(best_path):
        if ctx.args.mechanism_profile is None:
            return False
        raise FileNotFoundError("selected checkpoint is missing: %s" % best_path)
    checkpoint = torch.load(best_path, map_location="cpu")
    _validate_resume_mechanism_metadata(checkpoint, ctx.args)
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    unwrap_model(ctx.model).load_state_dict(state_dict, strict=True)
    ctx.best_metric = float(checkpoint.get("best_metric", ctx.best_metric))
    ctx.best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1)))
    if ctx.args.distributed:
        dist.barrier()
    return True


def make_log_path(args):
    if args.log_dir is None:
        args.log_dir = args.output_dir
    os.makedirs(args.log_dir, exist_ok=True)
    return os.path.join(args.log_dir, "log.jsonl")


def write_epoch_log(log_path, payload):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    resolve_mechanism_config(args, raw_argv)
    init_distributed_mode(args)
    validate_args(args)
    resolve_pretrained_provenance(args)
    resolve_git_provenance(args)
    ensure_mechanism_manifest(args)
    seed_everything(args.seed, args.rank)
    if is_main_process(args) and args.mechanism_profile is not None:
        print(
            "RESOLVED_MECHANISM_CONFIG "
            + json.dumps(
                mechanism_manifest_payload(args),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            "PRETRAINED_PROVENANCE "
            + json.dumps(
                {
                    "path": args.pretrained_checkpoint_path,
                    "sha256": args.pretrained_checkpoint_sha256,
                    "git_commit": args.git_commit,
                    "git_dirty": args.git_dirty,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    from spikingjelly.clock_driven import functional
    from utils.metric_V1 import IOUandSek

    try:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:%d" % args.local_rank if torch.cuda.is_available() else "cpu")
        RS, trainset, trainloader, train_sampler, valloader = build_dataloaders(args)
        model = build_model(args, RS, device)
        optimizer = build_timm_optimizer(model, args)
        assert_head_optimizer_registration(model, optimizer, args)
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
            best_metric=float("-inf"),
            best_epoch=-1,
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
            if (
                args.v4_mechanism_diagnostics
                and ctx.start_epoch - 1 != ctx.best_epoch
            ):
                raise RuntimeError(
                    "V4 diagnostics must load the unique selected best.pth, "
                    "not a later latest checkpoint"
                )

        log_path = make_log_path(args) if is_main_process(args) else None
        if args.eval_only:
            val_stats = validate_t1_t3(ctx, ctx.start_epoch)
            all_pairs_stats = validate_all_pairs(ctx, ctx.start_epoch)
            _assert_metric_dict_exact(
                val_stats,
                all_pairs_stats["per_pair"]["t1_to_t3"],
                "selected validation t1-to-t3",
            )
            payload = {
                "epoch": ctx.start_epoch,
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "selected_validation_t1_t3": val_stats,
                "best_score": float(ctx.best_metric),
                "best_epoch": int(ctx.best_epoch),
                "selected_validation_all_pairs": all_pairs_stats,
            }
            if args.v4_mechanism_diagnostics:
                payload["v4_mechanism_diagnostics"] = (
                    run_v4_mechanism_diagnostics(ctx)
                )
            if is_main_process(args):
                print("EPOCH_SUMMARY " + json.dumps(payload, ensure_ascii=False), flush=True)
            return

        start_time = datetime.now()
        for epoch in range(ctx.start_epoch, args.epochs):
            train_stats = train_one_epoch(ctx, epoch)
            val_stats = validate_t1_t3(ctx, epoch)
            step_scheduler_epoch(ctx, epoch, val_stats["score"] if val_stats else None)
            save_checkpoint(ctx, epoch, val_stats)

            if is_main_process(args):
                payload = {
                    "epoch": epoch,
                    "train": train_stats,
                    "val": val_stats,
                    "best_score": float(ctx.best_metric),
                    "best_epoch": int(ctx.best_epoch),
                }
                print("EPOCH_SUMMARY " + json.dumps(payload, ensure_ascii=False), flush=True)
                write_epoch_log(log_path, payload)

        selected_loaded = load_selected_best_model(ctx)
        if selected_loaded:
            selected_t1_t3 = validate_t1_t3(ctx, ctx.best_epoch)
            selected_all_pairs = validate_all_pairs(ctx, ctx.best_epoch)
            _assert_metric_dict_exact(
                selected_t1_t3,
                selected_all_pairs["per_pair"]["t1_to_t3"],
                "selected validation t1-to-t3",
            )
            if is_main_process(args):
                selected_payload = {
                    "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                    "best_epoch": int(ctx.best_epoch),
                    "best_score": float(ctx.best_metric),
                    "selected_validation_t1_t3": selected_t1_t3,
                    "selected_validation_all_pairs": selected_all_pairs,
                }
                print(
                    "SELECTED_VALIDATION_SUMMARY "
                    + json.dumps(selected_payload, ensure_ascii=False),
                    flush=True,
                )
                write_epoch_log(log_path, selected_payload)

        if is_main_process(args):
            print("Training time: %s" % (datetime.now() - start_time), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
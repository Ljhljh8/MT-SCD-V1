#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean WUSU MTSCD training entrypoint.

Design:
- DDP + AMP + gradient accumulation.
- Three-phase input: x = stack([img1,img2,img3], dim=0), shape [3,B,4,H,W].
- Model path: backbone -> FDPC encoder dendritic + PDCA -> PDCA-guided pair decoder.
- The training file only receives final outputs: seg1, seg2, seg3, change13.
- No PairRelAux, no PDCA-RAS, no pair-gate debug, no per-iteration TensorBoard/text logging.
- One JSON line per epoch with train summary and validation metrics.

Expected to be used after the small model/PDCA cleanup patches described in the answer.
It can still run with the current model as long as the current imports are available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


class FloatTupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Clean WUSU GSTMSCD DDP AMP Training")

    # dataset / runtime
    parser.add_argument("--data_name", "--data-name", dest="data_name", default="WUSU")
    parser.add_argument("--Net_name", "--net-name", dest="Net_name", default="GSTMSCD")
    parser.add_argument("--data_root", "--data-root", dest="data_root", default=None)
    parser.add_argument("--backbone", default="sdtv2")
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")

    # output
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="checkpoints_clean")
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", default=None)
    parser.add_argument("--save_latest", "--save-latest", dest="save_latest", type=str2bool, default=True)

    # checkpoint
    parser.add_argument("--pretrain_from", "--pretrain-from", dest="pretrain_from", default=None)
    parser.add_argument("--resume", default=None)

    # model switches retained for the current experiment
    parser.add_argument("--relation-mode", choices=["pdca", "none"], default="pdca")
    parser.add_argument("--use-pdca-guided-pair-decoder", action="store_true")
    parser.add_argument("--no-pdca-guidance", action="store_true")
    parser.add_argument("--no-detach-pdca-guidance", action="store_true")

    # dendritic prior retained; stats are disabled by default
    parser.add_argument(
        "--pdca-dend-prior-mode",
        default="offset_dual",
        choices=["none", "source", "offset_sim", "offset_dual"],
    )
    parser.add_argument("--pdca-dend-prior-alpha", type=float, default=1e-3)
    parser.add_argument("--pdca-dend-prior-detach", type=str2bool, default=True)
    parser.add_argument(
        "--pdca-dend-prior-descriptor",
        default="mean_std",
        choices=["mean", "mean_std", "raw"],
    )
    parser.add_argument(
        "--pdca-dend-prior-normalize",
        default="zscore",
        choices=["none", "zscore"],
    )
    parser.add_argument("--pdca-dend-prior-sim-weight", type=float, default=1.0)
    parser.add_argument("--pdca-dend-prior-diff-weight", type=float, default=0.25)
    parser.add_argument("--pdca-dend-prior-use-conf-gate", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-conf-beta", type=float, default=4.0)
    parser.add_argument("--pdca-dend-prior-conf-tau", type=float, default=0.10)
    parser.add_argument("--pdca-dend-prior-affect-null", type=str2bool, default=False)
    parser.add_argument("--pdca-dend-prior-stats", type=str2bool, default=False)

    # optimizer
    parser.add_argument("--opt", default="adamp")
    parser.add_argument("--opt-eps", default=None, type=float)
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", action=FloatTupleAction)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--filter-bias-and-bn", dest="filter_bias_and_bn", action="store_true")
    parser.add_argument("--no-filter-bias-and-bn", dest="filter_bias_and_bn", action="store_false")
    parser.set_defaults(filter_bias_and_bn=False)

    # simple poly schedule
    parser.add_argument("--sched", choices=["poly", "none"], default="poly")
    parser.add_argument("--sched-on-updates", dest="sched_on_updates", action="store_true")
    parser.add_argument("--sched-on-epochs", dest="sched_on_updates", action="store_false")
    parser.set_defaults(sched_on_updates=True)
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=5.0)
    parser.add_argument("--poly-power", dest="poly_power", type=float, default=1.0)

    # training control
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum_steps", "--accum-steps", dest="accum_steps", type=int, default=1)
    parser.add_argument("--sync_bn", "--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument("--grad_clip_norm", "--grad-clip-norm", "--clip-grad", dest="grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--find_unused_parameters", "--find-unused-parameters", dest="find_unused_parameters", action="store_true")
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.set_defaults(find_unused_parameters=False)

    # distributed
    parser.add_argument("--dist_url", "--dist-url", dest="dist_url", default="env://")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", "--world-size", dest="world_size", type=int, default=1)

    return parser


def validate_args(args) -> None:
    if args.use_pdca_guided_pair_decoder and not args.no_pdca_guidance and args.relation_mode != "pdca":
        raise ValueError("--use-pdca-guided-pair-decoder requires --relation-mode pdca unless --no-pdca-guidance is set.")
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr < 0 or args.weight_decay < 0 or args.grad_clip_norm < 0:
        raise ValueError("lr, weight_decay and grad_clip_norm must be non-negative")
    if args.opt_betas is not None and len(args.opt_betas) not in (2, 3):
        raise ValueError("--opt-betas must contain two values for Adam/AdamP or three values for Adan")


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(args) -> bool:
    return int(getattr(args, "rank", 0)) == 0


def init_distributed_mode(args) -> None:
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


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reset_snn_state(model: nn.Module, functional_module) -> None:
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


def load_model_weights(model: nn.Module, path: str, strict: bool = False):
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

    return RS, trainset, trainloader, train_sampler, valloader


def build_model(args, RS, device):
    from models.GSTMSCD_MTSCD_Snn_ForDecoder import GSTMSCD_WUSU as Net

    model = Net(
        args.backbone,
        args.pretrained,
        len(RS.ST_CLASSES),
        relation_mode=args.relation_mode,
        use_pdca_relation_aux=False,
        use_pairrel_aux=False,
        use_pdca_guided_pair_decoder=args.use_pdca_guided_pair_decoder,
        detach_pdca_guidance=not args.no_detach_pdca_guidance,
        use_pdca_guidance=not args.no_pdca_guidance,
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
    )

    if args.pretrain_from:
        incompatible = load_model_weights(model, args.pretrain_from, strict=False)
        if is_main_process(args):
            print(f"Loaded non-strict pretrain weights: {incompatible}")

    model.to(device)

    if args.sync_bn and args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process(args):
            print("Converted BatchNorm layers to SyncBatchNorm.")

    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )

    return model


def build_optimizer(model: nn.Module, args):
    from timm.optim import create_optimizer_v2, optimizer_kwargs

    kwargs = optimizer_kwargs(cfg=args)
    kwargs["filter_bias_and_bn"] = bool(args.filter_bias_and_bn)
    return create_optimizer_v2(unwrap_model(model), **kwargs)


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def poly_lr(args, update_idx: int, total_updates: int, warmup_updates: int) -> float:
    if args.sched == "none":
        return float(args.lr)
    update_idx = max(0, int(update_idx))
    total_updates = max(1, int(total_updates))
    warmup_updates = max(0, int(warmup_updates))

    if warmup_updates > 0 and update_idx < warmup_updates:
        return float(args.lr) * float(update_idx + 1) / float(warmup_updates)

    denom = max(1, total_updates - warmup_updates)
    progress = min(1.0, max(0.0, float(update_idx - warmup_updates) / float(denom)))
    return float(args.min_lr) + (float(args.lr) - float(args.min_lr)) * ((1.0 - progress) ** float(args.poly_power))


def build_loss_functions():
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
    from utils.loss import ChangeSimilarity, DiceLoss

    return {
        "seg": CrossEntropyLoss(ignore_index=-1),
        "bce": BCEWithLogitsLoss(reduction="none"),
        "dice": DiceLoss(activation="none"),
        "similarity": ChangeSimilarity(),
    }


def compute_binary_change_loss(change_logits, mask_bn, criterion_bce, criterion_dice):
    logits = change_logits.float()
    target = mask_bn.float()
    loss_bce = criterion_bce(logits, target)
    loss_bce[target == 1] *= 2
    loss_bce = loss_bce.mean()
    loss_dice = criterion_dice(torch.sigmoid(logits), target)
    return loss_bce + loss_dice


def compute_losses(outputs, masks, criteria):
    out1, out2, out3, change_logits = outputs
    mask1, mask2, mask3, mask_bn = masks

    loss1 = criteria["seg"](out1.float(), mask1 - 1)
    loss2 = criteria["seg"](out2.float(), mask2 - 1)
    loss3 = criteria["seg"](out3.float(), mask3 - 1)
    loss_seg = (loss1 + loss2 + loss3) / 3.0

    loss_bn = compute_binary_change_loss(
        change_logits,
        mask_bn,
        criteria["bce"],
        criteria["dice"],
    )
    loss_similarity = criteria["similarity"](out1.float(), out3.float(), mask_bn)
    loss = loss_seg + loss_bn + loss_similarity

    return loss, {
        "loss": float(loss.detach()),
        "seg": float(loss_seg.detach()),
        "bn": float(loss_bn.detach()),
        "similarity": float(loss_similarity.detach()),
    }


def should_update_optimizer(step: int, num_steps: int, accum_steps: int) -> bool:
    return ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)


def train_one_epoch(ctx, epoch: int) -> Dict[str, float]:
    args = ctx["args"]
    model = ctx["model"]
    device = ctx["device"]

    if ctx["train_sampler"] is not None:
        ctx["train_sampler"].set_epoch(epoch)

    model.train()
    ctx["optimizer"].zero_grad(set_to_none=True)

    totals = {"loss": 0.0, "seg": 0.0, "bn": 0.0, "similarity": 0.0}
    total_steps = len(ctx["trainloader"])

    for step, batch in enumerate(ctx["trainloader"]):
        img1, img2, img3, mask1, mask2, mask3, mask_bn, _sample_id = batch

        img1 = img1.float().to(device, non_blocking=True)
        img2 = img2.float().to(device, non_blocking=True)
        img3 = img3.float().to(device, non_blocking=True)
        mask1 = mask1.long().to(device, non_blocking=True)
        mask2 = mask2.long().to(device, non_blocking=True)
        mask3 = mask3.long().to(device, non_blocking=True)
        mask_bn = mask_bn.float().to(device, non_blocking=True)

        x = torch.stack([img1, img2, img3], dim=0)
        update_now = should_update_optimizer(step, total_steps, args.accum_steps)
        sync_ctx = model.no_sync() if args.distributed and not update_now else nullcontext()

        with sync_ctx:
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(x)  # important: no return_aux=True in clean training

            with torch.cuda.amp.autocast(enabled=False):
                loss, loss_stats = compute_losses(
                    outputs,
                    (mask1, mask2, mask3, mask_bn),
                    ctx["criteria"],
                )
                if not torch.isfinite(loss.detach()).all():
                    raise FloatingPointError(f"non-finite loss at epoch={epoch}, step={step}")
                loss_to_backward = loss / float(args.accum_steps)

            if args.amp:
                ctx["scaler"].scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        if update_now:
            if args.amp:
                ctx["scaler"].unscale_(ctx["optimizer"])
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip_norm)

            if args.amp:
                ctx["scaler"].step(ctx["optimizer"])
                ctx["scaler"].update()
            else:
                ctx["optimizer"].step()

            ctx["global_update_step"] += 1
            if args.sched_on_updates:
                lr = poly_lr(args, ctx["global_update_step"], ctx["total_updates"], ctx["warmup_updates"])
                set_lr(ctx["optimizer"], lr)

            ctx["optimizer"].zero_grad(set_to_none=True)

        reset_snn_state(model, ctx["functional"])

        for k, v in loss_stats.items():
            totals[k] += v

    denom = max(1, total_steps)
    stats = {k: v / denom for k, v in totals.items()}
    stats["lr"] = float(ctx["optimizer"].param_groups[0]["lr"])
    return stats


@torch.no_grad()
def validate_t1_t3(ctx, epoch: int) -> Dict[str, float]:
    args = ctx["args"]

    if not is_main_process(args):
        if args.distributed:
            dist.barrier()
        return {}

    model = unwrap_model(ctx["model"])
    model.eval()

    metric = ctx["IOUandSek"](num_classes=len(ctx["RS"].ST_CLASSES))
    device = ctx["device"]

    for batch in ctx["valloader"]:
        img1, img2, img3, mask1, _mask2, mask3, mask_bn, _sample_id = batch

        img1 = img1.float().to(device, non_blocking=True)
        img2 = img2.float().to(device, non_blocking=True)
        img3 = img3.float().to(device, non_blocking=True)
        x = torch.stack([img1, img2, img3], dim=0)

        with torch.cuda.amp.autocast(enabled=args.amp):
            out1, _out2, out3, change_logits = model(x)

        out1 = torch.argmax(out1.float(), dim=1).cpu().numpy() + 1
        out3 = torch.argmax(out3.float(), dim=1).cpu().numpy() + 1
        out_bn = (torch.sigmoid(change_logits.float()) > 0.5).cpu().numpy().astype(np.uint8)

        mask1 = mask1.clone()
        mask3 = mask3.clone()
        mask1[mask_bn == 0] = 0
        mask3[mask_bn == 0] = 0
        out1[out_bn == 0] = 0
        out3[out_bn == 0] = 0

        metric.add_batch(out1, mask1.numpy())
        metric.add_batch(out3, mask3.numpy())

        reset_snn_state(model, ctx["functional"])

    _, score, miou, sek, fscd, oa, sc_precision, sc_recall = metric.evaluate_SECOND()
    stats = {
        "score": float(score),
        "miou": float(miou),
        "sek": float(sek),
        "Fscd": float(fscd),
        "OA": float(oa),
        "SC_Precision": float(sc_precision),
        "SC_Recall": float(sc_recall),
    }

    save_checkpoint(ctx, epoch, stats)

    if args.distributed:
        dist.barrier()

    return stats


def checkpoint_dir(args) -> str:
    return os.path.join(args.output_dir, args.data_name, args.Net_name, args.backbone)


def checkpoint_payload(ctx, epoch: int) -> Dict:
    return {
        "epoch": epoch,
        "model": unwrap_model(ctx["model"]).state_dict(),
        "optimizer": ctx["optimizer"].state_dict(),
        "scaler": ctx["scaler"].state_dict() if ctx["args"].amp else None,
        "best_metric": ctx["best_metric"],
        "global_update_step": ctx["global_update_step"],
        "args": vars(ctx["args"]),
    }


def save_checkpoint(ctx, epoch: int, val_stats: Dict[str, float]) -> None:
    args = ctx["args"]
    if not is_main_process(args):
        return

    os.makedirs(checkpoint_dir(args), exist_ok=True)
    score = float(val_stats["score"])
    is_best = score >= float(ctx["best_metric"])
    if is_best:
        ctx["best_metric"] = score

    if args.save_latest:
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


def load_checkpoint(ctx, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    incompatible = unwrap_model(ctx["model"]).load_state_dict(state_dict, strict=False)

    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        ctx["optimizer"].load_state_dict(checkpoint["optimizer"])
    if ctx["args"].amp and checkpoint.get("scaler") is not None:
        ctx["scaler"].load_state_dict(checkpoint["scaler"])

    ctx["best_metric"] = float(checkpoint.get("best_metric", 0.0))
    ctx["start_epoch"] = int(checkpoint.get("epoch", -1)) + 1
    ctx["global_update_step"] = int(checkpoint.get("global_update_step", 0))

    if is_main_process(ctx["args"]):
        print(f"Resumed checkpoint from {checkpoint_path}: {incompatible}", flush=True)


def make_log_path(args) -> str:
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, args.data_name, args.Net_name, args.backbone)
    os.makedirs(args.log_dir, exist_ok=True)
    return os.path.join(args.log_dir, "log.jsonl")


def write_epoch_log(log_path: str, payload: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    init_distributed_mode(args)
    validate_args(args)
    seed_everything(args.seed, args.rank)

    from spikingjelly.clock_driven import functional
    from utils.metric import IOUandSek

    try:
        torch.backends.cudnn.benchmark = True

        device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
        RS, trainset, trainloader, train_sampler, valloader = build_dataloaders(args)
        model = build_model(args, RS, device)
        optimizer = build_optimizer(model, args)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        criteria = build_loss_functions()

        updates_per_epoch = max(1, math.ceil(len(trainloader) / args.accum_steps))
        total_updates = max(1, updates_per_epoch * args.epochs)
        warmup_updates = int(max(0, args.warmup_epochs) * updates_per_epoch)

        ctx = {
            "args": args,
            "RS": RS,
            "IOUandSek": IOUandSek,
            "functional": functional,
            "device": device,
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "criteria": criteria,
            "trainloader": trainloader,
            "train_sampler": train_sampler,
            "valloader": valloader,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "global_update_step": 0,
            "start_epoch": 0,
            "best_metric": 0.0,
        }

        if is_main_process(args):
            n_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
            print("args:", args, flush=True)
            print(
                "params_M=%.2f, batch_per_gpu=%d, world_size=%d, accum_steps=%d, effective_batch=%d, updates_per_epoch=%d"
                % (
                    n_params / 1e6,
                    args.batch_size,
                    args.world_size,
                    args.accum_steps,
                    args.batch_size * args.world_size * args.accum_steps,
                    updates_per_epoch,
                ),
                flush=True,
            )

        if args.resume:
            load_checkpoint(ctx, args.resume)

        log_path = make_log_path(args) if is_main_process(args) else None

        if args.eval_only:
            val_stats = validate_t1_t3(ctx, ctx["start_epoch"])
            if is_main_process(args):
                print("VAL_EPOCH_SUMMARY " + json.dumps(val_stats, ensure_ascii=False), flush=True)
            return

        start_time = datetime.now()
        for epoch in range(ctx["start_epoch"], args.epochs):
            if not args.sched_on_updates:
                lr = poly_lr(args, epoch, args.epochs, int(args.warmup_epochs))
                set_lr(optimizer, lr)

            train_stats = train_one_epoch(ctx, epoch)
            val_stats = validate_t1_t3(ctx, epoch)

            if is_main_process(args):
                payload = {
                    "epoch": epoch,
                    "train": train_stats,
                    "val": val_stats,
                    "best_score": float(ctx["best_metric"]),
                }
                print("EPOCH_SUMMARY " + json.dumps(payload, ensure_ascii=False), flush=True)
                write_epoch_log(log_path, payload)

        if is_main_process(args):
            print("Training time:", str(datetime.now() - start_time), flush=True)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

import argparse
import contextlib
import math
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import BCELoss, CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import datasets.MultiSiamese_RS_ST_TL as RS
from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU as Net
from spikingjelly.clock_driven import functional
from utils.loss import ChangeSimilarity, DiceLoss
from utils.metric import IOUandSek


working_path = os.path.dirname(os.path.abspath(__file__))


class TemporalLogicKLDivLoss(nn.Module):
    def __init__(
        self,
        margin_kl=0.5,
        margin_consistent=0.2,
        temperature=1.0,
        reduction="mean",
        epsilon=1e-8,
    ):
        super().__init__()
        self.margin_kl = margin_kl
        self.margin_cons = margin_consistent
        self.temp = temperature
        self.reduction = reduction
        self.eps = epsilon

    def kl_divergence(self, p, q):
        p = p.clamp(min=self.eps)
        q = q.clamp(min=self.eps)
        kl_pq = F.kl_div(q.log(), p, reduction="none").sum(dim=1)
        kl_qp = F.kl_div(p.log(), q, reduction="none").sum(dim=1)
        return (kl_pq + kl_qp) / 2

    def forward(self, feat_t1, feat_t2, feat_t3):
        kl_12 = self.kl_divergence(feat_t1, feat_t2)
        kl_23 = self.kl_divergence(feat_t2, feat_t3)
        kl_13 = self.kl_divergence(feat_t1, feat_t3)

        mask_rule1 = (kl_12 > self.margin_kl) & (kl_23 <= self.margin_kl)
        mask_rule2 = (kl_12 <= self.margin_kl) & (kl_23 > self.margin_kl)
        mask_rule3 = (kl_12 <= self.margin_kl) & (kl_23 <= self.margin_kl)

        loss_rule1 = torch.where(
            mask_rule1,
            (self.margin_cons - kl_13).clamp(min=0),
            torch.zeros_like(kl_12),
        )
        loss_rule2 = torch.where(
            mask_rule2,
            (self.margin_cons - kl_13).clamp(min=0),
            torch.zeros_like(kl_12),
        )
        loss_rule3 = torch.where(
            mask_rule3,
            (kl_13 - self.margin_cons).clamp(min=0),
            torch.zeros_like(kl_12),
        )

        total_loss = loss_rule1 + loss_rule2 + loss_rule3
        if self.reduction == "mean":
            return total_loss.mean()
        if self.reduction == "sum":
            return total_loss.sum()
        return total_loss


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser("Semantic Change Detection DDP Training")
    parser.add_argument("--data_name", type=str, default="WUSU")
    parser.add_argument("--Net_name", type=str, default="GSTMSCD")
    parser.add_argument("--backbone", type=str, default="GOST-Mamba")
    parser.add_argument("--data_root", type=str, default="/WUSU")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--val-batch-size", "--val_batch_size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--test-batch-size", "--test_batch_size", dest="test_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--lightweight", dest="lightweight", action="store_true")
    parser.add_argument("--pretrain_from", type=str, default=None, help="Load model weights before training.")
    parser.add_argument("--load_from", type=str, default=None, help="Load model weights for validation or finetuning.")
    parser.add_argument("--resume", type=str, default=None, help="Resume a full training checkpoint.")
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--tta", dest="tta", action="store_true")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--save_mask", dest="save_mask", action="store_true")
    parser.add_argument("--use_pseudo_label", dest="use_pseudo_label", action="store_true")
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--Lambda", type=float, default=0.00005)

    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--sync-bn", action="store_true", help="Convert BN to SyncBatchNorm in DDP mode.")
    parser.add_argument("--freeze-bn", action="store_true", help="Freeze BN running stats and affine params.")
    parser.add_argument(
        "--find-unused-parameters",
        dest="find_unused_parameters",
        action="store_true",
        default=True,
        help="Use DDP unused-parameter detection. Default is True for this model.",
    )
    parser.add_argument(
        "--no-find-unused-parameters",
        dest="find_unused_parameters",
        action="store_false",
        help="Disable DDP unused-parameter detection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        args.distributed = args.world_size > 1
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training with NCCL requires CUDA.")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


@contextlib.contextmanager
def suppress_stdout(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed, rank):
    def _worker_init_fn(worker_id):
        worker_seed = seed + rank * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    dropped_zero_size = []
    for key, value in state_dict.items():
        if key in model_state:
            filtered[key] = value
        elif torch.is_tensor(value) and value.numel() == 0:
            dropped_zero_size.append(key)
        else:
            filtered[key] = value
    if dropped_zero_size and is_main_process():
        print(f"Dropped zero-size checkpoint keys not present in model: {dropped_zero_size}")
    return filtered


def load_model_weights(model, path, strict=True):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    state_dict = filter_state_dict_for_model(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if is_main_process() and not strict:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing:
            print(f"Missing keys when loading {path}: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys when loading {path}: {len(unexpected)}")


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def freeze_batchnorm(module):
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            for param in child.parameters(recurse=False):
                param.requires_grad = False


def reset_snn_state(model):
    functional.reset_net(model)


def build_optimizer(model, lr, weight_decay):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    return AdamW(param_groups, lr=lr, weight_decay=weight_decay)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.rank = get_rank()
        self.device = torch.device(
            f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.main_process = is_main_process()

        args.log_dir = args.log_dir or os.path.join(
            working_path, "logs", args.data_name, args.Net_name, args.backbone
        )
        self.output_dir = os.path.join(
            working_path, args.output_dir, args.data_name, args.Net_name, args.backbone
        )
        if self.main_process:
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            self.writer = SummaryWriter(args.log_dir)
        else:
            self.writer = None

        with suppress_stdout(not self.main_process):
            trainset = RS.Data(mode="train", random_flip=True)
            valset = RS.Data(mode="val", random_flip=True) if self.main_process else None

        self.train_sampler = None
        if args.distributed:
            self.train_sampler = DistributedSampler(
                trainset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                drop_last=True,
            )

        self.trainloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            pin_memory=False,
            num_workers=args.workers,
            drop_last=True,
            worker_init_fn=make_worker_init_fn(args.seed, self.rank),
        )
        self.valloader = None
        if self.main_process:
            self.valloader = DataLoader(
                valset,
                batch_size=args.val_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers,
                drop_last=False,
                worker_init_fn=make_worker_init_fn(args.seed, 0),
            )

        with suppress_stdout(not self.main_process):
            model = Net(
                args.backbone,
                args.pretrained,
                len(RS.ST_CLASSES),
                args.lightweight,
                args.M,
                args.Lambda,
            )

        if args.pretrain_from:
            load_model_weights(model, args.pretrain_from, strict=False)
        if args.load_from:
            load_model_weights(model, args.load_from, strict=True)

        model = model.to(self.device)

        if args.sync_bn and args.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if self.main_process:
                print("Converted BatchNorm layers to SyncBatchNorm.")

        if args.freeze_bn:
            freeze_batchnorm(model)
            if self.main_process:
                print("Frozen BatchNorm running stats and affine parameters.")

        self.model_without_ddp = model
        if args.distributed:
            self.model = DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )
        else:
            self.model = model

        self.TCL = TemporalLogicKLDivLoss()
        self.criterion_seg = CrossEntropyLoss(ignore_index=-1)
        self.criterion_bn = BCELoss(reduction="none")
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()

        self.optimizer = build_optimizer(self.model, args.lr, args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self.total_updates_per_epoch = max(1, math.ceil(len(self.trainloader) / args.accum_steps))
        self.total_update_steps = max(1, self.total_updates_per_epoch * args.epochs)
        self.update_steps = 0
        self.previous_best = 0.0
        self.start_epoch = 0

        if args.resume:
            self.resume(args.resume)

    def resume(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = strip_module_prefix(extract_model_state(checkpoint))
        state_dict = filter_state_dict_for_model(self.model_without_ddp, state_dict)
        self.model_without_ddp.load_state_dict(state_dict, strict=True)

        if isinstance(checkpoint, dict):
            if checkpoint.get("optimizer") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                move_optimizer_state_to_device(self.optimizer, self.device)
            if checkpoint.get("scaler") is not None and self.args.amp:
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.previous_best = float(checkpoint.get("best_metric", 0.0))
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.update_steps = int(checkpoint.get("update_steps", 0))

        if self.main_process:
            print(f"Resumed checkpoint from {path}, start_epoch={self.start_epoch}")

    def compute_lr(self):
        if self.args.warmup:
            warmup_steps = self.total_update_steps / 5
            if warmup_steps and self.update_steps < warmup_steps:
                lr = self.args.lr * (self.update_steps / warmup_steps)
            else:
                progress = min(1.0, float(self.update_steps) / self.total_update_steps)
                lr = self.args.lr * (1.0 - progress) ** 1.5
        else:
            progress = min(1.0, float(self.update_steps) / self.total_update_steps)
            lr = self.args.lr * (1.0 - progress) ** 1.5
        return lr

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def compute_losses(self, out1, out2, out3, out_bn, mask1, mask2, mask3, mask_bn):
        out1 = out1.float()
        out2 = out2.float()
        out3 = out3.float()
        out_bn = out_bn.float()

        loss1 = self.criterion_seg(out1, mask1 - 1)
        loss2 = self.criterion_seg(out2, mask2 - 1)
        loss3 = self.criterion_seg(out3, mask3 - 1)
        loss_seg = (loss1 + loss2 + loss3) / 3

        loss_similarity = self.criterion_sc(out1[:, 0:], out3[:, 0:], mask_bn)

        loss_bn_1 = self.criterion_bn(out_bn, mask_bn)
        loss_bn_1[mask_bn == 1] *= 2
        loss_bn_1 = loss_bn_1.mean()
        loss_bn_2 = self.criterion_bn_2(out_bn, mask_bn)
        loss_bn = loss_bn_1 + loss_bn_2

        loss = loss_bn + loss_seg + loss_similarity
        return loss, loss_seg, loss_bn, loss_similarity

    def training(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        if self.args.freeze_bn:
            freeze_batchnorm(self.model_without_ddp)

        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0
        total_TCL = 0.0

        self.optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(self.trainloader, disable=not self.main_process)

        for step, batch in enumerate(iterator):
            img1, img2, img3, mask1, mask2, mask3, mask_bn, _ = batch
            img1 = img1.float().to(self.device, non_blocking=True)
            img2 = img2.float().to(self.device, non_blocking=True)
            img3 = img3.float().to(self.device, non_blocking=True)
            mask1 = mask1.long().to(self.device, non_blocking=True)
            mask2 = mask2.long().to(self.device, non_blocking=True)
            mask3 = mask3.long().to(self.device, non_blocking=True)
            mask_bn = mask_bn.float().to(self.device, non_blocking=True)
            x = torch.stack([img1, img2, img3], dim=1)

            is_last_step = (step + 1) == len(self.trainloader)
            need_update = ((step + 1) % self.args.accum_steps == 0) or is_last_step
            sync_context = (
                self.model.no_sync()
                if self.args.distributed and not need_update
                else contextlib.nullcontext()
            )

            with sync_context:
                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out1, out2, out3, out_bn = self.model(x)

                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_seg, loss_bn, loss_similarity = self.compute_losses(
                        out1, out2, out3, out_bn, mask1, mask2, mask3, mask_bn
                    )
                    backward_loss = loss / self.args.accum_steps

                reset_snn_state(self.model_without_ddp)
                if self.args.amp:
                    self.scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()

            total_loss += loss.item()
            total_loss_seg += loss_seg.item()
            total_loss_bn += loss_bn.item()
            total_loss_similarity += loss_similarity.item()

            if need_update:
                self.update_steps += 1
                lr = self.compute_lr()
                self.set_lr(lr)

                if self.args.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.main_process:
                seen = step + 1
                current_lr = self.optimizer.param_groups[0]["lr"]
                iterator.set_description(
                    "Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, "
                    "Similarity Loss: %.3f, TL Loss: %.3f"
                    % (
                        total_loss / seen,
                        total_loss_seg / seen,
                        total_loss_bn / seen,
                        total_loss_similarity / seen,
                        total_TCL / seen,
                    )
                )
                running_iter = epoch * len(self.trainloader) + seen
                self.writer.add_scalar("train total_loss", total_loss / seen, running_iter)
                self.writer.add_scalar("train seg_loss", total_loss_seg / seen, running_iter)
                self.writer.add_scalar("train bn_loss", total_loss_bn / seen, running_iter)
                self.writer.add_scalar("train sc_loss", total_loss_similarity / seen, running_iter)
                self.writer.add_scalar("train TL Loss", total_TCL / seen, running_iter)
                self.writer.add_scalar("lr", current_lr, running_iter)

    def validation(self, epoch):
        if not self.main_process:
            if self.args.distributed:
                dist.barrier()
            return None

        model = self.model_without_ddp
        model.eval()
        metric = IOUandSek(num_classes=len(RS.ST_CLASSES))
        iterator = tqdm(self.valloader)

        with torch.no_grad():
            for batch in iterator:
                img1, img2, img3, mask1, _, mask3, mask_bn, _ = batch
                img1 = img1.float().to(self.device, non_blocking=True)
                img2 = img2.float().to(self.device, non_blocking=True)
                img3 = img3.float().to(self.device, non_blocking=True)
                x = torch.stack([img1, img2, img3], dim=1)

                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out1, _, out3_logits, out_bn13 = model(x)

                out1 = torch.argmax(out1.float(), dim=1).cpu().numpy() + 1
                out3 = torch.argmax(out3_logits.float(), dim=1).cpu().numpy() + 1
                out_bn = (out_bn13.float() > 0.5).cpu().numpy().astype(np.uint8)

                mask_bn_np = mask_bn.numpy()
                mask1_np = mask1.numpy().copy()
                mask3_np = mask3.numpy().copy()

                out1[out_bn == 0] = 0
                out3[out_bn == 0] = 0
                mask1_np[mask_bn_np == 0] = 0
                mask3_np[mask_bn_np == 0] = 0

                metric.add_batch(out1, mask1_np)
                metric.add_batch(out3, mask3_np)
                reset_snn_state(model)

                change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = (
                    metric.evaluate_SECOND()
                )
                iterator.set_description(
                    "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, "
                    "SC_Precision: %.4f, SC_Recall: %.4f"
                    % (miou, sek, score, Fscd, OA, SC_Precision, SC_Recall)
                )

        change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = (
            metric.evaluate_SECOND()
        )

        is_best = score >= self.previous_best
        if is_best:
            self.previous_best = score
        self.save_checkpoint(
            epoch=epoch,
            metrics={
                "score": score,
                "miou": miou,
                "sek": sek,
                "Fscd": Fscd,
                "OA": OA,
                "SC_Precision": SC_Precision,
                "SC_Recall": SC_Recall,
                "change_ratio": change_ratio,
            },
            is_best=is_best,
        )

        self.writer.add_scalar("val_Score", score, epoch)
        self.writer.add_scalar("val_mIOU", miou, epoch)
        self.writer.add_scalar("val_Sek", sek, epoch)
        self.writer.add_scalar("val_Fscd", Fscd, epoch)
        self.writer.add_scalar("val_OA", OA, epoch)

        if self.args.distributed:
            dist.barrier()
        return score

    def save_checkpoint(self, epoch, metrics, is_best):
        checkpoint = {
            "epoch": epoch,
            "model": self.model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None,
            "scaler": self.scaler.state_dict() if self.args.amp else None,
            "best_metric": self.previous_best,
            "metrics": metrics,
            "update_steps": self.update_steps,
            "args": vars(self.args),
        }

        latest_path = os.path.join(self.output_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(
                self.output_dir,
                "epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth"
                % (
                    epoch,
                    metrics["score"] * 100,
                    metrics["miou"] * 100,
                    metrics["sek"] * 100,
                    metrics["Fscd"] * 100,
                    metrics["OA"] * 100,
                ),
            )
            torch.save(checkpoint, best_path)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    args = get_args()
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")

    init_distributed_mode(args)
    setup_seed(args.seed + get_rank())
    args.amp = bool(args.amp and torch.cuda.is_available())

    if is_main_process():
        print(args)
        if args.distributed:
            global_batch = args.batch_size * args.world_size
            effective_batch = global_batch * args.accum_steps
            print(
                f"DDP enabled: world_size={args.world_size}, "
                f"global_batch={global_batch}, effective_batch={effective_batch}"
            )

    trainer = Trainer(args)

    try:
        if args.load_from:
            trainer.validation(trainer.start_epoch)
            if args.eval_only:
                return

        for epoch in range(trainer.start_epoch, args.epochs):
            if is_main_process():
                lr = trainer.optimizer.param_groups[0]["lr"]
                print(
                    "\n==> Epoch %i, learning rate = %.6f, previous best = %.5f"
                    % (epoch, lr, trainer.previous_best)
                )
            trainer.training(epoch)
            trainer.validation(epoch)
    finally:
        trainer.close()
        cleanup_distributed()


if __name__ == "__main__":
    # Single GPU:
    #   python train_WUSU_ddp_accum.py --batch-size 2 --accum-steps 1
    #
    # 4 GPU DDP + SyncBN:
    #   torchrun --nproc_per_node=4 train_WUSU_ddp_accum.py --batch-size 2 --sync-bn
    #
    # 4 GPU DDP + SyncBN + gradient accumulation:
    #   torchrun --nproc_per_node=4 train_WUSU_ddp_accum.py --batch-size 2 --sync-bn --accum-steps 4
    main()

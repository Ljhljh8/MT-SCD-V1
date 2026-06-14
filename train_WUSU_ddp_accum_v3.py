import argparse
import math
import os
import random
from contextlib import nullcontext

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ModuleNotFoundError:
    torch = None
    dist = None
    nn = None
    F = None
    DistributedDataParallel = None
    DataLoader = None
    DistributedSampler = None


working_path = os.path.dirname(os.path.abspath(__file__))


if nn is not None:
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
else:
    class TemporalLogicKLDivLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to construct TemporalLogicKLDivLoss.")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser("WUSU GSTMSCD DDP SyncBN AMP Accum Training")

    parser.add_argument("--data_name", "--data-name", dest="data_name", type=str, default="WUSU")
    parser.add_argument("--Net_name", "--net-name", dest="Net_name", type=str, default="GSTMSCD")
    parser.add_argument("--backbone", type=str, default="sdtv2")
    parser.add_argument("--data_root", "--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", "--test-batch-size", dest="test_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="checkpoints")

    parser.add_argument("--lightweight", dest="lightweight", action="store_true")
    parser.add_argument("--pretrain_from", "--pretrain-from", dest="pretrain_from", type=str, default=None)
    parser.add_argument("--load_from", "--load-from", dest="load_from", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--tta", dest="tta", action="store_true")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--save_mask", "--save-mask", dest="save_mask", action="store_true")
    parser.add_argument("--use_pseudo_label", "--use-pseudo-label", dest="use_pseudo_label", action="store_true")
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--Lambda", type=float, default=0.00005)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum_steps", "--accum-steps", dest="accum_steps", type=int, default=1)
    parser.add_argument("--sync_bn", "--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument(
        "--change_output_api",
        "--change-output-api",
        dest="change_output_api",
        choices=("logits", "probability"),
        default="logits",
        help="Binary change head output API. Use probability only with legacy model code that still returns sigmoid probabilities.",
    )
    parser.add_argument(
        "--reference_batch_size",
        "--reference-batch-size",
        dest="reference_batch_size",
        type=int,
        default=None,
        help="Single-GPU reference batch size used to compute LR total update steps.",
    )
    parser.add_argument(
        "--reference_accum_steps",
        "--reference-accum-steps",
        dest="reference_accum_steps",
        type=int,
        default=1,
        help="Single-GPU reference accumulation steps used with --reference-batch-size.",
    )
    parser.add_argument(
        "--reference_total_updates",
        "--reference-total-updates",
        dest="reference_total_updates",
        type=int,
        default=None,
        help="Explicit LR schedule total optimizer updates; overrides --reference-batch-size.",
    )
    parser.add_argument(
        "--reference_warmup_updates",
        "--reference-warmup-updates",
        dest="reference_warmup_updates",
        type=float,
        default=None,
        help="Explicit warmup optimizer updates; defaults to one fifth of schedule total when warmup is enabled.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        "--grad-clip-norm",
        dest="grad_clip_norm",
        type=float,
        default=0.0,
        help="Clip gradient norm after AMP unscale; 0 disables clipping.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        "--find-unused-parameters",
        dest="find_unused_parameters",
        action="store_true",
        default=True,
        help="Current WUSU model has unused classification/transition heads; default keeps DDP robust.",
    )
    parser.add_argument(
        "--no-find-unused-parameters",
        dest="find_unused_parameters",
        action="store_false",
        help="Use only after confirming every trainable parameter participates in loss.",
    )
    parser.add_argument("--freeze_bn", "--freeze-bn", dest="freeze_bn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist_url", "--dist-url", dest="dist_url", type=str, default="env://")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", "--world-size", dest="world_size", type=int, default=1)
    parser.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")

    return parser


def should_update_optimizer(step, num_steps, accum_steps):
    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    return ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def compute_poly_warmup_lr(args, update_step, total_updates, warmup_updates):
    if total_updates <= 0:
        return args.lr
    update_step = min(max(int(update_step), 0), int(total_updates))
    if args.warmup and warmup_updates and update_step < warmup_updates:
        return args.lr * (float(update_step) / float(warmup_updates))
    progress = 1.0 - float(update_step) / float(total_updates)
    return args.lr * max(progress, 0.0) ** 1.5


def build_poly_warmup_lr_lambda(warmup_enabled, total_updates, warmup_updates):
    total_updates = max(1, int(total_updates))
    warmup_updates = float(warmup_updates or 0)

    def lr_lambda(step_index):
        update_step = min(max(int(step_index) + 1, 1), total_updates)
        if warmup_enabled and warmup_updates and update_step < warmup_updates:
            return float(update_step) / float(warmup_updates)
        progress = 1.0 - float(update_step) / float(total_updates)
        return max(progress, 0.0) ** 1.5

    return lr_lambda


def resolve_reference_update_counts(args, dataset_len, actual_total_updates):
    if args.reference_total_updates is not None:
        total_updates = int(args.reference_total_updates)
    elif args.reference_batch_size is not None:
        reference_micro_batches = int(dataset_len) // int(args.reference_batch_size)
        reference_updates_per_epoch = max(
            1,
            math.ceil(reference_micro_batches / int(args.reference_accum_steps)),
        )
        total_updates = reference_updates_per_epoch * int(args.epochs)
    else:
        total_updates = int(actual_total_updates)

    total_updates = max(1, total_updates)
    if args.reference_warmup_updates is not None:
        warmup_updates = float(args.reference_warmup_updates)
    else:
        warmup_updates = total_updates / 5.0 if args.warmup else 0.0
    return total_updates, warmup_updates


def dataset_random_flip(mode):
    return mode == "train"


def change_probability(out_bn, output_api):
    if output_api == "logits":
        return torch.sigmoid(out_bn)
    if output_api == "probability":
        return out_bn
    raise ValueError(f"Unsupported change output API: {output_api}")


def ensure_finite_tensor(name, tensor):
    if not torch.isfinite(tensor.detach()).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def trainable_parameters(model):
    return [param for param in unwrap_model(model).parameters() if param.requires_grad and param.grad is not None]


def compute_grad_norm(parameters):
    parameters = [param for param in parameters if param.grad is not None]
    if not parameters:
        return torch.zeros(())
    device = parameters[0].grad.device
    norms = [param.grad.detach().float().norm(2).to(device) for param in parameters]
    return torch.norm(torch.stack(norms), 2)


def is_dist_avail_and_initialized():
    if dist is None:
        return False
    return dist.is_available() and dist.is_initialized()


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def init_distributed_mode(args):
    if torch is None:
        raise ImportError("PyTorch is required to run train_WUSU_ddp_accum_v3.py.")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    args.distributed = args.world_size > 1
    if not args.distributed:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed WUSU training requires CUDA/NCCL.")

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
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def freeze_batch_norm(module):
    if nn is None:
        raise ImportError("PyTorch is required to freeze BatchNorm layers.")
    for layer in module.modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            layer.eval()
            if layer.weight is not None:
                layer.weight.requires_grad_(False)
            if layer.bias is not None:
                layer.bias.requires_grad_(False)


def reset_snn_state(model, functional_module):
    functional_module.reset_net(unwrap_model(model))


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def load_state_dict_compatible(model, checkpoint_path, strict, map_location="cpu"):
    if torch is None:
        raise ImportError("PyTorch is required to load checkpoints.")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    incompatible = unwrap_model(model).load_state_dict(state_dict, strict=strict)
    return checkpoint, incompatible


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Trainer:
    def __init__(self, args):
        if torch is None:
            raise ImportError("PyTorch is required to instantiate Trainer.")

        import datasets.MultiSiamese_RS_ST_TL as RS
        from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU as Net
        from spikingjelly.clock_driven import functional
        from tensorboardX import SummaryWriter
        from tqdm import tqdm
        from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        from utils.loss import ChangeSimilarity, DiceLoss
        from utils.metric import IOUandSek
        from utils.palette import color_map

        self.RS = RS
        self.functional = functional
        self.SummaryWriter = SummaryWriter
        self.tqdm = tqdm
        self.IOUandSek = IOUandSek
        self.color_map = color_map
        self.args = args

        if args.data_root is not None:
            self.RS.root = args.data_root

        if args.log_dir is None:
            args.log_dir = os.path.join(working_path, "logs", args.data_name, args.Net_name, args.backbone)
        self.writer = None
        if is_main_process(args):
            os.makedirs(args.log_dir, exist_ok=True)
            self.writer = SummaryWriter(args.log_dir)

        self.device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

        trainset = self.RS.Data(mode="train", random_flip=dataset_random_flip("train"))
        valset = self.RS.Data(mode="val", random_flip=dataset_random_flip("val")) if is_main_process(args) else None

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
            pin_memory=True,
            num_workers=args.workers,
            drop_last=True,
        )
        self.valloader = None
        if is_main_process(args):
            self.valloader = DataLoader(
                valset,
                batch_size=args.val_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers,
                drop_last=False,
            )

        model = Net(args.backbone, args.pretrained, len(self.RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
        if args.pretrain_from:
            _, incompatible = load_state_dict_compatible(model, args.pretrain_from, strict=False)
            if is_main_process(args):
                print(f"Loaded pretrain weights from {args.pretrain_from}: {incompatible}")
        if args.load_from:
            _, incompatible = load_state_dict_compatible(model, args.load_from, strict=True)
            if is_main_process(args):
                print(f"Loaded model weights from {args.load_from}: {incompatible}")

        model = model.to(self.device)
        if args.sync_bn and args.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if is_main_process(args):
                print("Converted BatchNorm layers to SyncBatchNorm.")
        if args.freeze_bn:
            freeze_batch_norm(model)

        self.model = model
        if args.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )

        self.TCL = TemporalLogicKLDivLoss()
        self.criterion_seg = CrossEntropyLoss(ignore_index=-1)
        if args.change_output_api == "logits":
            self.criterion_bn = BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion_bn = BCELoss(reduction="none")
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()
        self.optimizer = AdamW(
            [
                {
                    "params": [
                        param
                        for name, param in unwrap_model(self.model).named_parameters()
                        if "backbone" in name and param.requires_grad
                    ],
                    "lr": args.lr,
                },
                {
                    "params": [
                        param
                        for name, param in unwrap_model(self.model).named_parameters()
                        if "backbone" not in name and param.requires_grad
                    ],
                    "lr": args.lr,
                },
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self.train_micro_batches_per_epoch = len(self.trainloader)
        self.updates_per_epoch = max(1, math.ceil(self.train_micro_batches_per_epoch / args.accum_steps))
        self.actual_total_update_steps = self.updates_per_epoch * args.epochs
        self.total_update_steps, self.warmup_updates = resolve_reference_update_counts(
            args,
            dataset_len=len(trainset),
            actual_total_updates=self.actual_total_update_steps,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=build_poly_warmup_lr_lambda(
                warmup_enabled=args.warmup,
                total_updates=self.total_update_steps,
                warmup_updates=self.warmup_updates,
            ),
        )
        self.global_update_step = 0
        self.start_epoch = 0
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0

        if is_main_process(args):
            print(
                "train_micro_batches_per_epoch=%d, updates_per_epoch=%d, actual_total_updates=%d, "
                "lr_total_updates=%d, warmup_updates=%.1f"
                % (
                    self.train_micro_batches_per_epoch,
                    self.updates_per_epoch,
                    self.actual_total_update_steps,
                    self.total_update_steps,
                    self.warmup_updates,
                )
            )

        if args.resume:
            self.resume(args.resume)

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = strip_module_prefix(extract_model_state(checkpoint))
        incompatible = unwrap_model(self.model).load_state_dict(state_dict, strict=False)

        if isinstance(checkpoint, dict):
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint and checkpoint["scaler"] is not None and self.args.amp:
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.previous_best = float(checkpoint.get("best_metric", checkpoint.get("previous_best", 0.0)))
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.global_update_step = int(
                checkpoint.get("global_update_step", self.start_epoch * self.updates_per_epoch)
            )
            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            elif self.global_update_step > 0:
                self.scheduler.last_epoch = self.global_update_step - 1
                self.scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        if is_main_process(self.args):
            print(f"Resumed checkpoint from {checkpoint_path}: {incompatible}")

    def compute_losses(self, out1, out2, out3, out_bn, mask1, mask2, mask3, mask_bn):
        loss1 = self.criterion_seg(out1.float(), mask1 - 1)
        loss2 = self.criterion_seg(out2.float(), mask2 - 1)
        loss3 = self.criterion_seg(out3.float(), mask3 - 1)
        loss_seg = (loss1 + loss2 + loss3) / 3

        loss_similarity = self.criterion_sc(out1.float(), out3.float(), mask_bn)

        loss_bn_1 = self.criterion_bn(out_bn.float(), mask_bn)
        loss_bn_1[mask_bn == 1] *= 2
        loss_bn_1 = loss_bn_1.mean()
        loss_bn_2 = self.criterion_bn_2(out_bn.float(), mask_bn)
        loss_bn = loss_bn_1 + loss_bn_2

        loss = loss_bn + loss_seg + loss_similarity
        loss_tl = torch.zeros((), device=loss.device, dtype=loss.dtype)
        return loss, loss_seg, loss_bn, loss_similarity, loss_tl

    def training(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        if self.args.freeze_bn:
            freeze_batch_norm(unwrap_model(self.model))

        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0
        total_tl = 0.0
        last_grad_norm = 0.0
        last_amp_scale = float(self.scaler.get_scale()) if self.args.amp else 1.0
        amp_skipped_steps = 0
        total_micro_batches = len(self.trainloader)

        iterator = self.tqdm(self.trainloader) if is_main_process(self.args) else self.trainloader
        self.optimizer.zero_grad(set_to_none=True)

        for step, (img1, img2, img3, mask1, mask2, mask3, mask_bn, sample_id) in enumerate(iterator):
            del sample_id
            img1 = img1.float().to(self.device, non_blocking=True)
            img2 = img2.float().to(self.device, non_blocking=True)
            img3 = img3.float().to(self.device, non_blocking=True)
            mask1 = mask1.long().to(self.device, non_blocking=True)
            mask2 = mask2.long().to(self.device, non_blocking=True)
            mask3 = mask3.long().to(self.device, non_blocking=True)
            mask_bn = mask_bn.float().to(self.device, non_blocking=True)
            x = torch.stack([img1, img2, img3], dim=0)

            update_now = should_update_optimizer(step, total_micro_batches, self.args.accum_steps)
            sync_context = (
                self.model.no_sync()
                if self.args.distributed and not update_now
                else nullcontext()
            )

            with sync_context:
                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out1, out2, out3, out_bn = self.model(x)
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_seg, loss_bn, loss_similarity, loss_tl = self.compute_losses(
                        out1, out2, out3, out_bn, mask1, mask2, mask3, mask_bn
                    )
                    ensure_finite_tensor("loss", loss)
                    loss_to_backward = loss / self.args.accum_steps

                total_loss += float(loss.detach())
                total_loss_seg += float(loss_seg.detach())
                total_loss_bn += float(loss_bn.detach())
                total_loss_similarity += float(loss_similarity.detach())
                total_tl += float(loss_tl.detach())

                if self.args.amp:
                    self.scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

            if update_now:
                params_with_grad = trainable_parameters(self.model)
                if self.args.amp:
                    previous_scale = float(self.scaler.get_scale())
                    self.scaler.unscale_(self.optimizer)
                    grad_norm_tensor = compute_grad_norm(params_with_grad)
                    ensure_finite_tensor("grad_norm", grad_norm_tensor)
                    if self.args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            params_with_grad,
                            self.args.grad_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    last_amp_scale = float(self.scaler.get_scale())
                    stepped = last_amp_scale >= previous_scale
                    if not stepped:
                        amp_skipped_steps += 1
                else:
                    grad_norm_tensor = compute_grad_norm(params_with_grad)
                    ensure_finite_tensor("grad_norm", grad_norm_tensor)
                    if self.args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            params_with_grad,
                            self.args.grad_clip_norm,
                        )
                    self.optimizer.step()
                    stepped = True
                last_grad_norm = float(grad_norm_tensor.detach())
                if stepped:
                    self.global_update_step += 1
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            reset_snn_state(self.model, self.functional)

            if is_main_process(self.args):
                seen = step + 1
                iterator.set_description(
                    "Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f, TL Loss: %.3f"
                    % (
                        total_loss / seen,
                        total_loss_seg / seen,
                        total_loss_bn / seen,
                        total_loss_similarity / seen,
                        total_tl / seen,
                    )
                )
                running_iter = epoch * total_micro_batches + seen
                self.writer.add_scalar("train total_loss", total_loss / seen, running_iter)
                self.writer.add_scalar("train seg_loss", total_loss_seg / seen, running_iter)
                self.writer.add_scalar("train bn_loss", total_loss_bn / seen, running_iter)
                self.writer.add_scalar("train sc_loss", total_loss_similarity / seen, running_iter)
                self.writer.add_scalar("train TL Loss", total_tl / seen, running_iter)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], running_iter)
                self.writer.add_scalar("train grad_norm", last_grad_norm, running_iter)
                if self.args.amp:
                    self.writer.add_scalar("amp scale", last_amp_scale, running_iter)
                    self.writer.add_scalar("amp skipped_steps", amp_skipped_steps, running_iter)

    def validation(self, epoch):
        if not is_main_process(self.args):
            if self.args.distributed:
                dist.barrier()
            return self.previous_best

        model_for_eval = unwrap_model(self.model)
        model_for_eval.eval()
        metric = self.IOUandSek(num_classes=len(self.RS.ST_CLASSES))
        if self.args.save_mask:
            _ = self.color_map()

        score = miou = sek = fscd = oa = sc_precision = sc_recall = 0.0
        iterator = self.tqdm(self.valloader)

        with torch.no_grad():
            for img1, img2, img3, mask1, mask2, mask3, mask_bn, sample_id in iterator:
                del sample_id, mask2
                img1 = img1.float().to(self.device, non_blocking=True)
                img2 = img2.float().to(self.device, non_blocking=True)
                img3 = img3.float().to(self.device, non_blocking=True)
                x = torch.stack([img1, img2, img3], dim=0)

                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out1, out2, out3, out_bn13 = model_for_eval(x)

                out1 = torch.argmax(out1.float(), dim=1).cpu().numpy() + 1
                out3 = torch.argmax(out3.float(), dim=1).cpu().numpy() + 1
                out_bn_prob = change_probability(out_bn13.float(), self.args.change_output_api)
                out_bn = (out_bn_prob > 0.5).cpu().numpy().astype(np.uint8)

                mask1 = mask1.clone()
                mask3 = mask3.clone()
                mask1[mask_bn == 0] = 0
                mask3[mask_bn == 0] = 0
                out1[out_bn == 0] = 0
                out3[out_bn == 0] = 0

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out3, mask3.numpy())
                _, score, miou, sek, fscd, oa, sc_precision, sc_recall = metric.evaluate_SECOND()
                reset_snn_state(model_for_eval, self.functional)

                iterator.set_description(
                    "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f"
                    % (miou, sek, score, fscd, oa, sc_precision, sc_recall)
                )

        self.save_checkpoint(epoch, score, miou, sek, fscd, oa)
        self.writer.add_scalar("val_Score", score, epoch)
        self.writer.add_scalar("val_mIOU", miou, epoch)
        self.writer.add_scalar("val_Sek", sek, epoch)
        self.writer.add_scalar("val_Fscd", fscd, epoch)
        self.writer.add_scalar("val_OA", oa, epoch)

        if self.args.distributed:
            dist.barrier()
        return score

    def checkpoint_dir(self):
        return os.path.join(self.args.output_dir, self.args.data_name, self.args.Net_name, self.args.backbone)

    def checkpoint_payload(self, epoch):
        return {
            "epoch": epoch,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.args.amp else None,
            "scheduler": self.scheduler.state_dict(),
            "best_metric": self.previous_best,
            "global_update_step": self.global_update_step,
            "args": vars(self.args),
        }

    def save_checkpoint(self, epoch, score, miou, sek, fscd, oa):
        os.makedirs(self.checkpoint_dir(), exist_ok=True)

        is_best = score >= self.previous_best
        if is_best:
            self.previous_best = score

        latest_path = os.path.join(self.checkpoint_dir(), "latest.pth")
        torch.save(self.checkpoint_payload(epoch), latest_path)

        if is_best:
            best_name = "epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" % (
                epoch,
                score * 100,
                miou * 100,
                sek * 100,
                fscd * 100,
                oa * 100,
            )
            best_path = os.path.join(self.checkpoint_dir(), best_name)
            torch.save(self.checkpoint_payload(epoch), best_path)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def validate_args(args):
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.reference_batch_size is not None and args.reference_batch_size < 1:
        raise ValueError("--reference-batch-size must be >= 1")
    if args.reference_accum_steps < 1:
        raise ValueError("--reference-accum-steps must be >= 1")
    if args.reference_total_updates is not None and args.reference_total_updates < 1:
        raise ValueError("--reference-total-updates must be >= 1")
    if args.reference_warmup_updates is not None and args.reference_warmup_updates < 0:
        raise ValueError("--reference-warmup-updates must be >= 0")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be >= 0")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if torch is not None and args.amp and not torch.cuda.is_available():
        args.amp = False


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    init_distributed_mode(args)
    validate_args(args)
    seed_everything(args.seed, args.rank)

    if is_main_process(args):
        print(args)
        global_batch = args.batch_size * args.world_size
        effective_batch = global_batch * args.accum_steps
        print(
            "batch_size_per_gpu=%d, world_size=%d, global_batch_size=%d, accum_steps=%d, effective_batch_size=%d"
            % (args.batch_size, args.world_size, global_batch, args.accum_steps, effective_batch)
        )
        if args.reference_total_updates is not None:
            print(
                "LR schedule uses explicit reference_total_updates=%d, reference_warmup_updates=%s"
                % (args.reference_total_updates, str(args.reference_warmup_updates))
            )
        elif args.reference_batch_size is not None:
            print(
                "LR schedule uses single-GPU reference_batch_size=%d, reference_accum_steps=%d"
                % (args.reference_batch_size, args.reference_accum_steps)
            )
        if args.sync_bn and args.distributed:
            print("SyncBatchNorm synchronizes BN statistics across ranks; accumulation affects optimizer batch only.")

    trainer = None
    try:
        trainer = Trainer(args)
        if args.eval_only:
            trainer.validation(trainer.start_epoch)
            return

        for epoch in range(trainer.start_epoch, args.epochs):
            if is_main_process(args):
                print(
                    "\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.5f"
                    % (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best)
                )
            trainer.training(epoch)
            trainer.validation(epoch)
    finally:
        if trainer is not None:
            trainer.close()
        cleanup_distributed()


if __name__ == "__main__":
    # Single GPU:
    #   CUDA_VISIBLE_DEVICES=0 python train_WUSU_ddp_accum_v3.py --batch-size 2 --accum-steps 1 --amp
    # 4 GPU DDP + SyncBN:
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum_v3.py --batch-size 2 --sync-bn --amp
    # 4 GPU DDP + SyncBN + accumulation:
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum_v3.py --batch-size 2 --sync-bn --accum-steps 4 --amp
    main()

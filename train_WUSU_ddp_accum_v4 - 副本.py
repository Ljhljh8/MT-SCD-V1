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


class FloatTupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="checkpoints")
    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer passed to timm create_optimizer_v2, e.g. "adamw" or "adamp".')
    parser.add_argument("--opt-eps", default=None, type=float, metavar="EPSILON", help="Optimizer epsilon; None uses the timm optimizer default.")
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", action=FloatTupleAction, metavar="BETA", help="Optimizer betas; use two values for Adam/AdamP.")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="Momentum for optimizers that use it.")
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--filter-bias-and-bn", dest="filter_bias_and_bn", action="store_true", help="Let timm skip weight decay for bias and norm parameters.")
    parser.add_argument("--no-filter-bias-and-bn", dest="filter_bias_and_bn", action="store_false", help="Apply weight decay to all trainable parameters, matching the previous AdamW path.")
    parser.set_defaults(filter_bias_and_bn=False)
    # Learning rate schedule parameters
    parser.add_argument("--sched", default="poly", type=str, metavar="SCHEDULER", help='timm LR scheduler. Default "poly" preserves the previous poly-style schedule.')
    parser.add_argument("--sched-on-updates", dest="sched_on_updates", action="store_true", help="Step the timm scheduler on optimizer updates.")
    parser.add_argument("--sched-on-epochs", dest="sched_on_updates", action="store_false", help="Step the timm scheduler once per epoch.")
    parser.set_defaults(sched_on_updates=True)
    parser.add_argument("--lr", type=float, default=0.005, metavar="LR", help="Base learning rate.")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct", help="Learning rate noise on/off epoch percentages.")
    parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT", help="Learning rate noise limit percent.")
    parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV", help="Learning rate noise std-dev.")
    parser.add_argument("--lr-cycle-mul", type=float, default=1.0, metavar="MULT", help="Learning rate cycle length multiplier.")
    parser.add_argument("--lr-cycle-decay", type=float, default=0.1, metavar="MULT", help="Learning rate cycle decay factor.")
    parser.add_argument("--lr-cycle-limit", type=int, default=1, metavar="N", help="Learning rate cycle limit.")
    parser.add_argument("--lr-k-decay", type=float, default=1.0, help="k-decay factor for cosine/poly schedulers.")
    parser.add_argument("--warmup-lr", type=float, default=0.0, metavar="LR", help="Warmup starting LR.")
    parser.add_argument("--min-lr", type=float, default=0.0, metavar="LR", help="Lower LR bound.")
    parser.add_argument("--epochs", type=int, default=100, metavar="N", help="Number of epochs to train.")
    parser.add_argument("--decay-epochs", type=float, default=30, metavar="N", help="Epoch interval to decay LR for step schedulers.")
    parser.add_argument("--decay-milestones", type=int, nargs="+", default=[30, 60], metavar="M", help="Epoch milestones for multistep scheduler.")
    parser.add_argument("--warmup-epochs", type=float, default=5, metavar="N", help="Warmup epochs for epoch-based timm schedules.")
    parser.add_argument("--warmup-prefix", action="store_true", default=False, help="Exclude warmup time from cycle length for timm schedulers that support it.")
    parser.add_argument("--cooldown-epochs", type=int, default=0, metavar="N", help="Cooldown epochs after cyclic schedule ends.")
    parser.add_argument("--patience-epochs", type=int, default=10, metavar="N", help="Patience epochs for plateau scheduler.")
    parser.add_argument("--decay-rate", "--dr", type=float, default=None, metavar="RATE", help="Poly power when --sched poly; decay factor for step/multistep.")
    parser.add_argument("--eval-metric", type=str, default="score", help="Metric name used by timm plateau mode.")

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
    parser.add_argument(
        "--amp_debug_nonfinite",
        "--amp-debug-nonfinite",
        dest="amp_debug_nonfinite",
        action="store_true",
        help="Print the first non-finite gradient parameter when AMP skips an update.",
    )
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
        "--clip-grad",
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


def first_nonfinite_gradient_name(model):
    if torch is None:
        return None
    for name, param in unwrap_model(model).named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad.detach()).all():
            return name
    return None


def synchronize_finite_flag(is_finite, device):
    if not is_dist_avail_and_initialized():
        return bool(is_finite)
    flag = torch.tensor([1 if is_finite else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def backoff_grad_scaler(scaler, previous_scale):
    backoff = scaler.get_backoff_factor() if hasattr(scaler, "get_backoff_factor") else 0.5
    scaler.update(new_scale=float(previous_scale) * float(backoff))


def is_dist_avail_and_initialized():
    if dist is None:
        return False
    return dist.is_available() and dist.is_initialized()


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def init_distributed_mode(args):
    if torch is None:
        raise ImportError("PyTorch is required to run train_WUSU_ddp_accum_v4.py.")

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


try:
    from timm.optim import create_optimizer_v2, optimizer_kwargs
    from timm.scheduler import create_scheduler
except ModuleNotFoundError:
    create_optimizer_v2 = None
    optimizer_kwargs = None
    create_scheduler = None


def require_timm():
    if create_optimizer_v2 is None or optimizer_kwargs is None or create_scheduler is None:
        raise ImportError(
            "timm is required for train_WUSU_ddp_accum_v4.py optimizer/scheduler. "
            "Install timm in the training environment or use train_WUSU_ddp_accum_v3.py."
        )


def require_timm_optimizer():
    if create_optimizer_v2 is None or optimizer_kwargs is None:
        raise ImportError(
            "timm.optim is required for train_WUSU_ddp_accum_v4.py optimizer construction."
        )


def require_timm_scheduler():
    if create_scheduler is None:
        raise ImportError(
            "timm.scheduler is required for train_WUSU_ddp_accum_v4.py scheduler construction."
        )


def build_timm_optimizer(model, args):
    require_timm_optimizer()
    kwargs = optimizer_kwargs(cfg=args)
    kwargs["filter_bias_and_bn"] = bool(getattr(args, "filter_bias_and_bn", False))
    return create_optimizer_v2(unwrap_model(model), **kwargs)


def reference_updates_per_epoch(total_updates, epochs):
    return max(1, int(math.ceil(float(total_updates) / float(max(1, int(epochs))))))


def build_timm_scheduler_config(args, total_updates, warmup_updates):
    scheduler_args = argparse.Namespace(**vars(args))
    if not getattr(args, "warmup", True):
        warmup_updates = 0.0
        scheduler_args.warmup_epochs = 0.0
    if getattr(args, "sched_on_updates", True):
        updates_per_reference_epoch = reference_updates_per_epoch(total_updates, args.epochs)
        scheduler_args.epochs = int(total_updates)
        scheduler_args.warmup_epochs = float(warmup_updates or 0.0)
        scheduler_args.decay_epochs = float(getattr(args, "decay_epochs", 30)) * updates_per_reference_epoch
        scheduler_args.decay_milestones = [
            int(milestone * updates_per_reference_epoch)
            for milestone in getattr(args, "decay_milestones", [30, 60])
        ]
        return scheduler_args, 1
    return scheduler_args, 0


def normalize_timm_scheduler_result(result):
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("timm create_scheduler returned an unexpected tuple.")
        return result
    return result, None


def build_timm_scheduler(args, optimizer, total_updates, warmup_updates):
    require_timm_scheduler()
    scheduler_args, updates_per_epoch = build_timm_scheduler_config(
        args,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
    )
    try:
        result = create_scheduler(scheduler_args, optimizer, updates_per_epoch=updates_per_epoch)
    except TypeError as exc:
        if getattr(scheduler_args, "sched_on_updates", False):
            raise RuntimeError(
                "This timm version does not support update-based create_scheduler(..., updates_per_epoch=...)."
            ) from exc
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


def broadcast_main_float(value, device):
    if not is_dist_avail_and_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


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
        self.optimizer = build_timm_optimizer(self.model, args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self.train_micro_batches_per_epoch = len(self.trainloader)
        self.updates_per_epoch = max(1, math.ceil(self.train_micro_batches_per_epoch / args.accum_steps))
        self.actual_total_update_steps = self.updates_per_epoch * args.epochs
        self.total_update_steps, self.warmup_updates = resolve_reference_update_counts(
            args,
            dataset_len=len(trainset),
            actual_total_updates=self.actual_total_update_steps,
        )
        (
            self.scheduler,
            self.scheduler_num_epochs,
            self.scheduler_args,
            self.scheduler_updates_per_epoch,
        ) = build_timm_scheduler(
            args,
            self.optimizer,
            total_updates=self.total_update_steps,
            warmup_updates=self.warmup_updates,
        )
        self.scheduler_step_on_updates = bool(getattr(self.scheduler_args, "sched_on_updates", False))
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
                load_scheduler_state(self.scheduler, checkpoint["scheduler"])
            elif self.global_update_step > 0:
                sync_scheduler_to_resume_position(
                    self.scheduler,
                    self.global_update_step,
                    self.start_epoch,
                    self.scheduler_step_on_updates,
                )

        if is_main_process(self.args):
            print(f"Resumed checkpoint from {checkpoint_path}: {incompatible}")

    def step_scheduler_epoch(self, epoch, metric):
        if self.scheduler_step_on_updates:
            return
        scheduler_metric = metric if self.args.sched == "plateau" else None
        step_timm_scheduler_epoch(self.scheduler, epoch + 1, scheduler_metric)

    def compute_losses(self, out1, out2, out3, out_bn, mask1, mask2, mask3, mask_bn):
        loss1 = self.criterion_seg(out1.float(), mask1 - 1)
        loss2 = self.criterion_seg(out2.float(), mask2 - 1)
        loss3 = self.criterion_seg(out3.float(), mask3 - 1)
        loss_seg = (loss1 + loss2 + loss3) / 3

        loss_similarity = self.criterion_sc(out1.float(), out3.float(), mask_bn)

        loss_bn_1 = self.criterion_bn(out_bn.float(), mask_bn)
        loss_bn_1[mask_bn == 1] *= 2
        loss_bn_1 = loss_bn_1.mean()
        loss_bn_2 = self.criterion_bn_2(torch.sigmoid(out_bn.float()), mask_bn)
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
                    local_grad_finite = bool(torch.isfinite(grad_norm_tensor.detach()).all().item())
                    grad_finite = synchronize_finite_flag(local_grad_finite, self.device)
                    if grad_finite:
                        if self.args.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                params_with_grad,
                                self.args.grad_clip_norm,
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.args.amp_debug_nonfinite:
                            bad_name = first_nonfinite_gradient_name(self.model)
                            print(
                                "AMP non-finite gradient skipped: rank=%d, param=%s, grad_norm=%s, scale=%.1f"
                                % (
                                    self.args.rank,
                                    bad_name if bad_name is not None else "none_on_this_rank",
                                    str(float(grad_norm_tensor.detach())),
                                    previous_scale,
                                ),
                                flush=True,
                            )
                        backoff_grad_scaler(self.scaler, previous_scale)
                    last_amp_scale = float(self.scaler.get_scale())
                    stepped = grad_finite and last_amp_scale >= previous_scale
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
                    if self.scheduler_step_on_updates:
                        step_timm_scheduler_update(self.scheduler, self.global_update_step)
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
                return broadcast_main_float(0.0, self.device)
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
            score = broadcast_main_float(score, self.device)
        return score

    def checkpoint_dir(self):
        return os.path.join(self.args.output_dir, self.args.data_name, self.args.Net_name, self.args.backbone)

    def checkpoint_payload(self, epoch):
        return {
            "epoch": epoch,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.args.amp else None,
            "scheduler": scheduler_state_dict(self.scheduler),
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
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr < 0:
        raise ValueError("--lr must be >= 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.opt_betas is not None and len(args.opt_betas) not in (2, 3):
        raise ValueError("--opt-betas must contain two values for Adam/AdamP or three values for Adan.")
    if args.sched_on_updates and args.sched == "plateau":
        raise ValueError("timm plateau scheduler only supports --sched-on-epochs, not --sched-on-updates.")
    if args.decay_rate is None:
        args.decay_rate = 1.5 if args.sched == "poly" else 0.1
    if args.decay_rate <= 0:
        raise ValueError("--decay-rate must be > 0")
    if args.decay_epochs <= 0:
        raise ValueError("--decay-epochs must be > 0")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if not args.decay_milestones:
        raise ValueError("--decay-milestones must contain at least one milestone")
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
        print(
            "optimizer=%s, scheduler=%s, sched_on_updates=%s, filter_bias_and_bn=%s"
            % (args.opt, args.sched, args.sched_on_updates, args.filter_bias_and_bn)
        )

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
            score = trainer.validation(epoch)
            trainer.step_scheduler_epoch(epoch, score)
    finally:
        if trainer is not None:
            trainer.close()
        cleanup_distributed()


if __name__ == "__main__":
    # Single GPU:
    #   CUDA_VISIBLE_DEVICES=0 python train_WUSU_ddp_accum_v4.py --batch-size 2 --accum-steps 1 --amp
    # 4 GPU DDP + SyncBN:
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum_v4.py --batch-size 2 --sync-bn --amp
    # 4 GPU DDP + SyncBN + accumulation:
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum_v4.py --batch-size 2 --sync-bn --accum-steps 4 --amp
    main()

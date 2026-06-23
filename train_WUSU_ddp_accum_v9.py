"""MTSCD-PairRelAux-V1 experimental training entrypoint; v4 has no PairRelAux wiring."""

import argparse
import math
import os
import random
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield
from types import SimpleNamespace

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ModuleNotFoundError:
    torch = None
    dist = None
    nn = None
    DistributedDataParallel = None
    DataLoader = None
    DistributedSampler = None


working_path = os.path.dirname(os.path.abspath(__file__))


class FloatTupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


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
    parser = argparse.ArgumentParser("WUSU GSTMSCD DDP AMP Training V6")
    parser.add_argument("--data_name", "--data-name", dest="data_name", type=str, default="WUSU")
    parser.add_argument("--Net_name", "--net-name", dest="Net_name", type=str, default="GSTMSCD")
    parser.add_argument("--backbone", type=str, default="sdtv2")
    parser.add_argument("--data_root", "--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="checkpoints_v6")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")

    parser.add_argument("--lightweight", dest="lightweight", action="store_true")
    parser.add_argument("--pretrain_from", "--pretrain-from", dest="pretrain_from", type=str, default=None)
    parser.add_argument("--load_from", "--load-from", dest="load_from", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--Lambda", type=float, default=0.00005)
    parser.add_argument("--relation-mode", choices=["prg", "pdca", "none"], default="pdca")
    parser.add_argument("--enable-pairrel-aux", action="store_true")
    parser.add_argument("--pairrel-aux-weight", type=float, default=0.05)
    parser.add_argument("--pairrel-aux-warmup-epochs", type=int, default=5)
    parser.add_argument("--pairrel-aux-scales", type=str, default="3")
    parser.add_argument("--pairrel-margin", type=float, default=1.0)
    parser.add_argument("--pairrel-tau-unchanged", type=float, default=0.05)
    parser.add_argument("--pairrel-tau-changed", type=float, default=0.30)
    parser.add_argument("--pdca_aux", action="store_true", default=False)
    parser.add_argument("--pdca_aux_weight", type=float, default=0.05)
    parser.add_argument("--pdca_aux_warmup_epochs", type=float, default=5.0)
    parser.add_argument("--pdca_aux_scale_key", type=str, default="3")
    parser.add_argument("--pdca_aux_tau_neg", type=float, default=0.05)
    parser.add_argument("--pdca_aux_tau_pos", type=float, default=0.20)
    parser.add_argument("--pdca_aux_ambiguous_weight", type=float, default=0.25)

    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt-eps", default=None, type=float, metavar="EPSILON")
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", action=FloatTupleAction, metavar="BETA")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--filter-bias-and-bn", dest="filter_bias_and_bn", action="store_true")
    parser.add_argument("--no-filter-bias-and-bn", dest="filter_bias_and_bn", action="store_false")
    parser.set_defaults(filter_bias_and_bn=False)

    parser.add_argument("--sched", default="poly", type=str, metavar="SCHEDULER")
    parser.add_argument("--sched-on-updates", dest="sched_on_updates", action="store_true")
    parser.add_argument("--sched-on-epochs", dest="sched_on_updates", action="store_false")
    parser.set_defaults(sched_on_updates=True)
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct")
    parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT")
    parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV")
    parser.add_argument("--lr-cycle-mul", type=float, default=1.0, metavar="MULT")
    parser.add_argument("--lr-cycle-decay", type=float, default=0.1, metavar="MULT")
    parser.add_argument("--lr-cycle-limit", type=int, default=1, metavar="N")
    parser.add_argument("--lr-k-decay", type=float, default=1.0)
    parser.add_argument("--warmup-lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--min-lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--decay-epochs", type=float, default=30, metavar="N")
    parser.add_argument("--decay-milestones", type=int, nargs="+", default=[30, 60], metavar="M")
    parser.add_argument("--decay-rate", "--dr", type=float, default=None, metavar="RATE")
    parser.add_argument("--warmup-epochs", type=float, default=5, metavar="N")
    parser.add_argument("--warmup-prefix", action="store_true", default=False)
    parser.add_argument("--cooldown-epochs", type=int, default=0, metavar="N")
    parser.add_argument("--patience-epochs", type=int, default=10, metavar="N")
    parser.add_argument("--eval-metric", type=str, default="score")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_debug_nonfinite", "--amp-debug-nonfinite", dest="amp_debug_nonfinite", action="store_true")
    parser.add_argument("--accum_steps", "--accum-steps", dest="accum_steps", type=int, default=1)
    parser.add_argument("--sync_bn", "--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument("--freeze_bn", "--freeze-bn", dest="freeze_bn", action="store_true")
    parser.add_argument("--grad_clip_norm", "--grad-clip-norm", "--clip-grad", dest="grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--find_unused_parameters", "--find-unused-parameters", dest="find_unused_parameters", action="store_true", default=True)
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.add_argument("--reference_batch_size", "--reference-batch-size", dest="reference_batch_size", type=int, default=None)
    parser.add_argument("--reference_accum_steps", "--reference-accum-steps", dest="reference_accum_steps", type=int, default=1)
    parser.add_argument("--reference_total_updates", "--reference-total-updates", dest="reference_total_updates", type=int, default=None)
    parser.add_argument("--reference_warmup_updates", "--reference-warmup-updates", dest="reference_warmup_updates", type=float, default=None)

    parser.add_argument("--dist_url", "--dist-url", dest="dist_url", type=str, default="env://")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", "--world-size", dest="world_size", type=int, default=1)
    return parser


def validate_args(args):
    if args.pdca_aux and args.enable_pairrel_aux:
        raise ValueError("This experiment tests PDCA-RAS only. Please disable PairRelAux.")
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr < 0 or args.weight_decay < 0 or args.grad_clip_norm < 0:
        raise ValueError("lr, weight decay, and grad clip must be non-negative")

    if args.reference_total_updates is not None and args.reference_total_updates < 1:
        raise ValueError("--reference-total-updates must be >= 1")
    if args.reference_warmup_updates is not None and args.reference_warmup_updates < 0:
        raise ValueError("--reference-warmup-updates must be >= 0")
    if args.opt_betas is not None and len(args.opt_betas) not in (2, 3):
        raise ValueError("--opt-betas must contain two values for Adam/AdamP or three values for Adan")
    if args.sched_on_updates and args.sched == "plateau":
        raise ValueError("timm plateau scheduler only supports --sched-on-epochs")
    if args.decay_rate is None:
        args.decay_rate = 1.5 if args.sched == "poly" else 0.1
    if args.decay_rate <= 0 or args.decay_epochs <= 0:
        raise ValueError("decay settings must be positive")
    if not args.decay_milestones:
        raise ValueError("--decay-milestones must contain at least one milestone")
    try:
        args.pairrel_aux_scale_ids = tuple(
            int(value.strip()) for value in args.pairrel_aux_scales.split(",") if value.strip()
        )
    except ValueError:
        raise ValueError("--pairrel-aux-scales must be a comma-separated list of integers")
    if not args.pairrel_aux_scale_ids or any(scale < 0 for scale in args.pairrel_aux_scale_ids):
        raise ValueError("--pairrel-aux-scales must contain non-negative scale indices")
    if args.pairrel_aux_weight < 0 or args.pairrel_aux_warmup_epochs < 0 or args.pairrel_margin < 0:
        raise ValueError("pairrel weight, warmup epochs, and margin must be non-negative")
    if not 0 <= args.pairrel_tau_unchanged <= args.pairrel_tau_changed <= 1:
        raise ValueError("pairrel thresholds must satisfy 0 <= unchanged <= changed <= 1")
    if args.pdca_aux_weight < 0 or args.pdca_aux_warmup_epochs < 0:
        raise ValueError("PDCA aux weight and warmup epochs must be non-negative")
    if not 0 <= args.pdca_aux_tau_neg <= args.pdca_aux_tau_pos <= 1:
        raise ValueError("PDCA aux thresholds must satisfy 0 <= tau_neg <= tau_pos <= 1")
    if args.pdca_aux_ambiguous_weight < 0:
        raise ValueError("PDCA aux ambiguous weight must be non-negative")
    if args.pdca_aux and args.relation_mode != "pdca":
        raise ValueError("--pdca_aux requires --relation-mode pdca")
    if args.enable_pairrel_aux and args.relation_mode != "pdca":
        if is_main_process(args):
            print("WARNING: PairRelAux requires --relation-mode pdca; auxiliary loss disabled.")
        args.enable_pairrel_aux = False
    if torch is not None and args.amp and not torch.cuda.is_available():
        args.amp = False


def should_update_optimizer(step, num_steps, accum_steps):
    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    return ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)


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
        warmup_updates = total_updates // int(args.epochs) * int(args.warmup_epochs) if args.warmup else 0.0

    # else:
    #     reference_micro_batches = int(dataset_len) // int(args.reference_batch_size)
    #     reference_updates = math.ceil(reference_micro_batches / int(args.reference_accum_steps))
    #     total_updates = max(1, reference_updates) * int(args.epochs)
    # if args.reference_warmup_updates is not None:
    #     warmup_updates = float(args.reference_warmup_updates)
    # else:
    #     # warmup_updates = float(total_updates) / 5.0 if args.warmup else 0.0
    #     warmup_updates = total_updates // int(args.epochs) * int(args.warmup_epochs) if args.warmup else 0.0

    return max(1, int(total_updates)), warmup_updates


def is_dist_avail_and_initialized():
    return dist is not None and dist.is_available() and dist.is_initialized()


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def init_distributed_mode(args):
    if torch is None:
        raise ImportError("PyTorch is required to run train_WUSU_ddp_accum_v6.py")
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
        raise RuntimeError("Distributed WUSU training requires CUDA/NCCL")
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


def strip_module_prefix(state_dict):
    if not state_dict or not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_state_dict_compatible(model, checkpoint_path, strict, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    incompatible = unwrap_model(model).load_state_dict(state_dict, strict=strict)
    return checkpoint, incompatible


def validate_pdca_pretrain_incompatible(incompatible, pdca_aux):
    if not pdca_aux:
        return
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [key for key in missing if ".relation_aux_head." not in key]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Unsafe --pretrain_from state for PDCA-RAS: invalid missing keys=%r, unexpected keys=%r"
            % (invalid_missing, unexpected)
        )


def validate_pdca_resume_state(state_dict, model_keys, pdca_aux):
    if not pdca_aux:
        return
    required = {key for key in model_keys if ".relation_aux_head." in key}
    missing = sorted(required.difference(state_dict))
    if not required or missing:
        raise RuntimeError(
            "Cannot --resume --pdca_aux from a checkpoint without complete relation_aux_head state; "
            "use --pretrain_from instead. Missing keys: %r" % missing
        )


def broadcast_main_float(value, device):
    if not is_dist_avail_and_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def ensure_finite_tensor(name, tensor):
    if not torch.isfinite(tensor.detach()).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


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


def should_advance_after_amp_step(grad_finite, previous_scale, current_scale):
    return bool(grad_finite) and float(current_scale) >= float(previous_scale)


try:
    from timm.optim import create_optimizer_v2, optimizer_kwargs
    from timm.scheduler import create_scheduler
except ModuleNotFoundError:
    create_optimizer_v2 = None
    optimizer_kwargs = None
    create_scheduler = None


def require_timm_optimizer():
    if create_optimizer_v2 is None or optimizer_kwargs is None:
        raise ImportError("timm.optim is required for V6 optimizer construction")


def require_timm_scheduler():
    if create_scheduler is None:
        raise ImportError("timm.scheduler is required for V6 scheduler construction")


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
    # from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU as Net
    from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU as Net

    model = Net(
        args.backbone,
        args.pretrained,
        len(RS.ST_CLASSES),
        args.lightweight,
        args.M,
        args.Lambda,
        relation_mode=args.relation_mode,
        use_pdca_relation_aux=args.pdca_aux,
        use_pairrel_aux=args.enable_pairrel_aux,
    )
    if args.pretrain_from:
        _, incompatible = load_state_dict_compatible(model, args.pretrain_from, strict=False)
        validate_pdca_pretrain_incompatible(incompatible, args.pdca_aux)
        if is_main_process(args):
            print(f"Loaded pretrain weights from {args.pretrain_from}: {incompatible}")
    if args.load_from:
        try:
            _, incompatible = load_state_dict_compatible(model, args.load_from, strict=True)
        except RuntimeError as exc:
            if args.pdca_aux:
                raise RuntimeError(
                    "Strict --load_from failed for PDCA-RAS. Old checkpoints do not contain "
                    "relation_aux_head; use --pretrain_from for warm-start."
                ) from exc
            raise
        if is_main_process(args):
            print(f"Loaded model weights from {args.load_from}: {incompatible}")
    model = model.to(device)
    if args.sync_bn and args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process(args):
            print("Converted BatchNorm layers to SyncBatchNorm.")
    if args.freeze_bn:
        freeze_batch_norm(model)
    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )
    return model


def build_loss_functions(args):
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
    from utils.loss import ChangeSimilarity, DiceLoss, PairwiseRelationAuxLoss
    from utils.pdca_aux_loss import pdca_relation_aux_loss

    criteria = {
        "seg": CrossEntropyLoss(ignore_index=-1),
        "bce": BCEWithLogitsLoss(reduction="none"),
        "dice": DiceLoss(activation="none"),
        "similarity": ChangeSimilarity(),
    }
    criteria["pairrel"] = (
        PairwiseRelationAuxLoss(
            scales=args.pairrel_aux_scale_ids,
            margin=args.pairrel_margin,
            tau_unchanged=args.pairrel_tau_unchanged,
            tau_changed=args.pairrel_tau_changed,
        )
        if args.enable_pairrel_aux
        else None
    )
    criteria["pdca_aux"] = pdca_relation_aux_loss if args.pdca_aux else None
    return criteria


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
    loss_seg = (loss1 + loss2 + loss3) / 3
    loss_similarity = criteria["similarity"](out1.float(), out3.float(), mask_bn)
    loss_bn = compute_binary_change_loss(change_logits, mask_bn, criteria["bce"], criteria["dice"])
    loss = loss_bn + loss_seg + loss_similarity
    loss_tl = torch.zeros((), device=loss.device, dtype=loss.dtype)
    return loss, loss_seg, loss_bn, loss_similarity, loss_tl


def train_one_epoch(ctx, epoch):
    args = ctx.args
    if ctx.train_sampler is not None:
        ctx.train_sampler.set_epoch(epoch)
    ctx.model.train()
    if args.freeze_bn:
        freeze_batch_norm(unwrap_model(ctx.model))

    totals = {
        "loss": 0.0,
        "seg": 0.0,
        "bn": 0.0,
        "similarity": 0.0,
        "tl": 0.0,
        "pairrel": 0.0,
        "pairrel_warm": 0.0,
        "pdca_aux_weight_eff": 0.0,
    }
    pairrel_stat_keys = []
    if args.enable_pairrel_aux:
        for scale in args.pairrel_aux_scale_ids:
            pairrel_stat_keys.extend(
                [
                    "pairrel_loss_s%d" % scale,
                    "pairrel_valid_ratio_s%d" % scale,
                    "pairrel_valid_weight_sum_s%d" % scale,
                    "pairrel_dist_unchanged_s%d" % scale,
                    "pairrel_dist_changed_s%d" % scale,
                    "pairrel_skipped_s%d" % scale,
                ]
            )
        totals.update({key: 0.0 for key in pairrel_stat_keys})
    pdca_stat_keys = [
        "pdca_aux_loss",
        "pdca_aux_target_mean",
        "pdca_aux_logit_mean",
        "pdca_aux_positive_ratio",
        "pdca_aux_prob_mean",
        "pdca_aux_changed_loss",
        "pdca_aux_unchanged_loss",
    ]
    if args.pdca_aux:
        totals.update({key: 0.0 for key in pdca_stat_keys})
    last_grad_norm = 0.0
    last_amp_scale = float(ctx.scaler.get_scale()) if args.amp else 1.0
    amp_skipped_steps = 0
    total_micro_batches = len(ctx.trainloader)
    iterator = ctx.tqdm(ctx.trainloader) if is_main_process(args) else ctx.trainloader
    ctx.optimizer.zero_grad(set_to_none=True)

    for step, (img1, img2, img3, mask1, mask2, mask3, mask_bn, sample_id) in enumerate(iterator):
        del sample_id
        img1 = img1.float().to(ctx.device, non_blocking=True)
        img2 = img2.float().to(ctx.device, non_blocking=True)
        img3 = img3.float().to(ctx.device, non_blocking=True)
        mask1 = mask1.long().to(ctx.device, non_blocking=True)
        mask2 = mask2.long().to(ctx.device, non_blocking=True)
        mask3 = mask3.long().to(ctx.device, non_blocking=True)
        mask_bn = mask_bn.float().to(ctx.device, non_blocking=True)
        x = torch.stack([img1, img2, img3], dim=0)
        update_now = should_update_optimizer(step, total_micro_batches, args.accum_steps)
        sync_context = ctx.model.no_sync() if args.distributed and not update_now else nullcontext()

        with sync_context:
            with torch.cuda.amp.autocast(enabled=args.amp):
                aux = None
                if args.pdca_aux or args.enable_pairrel_aux:
                    out1, out2, out3, change_logits, aux = ctx.model(x, return_aux=True)
                    outputs = (out1, out2, out3, change_logits)
                else:
                    outputs = ctx.model(x)
            with torch.cuda.amp.autocast(enabled=False):
                loss_main, loss_seg, loss_bn, loss_similarity, loss_tl = compute_losses(
                    outputs,
                    (mask1, mask2, mask3, mask_bn),
                    ctx.criteria,
                )
                loss_pairrel = loss_main.new_zeros(())
                pairrel_warm = 0.0
                pairrel_stats = {}
                if args.enable_pairrel_aux:
                    loss_pairrel, pairrel_stats = ctx.criteria["pairrel"](
                        aux["encoder_features"],
                        mask_bn,
                    )
                    pairrel_warm = min(
                        1.0,
                        float(epoch + 1) / max(1, args.pairrel_aux_warmup_epochs),
                    )
                loss_pdca_aux = loss_main.new_zeros(())
                pdca_aux_weight_eff = 0.0
                pdca_stats = {}
                if args.pdca_aux:
                    loss_pdca_aux, pdca_stats = ctx.criteria["pdca_aux"](
                        aux["encoder_aux"],
                        mask_bn,
                        scale_key=args.pdca_aux_scale_key,
                        tau_neg=args.pdca_aux_tau_neg,
                        tau_pos=args.pdca_aux_tau_pos,
                        ambiguous_weight=args.pdca_aux_ambiguous_weight,
                    )
                    pdca_aux_weight_eff = args.pdca_aux_weight * min(
                        1.0,
                        float(epoch) / max(float(args.pdca_aux_warmup_epochs), 1e-6),
                    )
                loss = (
                    loss_main
                    + args.pairrel_aux_weight * pairrel_warm * loss_pairrel
                    + pdca_aux_weight_eff * loss_pdca_aux
                )
                ensure_finite_tensor("loss", loss)
                loss_to_backward = loss / args.accum_steps

            totals["loss"] += float(loss.detach())
            totals["seg"] += float(loss_seg.detach())
            totals["bn"] += float(loss_bn.detach())
            totals["similarity"] += float(loss_similarity.detach())
            totals["tl"] += float(loss_tl.detach())
            totals["pairrel"] += float(loss_pairrel.detach())
            totals["pairrel_warm"] += float(pairrel_warm)
            for key in pairrel_stat_keys:
                totals[key] += float(pairrel_stats[key].detach())
            if args.pdca_aux:
                totals["pdca_aux_weight_eff"] += float(pdca_aux_weight_eff)
                for key in pdca_stat_keys:
                    totals[key] += float(pdca_stats[key])
            if aux is not None:
                del aux

            if args.amp:
                ctx.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        if update_now:
            params_with_grad = trainable_parameters(ctx.model)
            if args.amp:
                previous_scale = float(ctx.scaler.get_scale())
                ctx.scaler.unscale_(ctx.optimizer)
                grad_norm_tensor = compute_grad_norm(params_with_grad)
                local_grad_finite = bool(torch.isfinite(grad_norm_tensor.detach()).all().item())
                grad_finite = synchronize_finite_flag(local_grad_finite, ctx.device)
                if grad_finite:
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params_with_grad, args.grad_clip_norm)
                    ctx.scaler.step(ctx.optimizer)
                    ctx.scaler.update()
                else:
                    if args.amp_debug_nonfinite:
                        bad_name = first_nonfinite_gradient_name(ctx.model)
                        print(
                            "AMP non-finite gradient skipped: rank=%d, param=%s, grad_norm=%s, scale=%.1f"
                            % (args.rank, bad_name or "none_on_this_rank", str(float(grad_norm_tensor.detach())), previous_scale),
                            flush=True,
                        )
                    backoff_grad_scaler(ctx.scaler, previous_scale)
                last_amp_scale = float(ctx.scaler.get_scale())
                stepped = should_advance_after_amp_step(grad_finite, previous_scale, last_amp_scale)
                if not stepped:
                    amp_skipped_steps += 1
            else:
                grad_norm_tensor = compute_grad_norm(params_with_grad)
                ensure_finite_tensor("grad_norm", grad_norm_tensor)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, args.grad_clip_norm)
                ctx.optimizer.step()
                stepped = True

            last_grad_norm = float(grad_norm_tensor.detach())
            if stepped:
                ctx.global_update_step += 1
                if ctx.scheduler_step_on_updates:
                    step_timm_scheduler_update(ctx.scheduler, ctx.global_update_step)
            ctx.optimizer.zero_grad(set_to_none=True)

        reset_snn_state(ctx.model, ctx.functional)
        if is_main_process(args):
            seen = step + 1
            description = (
                "Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f, TL Loss: %.3f"
                % (
                    totals["loss"] / seen,
                    totals["seg"] / seen,
                    totals["bn"] / seen,
                    totals["similarity"] / seen,
                    totals["tl"] / seen,
                )
            )
            if args.enable_pairrel_aux:
                description += ", Loss_pairrel: %.3f, PairRel_warm: %.3f" % (
                    totals["pairrel"] / seen,
                    totals["pairrel_warm"] / seen,
                )
                for key in pairrel_stat_keys:
                    description += ", %s: %.3f" % (key, totals[key] / seen)
            if args.pdca_aux:
                description += ", PDCA-RAS: %.3f, PDCA_w: %.3f" % (
                    totals["pdca_aux_loss"] / seen,
                    totals["pdca_aux_weight_eff"] / seen,
                )
            iterator.set_description(description)
            if ctx.writer is not None:
                running_iter = epoch * total_micro_batches + seen
                ctx.writer.add_scalar("train total_loss", totals["loss"] / seen, running_iter)
                ctx.writer.add_scalar("train seg_loss", totals["seg"] / seen, running_iter)
                ctx.writer.add_scalar("train bn_loss", totals["bn"] / seen, running_iter)
                ctx.writer.add_scalar("train sc_loss", totals["similarity"] / seen, running_iter)
                if args.enable_pairrel_aux:
                    ctx.writer.add_scalar("Loss_pairrel", totals["pairrel"] / seen, running_iter)
                    ctx.writer.add_scalar("PairRel_warm", totals["pairrel_warm"] / seen, running_iter)
                    for key in pairrel_stat_keys:
                        ctx.writer.add_scalar(key, totals[key] / seen, running_iter)
                if args.pdca_aux:
                    ctx.writer.add_scalar(
                        "train pdca_aux_weight_eff",
                        totals["pdca_aux_weight_eff"] / seen,
                        running_iter,
                    )
                    for key in pdca_stat_keys:
                        ctx.writer.add_scalar("train " + key, totals[key] / seen, running_iter)
                ctx.writer.add_scalar("lr", ctx.optimizer.param_groups[0]["lr"], running_iter)
                ctx.writer.add_scalar("train grad_norm", last_grad_norm, running_iter)
                if args.amp:
                    ctx.writer.add_scalar("amp scale", last_amp_scale, running_iter)
                    ctx.writer.add_scalar("amp skipped_steps", amp_skipped_steps, running_iter)


def validate_t1_t3(ctx, epoch):
    args = ctx.args
    if not is_main_process(args):
        if args.distributed:
            dist.barrier()
            return broadcast_main_float(0.0, ctx.device)
        return ctx.previous_best

    model_for_eval = unwrap_model(ctx.model)
    model_for_eval.eval()
    metric = ctx.IOUandSek(num_classes=len(ctx.RS.ST_CLASSES))
    score = miou = sek = fscd = oa = sc_precision = sc_recall = 0.0
    iterator = ctx.tqdm(ctx.valloader)

    with torch.no_grad():
        for img1, img2, img3, mask1, mask2, mask3, mask_bn, sample_id in iterator:
            del sample_id, mask2
            img1 = img1.float().to(ctx.device, non_blocking=True)
            img2 = img2.float().to(ctx.device, non_blocking=True)
            img3 = img3.float().to(ctx.device, non_blocking=True)
            x = torch.stack([img1, img2, img3], dim=0)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out1, _out2, out3, change_logits = model_for_eval(x)

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
            _, score, miou, sek, fscd, oa, sc_precision, sc_recall = metric.evaluate_SECOND()
            reset_snn_state(model_for_eval, ctx.functional)
            iterator.set_description(
                "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f"
                % (miou, sek, score, fscd, oa, sc_precision, sc_recall)
            )

    save_checkpoint(ctx, epoch, score, miou, sek, fscd, oa)
    if ctx.writer is not None:
        ctx.writer.add_scalar("val_Score", score, epoch)
        ctx.writer.add_scalar("val_mIOU", miou, epoch)
        ctx.writer.add_scalar("val_Sek", sek, epoch)
        ctx.writer.add_scalar("val_Fscd", fscd, epoch)
        ctx.writer.add_scalar("val_OA", oa, epoch)

    if args.distributed:
        dist.barrier()
        score = broadcast_main_float(score, ctx.device)
    return score


def checkpoint_dir(args):
    return os.path.join(args.output_dir, args.data_name, args.Net_name, args.backbone)


def checkpoint_payload(ctx, epoch):
    return {
        "epoch": epoch,
        "model": unwrap_model(ctx.model).state_dict(),
        "optimizer": ctx.optimizer.state_dict(),
        "scaler": ctx.scaler.state_dict() if ctx.args.amp else None,
        "scheduler": scheduler_state_dict(ctx.scheduler),
        "best_metric": ctx.previous_best,
        "global_update_step": ctx.global_update_step,
        "args": vars(ctx.args),
    }


def save_checkpoint(ctx, epoch, score, miou, sek, fscd, oa):
    os.makedirs(checkpoint_dir(ctx.args), exist_ok=True)
    is_best = score >= ctx.previous_best
    if is_best:
        ctx.previous_best = score
    latest_path = os.path.join(checkpoint_dir(ctx.args), "latest.pth")
    torch.save(checkpoint_payload(ctx, epoch), latest_path)
    if is_best:
        best_name = "epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" % (
            epoch,
            score * 100,
            miou * 100,
            sek * 100,
            fscd * 100,
            oa * 100,
        )
        torch.save(checkpoint_payload(ctx, epoch), os.path.join(checkpoint_dir(ctx.args), best_name))


def load_checkpoint(ctx, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    model = unwrap_model(ctx.model)
    validate_pdca_resume_state(state_dict, model.state_dict().keys(), ctx.args.pdca_aux)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            ctx.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None and ctx.args.amp:
            ctx.scaler.load_state_dict(checkpoint["scaler"])
        ctx.previous_best = float(checkpoint.get("best_metric", checkpoint.get("previous_best", 0.0)))
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
        print(f"Resumed checkpoint from {checkpoint_path}: {incompatible}")


def step_scheduler_epoch(ctx, epoch, metric):
    if ctx.scheduler_step_on_updates:
        return
    scheduler_metric = metric if ctx.args.sched == "plateau" else None
    step_timm_scheduler_epoch(ctx.scheduler, epoch + 1, scheduler_metric)


def make_writer(args):
    if not is_main_process(args):
        return None
    if args.log_dir is None:
        args.log_dir = os.path.join(working_path, "logs", args.data_name, args.Net_name, args.backbone + "_v6")
    os.makedirs(args.log_dir, exist_ok=True)
    from tensorboardX import SummaryWriter

    return SummaryWriter(args.log_dir)


def print_startup(args, train_micro_batches, updates_per_epoch, actual_updates, total_updates, warmup_updates):
    if not is_main_process(args):
        return
    global_batch = args.batch_size * args.world_size
    effective_batch = global_batch * args.accum_steps
    # reference_effective = args.reference_batch_size * args.reference_accum_steps
    print(args)
    print(
        "batch_size_per_gpu=%d, world_size=%d, global_batch_size=%d, accum_steps=%d, effective_batch_size=%d"
        % (args.batch_size, args.world_size, global_batch, args.accum_steps, effective_batch)
    )
    if args.reference_batch_size is not None:
        print(
            "reference_batch_size=%d, reference_accum_steps=%d"
            % (args.reference_batch_size, args.reference_accum_steps)
        )
    print(
        "train_micro_batches_per_epoch=%d, updates_per_epoch=%d, actual_total_updates=%d, "
        "lr_total_updates=%d, warmup_updates=%.1f"
        % (train_micro_batches, updates_per_epoch, actual_updates, total_updates, warmup_updates)
    )
    print(
        "optimizer=%s, scheduler=%s, sched_on_updates=%s, lr=%.6g, amp=%s, sync_bn=%s"
        % (args.opt, args.sched, args.sched_on_updates, args.lr, args.amp, args.sync_bn)
    )



def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    init_distributed_mode(args)
    validate_args(args)
    seed_everything(args.seed, args.rank)

    from spikingjelly.clock_driven import functional
    from tqdm import tqdm
    from utils.metric import IOUandSek

    writer = None
    ctx = None
    try:
        device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
        RS, trainset, trainloader, train_sampler, valloader = build_dataloaders(args)
        model = build_model(args, RS, device)
        criteria = build_loss_functions(args)
        optimizer = build_timm_optimizer(model, args)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        train_micro_batches = len(trainloader)
        updates_per_epoch = max(1, math.ceil(train_micro_batches / args.accum_steps))
        actual_total_updates = updates_per_epoch * args.epochs
        total_updates, warmup_updates = resolve_reference_update_counts(args, len(trainset), actual_total_updates)
        scheduler, _num_epochs, scheduler_args, _scheduler_updates_per_epoch = build_timm_scheduler(
            args,
            optimizer,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )
        writer = make_writer(args)
        ctx = SimpleNamespace(
            args=args,
            RS=RS,
            IOUandSek=IOUandSek,
            functional=functional,
            tqdm=tqdm,
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
            writer=writer,
            train_micro_batches_per_epoch=train_micro_batches,
            updates_per_epoch=updates_per_epoch,
            global_update_step=0,
            start_epoch=0,
            previous_best=0.0,
        )
        print_startup(args, train_micro_batches, updates_per_epoch, actual_total_updates, total_updates, warmup_updates)
        if args.resume:
            load_checkpoint(ctx, args.resume)
        if args.eval_only:
            validate_t1_t3(ctx, ctx.start_epoch)
            return
        for epoch in range(ctx.start_epoch, args.epochs):
            if is_main_process(args):
                print(
                    "\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.5f"
                    % (epoch, ctx.optimizer.param_groups[0]["lr"], ctx.previous_best)
                )
            train_one_epoch(ctx, epoch)
            score = validate_t1_t3(ctx, epoch)
            step_scheduler_epoch(ctx, epoch, score)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()

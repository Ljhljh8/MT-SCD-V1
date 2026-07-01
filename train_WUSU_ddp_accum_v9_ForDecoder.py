"""MTSCD-PairRelAux-V1.1 experimental training entrypoint."""

import argparse
import math
import os
import random
from datetime import datetime
import sys

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
def default_log_dir(args):
    return os.path.join(
        working_path,
        "logs",
        args.data_name,
        args.Net_name,
        args.backbone + "_v6",
    )


class TeeStream:
    """Write console output to both terminal and a log file."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.primary, name)


def setup_text_log(args):
    """Create a timestamped text log and tee stdout/stderr on the main process."""
    if not getattr(args, "save_text_log", True):
        return None
    if not is_main_process(args):
        return None

    base_log_dir = args.text_log_dir
    if base_log_dir is None:
        base_log_dir = os.path.join(args.log_dir or default_log_dir(args), "text_logs")

    os.makedirs(base_log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_path = os.path.join(base_log_dir, f"train_{timestamp}.log")

    log_file = open(log_path, mode="a", buffering=1, encoding="utf-8")

    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    args.text_log_path = log_path
    print("Text log file: %s" % log_path, flush=True)
    print("Command: %s" % " ".join(sys.argv), flush=True)

    return log_file


def close_text_log(log_file):
    if log_file is None:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


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
    # parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="checkpoints_v6")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")

    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--text_log_dir", "--text-log-dir", dest="text_log_dir", type=str, default=None)
    parser.add_argument("--save-text-log", dest="save_text_log", action="store_true")
    parser.add_argument("--no-save-text-log", dest="save_text_log", action="store_false")
    parser.set_defaults(save_text_log=True)

    # parser.add_argument("--lightweight", dest="lightweight", action="store_true")
    parser.add_argument("--pretrain_from", "--pretrain-from", dest="pretrain_from", type=str, default=None)
    parser.add_argument("--load_from", "--load-from", dest="load_from", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str2bool, default=True)

    # parser.add_argument("--Lambda", type=float, default=0.00005)
    parser.add_argument("--relation-mode", choices=["prg", "pdca", "none"], default="pdca")
    parser.add_argument("--enable-pairrel-aux", action="store_true")
    parser.add_argument(
        "--pairrel-mode",
        choices=["unchanged_only", "weak_contrastive", "contrastive"],
        default="unchanged_only",
    )
    parser.add_argument("--pairrel-aux-weight", type=float, default=0.02)
    parser.add_argument("--pairrel-aux-start-epoch", type=int, default=10)
    parser.add_argument("--pairrel-aux-warmup-epochs", type=int, default=10)
    parser.add_argument("--pairrel-aux-scales", type=str, default="3")
    parser.add_argument("--pairrel-margin", type=float, default=0.5)
    parser.add_argument("--pairrel-tau-unchanged", type=float, default=0.05)
    parser.add_argument("--pairrel-tau-changed", type=float, default=0.50)
    parser.add_argument(
        "--pairrel-changed-weight",
        type=float,
        default=0.0,
        help="Use a small value such as 0.1 with weak_contrastive.",
    )
    parser.add_argument("--pdca_aux", action="store_true", default=False)
    parser.add_argument("--pdca_aux_weight", type=float, default=0.05)
    parser.add_argument("--pdca_aux_warmup_epochs", type=float, default=5.0)
    parser.add_argument("--pdca_aux_scale_key", type=str, default="3")
    parser.add_argument("--pdca_aux_tau_neg", type=float, default=0.05)
    parser.add_argument("--pdca_aux_tau_pos", type=float, default=0.20)
    parser.add_argument("--pdca_aux_ambiguous_weight", type=float, default=0.25)
    parser.add_argument("--use-pdca-guided-pair-decoder", action="store_true")
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
    parser.add_argument("--pdca-dend-prior-stats", type=str2bool, default=True)

    parser.add_argument("--no-pdca-guidance", action="store_true")
    parser.add_argument("--no-detach-pdca-guidance", action="store_true")
    parser.add_argument("--pair-bcd-lambda-adj", type=float, default=1.0)
    parser.add_argument("--pair-bcd-lambda-13", type=float, default=1.0)
    parser.add_argument("--pair-bcd-dice-weight", type=float, default=1.0)

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

    parser.add_argument("--pdca-context-spike-mode", default="none",
                        choices=["none", "weights", "values", "both", "context"])
    parser.add_argument("--pdca-context-spike-capacity", default=8, type=int)
    parser.add_argument("--pdca-context-spike-threshold", default=1.0, type=float)
    parser.add_argument("--pdca-context-spike-signed", type=str2bool, default=True)
    parser.add_argument("--pdca-context-spike-detach", action="store_true")
    parser.add_argument("--pdca-context-spike-topk", default=2, type=int)
    parser.add_argument("--pdca-context-spike-tau", default=1.0, type=float)
    parser.add_argument("--pdca-context-spike-warmup-epoch", default=0, type=int)

    return parser


def validate_args(args):
    if args.pdca_aux and args.enable_pairrel_aux:
        raise ValueError("This experiment tests PDCA-RAS only. Please disable PairRelAux.")
    if args.use_pdca_guided_pair_decoder and (args.pdca_aux or args.enable_pairrel_aux):
        raise ValueError(
            "PDCA-guided pair decoder V1 cannot be combined with --pdca_aux or --enable-pairrel-aux."
        )
    if (
        args.use_pdca_guided_pair_decoder
        and not args.no_pdca_guidance
        and args.relation_mode != "pdca"
    ):
        raise ValueError(
            "--use-pdca-guided-pair-decoder requires --relation-mode pdca unless --no-pdca-guidance is set."
        )
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr < 0 or args.weight_decay < 0 or args.grad_clip_norm < 0:
        raise ValueError("lr, weight decay, and grad clip must be non-negative")
    if args.pdca_context_spike_capacity <= 0:
        raise ValueError("--pdca-context-spike-capacity must be > 0")
    if args.pdca_context_spike_threshold <= 0:
        raise ValueError("--pdca-context-spike-threshold must be > 0")
    if args.pdca_context_spike_topk <= 0:
        raise ValueError("--pdca-context-spike-topk must be > 0")
    if args.pdca_context_spike_tau <= 0:
        raise ValueError("--pdca-context-spike-tau must be > 0")
    if args.pdca_context_spike_warmup_epoch < 0:
        raise ValueError("--pdca-context-spike-warmup-epoch must be >= 0")

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
    if (
        args.pairrel_aux_weight < 0
        or args.pairrel_aux_start_epoch < 0
        or args.pairrel_aux_warmup_epochs < 0
        or args.pairrel_margin < 0
        or args.pairrel_changed_weight < 0
    ):
        raise ValueError(
            "pairrel weight, start epoch, warmup epochs, margin, and changed weight must be non-negative"
        )
    if not 0 <= args.pairrel_tau_unchanged <= args.pairrel_tau_changed <= 1:
        raise ValueError("pairrel thresholds must satisfy 0 <= unchanged <= changed <= 1")
    if args.pdca_aux_weight < 0 or args.pdca_aux_warmup_epochs < 0:
        raise ValueError("PDCA aux weight and warmup epochs must be non-negative")
    if (
        args.pair_bcd_lambda_adj < 0
        or args.pair_bcd_lambda_13 < 0
        or args.pair_bcd_dice_weight < 0
    ):
        raise ValueError("pair BCD weights must be non-negative")
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

def set_pdca_context_spike_mode(model, mode: str):
    module = model.module if hasattr(model, "module") else model
    for m in module.modules():
        if hasattr(m, "pdca_context_spike_runtime_mode"):
            m.pdca_context_spike_runtime_mode = mode


def should_update_optimizer(step, num_steps, accum_steps):
    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    return ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)


def pairrel_warmup(epoch, start_epoch, warmup_epochs):
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, float(epoch - start_epoch + 1) / float(warmup_epochs))


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


def add_finite_scalar(writer, tag, value, step):
    value = float(value)
    if math.isfinite(value):
        writer.add_scalar(tag, value, step)


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
    from models.GSTMSCD_MTSCD_Snn_ForDecoder import GSTMSCD_WUSU as Net

    model = Net(
        args.backbone,
        args.pretrained,
        len(RS.ST_CLASSES),
        # args.lightweight,
        # args.M,
        # args.Lambda,
        relation_mode=args.relation_mode,
        use_pdca_relation_aux=args.pdca_aux,
        use_pairrel_aux=args.enable_pairrel_aux,
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
        # pdca_context_spike_mode=args.pdca_context_spike_mode,
        # pdca_context_spike_capacity=args.pdca_context_spike_capacity,
        # pdca_context_spike_threshold=args.pdca_context_spike_threshold,
        # pdca_context_spike_signed=args.pdca_context_spike_signed,
        # pdca_context_spike_detach=args.pdca_context_spike_detach,
        # pdca_context_spike_topk=args.pdca_context_spike_topk,
        # pdca_context_spike_tau=args.pdca_context_spike_tau,
        # pdca_context_spike_stats=True,
    )
    if is_main_process(args):
        if args.use_pdca_guided_pair_decoder:
            print(
                "PDCA-guided pair decoder enabled: legacy change_decoder/change_head are not built. "
                "Old full-checkpoint resume should use a fresh experiment or non-strict warm start."
            )
        else:
            print(
                "PDCA-guided pair decoder disabled: legacy change_decoder/change_head are built; "
                "strict old checkpoint loading remains compatible."
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
            if args.use_pdca_guided_pair_decoder:
                raise RuntimeError(
                    "Strict --load_from failed for PDCA-guided pair decoder. Old legacy checkpoints "
                    "contain change_decoder/change_head instead of pair_change_decoder; use --pretrain_from "
                    "for non-strict warm-start or start a fresh experiment."
                ) from exc
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
    from utils.loss import ChangeSimilarity, DiceLoss, PairwiseBinaryChangeLoss, PairwiseRelationAuxLoss_V11
    from utils.pdca_aux_loss import pdca_relation_aux_loss

    criteria = {
        "seg": CrossEntropyLoss(ignore_index=-1),
        "bce": BCEWithLogitsLoss(reduction="none"),
        "dice": DiceLoss(activation="none"),
        "similarity": ChangeSimilarity(),
    }
    criteria["pairrel"] = (
        PairwiseRelationAuxLoss_V11(
            scales=args.pairrel_aux_scale_ids,
            mode=args.pairrel_mode,
            changed_weight=args.pairrel_changed_weight,
            margin=args.pairrel_margin,
            tau_unchanged=args.pairrel_tau_unchanged,
            tau_changed=args.pairrel_tau_changed,
        )
        if args.enable_pairrel_aux
        else None
    )
    criteria["pdca_aux"] = pdca_relation_aux_loss if args.pdca_aux else None
    criteria["pair_bcd"] = (
        PairwiseBinaryChangeLoss(
            lambda_adj=args.pair_bcd_lambda_adj,
            lambda_13=args.pair_bcd_lambda_13,
            dice_weight=args.pair_bcd_dice_weight,
        )
        if args.use_pdca_guided_pair_decoder
        else None
    )
    return criteria


def compute_binary_change_loss(change_logits, mask_bn, criterion_bce, criterion_dice):
    logits = change_logits.float()
    target = mask_bn.float()
    loss_bce = criterion_bce(logits, target)
    loss_bce[target == 1] *= 2
    loss_bce = loss_bce.mean()
    loss_dice = criterion_dice(torch.sigmoid(logits), target)
    return loss_bce + loss_dice


def compute_losses(outputs, masks, criteria, pair_targets=None, change_logits_dict=None):
    out1, out2, out3, change_logits = outputs
    mask1, mask2, mask3, mask_bn = masks
    loss1 = criteria["seg"](out1.float(), mask1 - 1)
    loss2 = criteria["seg"](out2.float(), mask2 - 1)
    loss3 = criteria["seg"](out3.float(), mask3 - 1)
    loss_seg = (loss1 + loss2 + loss3) / 3
    # loss_similarity = criteria["similarity"](out1.float(), out3.float(), mask_bn)
    pair_bcd_stats = {}
    if criteria.get("pair_bcd") is not None:
        if pair_targets is None or change_logits_dict is None:
            raise RuntimeError("pairwise BCD loss requires pair_targets and change_logits_dict")
        loss_similarity_1 = criteria["similarity"](out1.float(), out2.float(), pair_targets["t1_to_t2"]["target"])
        loss_similarity_2 = criteria["similarity"](out2.float(), out3.float(), pair_targets["t2_to_t3"]["target"])
        loss_similarity_3 = criteria["similarity"](out1.float(), out3.float(), pair_targets["t1_to_t3"]["target"])
        loss_similarity = (loss_similarity_1+loss_similarity_2+loss_similarity_3).mean()
        loss_bn, pair_bcd_stats = criteria["pair_bcd"](change_logits_dict, pair_targets)
    else:
        loss_similarity = criteria["similarity"](out1.float(), out3.float(), mask_bn)
        loss_bn = compute_binary_change_loss(change_logits, mask_bn, criteria["bce"], criteria["dice"])
    loss = loss_bn + loss_seg + loss_similarity
    loss_tl = torch.zeros((), device=loss.device, dtype=loss.dtype)
    return loss, loss_seg, loss_bn, loss_similarity, loss_tl, pair_bcd_stats


def collect_pair_gate_mean_stats(aux):
    gate_debug = (aux or {}).get("pair_gate_debug", {}).get("gate", {})
    stats = {}
    for pair_key in ("t1_to_t2", "t2_to_t3", "t1_to_t3"):
        gates = gate_debug.get(pair_key, [])
        if gates:
            stats["gate_mean_" + pair_key] = torch.stack(
                [gate.detach().float().mean() for gate in gates]
            ).mean()
    return stats


def train_one_epoch(ctx, epoch):
    args = ctx.args
    from utils.loss import make_pairwise_change_targets, pairwise_c13_mismatch_stats

    if ctx.train_sampler is not None:
        ctx.train_sampler.set_epoch(epoch)
    ctx.model.train()
    if args.freeze_bn:
        freeze_batch_norm(unwrap_model(ctx.model))
    pairrel_warm = (
        pairrel_warmup(
            epoch,
            args.pairrel_aux_start_epoch,
            args.pairrel_aux_warmup_epochs,
        )
        if args.enable_pairrel_aux
        else 0.0
    )
    pairrel_active = bool(
        args.enable_pairrel_aux
        and args.relation_mode == "pdca"
        and pairrel_warm > 0.0
    )

    totals = {
        "loss": 0.0,
        "seg": 0.0,
        "bn": 0.0,
        "similarity": 0.0,
        "tl": 0.0,
        "pairrel": 0.0,
        "pairrel_effective": 0.0,
        "pairrel_warm": 0.0,
        "pdca_aux_weight_eff": 0.0,
    }
    pair_bcd_stat_keys = []
    gate_stat_keys = []
    if args.use_pdca_guided_pair_decoder:
        totals["pair_bcd"] = 0.0
        for pair_key in ("t1_to_t2", "t2_to_t3", "t1_to_t3"):
            pair_bcd_stat_keys.extend(
                [
                    "pair_bcd_loss_" + pair_key,
                    "pair_bcd_valid_ratio_" + pair_key,
                    "pair_bcd_pos_ratio_" + pair_key,
                ]
            )
            gate_stat_keys.append("gate_mean_" + pair_key)
        pair_bcd_stat_keys.extend(
            [
                "pair_bcd_c13_mismatch_ratio",
                "pair_bcd_c13_valid_ratio",
            ]
        )
        totals.update({key: 0.0 for key in pair_bcd_stat_keys + gate_stat_keys})
    pairrel_stat_keys = []
    if args.enable_pairrel_aux:
        for scale in args.pairrel_aux_scale_ids:
            pairrel_stat_keys.extend(
                [
                    "pairrel_loss_s%d" % scale,
                    "pairrel_loss_unchanged_s%d" % scale,
                    "pairrel_loss_changed_s%d" % scale,
                    "pairrel_valid_ratio_s%d" % scale,
                    "pairrel_valid_weight_sum_s%d" % scale,
                    "pairrel_unchanged_ratio_s%d" % scale,
                    "pairrel_changed_ratio_s%d" % scale,
                    "pairrel_ambiguous_ratio_s%d" % scale,
                    "pairrel_unchanged_weight_sum_s%d" % scale,
                    "pairrel_changed_weight_sum_s%d" % scale,
                    "pairrel_dist_unchanged_s%d" % scale,
                    "pairrel_dist_changed_s%d" % scale,
                    "pairrel_dist_gap_changed_minus_unchanged_s%d" % scale,
                    "pairrel_hinge_active_ratio_s%d" % scale,
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
    warned_c13_mismatch = False
    # epoch loop
    if epoch < args.pdca_context_spike_warmup_epoch:
        set_pdca_context_spike_mode(ctx.model, "none")
    else:
        set_pdca_context_spike_mode(ctx.model, args.pdca_context_spike_mode)

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
                pairrel_needs_aux = pairrel_active
                pdca_needs_aux = bool(args.pdca_aux)
                pair_decoder_needs_aux = bool(args.use_pdca_guided_pair_decoder)
                if pairrel_needs_aux or pdca_needs_aux or pair_decoder_needs_aux:
                    out1, out2, out3, change_logits, aux = ctx.model(x, return_aux=True)
                    outputs = (out1, out2, out3, change_logits)
                else:
                    outputs = ctx.model(x)
            with torch.cuda.amp.autocast(enabled=False):
                pair_targets = None
                pair_bcd_stats = {}
                if args.use_pdca_guided_pair_decoder:
                    sem_targets = {
                        "t1": mask1 - 1,
                        "t2": mask2 - 1,
                        "t3": mask3 - 1,
                    }
                    pair_targets = make_pairwise_change_targets(sem_targets, ignore_index=-1)
                    pair_bcd_stats.update(pairwise_c13_mismatch_stats(pair_targets, mask_bn))
                    mismatch_ratio = float(pair_bcd_stats["pair_bcd_c13_mismatch_ratio"].detach().item())
                    if mismatch_ratio > 0.2 and is_main_process(args) and not warned_c13_mismatch:
                        print(
                            "NEEDS_DATA_AUDIT: pair_bcd_c13_mismatch_ratio=%.4f exceeds 0.2"
                            % mismatch_ratio
                        )
                        warned_c13_mismatch = True
                loss_main, loss_seg, loss_bn, loss_similarity, loss_tl, loss_pair_bcd_stats = compute_losses(
                    outputs,
                    (mask1, mask2, mask3, mask_bn),
                    ctx.criteria,
                    pair_targets=pair_targets,
                    change_logits_dict=(aux or {}).get("change_logits_dict"),
                )
                pair_bcd_stats.update(loss_pair_bcd_stats)
                if args.use_pdca_guided_pair_decoder:
                    pair_bcd_stats.update(collect_pair_gate_mean_stats(aux))
                loss_pairrel = loss_main.new_zeros(())
                loss_pairrel_effective = loss_main.new_zeros(())
                pairrel_stats = {}
                if pairrel_active:
                    loss_pairrel, pairrel_stats = ctx.criteria["pairrel"](
                        aux["encoder_features"],
                        mask_bn,
                    )
                    loss_pairrel_effective = (
                        args.pairrel_aux_weight * pairrel_warm * loss_pairrel
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
                    + loss_pairrel_effective
                    + pdca_aux_weight_eff * loss_pdca_aux
                )
                ensure_finite_tensor("loss", loss)
                loss_to_backward = loss / args.accum_steps

            totals["loss"] += float(loss.detach())
            totals["seg"] += float(loss_seg.detach())
            totals["bn"] += float(loss_bn.detach())
            totals["similarity"] += float(loss_similarity.detach())
            totals["tl"] += float(loss_tl.detach())
            if args.use_pdca_guided_pair_decoder:
                totals["pair_bcd"] += float(loss_bn.detach())
                for key in pair_bcd_stat_keys + gate_stat_keys:
                    totals[key] += float(pair_bcd_stats[key].detach())
            totals["pairrel"] += float(loss_pairrel.detach())
            totals["pairrel_effective"] += float(loss_pairrel_effective.detach())
            totals["pairrel_warm"] += float(pairrel_warm)
            if pairrel_active:
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
                description += (
                    ", Loss_pairrel: %.3f, Loss_pairrel_effective: %.3f, "
                    "PairRel_warm: %.3f, PairRel_active: %d"
                ) % (
                    totals["pairrel"] / seen,
                    totals["pairrel_effective"] / seen,
                    totals["pairrel_warm"] / seen,
                    int(pairrel_active),
                )
                for key in pairrel_stat_keys:
                    description += ", %s: %.3f" % (key, totals[key] / seen)
            if args.use_pdca_guided_pair_decoder:
                description += ", Loss_pair_bcd: %.3f, C13_mismatch: %.3f" % (
                    totals["pair_bcd"] / seen,
                    totals["pair_bcd_c13_mismatch_ratio"] / seen,
                )
                if totals["pair_bcd_c13_mismatch_ratio"] / seen > 0.2:
                    description += ", NEEDS_DATA_AUDIT"
                for key in pair_bcd_stat_keys + gate_stat_keys:
                    description += ", %s: %.3f" % (key, totals[key] / seen)
            if args.pdca_aux:
                description += ", PDCA-RAS: %.3f, PDCA_w: %.3f" % (
                    totals["pdca_aux_loss"] / seen,
                    totals["pdca_aux_weight_eff"] / seen,
                )
            iterator.set_description(description)
            if ctx.writer is not None:
                running_iter = epoch * total_micro_batches + seen
                add_finite_scalar(ctx.writer, "train total_loss", totals["loss"] / seen, running_iter)
                add_finite_scalar(ctx.writer, "train seg_loss", totals["seg"] / seen, running_iter)
                add_finite_scalar(ctx.writer, "train bn_loss", totals["bn"] / seen, running_iter)
                add_finite_scalar(ctx.writer, "train sc_loss", totals["similarity"] / seen, running_iter)
                if args.use_pdca_guided_pair_decoder:
                    add_finite_scalar(ctx.writer, "Loss_pair_bcd", totals["pair_bcd"] / seen, running_iter)
                    for key in pair_bcd_stat_keys + gate_stat_keys:
                        add_finite_scalar(ctx.writer, key, totals[key] / seen, running_iter)
                if args.enable_pairrel_aux:
                    add_finite_scalar(ctx.writer, "Loss_pairrel", totals["pairrel"] / seen, running_iter)
                    add_finite_scalar(
                        ctx.writer,
                        "Loss_pairrel_effective",
                        totals["pairrel_effective"] / seen,
                        running_iter,
                    )
                    add_finite_scalar(ctx.writer, "PairRel_warm", totals["pairrel_warm"] / seen, running_iter)
                    add_finite_scalar(ctx.writer, "PairRel_active", int(pairrel_active), running_iter)
                    for key in pairrel_stat_keys:
                        add_finite_scalar(ctx.writer, key, totals[key] / seen, running_iter)
                if args.pdca_aux:
                    add_finite_scalar(
                        ctx.writer,
                        "train pdca_aux_weight_eff",
                        totals["pdca_aux_weight_eff"] / seen,
                        running_iter,
                    )
                    for key in pdca_stat_keys:
                        add_finite_scalar(ctx.writer, "train " + key, totals[key] / seen, running_iter)
                add_finite_scalar(ctx.writer, "lr", ctx.optimizer.param_groups[0]["lr"], running_iter)
                add_finite_scalar(ctx.writer, "train grad_norm", last_grad_norm, running_iter)
                if args.amp:
                    add_finite_scalar(ctx.writer, "amp scale", last_amp_scale, running_iter)
                    add_finite_scalar(ctx.writer, "amp skipped_steps", amp_skipped_steps, running_iter)
    if is_main_process(args):
        denom = max(1, total_micro_batches)
        summary = (
            "TRAIN_EPOCH_SUMMARY epoch=%d, loss=%.6f, seg=%.6f, bn=%.6f, "
            "similarity=%.6f, tl=%.6f, lr=%.8g, grad_norm=%.6f"
            % (
                epoch,
                totals["loss"] / denom,
                totals["seg"] / denom,
                totals["bn"] / denom,
                totals["similarity"] / denom,
                totals["tl"] / denom,
                ctx.optimizer.param_groups[0]["lr"],
                last_grad_norm,
            )
        )
        if args.enable_pairrel_aux:
            summary += (
                ", pairrel=%.6f, pairrel_effective=%.6f, pairrel_warm=%.6f"
                % (
                    totals["pairrel"] / denom,
                    totals["pairrel_effective"] / denom,
                    totals["pairrel_warm"] / denom,
                )
            )
        if args.pdca_aux:
            summary += (
                ", pdca_aux=%.6f, pdca_aux_weight_eff=%.6f"
                % (
                    totals["pdca_aux_loss"] / denom,
                    totals["pdca_aux_weight_eff"] / denom,
                )
            )
        if args.use_pdca_guided_pair_decoder:
            summary += ", Loss_pair_bcd: %.3f, C13_mismatch: %.3f" % (
                totals["pair_bcd"] / seen,
                totals["pair_bcd_c13_mismatch_ratio"] / seen,
            )
            if totals["pair_bcd_c13_mismatch_ratio"] / seen > 0.2:
                description += ", NEEDS_DATA_AUDIT"
            for key in pair_bcd_stat_keys + gate_stat_keys:
                description += ", %s: %.3f" % (key, totals[key] / seen)

        if args.amp:
            summary += ", amp_scale=%.1f, amp_skipped_steps=%d" % (
                last_amp_scale,
                amp_skipped_steps,
            )
        print(summary, flush=True)


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
    print(
        "VAL_EPOCH_SUMMARY epoch=%d, score=%.6f, miou=%.6f, sek=%.6f, "
        "Fscd=%.6f, OA=%.6f, SC_Precision=%.6f, SC_Recall=%.6f"
        % (epoch, score, miou, sek, fscd, oa, sc_precision, sc_recall),
        flush=True,
    )
    if ctx.writer is not None:
        add_finite_scalar(ctx.writer, "val_Score", score, epoch)
        add_finite_scalar(ctx.writer, "val_mIOU", miou, epoch)
        add_finite_scalar(ctx.writer, "val_Sek", sek, epoch)
        add_finite_scalar(ctx.writer, "val_Fscd", fscd, epoch)
        add_finite_scalar(ctx.writer, "val_OA", oa, epoch)

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
        # "best_epoch": getattr(ctx, "best_epoch", -1),
        # "best_metrics": getattr(ctx, "best_metrics", {}),
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
    if ctx.args.use_pdca_guided_pair_decoder and is_main_process(ctx.args):
        print(
            "PDCA-guided pair decoder resume used strict=False; missing keys=%r, unexpected keys=%r. "
            "Old optimizer state may be incompatible with the new decoder."
            % (list(incompatible.missing_keys), list(incompatible.unexpected_keys))
        )
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            try:
                ctx.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                if ctx.args.use_pdca_guided_pair_decoder:
                    raise RuntimeError(
                        "Cannot resume old optimizer state with PDCA-guided pair decoder; "
                        "start a fresh experiment or use --pretrain_from."
                    ) from exc
                raise
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


# def make_writer(args):
#     if not is_main_process(args):
#         return None
#     if args.log_dir is None:
#         args.log_dir = os.path.join(working_path, "logs", args.data_name, args.Net_name, args.backbone + "_v6")
#     os.makedirs(args.log_dir, exist_ok=True)
#     from tensorboardX import SummaryWriter
#
#     return SummaryWriter(args.log_dir)
def make_writer(args):
    if not is_main_process(args):
        return None
    if args.log_dir is None:
        args.log_dir = default_log_dir(args)
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

    text_log_file = None
    writer = None
    ctx = None
    try:
        text_log_file = setup_text_log(args)

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
                if args.enable_pairrel_aux:
                    print(
                        "PairRel: mode=%s, start_epoch=%d, aux_weight=%.4f, changed_weight=%.4f"
                        % (
                            args.pairrel_mode,
                            args.pairrel_aux_start_epoch,
                            args.pairrel_aux_weight,
                            args.pairrel_changed_weight,
                        )
                    )
            train_one_epoch(ctx, epoch)
            score = validate_t1_t3(ctx, epoch)
            step_scheduler_epoch(ctx, epoch, score)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()
        close_text_log(text_log_file)


if __name__ == "__main__":
    main()

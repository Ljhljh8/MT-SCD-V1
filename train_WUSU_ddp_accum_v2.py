import argparse
import contextlib
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                print("Warning: tensorboardX/tensorboard is unavailable; TensorBoard logging is disabled.")

            def add_scalar(self, *args, **kwargs):
                return None

            def close(self):
                return None

import datasets.MultiSiamese_RS_ST_TL as RS
from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU as Net
from spikingjelly.clock_driven import functional
from utils.loss import ChangeSimilarity, DiceLoss
from utils.metric import IOUandSek
from utils.palette import color_map


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

    @staticmethod
    def visual_debug(feat_t1, feat_t2, feat_t3):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        loss_fn = TemporalLogicKLDivLoss()
        kl_12 = loss_fn.kl_divergence(feat_t1, feat_t2)[0].detach().cpu().numpy()
        axes[0].imshow(kl_12, cmap="jet")
        axes[0].set_title("T1-T2 KL Divergence")

        kl_23 = loss_fn.kl_divergence(feat_t2, feat_t3)[0].detach().cpu().numpy()
        axes[1].imshow(kl_23, cmap="jet")
        axes[1].set_title("T2-T3 KL Divergence")

        kl_13 = loss_fn.kl_divergence(feat_t1, feat_t3)[0].detach().cpu().numpy()
        axes[2].imshow(kl_13, cmap="jet")
        axes[2].set_title("T1-T3 KL Divergence")

        plt.tight_layout()
        plt.show()


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser("Semantic Change Detection")
        parser.add_argument("--data_name", type=str, default="WUSU")
        parser.add_argument("--Net_name", type=str, default="GSTMSCD")
        parser.add_argument("--backbone", type=str, default="GOST-Mamba")
        parser.add_argument("--data_root", type=str, default="/WUSU")
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
        parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
        parser.add_argument("--test_batch_size", "--test-batch-size", dest="test_batch_size", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument(
            "--lightweight",
            dest="lightweight",
            action="store_true",
            help="lightweight head for fewer parameters and faster speed",
        )
        parser.add_argument("--pretrain_from", type=str, help="train from a checkpoint")
        parser.add_argument("--load_from", type=str, help="load trained model to generate predictions of validation set")
        parser.add_argument("--pretrained", type=bool, default=True, help="initialize the backbone with pretrained parameters")
        parser.add_argument("--tta", dest="tta", action="store_true", help="test_time augmentation")
        parser.add_argument("--warmup", dest="warmup", default=True, action="store_true", help="warm up")
        parser.add_argument("--save_mask", dest="save_mask", action="store_true", help="save predictions of validation set during training")
        parser.add_argument(
            "--use_pseudo_label",
            dest="use_pseudo_label",
            action="store_true",
            help="use pseudo labels for re-training (must pseudo label first)",
        )
        parser.add_argument("--M", type=int, default=6)
        parser.add_argument("--Lambda", type=float, default=0.00005)
        parser.add_argument("--expander_K", "--expander-k", dest="expander_K", type=int, default=2, help="PAENTE intra-phase steps per phase")
        parser.add_argument("--expander_R", "--expander-r", dest="expander_R", type=int, default=0, help="PAENTE transition steps between adjacent phases")

        parser.add_argument("--accum_steps", "--accum-steps", dest="accum_steps", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--sync_bn", "--sync-bn", dest="sync_bn", action="store_true", help="convert BatchNorm to SyncBatchNorm under DDP")
        parser.add_argument("--freeze_bn", "--freeze-bn", dest="freeze_bn", action="store_true", help="freeze BatchNorm running statistics during training")
        parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--find_unused_parameters",
            "--find-unused-parameters",
            dest="find_unused_parameters",
            action="store_true",
            help="use DDP find_unused_parameters=True if the model has conditionally unused branches",
        )
        parser.add_argument(
            "--local_rank",
            "--local-rank",
            dest="local_rank",
            type=int,
            default=0,
            help="kept for compatibility; torchrun mainly uses LOCAL_RANK env var",
        )
        parser.add_argument("--amp", action="store_true", help="optional mixed precision training")
        parser.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", type=float, default=0.0, help="clip grad norm if > 0")
        parser.add_argument("--debug_iters", "--debug-iters", dest="debug_iters", type=int, default=0, help="run only N training iterations per epoch for debugging; 0 means full epoch")
        parser.add_argument("--disable_cudnn", "--disable-cudnn", dest="disable_cudnn", action="store_true", help="disable cuDNN convolution kernels as a fallback for cuDNN algorithm failures")
        parser.add_argument("--cudnn_benchmark", "--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", help="enable cuDNN benchmark for fixed-size inputs")
        parser.add_argument("--disable_mmcv_deform", "--disable-mmcv-deform", dest="disable_mmcv_deform", action="store_true", help="force DendFADCConv2d to use grouped Conv2d fallback instead of MMCV modulated deform conv")
        parser.add_argument("--disable_dendfadc", "--disable-dendfadc", dest="disable_dendfadc", action="store_true", help="disable DendFADC replacements in the SNN backbone")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()


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
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    args.device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")


def is_main_process(args):
    return (not getattr(args, "distributed", False)) or args.rank == 0


def cleanup_distributed(args):
    if getattr(args, "distributed", False) and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed, rank=0):
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)


def configure_cudnn(args):
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = args.cudnn_benchmark


def configure_dendfadc_backend(args):
    if args.disable_mmcv_deform:
        os.environ["DENDSN_FORCE_FALLBACK_GROUP_CONV"] = "1"
    else:
        os.environ.pop("DENDSN_FORCE_FALLBACK_GROUP_CONV", None)


def get_raw_model(model):
    return model.module if hasattr(model, "module") else model


def reset_spiking_state(model):
    functional.reset_net(get_raw_model(model))


def freeze_batchnorm_modules(model):
    raw_model = get_raw_model(model)
    for module in raw_model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


@contextlib.contextmanager
def suppress_stdout(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_model_weights(model, checkpoint_path, strict):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    model.load_state_dict(state_dict, strict=strict)


def build_optimizer(model, args):
    backbone_params = []
    other_params = []
    for name, param in get_raw_model(model).named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr * 1.0})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.main_process = is_main_process(args)

        args.log_dir = args.log_dir or os.path.join(working_path, "logs", args.data_name, args.Net_name, args.backbone)
        if self.main_process:
            os.makedirs(args.log_dir, exist_ok=True)
            self.writer = SummaryWriter(args.log_dir)
        else:
            self.writer = None

        with suppress_stdout(not self.main_process):
            trainset = RS.Data(mode="train", random_flip=True)
            valset = RS.Data(mode="val", random_flip=True) if self.main_process else None

        if args.distributed:
            self.train_sampler = DistributedSampler(
                trainset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.train_sampler = None

        self.trainloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        self.valloader = None
        if self.main_process:
            self.valloader = DataLoader(
                valset,
                batch_size=args.val_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=args.num_workers,
                drop_last=False,
            )

        with suppress_stdout(not self.main_process):
            self.model = Net(
                args.backbone,
                args.pretrained,
                len(RS.ST_CLASSES),
                args.lightweight,
                args.M,
                args.Lambda,
                expander_K=args.expander_K,
                expander_R=args.expander_R,
                use_dendfadc=not args.disable_dendfadc,
            )

        if args.pretrain_from:
            load_model_weights(self.model, args.pretrain_from, strict=False)
        if args.load_from:
            load_model_weights(self.model, args.load_from, strict=True)

        self.model = self.model.to(args.device)

        if args.distributed and args.sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.main_process:
                print("Converted BatchNorm layers to SyncBatchNorm.")

        if args.freeze_bn:
            freeze_batchnorm_modules(self.model)
            if self.main_process:
                print("Frozen BatchNorm modules.")

        self.optimizer = build_optimizer(self.model, args)

        if args.distributed:
            self.model = DDP(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )

        self.TCL = TemporalLogicKLDivLoss()
        self.criterion_seg = CrossEntropyLoss(ignore_index=-1)
        self.criterion_bn = BCELoss(reduction="none")
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self.train_micro_batches_per_epoch = len(self.trainloader)
        if args.debug_iters > 0:
            self.train_micro_batches_per_epoch = min(args.debug_iters, self.train_micro_batches_per_epoch)
        self.total_update_steps = math.ceil(self.train_micro_batches_per_epoch / args.accum_steps) * args.epochs
        self.total_update_steps = max(1, self.total_update_steps)
        self.update_steps = 0
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0

    def adjust_learning_rate(self):
        if self.args.warmup:
            warmup_steps = max(1, int(self.total_update_steps / 5))
            if self.update_steps < warmup_steps:
                lr = self.args.lr * float(self.update_steps + 1) / float(warmup_steps)
            else:
                progress = float(self.update_steps - warmup_steps) / float(max(1, self.total_update_steps - warmup_steps))
                lr = self.args.lr * (1.0 - progress) ** 1.5
        else:
            progress = float(self.update_steps) / float(max(1, self.total_update_steps))
            lr = self.args.lr * (1.0 - progress) ** 1.5

        for index, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr if index == 0 else lr * 1.0
        return lr

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

        loss_bn_1 = self.criterion_bn(out_bn.float(), mask_bn)
        loss_bn_1[mask_bn == 1] *= 2
        loss_bn_1 = loss_bn_1.mean()

        loss_bn_2 = self.criterion_bn_2(out_bn.float(), mask_bn)
        loss_bn = loss_bn_1 + loss_bn_2

        loss = loss_bn + loss_seg + loss_similarity
        return loss, loss_seg, loss_bn, loss_similarity

    def training(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        if self.args.freeze_bn:
            freeze_batchnorm_modules(self.model)

        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0
        total_TCL = 0.0
        curr_iter = epoch * self.train_micro_batches_per_epoch
        total_micro_batches = self.train_micro_batches_per_epoch

        self.optimizer.zero_grad(set_to_none=True)
        tbar = tqdm(self.trainloader, disable=not self.main_process)

        for i, (img1, img2, img3, mask1, mask2, mask3, mask_bn, _) in enumerate(tbar):
            if i >= total_micro_batches:
                break

            running_iter = curr_iter + i + 1
            img1 = img1.float().to(self.args.device, non_blocking=True)
            img2 = img2.float().to(self.args.device, non_blocking=True)
            img3 = img3.float().to(self.args.device, non_blocking=True)
            mask1 = mask1.long().to(self.args.device, non_blocking=True)
            mask2 = mask2.long().to(self.args.device, non_blocking=True)
            mask3 = mask3.long().to(self.args.device, non_blocking=True)
            mask_bn = mask_bn.float().to(self.args.device, non_blocking=True)
            x = torch.stack([img1, img2, img3], dim=1)

            update_now = ((i + 1) % self.args.accum_steps == 0) or ((i + 1) == total_micro_batches)
            sync_context = contextlib.nullcontext()
            if self.args.distributed and hasattr(self.model, "no_sync") and not update_now:
                sync_context = self.model.no_sync()

            with sync_context:
                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out1, out2, out3, out_bn = self.model(x)

                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_seg, loss_bn, loss_similarity = self.compute_losses(
                        out1,
                        out2,
                        out3,
                        out_bn,
                        mask1,
                        mask2,
                        mask3,
                        mask_bn,
                    )
                    loss_to_backward = loss / self.args.accum_steps

                if self.args.amp:
                    self.scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

            reset_spiking_state(self.model)

            total_loss_seg += loss_seg.item()
            total_loss_similarity += loss_similarity.item()
            total_loss_bn += loss_bn.item()
            total_loss += loss.item()

            if update_now:
                lr = self.adjust_learning_rate()

                if self.args.grad_clip > 0:
                    if self.args.amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(get_raw_model(self.model).parameters(), self.args.grad_clip)

                if self.args.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.update_steps += 1
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            if self.main_process:
                seen = i + 1
                tbar.set_description(
                    "Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f, TL Loss: %.3f"
                    % (
                        total_loss / seen,
                        total_loss_seg / seen,
                        total_loss_bn / seen,
                        total_loss_similarity / seen,
                        total_TCL / seen,
                    )
                )

                self.writer.add_scalar("train total_loss", total_loss / seen, running_iter)
                self.writer.add_scalar("train seg_loss", total_loss_seg / seen, running_iter)
                self.writer.add_scalar("train bn_loss", total_loss_bn / seen, running_iter)
                self.writer.add_scalar("train sc_loss", total_loss_similarity / seen, running_iter)
                self.writer.add_scalar("train TL Loss", total_TCL / seen, running_iter)
                self.writer.add_scalar("lr", lr, running_iter)

    def validation(self, epoch=0):
        if not self.main_process:
            if self.args.distributed:
                dist.barrier()
            return None

        curr_epoch = epoch
        tbar = tqdm(self.valloader)
        raw_model = get_raw_model(self.model)
        raw_model.eval()
        metric = IOUandSek(num_classes=len(RS.ST_CLASSES))
        if self.args.save_mask:
            _ = color_map()

        score = 0.0
        miou = 0.0
        sek = 0.0
        Fscd = 0.0
        OA = 0.0
        SC_Precision = 0.0
        SC_Recall = 0.0

        with torch.no_grad():
            for img1, img2, img3, mask1, _, mask3, mask_bn, _ in tbar:
                img1 = img1.float().to(self.args.device, non_blocking=True)
                img2 = img2.float().to(self.args.device, non_blocking=True)
                img3 = img3.float().to(self.args.device, non_blocking=True)
                x = torch.stack([img1, img2, img3], dim=1)

                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out1, _, out3, out_bn13 = raw_model(x)

                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out3 = torch.argmax(out3, dim=1).cpu().numpy() + 1
                out_bn = (out_bn13 > 0.5).cpu().numpy().astype(np.uint8)

                mask_bn_np = mask_bn.numpy()
                mask1_np = mask1.numpy().copy()
                mask3_np = mask3.numpy().copy()
                out1[out_bn == 0] = 0
                out3[out_bn == 0] = 0
                mask1_np[mask_bn_np == 0] = 0
                mask3_np[mask_bn_np == 0] = 0

                metric.add_batch(out1, mask1_np)
                metric.add_batch(out3, mask3_np)
                change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()
                reset_spiking_state(raw_model)
                tbar.set_description(
                    "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f"
                    % (miou, sek, score, Fscd, OA, SC_Precision, SC_Recall)
                )

        if score >= self.previous_best:
            model_path = os.path.join("checkpoints", self.args.data_name, self.args.Net_name, self.args.backbone)
            os.makedirs(model_path, exist_ok=True)
            save_path = os.path.join(
                model_path,
                "epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth"
                % (curr_epoch, score * 100, miou * 100, sek * 100, Fscd * 100, OA * 100),
            )
            model_to_save = get_raw_model(self.model)
            torch.save(model_to_save.state_dict(), save_path)
            self.previous_best = score

        self.writer.add_scalar("val_Score", score, curr_epoch)
        self.writer.add_scalar("val_mIOU", miou, curr_epoch)
        self.writer.add_scalar("val_Sek", sek, curr_epoch)
        self.writer.add_scalar("val_Fscd", Fscd, curr_epoch)
        self.writer.add_scalar("val_OA", OA, curr_epoch)

        if self.args.distributed:
            dist.barrier()
        return score

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    args = Options().parse()
    if args.accum_steps < 1:
        raise ValueError("--accum_steps must be >= 1")
    if args.debug_iters < 0:
        raise ValueError("--debug_iters must be >= 0")

    init_distributed_mode(args)
    seed_everything(args.seed, args.rank)
    configure_cudnn(args)
    configure_dendfadc_backend(args)
    args.amp = bool(args.amp and torch.cuda.is_available())

    if is_main_process(args):
        print(args)
        if args.disable_cudnn:
            print("cuDNN is disabled; convolution fallback may be slower but can avoid cuDNN algorithm-selection failures.")
        if args.disable_mmcv_deform:
            print("MMCV modulated deform conv is disabled; DendFADCConv2d will use grouped Conv2d fallback.")
        if args.disable_dendfadc:
            print("DendFADC replacements are disabled in the SNN backbone.")
        world_size = args.world_size if getattr(args, "distributed", False) else 1
        bn_stat_batch = args.batch_size * world_size if args.distributed and args.sync_bn else args.batch_size
        effective_batch = args.batch_size * world_size * args.accum_steps
        print(
            "Batch summary: BN statistical batch = %d, optimizer effective batch = %d"
            % (bn_stat_batch, effective_batch)
        )

    trainer = None
    try:
        trainer = Trainer(args)
        if args.load_from:
            trainer.validation(0)

        for epoch in range(args.epochs):
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
        cleanup_distributed(args)


if __name__ == "__main__":
    main()

import argparse

import train_WUSU_ddp_accum_v4 as v4


torch = v4.torch


def build_parser():
    parser = argparse.ArgumentParser("WUSU GSTMSCD DDP AMP Training V5")

    parser.add_argument("--data_name", "--data-name", dest="data_name", type=str, default="WUSU")
    parser.add_argument("--Net_name", "--net-name", dest="Net_name", type=str, default="GSTMSCD")
    parser.add_argument("--backbone", type=str, default="sdtv2")
    parser.add_argument("--data_root", "--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="checkpoints_v5")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", "--val-batch-size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--lightweight", dest="lightweight", action="store_true")
    parser.add_argument("--pretrain_from", "--pretrain-from", dest="pretrain_from", type=str, default=None)
    parser.add_argument("--load_from", "--load-from", dest="load_from", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=v4.str2bool, default=True)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--Lambda", type=float, default=0.00005)

    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt-eps", default=None, type=float, metavar="EPSILON")
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", action=v4.FloatTupleAction, metavar="BETA")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--filter-bias-and-bn", dest="filter_bias_and_bn", action="store_true")
    parser.add_argument("--no-filter-bias-and-bn", dest="filter_bias_and_bn", action="store_false")
    parser.set_defaults(filter_bias_and_bn=False)

    parser.add_argument("--sched", default="poly", type=str, metavar="SCHEDULER")
    parser.add_argument("--sched-on-updates", dest="sched_on_updates", action="store_true")
    parser.add_argument("--sched-on-epochs", dest="sched_on_updates", action="store_false")
    parser.set_defaults(sched_on_updates=True)
    parser.add_argument("--lr", type=float, default=0.005, metavar="LR")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct")
    parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT")
    parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV")
    parser.add_argument("--lr-cycle-mul", type=float, default=1.0, metavar="MULT")
    parser.add_argument("--lr-cycle-decay", type=float, default=0.1, metavar="MULT")
    parser.add_argument("--lr-cycle-limit", type=int, default=1, metavar="N")
    parser.add_argument("--lr-k-decay", type=float, default=1.0)
    parser.add_argument("--warmup-lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--min-lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--epochs", type=int, default=100, metavar="N")
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
    parser.add_argument("--reference_batch_size", "--reference-batch-size", dest="reference_batch_size", type=int, default=None)
    parser.add_argument("--reference_accum_steps", "--reference-accum-steps", dest="reference_accum_steps", type=int, default=1)
    parser.add_argument("--reference_total_updates", "--reference-total-updates", dest="reference_total_updates", type=int, default=None)
    parser.add_argument("--reference_warmup_updates", "--reference-warmup-updates", dest="reference_warmup_updates", type=float, default=None)
    parser.add_argument("--grad_clip_norm", "--grad-clip-norm", "--clip-grad", dest="grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--find_unused_parameters", "--find-unused-parameters", dest="find_unused_parameters", action="store_true", default=True)
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.add_argument("--freeze_bn", "--freeze-bn", dest="freeze_bn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist_url", "--dist-url", dest="dist_url", type=str, default="env://")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", "--world-size", dest="world_size", type=int, default=1)
    parser.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")

    return parser


def fill_v4_defaults(args):
    hidden_defaults = {
        "test_batch_size": args.val_batch_size,
        "tta": False,
        "save_mask": False,
        "use_pseudo_label": False,
        "change_output_api": "logits",
    }
    for name, value in hidden_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def build_binary_criteria():
    if torch is None:
        raise ImportError("PyTorch is required to construct V5 loss functions.")
    from torch.nn import BCEWithLogitsLoss
    from utils.loss import DiceLoss

    return BCEWithLogitsLoss(reduction="none"), DiceLoss(activation="none")


def compute_binary_change_loss(change_logits, mask_bn, criterion_bce, criterion_dice):
    logits = change_logits.float()
    target = mask_bn.float()
    loss_bce = criterion_bce(logits, target)
    loss_bce[target == 1] *= 2
    loss_bce = loss_bce.mean()
    loss_dice = criterion_dice(torch.sigmoid(logits), target)
    return loss_bce + loss_dice


class Trainer(v4.Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.criterion_bn, self.criterion_bn_2 = build_binary_criteria()

    def compute_losses(self, out1, out2, out3, out_bn, mask1, mask2, mask3, mask_bn):
        loss1 = self.criterion_seg(out1.float(), mask1 - 1)
        loss2 = self.criterion_seg(out2.float(), mask2 - 1)
        loss3 = self.criterion_seg(out3.float(), mask3 - 1)
        loss_seg = (loss1 + loss2 + loss3) / 3
        loss_similarity = self.criterion_sc(out1.float(), out3.float(), mask_bn)
        loss_bn = compute_binary_change_loss(out_bn, mask_bn, self.criterion_bn, self.criterion_bn_2)
        loss = loss_bn + loss_seg + loss_similarity
        loss_tl = torch.zeros((), device=loss.device, dtype=loss.dtype)
        return loss, loss_seg, loss_bn, loss_similarity, loss_tl


def main(argv=None):
    args = build_parser().parse_args(argv)
    v4.init_distributed_mode(args)
    fill_v4_defaults(args)
    v4.validate_args(args)
    v4.seed_everything(args.seed, args.rank)

    if v4.is_main_process(args):
        global_batch = args.batch_size * args.world_size
        effective_batch = global_batch * args.accum_steps
        print(args)
        print(
            "batch_size_per_gpu=%d, world_size=%d, global_batch_size=%d, accum_steps=%d, effective_batch_size=%d"
            % (args.batch_size, args.world_size, global_batch, args.accum_steps, effective_batch)
        )
        print("change_output_api=logits, dice_activation=none, output_dir=%s" % args.output_dir)

    trainer = None
    try:
        trainer = Trainer(args)
        if args.eval_only:
            trainer.validation(trainer.start_epoch)
            return
        for epoch in range(trainer.start_epoch, args.epochs):
            if v4.is_main_process(args):
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
        v4.cleanup_distributed()


if __name__ == "__main__":
    main()

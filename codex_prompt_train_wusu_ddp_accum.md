# Codex Prompt: Refactor `train_WUSU.py` for 4-GPU Training, SyncBatchNorm, and Gradient Accumulation

## Role

You are a senior PyTorch / distributed-training engineer and a deep-learning code reviewer. You must modify the current WUSU training script in a conservative, runnable, and testable way.

The current problem is:

- The training batch size can only be set to `2` on one GPU because of GPU memory limits.
- The model contains BatchNorm operations, so per-GPU `BS=2` is likely too small and may make training unstable.
- The server has 4 GPUs available.
- I want to use the 4 GPUs to increase the effective batch size and improve BatchNorm statistics.
- I also want an optional gradient accumulation scheme to further increase the optimizer-level effective batch size.

You must first produce an implementation plan, then modify the training code.

---

## Current File to Read First

Read the existing training script first:

```text
train_WUSU.py
```

The current script already contains:

- dataset import:
  ```python
  import datasets.MultiSiamese_RS_ST_TL as RS
  ```
- model import:
  ```python
  from models.GSTMSCD_MTSCD_Snn import GSTMSCD_WUSU as Net
  ```
- model input format:
  ```python
  x = torch.stack([img1, img2, img3], dim=1)
  out1, out2, out3, out_bn = self.model(x)
  ```
- losses:
  - `CrossEntropyLoss(ignore_index=-1)` for semantic segmentation of three phases;
  - `BCELoss(reduction='none') + DiceLoss()` for binary change;
  - `ChangeSimilarity()` for semantic consistency between phase 1 and phase 3;
  - an optional/commented `TemporalLogicKLDivLoss`.
- SpikingJelly reset:
  ```python
  from spikingjelly.clock_driven import functional
  functional.reset_net(self.model)
  ```
- current training behavior:
  - single-process single-GPU `.cuda()`;
  - `DataLoader(..., shuffle=True, batch_size=args.batch_size)`;
  - `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` every iteration;
  - no `DistributedSampler`;
  - no `DistributedDataParallel`;
  - no `SyncBatchNorm`;
  - no gradient accumulation.

---

## Important Diagnosis

Before editing code, explain the following clearly:

1. **Gradient accumulation alone does not solve the BatchNorm problem.**
   - With per-GPU batch size 2, BatchNorm still computes statistics from only 2 samples per forward pass.
   - Gradient accumulation only increases the optimizer-level effective batch size.
   - It does not increase BN's instantaneous batch statistics.

2. **Preferred solution: DDP + SyncBatchNorm.**
   - Use 4 GPUs with `DistributedDataParallel`.
   - Keep per-GPU `batch_size=2`.
   - Convert BN to `SyncBatchNorm` so BN statistics are synchronized across GPUs.
   - The BN statistical batch becomes:
     ```text
     per_gpu_batch_size × world_size = 2 × 4 = 8
     ```

3. **Optional additional solution: gradient accumulation.**
   - With 4 GPUs, per-GPU BS=2, and `accum_steps=2`, the optimizer effective batch size becomes:
     ```text
     effective_batch_size = per_gpu_batch_size × world_size × accum_steps
                          = 2 × 4 × 2
                          = 16
     ```
   - However, the BN statistics are still based on:
     ```text
     per_gpu_batch_size × world_size = 8
     ```
     when SyncBatchNorm is enabled.

4. **Fallback solution if SyncBatchNorm is not usable.**
   - Add an optional `--freeze_bn` flag.
   - When enabled, set all BatchNorm modules to eval mode during training.
   - Do not replace BN with GroupNorm automatically unless explicitly requested.

---

## Required Output

Create a new modified training script instead of destructively overwriting the original file.

Recommended output filename:

```text
train_WUSU_ddp_accum.py
```

You may reuse most of the original code, but the new script must be complete and runnable.

At the end, report:

1. the implementation plan;
2. the modified files;
3. the exact launch commands;
4. the main changes;
5. any unresolved assumptions or risks;
6. the final test result.

---

## Required Code Modifications

### 1. Add distributed-training imports

Add:

```python
import math
import random
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

Keep existing imports if still needed.

---

### 2. Add command-line arguments

Extend `Options` with at least:

```python
parser.add_argument("--accum_steps", type=int, default=1,
                    help="number of gradient accumulation steps")
parser.add_argument("--sync_bn", action="store_true",
                    help="convert BatchNorm to SyncBatchNorm under DDP")
parser.add_argument("--freeze_bn", action="store_true",
                    help="freeze BatchNorm running statistics during training")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--find_unused_parameters", action="store_true",
                    help="use DDP find_unused_parameters=True if the model has conditionally unused branches")
parser.add_argument("--local_rank", type=int, default=0,
                    help="kept for compatibility; torchrun mainly uses LOCAL_RANK env var")
parser.add_argument("--amp", action="store_true",
                    help="optional mixed precision training")
parser.add_argument("--grad_clip", type=float, default=0.0,
                    help="clip grad norm if > 0")
parser.add_argument("--debug_iters", type=int, default=0,
                    help="run only N training iterations per epoch for debugging; 0 means full epoch")
```

Do not remove the existing arguments unless they are truly unused and safe to remove.

---

### 3. Add distributed utility functions

Implement utilities similar to:

```python
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
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    args.device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")


def is_main_process(args):
    return (not getattr(args, "distributed", False)) or args.rank == 0


def cleanup_distributed(args):
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
```

Use these utilities in `main`.

---

### 4. Update DataLoader and sampler logic

Replace the current single-process `DataLoader(..., shuffle=True)` training loader with distributed-aware logic:

```python
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
```

Then:

```python
self.trainloader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=(self.train_sampler is None),
    sampler=self.train_sampler,
    pin_memory=True,
    num_workers=args.num_workers,
    drop_last=True,
)
```

For validation, use a normal non-distributed loader on rank 0 only, or implement distributed validation correctly. The simpler required solution is:

- Only rank 0 creates and uses the full validation loader.
- Other ranks skip validation and wait at a barrier.

This is acceptable because validation does not need gradients.

---

### 5. Update model initialization

Load checkpoints before DDP wrapping.

The correct order should be:

1. create model;
2. load `pretrain_from` or `load_from` if provided;
3. move model to device;
4. if distributed and `--sync_bn` is set, call:
   ```python
   self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
   ```
5. wrap with DDP:
   ```python
   self.model = DDP(
       self.model,
       device_ids=[args.local_rank],
       output_device=args.local_rank,
       find_unused_parameters=args.find_unused_parameters,
   )
   ```

When saving checkpoints, save:

```python
model_to_save = self.model.module if hasattr(self.model, "module") else self.model
torch.save(model_to_save.state_dict(), save_path)
```

---

### 6. Add helper for model reset

Because the model may be wrapped in DDP, add:

```python
def get_raw_model(model):
    return model.module if hasattr(model, "module") else model


def reset_spiking_state(model):
    functional.reset_net(get_raw_model(model))
```

Use this after every forward/backward micro-batch to avoid state leakage in the SNN.

Important:

- Do not reset the spiking state only once per optimizer step.
- Reset after each micro-batch.
- Prefer resetting after `backward()` rather than before it.

---

### 7. Add optional BatchNorm freezing

Implement:

```python
def freeze_batchnorm_modules(model):
    raw_model = get_raw_model(model)
    for m in raw_model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
```

In `training()`, after `self.model.train()`, call:

```python
if self.args.freeze_bn:
    freeze_batchnorm_modules(self.model)
```

This ensures that `model.train()` does not reactivate BN.

---

### 8. Implement gradient accumulation correctly

The current script calls:

```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

inside every iteration.

Replace this with accumulation logic:

```python
self.optimizer.zero_grad(set_to_none=True)

for i, batch in enumerate(tbar):
    ...
    loss = loss_bn + loss_seg + loss_similarity
    loss_to_backward = loss / args.accum_steps

    update_now = ((i + 1) % args.accum_steps == 0) or ((i + 1) == len(self.trainloader))
```

For DDP, avoid gradient synchronization on non-update micro-steps:

```python
sync_context = contextlib.nullcontext()
if args.distributed and hasattr(self.model, "no_sync") and not update_now:
    sync_context = self.model.no_sync()

with sync_context:
    loss_to_backward.backward()
```

Then:

```python
reset_spiking_state(self.model)

if update_now:
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(get_raw_model(self.model).parameters(), args.grad_clip)

    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    self.update_steps += 1
```

If AMP is implemented, use `torch.cuda.amp.autocast` and `GradScaler`. AMP is optional but should not break the default FP32 path.

---

### 9. Fix LR schedule for accumulation

The old code updates LR based on every micro iteration:

```python
self.iters += 1
self.total_iters = len(self.trainloader) * args.epochs
```

With gradient accumulation, LR should be based on optimizer update steps, not raw micro-batches.

Implement:

```python
self.total_update_steps = math.ceil(len(self.trainloader) / args.accum_steps) * args.epochs
self.update_steps = 0
```

Then add:

```python
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

    self.optimizer.param_groups[0]["lr"] = lr
    self.optimizer.param_groups[1]["lr"] = lr * 1.0
    return lr
```

Call this before each optimizer step.

---

### 10. Preserve the original loss behavior

Do not change the loss definitions unless required for correctness.

Preserve:

```python
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
```

If you decide to expose `TemporalLogicKLDivLoss`, add a separate flag such as:

```python
--use_tcl
--tcl_weight
```

But do not silently enable it.

Also fix the commented TCL branch if you touch it:

```python
total_TCL += loss_TL.item()
```

not:

```python
total_TCL += loss_TL
```

---

### 11. Fix existing minor issues

Fix these issues while refactoring:

1. Avoid using the global `args` variable inside class methods. Use `self.args`.
2. Current `if args.load_from: trainer.validation()` misses the `epoch` argument. Either:
   - make `validation(self, epoch=0)`, or
   - call `trainer.validation(0)`.
3. Avoid creating unused `testset` if not needed.
4. Use `non_blocking=True` when moving tensors to CUDA if `pin_memory=True`.
5. Only rank 0 should write TensorBoard logs and save checkpoints.
6. Only rank 0 should print tqdm progress bars. Other ranks can disable tqdm:
   ```python
   tbar = tqdm(self.trainloader, disable=not is_main_process(self.args))
   ```
7. In DDP mode, call:
   ```python
   self.train_sampler.set_epoch(epoch)
   ```
   at the beginning of each training epoch.

---

## Recommended Training Commands

### A. Four GPUs, best default for BN stability

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum.py \
  --batch_size 2 \
  --val_batch_size 2 \
  --sync_bn \
  --accum_steps 1
```

Expected:

```text
BN statistical batch = 2 × 4 = 8
Optimizer effective batch = 2 × 4 × 1 = 8
```

---

### B. Four GPUs + gradient accumulation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum.py \
  --batch_size 2 \
  --val_batch_size 2 \
  --sync_bn \
  --accum_steps 2
```

Expected:

```text
BN statistical batch = 2 × 4 = 8
Optimizer effective batch = 2 × 4 × 2 = 16
```

---

### C. Single GPU fallback with gradient accumulation

```bash
python train_WUSU_ddp_accum.py \
  --batch_size 2 \
  --accum_steps 4
```

Expected:

```text
BN statistical batch = 2
Optimizer effective batch = 2 × 1 × 4 = 8
```

Explain clearly that this does not solve the BN statistics issue.

---

### D. If BN remains unstable

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum.py \
  --batch_size 2 \
  --sync_bn \
  --accum_steps 2 \
  --freeze_bn
```

Use this only as a fallback.

---

## Required Validation / Smoke Tests

Do not run full training first.

Run smoke tests:

### 1. Python syntax check

```bash
python -m py_compile train_WUSU_ddp_accum.py
```

### 2. Single-GPU short run

```bash
CUDA_VISIBLE_DEVICES=0 python train_WUSU_ddp_accum.py \
  --batch_size 2 \
  --accum_steps 2 \
  --debug_iters 2
```

### 3. Four-GPU short run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_WUSU_ddp_accum.py \
  --batch_size 2 \
  --sync_bn \
  --accum_steps 2 \
  --debug_iters 2
```

Report:

- exact command;
- whether import succeeded;
- whether dataset initialization succeeded;
- whether model initialization succeeded;
- whether one forward pass succeeded;
- whether backward succeeded;
- whether optimizer step succeeded;
- whether validation/saving works on rank 0.

---

## Acceptance Criteria

The task is complete only if:

1. `train_WUSU_ddp_accum.py` is generated as a full runnable script.
2. The script supports both single-GPU and 4-GPU DDP execution.
3. DDP mode uses `DistributedSampler`.
4. Optional `--sync_bn` converts BN to SyncBatchNorm before DDP wrapping.
5. Optional `--accum_steps` works without calling `optimizer.zero_grad()` every micro-batch.
6. LR schedule is based on optimizer update steps, not raw micro-batches.
7. SNN state is reset after every micro-batch.
8. Only rank 0 writes logs, saves checkpoints, and performs full validation.
9. The original model input/output and loss behavior are preserved.
10. Smoke-test commands and results are reported.

---

## Final Response Format

After finishing the code modification, respond in this structure:

```markdown
# Implementation Plan

...

# Modified Files

...

# Key Code Changes

...

# Launch Commands

...

# Smoke-Test Results

...

# Notes and Risks

...
```

Do not provide only a patch summary. Provide the actual complete modified training file or state clearly where it was written.

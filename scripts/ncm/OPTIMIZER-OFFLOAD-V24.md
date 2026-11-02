# Two-replica training with host optimizer state

This stage addresses training memory. It does not offload OxCoder inference to
SSD, change model quantization, or establish a usable NCM checkpoint.

## Observed limitation of the resident optimizer

The version 23 three-update smoke completed, but the longer run stalled on
update 354. Reproduction with immediate worker-error reporting exposed a CUDA
out-of-memory exception while unpacking a 96 MiB matrix on GPU0. The allocator
allowed 6.52 GiB, with 6.35 GiB allocated and 91.58 MiB reserved but unused.
Physical GPU memory still had 3.90 GiB free; that space was retained for the
existing applications and runtime reserve. No unrelated application was stopped.
The [recorded failure](validation/v24-controls/resident-optimizer-failure.json)
includes the original exception and log checksum.

A short optimizer smoke was therefore insufficient to qualify all training
contexts. The follow-up smoke traverses the previously failing update and
records each worker's allocated, reserved, and peak allocated CUDA memory.

The [completed recovery smoke](validation/v24-controls/optimizer-smoke.json)
ran updates 351–360, including the previously failing batch and prompts up to
2,032 tokens. Both replicas agreed on the saved weights. Peak PyTorch allocated
memory was 6,596,718,592 bytes on GPU0 and 6,328,131,584 bytes on GPU1. These
exclude driver allocations and other processes. The saved candidate remained
0/12 on the development tasks: execution recovery is not coding success.

## CPU AdamW for the output heads

`cpu_adamw.py` keeps the FP32 concept-classifier and termination parameters on
the GPU for forward/backward execution. Each update copies their summed,
clipped gradients to CPU parameters, applies ordinary AdamW using the existing
moments and step counters, and copies updated FP32 parameters back. Muon and
the smaller interface optimizer retain their existing placement and behavior.
Both replicas use the same optimizer placement.

This removes 537,034,760 bytes of resident CUDA optimizer state per worker in
the measured head inventory. It adds host copies of the parameters, gradients,
and moments, plus transfer and CPU update time. It does not reduce the model's
inference footprint. The training services prohibit additional swap use.

The wrapper accepts FP32 parameters only. It does not silently introduce master
precision for BF16 parameters. Legacy AdamW migration is explicit. Save/reload
retains its moments and counters; model weights provide the authoritative
FP32 parameter copy, avoiding another copy in every optimizer checkpoint.

## Numerical control

[Eight updates](validation/v24-controls/cpu-adamw-audit.json) compare CUDA AdamW
with CPU AdamW using actual saved NCM head parameters and optimizer states.
Both arms receive the same deterministic directions derived from saved first
moments. This tests optimizer numerics, not newly computed model gradients.
The CPU arm is serialized and restored midway through the comparison.

Every step passed the fixed relative-update L2 limit of 0.001 and
relative-parameter L2 limit of 0.000002. The report retains per-step errors and
timings. CPU updates take approximately 0.6 seconds in this control, compared
with approximately 0.03 seconds for resident CUDA updates after the first step.
This is a memory tradeoff, not an isolated optimizer speedup. The complete
two-replica run still needs its own execution and generation checks.

The trainer validates the audit's source/codec identities, head configuration,
head learning rate, implementation hashes, numerical errors, and state
placement before using the option. Its checkpoint identity records the
execution change. BF16 forward/backward, gradient synchronization, data,
conditioning noise, and the 12-task acceptance test remain unchanged.

```bash
python -m scripts.ncm.audit_cpu_adamw \
  --checkpoint /path/to/immutable-fp32-master-checkpoint.pt \
  --output results/new-cpu-optimizer-audit --device cuda:1
```

After that control passes, append these arguments to the two-replica training
command from [the parallel guide](PARALLEL-TRAINING-V23.md):

```bash
--cpu-head-optimizer \
--cpu-optimizer-audit results/new-cpu-optimizer-audit/report.json
```

Use a fresh experiment directory for this explicit optimizer migration. Resume
that new experiment with exactly the same arguments and `--resume`. Do not
reuse the previous experiment directory with a changed implementation identity.

`run_cpu_optimizer_stage.py` coordinates the recorded local continuation: wait
for the recovery smoke, validate its data and implementation identities,
preserve its evaluated checkpoint, and then launch the full stage from the
unchanged update-350 parent. It monitors `ncm-parallel-v24.service` and performs
the cold replay only for a preserved 12/12 candidate. Resuming this coordinator
with `--monitor-only` observes the existing trainer without starting it again.

## Worker failure handling and monitoring

On fatal worker errors, the trainer prints the original traceback and exits
that worker immediately. `torchrun` then terminates the peer. Waiting for NCCL
shutdown inside an exception handler previously hid the original allocation
error while another worker waited in a collective. Normal completion still
destroys the process group cleanly. Existing atomic checkpoints remain intact.

The monitor supports `--monitor-only --training-unit ncm-<experiment-name>` for
an existing owned training unit. It does not relaunch training. Every completed
12-task report is paired with the checkpoint checksum before preservation;
12/12 still requires a separate cold replay from that preserved file.

## Separate learning-rate experiment

`change_body_lr.py` can prepare a separate branch from an immutable checkpoint.
It changes only the body learning rate in the identity and the matching
interface-AdamW and FP32-Muon parameter groups. Heads, model weights, master
weights, moments, counters, completed updates, and RNG states remain unchanged.
The branch records its parent checksum. It is not enabled by optimizer
offloading and has no demonstrated quality benefit yet.

```bash
python -m scripts.ncm.change_body_lr \
  --checkpoint /path/to/immutable-checkpoint.pt \
  --output results/explicit-lr-branch.pt --body-lr 0.0001
```

Compare branches from the same parent, with the same data and example sequence,
before selecting a continuation. A lower training loss alone is insufficient;
compare held-out concept prediction and actual generation. Never change the
test answers, substitute teacher output, or label an optimizer check as 12/12.

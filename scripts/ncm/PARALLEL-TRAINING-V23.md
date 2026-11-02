# Two-replica NCM training

The two-GPU compact training path computes disjoint portions of the same
four-example batch concurrently. It retains the frozen OxCoder GGUF weights,
the learned concept codec, independent concept heads, full prompt gradients,
FP32 Muon master parameters, AdamW states, and canonical gradient-clipping
groups. It does not add a teacher at inference or substitute an AR answer.

## Measured numerical and speed control

The [full report](validation/v23-controls/parallel-gradient-audit.json) uses the
immutable v22 update-250 checkpoint, SHA256
`3c30033627362846b1ee1bed36987e42bb92424e5d99472223c694a94e4119f3`.
Both RTX 5060 Ti GPUs hold one compact replica. No unrelated GPU application
was stopped. The reference computes the same four examples sequentially on
GPU0; the parallel path includes summed-gradient communication. Both paths
use identical per-example corruption seeds. One warmup precedes three timed
repeats. Optimizer updates, checkpoint writes, startup, and evaluation are not
included in these timings.

| Prompt lengths in the global batch | Sequential median | Parallel median | Speed ratio |
|---|---:|---:|---:|
| 64, 69, 128, 256 | 2.8632 s | 2.0444 s | 1.4005× |
| 64, 512, 1,024, 2,048 | 21.4965 s | 13.1063 s | 1.6402× |

Logits and classifier gradients match exactly in both controls. Global
gradient relative L2 errors are 0.339% and 0.535%; the largest parameter-group
error is 1.931%, for interfaces in the mixed-length batch. Every group passes
the existing relative-L2 limit of 5% and cosine threshold of 0.99. This is a
finite BF16 numerical comparison, not bit-exact training equivalence or a
model-quality result. The complete optimizer continuation still requires its
separate smoke run with resident optimizer states.

## Why the synchronization is explicit

One example can use a single shared-prefix backward; another can require four
independent level backwards. Naively synchronizing every backward would give
workers different collective counts. `parallel_training.py` instead sums
gradients once all local examples are complete. Each example's loss is already
divided by the global batch size, so the collective is **SUM**, not a second
average. Frozen GGUF buffers never participate in broadcasts or gradient
communication.

A deterministic longest-first assignment balances the fixed batch across
workers without changing its examples or loss weights. Gradient transfer uses
bounded FP32 buffers before copying back into each parameter's gradient dtype.
Workers first agree on gradient validity; a missing gradient causes both to
fail instead of entering mismatched collectives. Unit tests cover unequal
local backward counts and compare against a sequential global mean.

## State and data continuity

`train_parallel.py` requires a saved FP32-master checkpoint. It restores learned
weights, optimizer/master states, completed update count, and Python sampler
state. Noise generation uses a new explicit schedule keyed by run seed, global
step, and global example index, independent of assignment to a worker. This is
recorded as a new stage, not claimed to reproduce the exact old stochastic
trajectory. Labels stay clean. Checkpoints contain both workers' local CUDA
RNG states without initializing a second CUDA context inside either worker.

Workers compare parameter inventories, curriculum IDs, selected example IDs,
and saved trainable-weight hashes. Checkpoints stay loadable through the
original resident BF16 loader. Shared-prefix execution remains restricted to
64–512 prompt tokens. Longer contexts retain independent reference levels.

An explicit `--data-expansion` may append training-only examples to the retained
base. It verifies the base hash and unchanged evaluation fingerprint. Numerical
audit metadata continues to name the original base examples; it is not
relabeled as a test on new data. Every training prompt must remain within the
qualified 2,048-token bound.

`prepare_curated_expansion.py` preserves inherited archive declarations as well
as new additions. This keeps the teacher filter's previous exclusions active
after a second expansion. Source archives and manifests are checksum-checked.
Both creator datasets and their existing teacher responses remain in the base.

The completed additional NVIDIA collection attempted 512 prompts and admitted
442 complete teacher responses. Executing the publisher checks against both
the published solution and the teacher response selected 188 responses.
Another 73 failed the teacher checks, 169 had unsupported or invalid code/test
representations, and 12 failed the published-answer control. Unsupported cases
are excluded rather than counted as demonstrated incorrect answers. The
[completed verification record](validation/v23-controls/publisher-test-complete.json)
pins the report, source identity, curated manifest, and archive hashes. These
are training-data checks, not student evaluation scores.

## Reproduction

Use the established CUDA/FLA/Triton environment. Stop only owned training jobs
before a two-GPU audit; do not run it beside another full replica on the same
GPU. Pass immutable checkpoint paths, not a changing `latest.pt`.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m scripts.ncm.audit_parallel_training \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint /path/to/immutable-fp32-master-checkpoint.pt \
  --data results/ncm-expanded-training-v19 \
  --output results/new-parallel-audit --repeats 3
```

After both numerical and measured speed gates pass:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m scripts.ncm.train_parallel \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init /path/to/immutable-fp32-master-checkpoint.pt \
  --data results/ncm-expanded-training-v19 \
  --runtime-audit results/ncm-runtime-audit-v19.json \
  --parallel-audit results/new-parallel-audit/report.json \
  --output results/new-parallel-training \
  --steps 5000 --save-every 50 --eval-every 250
```

First run a short continuation ending three updates after the parent. Confirm
optimizer restoration, finite gradients, replica agreement, saved checkpoints,
memory headroom, and complete evaluation. A failed smoke cannot authorize the
long run. Use a fresh output directory for the full run from the same parent;
the smoke updates are qualification work and are not counted twice. To resume
an interrupted run, preserve every argument and add `--resume`.

## Coordinated next stage

`run_parallel_stage.py` finishes the already selected 512-request NVIDIA teacher
collection, stops that owned teacher server, verifies the completed archive,
materializes passing answers, and prepares the explicit expansion. It defers
the optional additional 256-per-source creator-teacher requests to free both
GPUs for student training; the two creator datasets and their previous
distillation are retained. The existing student trains during CPU verification
and preparation. Handoff occurs at a saved update boundary.

The coordinator runs the optimizer smoke before launching the long stage.
Every 12/12 candidate is copied with its SHA256 and reloaded through the
original backend for the unchanged 12 tasks. Completion of a training schedule
without that result records `stage_complete_requires_next_experiment`; it does
not mark the model usable. The 12 elementary tasks also do not establish broad
professional coding ability.

Checkpoint preservation now uses a nonblocking filesystem lock so concurrent
observers cannot race to create the same snapshot. The original master-stage
monitor supports `--monitor-only` to observe an existing trainer without
restarting its optimization.

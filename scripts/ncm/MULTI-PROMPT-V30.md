# Variable-prompt batching for original-backbone NCM training

This experiment preserves the frozen OxCoder source, trained adapters,
residual-VQ concept vocabulary and native generation objective of v29. It
batches independent prompts and concept levels to reduce repeated layer
execution. It introduces no autoregressive answer fallback and does not change
the twelve executable acceptance tasks.

## Execution and state layout

`frozen_tower_multi.prepare` processes each prompt independently without
padding or adapters. Full-attention keys and values are padded on the right
only after prefill. A per-row mask excludes those padding positions.
GatedDeltaNet convolution and recurrent states are stacked directly: padding
tokens never alter them. Unlike the v29 aligned-prefix path, this path caches
the entire prompt, with alignment one.

Queries use each row's actual prompt length for rotary positions. Concept
levels are padded to the largest level and concatenated in level-major order.
Each level gets an independent fork of the prompt state. Full attention can
see valid positions within that level; recurrent layers retain their causal
within-pass operation. Backward checkpoint recomputation creates fresh forks.
The weak-reference cache lifetime fix remains in use.

`frozen_multi_objective` chooses clean or student-conditioned trajectories per
example. Clean rows use target residual labels to construct earlier levels.
Student-conditioned rows choose their own earlier concepts; target bytes only
determine the residual supervision. CPU tests compare independent and batched
trajectories, check all trainable gradients, and confirm that changing target
bytes cannot alter student-conditioned inputs or predictions. The target
encoder still runs one example at a time to preserve nearest-code boundaries.

## Full-model numerical and timing evidence

The execution audit uses the preserved v24 initialization, FP16 computation,
full-precision matrix reductions, loss scale 64 and the existing packed source
weights. Every trainable gradient and parameter subgroup is compared against
an uncached control. Acceptance thresholds remain gradient relative L2 below
0.05, gradient cosine above 0.99, and logit relative L2 below 0.02.

| Four prompt lengths | Uncached seconds | v29 seconds | Batched seconds | Ratio over v29 | Batched peak allocation |
|---|---:|---:|---:|---:|---:|
| 64, 69, 128, 256 | 7.567 | 3.719 | 3.333 | 1.116× | 6.70 GB |
| 64, 512, 1,024, 2,048 | 22.730 | 4.934 | 4.696 | 1.051× | 7.49 GB |

Both numerical cases pass. The largest interface-gradient relative L2 is
0.01537. The 1,024- and 2,048-token examples are explicitly synthetic
truncation/extension controls, not natural quality examples. Timings contain
one warmup and one measured iteration, including prefill and backward but
excluding optimizer work. The small advantage over v29 requires further
full-update measurements; these numbers do not justify a large speed claim.
GPU allocations exclude host memory and are not an 8 GB deployment result.
See [the complete execution report](validation/v30-controls/multi-prompt-execution.json).

## Qualification and continuation

`frozen_multi_qualification` rejects incomplete reports, changed implementation
hashes, missing shapes, failed numerical subgroups or nonpositive measured
speed advantages. A separate `audit_frozen_multi_optimizer` must complete two
updates, one clean and one student-conditioned, before training is admitted.
It verifies finite changes in adapters, interfaces and classifiers, immutable
source weights, and a gradient-free codec. Its test weights are discarded.

The [full optimizer control](validation/v30-controls/multi-prompt-optimizer.json)
passed both updates: 8.74 seconds for clean inputs and 7.31 seconds for
student-conditioned inputs, including parameter comparisons. Peak allocation
was 7,563,902,976 bytes. The corresponding v29 smoke was 9.63/9.50 seconds;
these short separate runs suggest a useful gain but are not a sustained
training throughput benchmark. The CPU suite passed 226 tests.

`train_frozen_multi` accepts an explicit v29 checkpoint as its parent. The
import preserves trainable weights, FP32 optimizer masters and moments, global
update number, Python sampler RNG, and CPU/CUDA RNG states. It rejects changed
data admission, schedule, batch size, seed, precision, loss scale and parent
implementation hashes. Subsequent resumes also require an identical v30
identity. A CPU regression checks the next random draws and next optimizer
update against uninterrupted execution. This is state-preserving continuation
with a numerically qualified execution change, not bit-exact equivalence of
the entire future training trajectory.

The following commands assume the pinned environment described in
[the v29 guide](FROZEN-TOWER-V29.md), admitted complete-response corpus, and
available immutable source/codec/parent checkpoints. Run GPU qualification
without competing full-model workers:

```bash
"$PYTHON" -m scripts.ncm.audit_frozen_multi \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-parallel-v24/latest.pt \
  --selection results/ncm-complete-selection-v27 \
  --output results/ncm-frozen-multi-audit-v30 --device cuda:1

"$PYTHON" -m scripts.ncm.audit_frozen_multi_optimizer \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-parallel-v24/latest.pt \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --execution-audit results/ncm-frozen-multi-audit-v30/report.json \
  --output results/ncm-frozen-multi-optimizer-v30 --device cuda:1

"$PYTHON" -m scripts.ncm.train_frozen_multi \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --init results/ncm-frozen-tower-training-v29/probe-100.pt \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --execution-audit results/ncm-frozen-multi-audit-v30/report.json \
  --optimizer-audit results/ncm-frozen-multi-optimizer-v30/report.json \
  --output results/ncm-frozen-multi-training-v30 \
  --device cuda:1 --session-steps 100 --eval-every 250
```

The 10,000-update schedule continues from the imported update rather than
restarting warmup. Both creator datasets and verified distillation remain in
the same admitted corpus. The supervisor `run_frozen_multi_stage.py` uses
explicit Linux service names to serialize full-model GPU work with the v28
evaluator. Its parent path is the immutable v29 step-100 snapshot. Stop the
v29 supervisor at a completed session boundary before replacing it with this
supervisor. Do not run both supervisors against the same device.

Validation uses the same full-prefix inference path as batched training,
with a batch of one. A twelve-task pass is still only a candidate until a
separate process reloads the exact checkpoint and repeats all checks and text.
The preserved [v29 step-100 probe](validation/v30-controls/parent-step-100-coding.json)
scored 0/12; successful execution audits are not coding-quality evidence.

## Diagnostic interpretation

`diagnose_frozen_multi.py` samples fixed training and validation windows,
without creating an optimizer. It reports teacher-forced code loss/accuracy,
autonomous reconstruction, reconstruction with correct earlier concepts but
a student-predicted final level, and reconstruction with every oracle code.
The oracle branches identify where errors occur; they cannot qualify an
inference model. The regression test changes diagnostic targets and checks
that autonomous text and termination predictions stay unchanged.

```bash
"$PYTHON" -m scripts.ncm.diagnose_frozen_multi \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-frozen-multi-training-v30/evaluated-250.pt \
  --execution-audit results/ncm-frozen-multi-audit-v30/report.json \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --output results/ncm-frozen-multi-diagnostics-250.json --device cuda:1
```

Run this after the checkpoint exists, with full-model GPU work serialized.
Low teacher-forced accuracy indicates a prediction-learning problem even
under correct earlier concepts. Good teacher-forced accuracy with poor free
generation instead identifies sensitivity to the student's earlier errors.
Oracle reconstruction failures identify a codec limitation. These probes
do not change the frozen executable test suite or supply its answers.

The [upstream paper's loss-weighting and corruption ablations](https://arxiv.org/html/2610.14162v1#A6.SS2.SSS3)
use much larger batches and a twelve-level tokenizer. The current four-level
byte codec and adaptive residual targets are independent experimental choices;
equal step counts do not imply equal data exposure or expected quality.
The [independent XSum-to-chat demonstration](https://huggingface.co/danielharkin21/ncm-chat-experimental-generator)
reports 4,000 fine-tuning steps on 21,600 examples, but explicitly describes
its short replies as often repetitive or wrong. Its reduced code loss is not
evidence that an OxCoder architectural conversion will pass executable coding.

### Measured update-250 checkpoint

The [unchanged executable suite](validation/v30-controls/step-250-coding.json)
scored **0/12**, and [autonomous validation](validation/v30-controls/step-250-validation.json)
reconstructed 0/64 windows exactly, with mean nonpadding byte accuracy 0.08788.
This checkpoint is not usable and is not a release candidate.

The [read-only diagnostic](validation/v30-controls/step-250-diagnostics.json)
sampled 16 training and 16 validation windows. All 32 reconstructed exactly
when supplied with oracle codes. Autonomous predictions reconstructed none.
With the first three levels supplied correctly and only the final level
predicted, mean byte accuracy was 0.68488 on training and 0.59912 on validation,
again with zero exact windows. Thus incorrect earlier levels contribute to
the problem, but final-level prediction is also inadequate even under clean
conditioning. The codec's exact oracle results do not establish predictive
quality. Training and validation have different source compositions, so this
small diagnostic does not isolate overfitting from distribution differences.

Continuation retains the full optimizer and sampling state. Execution audits
and training speed measurements cannot replace the executable acceptance gate.

## Frozen-source transfer control

`audit_frozen_source.py` runs the FP16 frozen prompt path against the existing
llama.cpp fixtures. It bypasses adapters even if their weights are nonzero and
temporarily computes vocabulary logits using bounded output-head chunks. This
head is used only in the isolated audit; it is absent from NCM answer generation
and training objectives. A CPU regression verifies adapter bypass, failed-token
reporting, source-checksum rejection and hook cleanup.

The full-model control matched 33/33 teacher-forced token positions across the
two original fixtures. This checks the transferred source's short-prompt
behavior; it does not qualify NCM coding or imply broad bitwise parity.
The [report](validation/v30-controls/frozen-source-parity.json) pins the source,
reference, checkpoint and implementation checksums.

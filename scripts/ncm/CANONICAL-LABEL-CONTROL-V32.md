# Canonical-label control for the frozen OxCoder tower

This branch tests a different supervision policy while the adaptive-residual
v31 run continues. It starts from the same immutable update-500 checkpoint,
restores its optimizer and schedule, and retains the entire admitted corpus,
both creator datasets and verified distillation. It does not add benchmark
answers or change the twelve executable tasks.

## Reason for the control

The [NCM paper](https://arxiv.org/html/2610.14162v1#A6.SS3) trains against clean
code labels while corrupting a portion of earlier conditioning codes with
uniform codebook samples. The v31 experiment instead obtains earlier codes
from the student on half its examples and recomputes residual labels. That
policy is an independent extension, and its labels change as the student
changes. The new control returns to canonical labels to test their learnability
with the frozen source tower. It also avoids training-time student proposal
passes. It does not reproduce the paper's model, data scale or twelve-level
tokenizer, and no published quality gain is assumed to transfer.

## Supervision and information boundaries

`frozen_clean_objective.supervision` encodes each target separately with the
frozen codec, preserving its established nearest-code arithmetic. The clean
codes are classification labels throughout the update. A copy supplies
conditioning: replacement probabilities across the four levels are
`[0.15, 0.10, 0.05, 0.0]`. Each selected replacement is drawn uniformly from
that level's codebook; it can coincidentally equal the original code. The last
level never conditions a later level. First-level input remains zero.

Prefix prefill remains frozen and runs in evaluation mode. The unpadded v31
gradient pass predicts all four levels with independent attention/recurrent
segments. Equal-level cross entropy and termination BCE weighted 0.1 remain
unchanged. There is no gradient into the codec or original source weights.

Target-derived codes exist only in training. Standalone inference retains the
same autonomous v30 trajectory and v31 runtime qualification; it receives no
labels, target latent, corruption schedule or teacher answer. Cross-entropy
values from this branch and adaptive-label training are not directly comparable,
because they can refer to different labels. Compare fixed validation and
executable results instead.

## Matched continuation

The parent is `results/ncm-frozen-ragged-training-v31/evaluated-500.pt`. The
branch retains global update number, body/head learning rates, warmup, cosine
schedule, batch four, FP32 optimizer masters/moments, and Python/CPU/CUDA RNG
state. This is an explicit objective change, not an identical training resume.
Subsequent resumes of the branch require its exact identity and code hashes.

Response sampling consumes the original four conditioning RNG draws after each
batch even though their decisions are unused. This keeps the selected examples
aligned with v31 at the same update number. Corruption uses the separate Torch
RNG, whose state is checkpointed. No forward pass is added to imitate the old
proposal computations. Training logs retain example IDs so matched exposure
can be checked rather than inferred from the seed alone.

CPU tests verify zero-corruption loss and every trainable gradient against the
existing teacher-forced path, unchanged clean labels under corruption, RNG
replay, frozen parameters and rejection of validation rows. The NCM suite passed
235 tests before the queued GPU qualification. The GPU optimizer smoke is a
separate mandatory gate; CPU tests alone do not admit the branch to training.

## Execution

Use the environment and source/codec paths from
[the unpadded training guide](UNPADDED-TRAINING-V31.md). First run two full-shape
optimizer updates with zero and 0.15 maximum corruption. Their weights are
discarded. Training rejects a missing or failed smoke, changed implementation,
different initialization, or different admitted corpus.

```bash
"$PYTHON" -m scripts.ncm.audit_frozen_clean_optimizer \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-frozen-ragged-training-v31/evaluated-500.pt \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --execution-audit results/ncm-frozen-ragged-audit-v31/report.json \
  --output results/ncm-frozen-clean-optimizer-v32 --device cuda:0

"$PYTHON" -m scripts.ncm.train_frozen_clean \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --init results/ncm-frozen-ragged-training-v31/evaluated-500.pt \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --execution-audit results/ncm-frozen-ragged-audit-v31/report.json \
  --optimizer-audit results/ncm-frozen-clean-optimizer-v32/report.json \
  --output results/ncm-frozen-clean-training-v32 \
  --device cuda:0 --session-steps 500 --eval-every 250
```

The experiment-specific `run_frozen_clean_stage` coordinator waits for the named
older GPU0 trainer to finish and for the parent checkpoint's evaluation to
complete. It then runs the smoke and, only on success, starts training. It
does not stop the v31 arm on GPU1. Host/GPU admission reserves remain enforced;
waiting, successful qualification and actual training are distinct states in
`progress.json` and the worker log. The coordinator uses Linux user systemd;
standalone commands do not require it.

Each 500-update session preserves full state, while fixed validation and the
twelve executable tasks run every 250 updates. A 12/12 candidate must pass an
independent reload with identical answers before `acceptance.json` is written.
No audit, training loss, planned run or queued service qualifies a usable model.

Training records include the four clean-label code accuracies. They are
teacher-conditioned diagnostics, not decoded-byte accuracy or executable
success. A CUDA synchronization separates the objective/backward interval from
the optimizer/invariant interval. The first includes sample selection, target
encoding and frozen-prefix prefill; the second includes host-resident AdamW,
Muon, parameter transfers and invariant checks. These measurements identify
where further training acceleration would actually matter. They do not measure
inference throughput.

## External conversion evidence checked on 2026-11-03

[PlanSpec v2](https://zenodo.org/records/23607317) reports a shared-weight
SmolLM2-135M experiment with AR and parallel masked-prediction modes. Its
parallel mode remained weak, and the author explicitly leaves a pretrained
model extended with RVQ/NCM level heads as follow-up work. The reported usable
translation improvement depends on an AR path. This is useful evidence about
precision and training difficulty, but it is not an available successful
native OxCoder-to-NCM conversion. It does not replace this experiment's
autonomous acceptance test.

The [official training-code issue](https://github.com/HazyResearch/ncm/issues/10)
still contains a promise to release training code; the
[coding-agent discussion](https://github.com/HazyResearch/ncm/issues/17)
contains community training logs, rather than a published coding checkpoint
with executable results. Repository execution audits and loss improvements
must therefore remain distinct from claims of a demonstrated coding model.

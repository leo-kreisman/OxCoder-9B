# OxCoder-derived NCM generator experiment, version 6

This is an architectural adaptation experiment, not a released coding model.
The source remains the unchanged OxCoder-9B Q4_K_M checkpoint. Its 32 hybrid
transformer layers are retained as frozen BF16 weights during CUDA training.
A candidate requires the original source, a codec, and the trained generator
parameters. This reference implementation is not the 8GB deployment engine.

## Codec repair and measured results

The previous Muon continuation reconstructed only 1/30 complete development
responses exactly. The new recipe continues a tokenwise learned residual-VQ
codec with natural training chunks and synthetic byte coverage, removes
unnecessary positional embeddings, reinitializes unused fine-code entries from
actual residuals, then freezes the representation and fits its decoder.
Reinitialization resets optimizer moments and targets are recomputed. There is
no raw-text side channel, answer lookup, autoregressive repair or teacher fallback.

The selected `results/ncm-coverage-decoder-v6/best.pt` reconstructs **28/30**
complete development responses exactly, with **99.998596%** non-padding symbol
accuracy, and passes **6/6** syntax/Unicode probes. All thirty responses decode
as valid UTF-8 with termination; two still contain errors. The broad report
`complete-gate.json` therefore remains **failed**. These are reconstruction
metrics, not generated-code scores.

This tokenwise codec is an independent multiscale residual-VQ variant, not an
upstream NCM tokenizer checkpoint. Its coarse-code semantics are unproven.
AdamW trains these tokenwise controls because their parameters have no eligible
hidden matrices for Muon under the documented partition.

## Exact-target training curriculum

Default generator admission still requires the broad reconstruction gate.
Explicit `--exact-target-curriculum` permits a narrower research experiment:

- Require a measured report matching the codec and complete corpus hashes.
- Require the short-chunk gate to pass.
- Use full source messages and complete preceding assistant chunks, rendered
  with the source chat template. Count and reject oversized prompts; never trim.
- Admit only training targets that individually reconstruct exactly through the
  frozen codec. Record rejected IDs and retained source/teacher counts.
- Preserve all eligible validation examples, including reconstruction failures.
- Require both source datasets to remain represented after training selection.
- Keep every checkpoint `quality_ready=false` until independent generation,
  executable coding, deployment memory and latency tests justify qualification.

This changes a training prerequisite, not the release criterion. It enables an
empirical generator test without pretending the codec is lossless.

## Generator implementation

Twelve frozen layers reside on CUDA device 0, twenty on device 1. Each has a
rank-32 residual adapter. Muon updates eligible hidden matrices; AdamW handles
heads, embeddings and other parameters. The original vocabulary head is still
replaced by a learned concept head: retaining backbone weights does not prove
preservation of coding ability. An AR transfer audit precedes training.

Full-attention layers use a causal prefix and bidirectional attention within a
concept level. Prefix positions cannot read targets. Target inputs contain only
earlier-level concepts. Gated-DeltaNet layers remain causal, an explicit
departure from the paper. Recurrent state resets between levels and examples.

A separate binary head predicts the end of a complete response; the codec's
per-chunk EOS alone cannot express that. Labels come from complete responses.
The objective is mean code cross-entropy across levels plus 0.1 times final-level
termination BCE. Inference generates levels in parallel, then may generate a
new chunk conditioned on previous generated text.

## Launch, progress and resumption

The 512-token conditioning attempt excluded every agent-source example and was
stopped before training. The revised finite run uses **1,024 prompt tokens**,
300 updates, held-out generation at initialization/every 100 updates, and saves
every 25 updates. This context is a training experiment limit, not OxCoder's
maximum supported context. The service has a three-hour runtime cap, 24 GiB host
memory cap, no additional swap, four-core CPU quota and GPU headroom checks.

```bash
systemctl --user status ncm-generator-v6b
journalctl --user -u ncm-generator-v6b -n 20 --no-pager
cat results/ncm-generator-v6b/curriculum.json
cat results/ncm-generator-v6b/training.json
cat results/ncm-generator-v6b/generation-evaluations.json
```

`conditioning.json` records prompt exclusions. `curriculum.json` records exact
training-target admission. `broad-reconstruction.json` preserves failed broad
validation. Generation diagnostics cover up to two development chunks per
represented source and include free-running output, termination probability and
teacher-forced code loss. They are not a coding benchmark.

From the repository root with the documented Python environment:

```bash
python -u -m scripts.ncm.train generator \
  --data results/ncm-muon-distillation-v5/data --output results/ncm-generator-v6b \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --complete-gate results/ncm-coverage-decoder-v6/complete-gate.json \
  --exact-target-curriculum --complete-data results/ncm-corpus-v2 \
  --teacher-data results/ncm-muon-distillation-v5/teacher \
  --parallel-attention --layer-split 12,20 --devices cuda:0,cuda:1 \
  --rank 32 --prompt-tokens 1024 --optimizer muon --lr 0.0001 \
  --steps 300 --eval-every 100 --save-every 25 \
  --audit-reference scripts/kindi_q4/kindi/reference-oxcoder-q4.json
```

Resume with the same identity-bound options and `--resume`, after stopping the
prior process. Original source weights are never updated. Optimizers and random
states restore together. Check service exit status as well as checkpoint files.

## Reference generation

After a generator checkpoint exists, write a messages-array JSON and run:

```bash
python -m scripts.ncm.generate_candidate \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --checkpoint results/ncm-generator-v6b/latest.pt \
  --messages /absolute/path/messages.json --output results/ncm-candidate-output.json
```

Source, codec and parameter inventories are checked. Invalid byte output,
context exhaustion, chunk limit and predicted end are reported explicitly.
There is no AR fallback. Generated code is data and is not executed by this
command. The CUDA BF16 path is for research evaluation, not a Mac release.

## Automatic executable-code diagnostic

`ncm-generator-v6b-evaluate` waits for the training service to finish, requires a
completed run and an exact checkpoint reload check, then runs `coding_eval.py`.
The resulting `coding-evaluation.json` covers twelve elementary Python tasks.
Function calls execute in bubblewrap with no network, no home-directory mount,
a cleared environment and CPU, memory, output-size and process limits. Expected
results are compared outside the worker. Early exits cannot report a pass, and
the interval-merging task also checks input preservation. Generated code never
executes in the model/training process.

The evaluator records invalid output and incomplete termination as failures.
Its report and `evaluation-status.json` remain `quality_ready=false`: these
small function tasks alone cannot establish professional quality. No baseline
comparison, Mac inference speed or 8GB deployment result is implied.

The first 300-update generator run completed its exact reload check and recorded
gradients through all 32 adapters. It passed **0/12** function tasks; all four
development chunks also failed valid terminated decoding. Its checkpoint is a
failed research candidate, not a converted coding model.

## Explicit continuation controls

The initial curriculum has 7,457 Alpha chunks and 84 agent chunks; 380 of the
7,541 admitted chunks are teacher responses. Uniform chunk sampling therefore
rarely visits agent examples or complete-response endings. The optional
`--balanced-curriculum` instead samples uniformly among represented combinations
of source, distilled/non-distilled provenance and final/non-final chunk, then
uniformly within that combination. `sampling.json` records each stratum. This
deliberately changes the training distribution and can overfit small strata;
validation must remain unfiltered and must not adopt that sampling distribution.

`--byte-loss 0.2` adds a final-level decoded-byte cross-entropy term. Softmax
concept probabilities form an expected codebook vector, which is added to the
known earlier-level latent and passed through the frozen codec decoder. Gradients
reach generator logits through the decoder; codec weights remain frozen. Target
bytes appear only in the loss. This supplements, rather than replaces, the
discrete code loss. It is a training surrogate: inference still selects discrete
concepts and must be evaluated with actual generated text.

Use `--generator-init /path/to/immutable-checkpoint.pt` for a changed recipe in a
new output directory. It verifies source, codec, corpus, architecture and
parameter inventory, restores weights only, and records the initialization hash
and source step. A new optimizer and update counter distinguish continuation
experiments from an exact `--resume`. Resume retains prior development results.

### Optional FLA backend

The reference run does not use Flash Linear Attention. An isolated installation
of `flash-linear-attention==0.6.0` and `fla-core==0.6.0` was tested separately with
PyTorch 2.11.0+cu128 on an RTX 5060 Ti. Do not replace dependencies under a running
trainer. These packages optimize Gated-DeltaNet operations; they do not convert
an autoregressive model into NCM.

```bash
python -m scripts.ncm.audit_kernels --output /path/to/kernel-audit.json
```

The audit compares BF16 outputs and all five input gradients with the unwrapped
Transformers reference at lengths 64, 128 and 512, using a declared relative-L2
tolerance of 0.02. The measured errors were below 0.005. At length 512, one
operator's forward/backward averaged 14.38 ms for the reference and 2.95 ms for
FLA. Length 64 was slower with FLA. These timings were collected while another
training job was active and are not an isolated end-to-end benchmark.

An FLA training stage must specify `--kernel-backend fla --kernel-audit PATH`.
The audit must pass and match installed FLA/PyTorch versions. FLA availability
must agree with the explicit backend choice. The existing whole-backbone
autoregressive transfer audit remains required by the experiment launch; the
operator check cannot replace it. Backend and audit hash enter checkpoint
identity, so switching kernels is a new stage, not an undisclosed resume.

Completed continuation recipe: initialize from the completed 300-update
checkpoint, retain the same source/corpus/codec and 1,024-token conditioning,
enable balanced sampling and byte-loss weight 0.2, and use a freshly audited FLA
backend with Muon/AdamW learning rate 0.0001 for 1,000 additional updates.
The continuation additionally uses `--concept-input-norm 0.8`: projected concept,
level and position vectors are L2-normalized before entering the backbone,
while original prompt embeddings remain unchanged. A fixed source-text sample
had mean embedding norm 0.781 (range 0.536–0.998); the first candidate's learned
level embeddings alone had norms 63.48–64.20. This is a substantial input-scale
mismatch. The 0.8 setting is an explicitly recorded experimental correction,
not a proven universal calibration. Gradients pass through normalization and
the same setting is restored by the candidate loader during evaluation.
Development generation is evaluated every 250 updates, including an ending
chunk from each source alongside the original chunk selections. Checkpoints
are saved every 100 updates. The `ncm-generator-v6c` service waits for the first
stage's function evaluation, snapshots its checkpoint and implementation, then
trains and runs the same function tests. These changes are experimental;
queuing them does not establish that the continued model can write code.

The first continuation checkpoint at update 100 loaded successfully and all
26,247,169 trainable parameters were finite. Its first 100 updates included 46
Alpha-source chunks and 54 agent-source chunks; 50 updates used cached teacher
responses (these counts overlap the source counts). These are samples from the
bounded corpus described above, not training on the entirety of either dataset.
This integrity check does not qualify generated code. The job finished all
1,000 updates and passed **0/12** executable function tasks. The final loss was
3.319997; generated text remained unusable. The checkpoint is a failed research
candidate. The [version 7 diagnostic and adaptation stage](GENERATOR-V7.md)
records the subsequent investigation.

```bash
systemctl --user status ncm-generator-v6c
journalctl --user -u ncm-generator-v6c -n 20 --no-pager
cat results/ncm-generator-v6c/checkpoint-integrity-100.json
cat results/ncm-generator-v6c/generation-evaluations.json
# Created when the post-training evaluation runs:
cat results/ncm-generator-v6c/coding-evaluation.json
```

Check both the service result and the evaluation files. `training.json` becoming
complete does not mean coding evaluation passed; `quality_ready` remains false.
The local checkpoint is `results/ncm-generator-v6c/latest.pt`, paired with the
original source GGUF and `results/ncm-coverage-decoder-v6/best.pt`. It is a
research adapter/interface checkpoint, not a standalone GGUF or Mac release.

The custom codec still uses linear interpolation for downsampling, whereas
upstream `models/quant.py::_down_for_quant` defaults to area averaging. Combined
with the tokenwise encoder, this means the custom coarse codes are not known to
contain semantic summaries of their whole span. This remains a representation
limitation, separate from reconstruction accuracy. Changing that convention
requires a new codec, newly encoded targets and an explicitly new generator
experiment; it cannot be silently changed in a running checkpoint.

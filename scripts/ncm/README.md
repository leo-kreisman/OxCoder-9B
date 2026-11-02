# OxCoder-initialized next-concept conversion experiment

The [frozen-encoder/native-decoder experiment](CONTEXT-DECODER-V25.md) caches
prompt-only states from all original OxCoder layers and trains a separate
bidirectional multiscale decoder with larger batches. This is distinct from
converting the original layers into a concept generator. Its data, inference
boundary and unchanged twelve-task acceptance test are documented explicitly.

The [paired learning-rate continuation](LEARNING-RATE-COMPARISON-V24.md) compares
two rates using identical training examples and a frozen 64-conversation
validation selection. The [prompt-conditioning control](PROMPT-CONDITIONING-CONTROL.md)
tests whether predictions benefit from the correct request. Neither diagnostic
substitutes for the unchanged executable acceptance and reload tests.
The optional [prepared-data JSON cache](PREPARED-DATA-CACHE.md) reduces corpus
startup cost without changing training records or the active comparison arms.

The [host optimizer stage](OPTIMIZER-OFFLOAD-V24.md) addresses the GPU0 memory
failure encountered beyond the initial parallel smoke. Its CPU AdamW path
retains learned moments and removes about 512 MiB of GPU optimizer state per
replica. This is a training-memory adjustment, with separately measured
numerical error and update cost; it does not establish NCM coding quality.

The [two-replica training path](PARALLEL-TRAINING-V23.md) passed numerical controls
and measured 1.40–1.64× faster four-example backwards, including gradient
communication. Its optimizer continuation and model-quality gates remain separate.
The [publisher-test filter](DISTILLATION-VERIFICATION-V22.md) independently checks
offline teacher answers before a new training-data expansion. The
[FP32-master Muon stage](MASTER-OPTIMIZER-V22.md) preserves small adapter updates
that previously rounded away in BF16.

The [compact frozen-weight training experiment](COMPACT-TRAINING-V20.md) records
exact GGUF matrix checks, complete-model numerical controls, independent-example
packing, and the optimizer/reload gates for the expanded corpus.

The [instruction and distillation expansion](TRAINING-EXPANSION-V19.md) adds a
pinned Python instruction supplement, checksum-bound offline teacher collection,
and an accumulated-gradient training entry point. It preserves both creator
datasets and the frozen evaluation inventory. Preparation and diagnostic results
are recorded separately from the unqualified student model.

The [training-acceleration implementation](TRAINING-ACCELERATION-V18.md) adapts
VeOmni's differentiable shared-prefix layout to the four NCM concept levels.
It includes full-model numerical/performance gates and optimizer-preserving
continuation. A passing execution audit is separate from model quality.

The [expanded-data stage](EXPANDED-GENERATOR-V17.md) adds 31,050 prepared creator
chunks while preserving evaluation and existing teacher responses. The original
AR source passed 12/12 elementary coding tasks; the preceding NCM candidate
passed 0/12. No usable NCM coding model has been qualified.

The [independent-classifier adaptation](INDEPENDENT-GENERATOR-V16.md) follows
an actual OxCoder control that memorized 8/8 training chunks with new classifiers
versus 0/8 with the tied heads. This is not held-out coding success.
[Complete creator-source acquisition and admission](CREATOR-EXPANSION-V15.md)
records the larger data expansion separately from the current training inputs.

The [reconstruction and output-head investigation](REFINEMENT-V12-V13.md)
continues from the completed, failed version 11 baseline. Decoder-only
calibration remains insufficient; a training-only control isolates a successful
alternative output interface. Neither result qualifies a coding model.

The [open-data baseline](OPEN-BASELINE-V11.md) tests the released NCM modules
on a bounded Python corpus, with code-only/mixed codec arms and downstream
small-generator diagnostics. This is a prerequisite experiment, not a converted
OxCoder checkpoint. Version 10 stopped at update 2,077 on its GPU reserve check.

The [version 10 generator run](GENERATOR-V10.md) uses the new contextual codec,
fresh interfaces and source/reload checks. Its new prefix audit failed, so the
replacement training path explicitly disables caching and uses full context.
Its first executable evaluation scored **0/12**. It remains an unqualified
research run, with both datasets and teacher responses.
The [contextual codec stage](CONTEXTUAL-CODEC-V9.md) addresses the measured
two-position dependency of the preceding first-level concept. It retrains the
representation; no usable NCM coding checkpoint has been qualified.
The [version 8 experiment](GENERATOR-V8.md) removes conflicting class-frequency
biases from the residual head and adds decoder supervision. Both creator datasets
and cached OxCoder distillation were used; its two coding evaluations failed.
The [version 7 investigation](GENERATOR-V7.md) follows two failed generator
stages with oracle controls, audited prefix reuse, longer conditioning and
codebook-tied output heads. No usable NCM coding checkpoint has been qualified.
The [version 6 generator experiment](GENERATOR-V6.md) documents improved codec
reconstruction, exact-target curriculum admission, full prompt conditioning,
response termination and actual generator training. It remains unqualified for
release; historical failed results below are retained.

This directory implements **an architectural adaptation**, not speculative
decoding. A learned multiscale residual vector-quantization codec maps output
text to discrete concept levels. The generator reuses all 32 OxCoder text
backbone layers and predicts an entire level in a forward pass. A learned
decoder reconstructs text after the last level. There is no autoregressive
OxCoder verifier in this path.

**Status: experimental conversion/training pipeline. It is not a usable converted
coding checkpoint.** A successful load or gradient check establishes plumbing,
not coding competence. See [RESULTS.md](RESULTS.md) for actual measurements and
the gates still failing. The original GGUF release is unchanged.

The [combined-data follow-up](FOLLOWUP.md) adds complete responses from both
datasets, cached OxCoder distillation, official NCM codec audits and controlled
optimizer/architecture experiments. It is the current continuation guide; the
phases below describe the original structural byte-codec pilot.
The [Muon integration](MUON.md) provides a separately selectable official-PyTorch
optimizer for hidden matrices while retaining AdamW for embeddings and heads.

## Architecture and explicit differences from the NCM paper

Reference: [Next-Concept Modeling](https://arxiv.org/abs/2610.14162),
[official implementation](https://github.com/HazyResearch/ncm), inspected at
`c70274273f489abc73832b2b98584a8483b6b203`.

The initial byte-codec implementation below is independent and does not load
upstream NCM weights. The separate follow-up wrapper loads pinned official codecs.
It implements residual quantization and coarse-to-fine generation, while making
three substantial experimental changes:

1. **UTF-8 bytes as codec symbols.** Bytes 0–255, EOS 256 and PAD 257 preserve
   whitespace, punctuation, identifiers and Unicode without text normalization.
   This is not a claim that the learned codec reconstructs them without errors.
   Invalid UTF-8, premature padding, missing EOS and incorrect bytes are failures.
2. **OxCoder's existing hybrid causal mixers remain.** Upstream NCM uses a
   different attention pattern. Retaining Qwen3.5's Gated DeltaNet and attention
   modules avoids discarding their pretrained weights. Each scale gets its own
   fresh recurrent state. The known, upsampled previous-scale representation is
   the input; no current-scale target codes are fed into the generator. All
   current-scale predictions come from the same forward evaluation. A recurrent
   implementation may still scan positions internally; parallel output does not
   guarantee fast execution.
3. **Parameter-efficient adaptation first.** Frozen backbone weights retain the
   source checkpoint. Trainable rank-8 residual adapters at every layer, latent
   projection, positions, level embeddings and concept head learn the new task.
   These are residual bottleneck adapters, not LoRA applied to every matrix.
   Whether adapters suffice is an open experiment; this can fail even when all
   numerical tests pass. Full backbone training is not implemented by this path.

The initial codec window is 128 bytes, with levels `[1,4,16,64,128]`. This is a
short-window proof of concept, **not 128K context support**. Training examples are
explicit response chunks; the last part of the prior response is conditioning.
Long-answer rolling generation, prefix-state reuse and Mac deployment remain
unimplemented. Do not extrapolate a short-window timing to a coding agent.
This describes the original pilot. Version 6 adds rolling chunks and a learned
response-ending head. Version 7 adds experimentally audited prefix-state reuse.
These are not quality-qualified; Mac NCM deployment remains unimplemented.
The five levels contain 213 code positions in total. This pilot does not claim
compressed storage relative to 128 raw bytes; it tests hierarchical prediction.
Five backbone passes, prompt recomputation and codec decoding must all be counted
when comparing latency with normal token generation.

## Weight provenance and transfer

The supported input is the existing OxCoder Q4_K_M GGUF:

```
SHA256 a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f
size   5,629,117,536 bytes
```

`backbone.py` uses the standard `gguf` dequantizer. It reads matrices individually,
undoes the GGUF value-head reordering, converts folded `-exp(A_log)` back to
`A_log`, and subtracts one from zero-centered RMSNorm weights. Gated DeltaNet's
ordinary normalization weight must **not** receive that subtraction. Convolution
dimensions are restored. Every target parameter must have the expected shape.
The SHA check rejects alternative checkpoints until they have been audited.

All 32 text blocks and the final norm are reused. The original output vocabulary
head is replaced with a concept-code head. The original token embedding is used
for conditioning prompts: its **packed CPU copy is resident**, and selected rows
are dequantized for each prompt. No dense-layer SSD streaming occurs during the
forward or backward pass. The quantized GGUF is read during loading only.

Backbone matrices are resident BF16 during training, split across the selected
CUDA devices. Small norms and decay parameters retain FP32. Existing Q4/Q6
quantization error is not reversed; this does not recreate the original BF16
checkpoint. The name “OxCoder-initialized” describes the source, not an unchanged
9B parameter count: the removed vocabulary head and newly introduced modules
change that count. `transfer.json` reports the actual reused parameter count.

The original BF16 source revision used for bounded audit reads is
`OrionLLM/OxCoder-9B@07ba04e2149bace1b43c66448471a70378a7a708`.
`audit_source.py` compares selected original norms, decay parameters and a gate
projection with inversely transformed GGUF tensors. It requires HTTP range
responses and downloads only small tensor ranges. This audit is not full-forward
parity with the original model. Full autoregressive baseline parity remains an
additional prerequisite before any claim that all pretrained behavior survived.

## Environment and resource accounting

Training runs on Linux/CUDA; this is not an MLX implementation. Install in a
dedicated virtual environment from the repository root:

```bash
python3 -m venv .venv-ncm
.venv-ncm/bin/python -m pip install --upgrade pip
# Select a PyTorch CUDA wheel compatible with the local GPU/driver first.
.venv-ncm/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-ncm/bin/python -m pip install -r scripts/ncm/requirements.txt
.venv-ncm/bin/python -m unittest discover -s tests/ncm -v
```

The recorded environment uses PyTorch 2.11.0+cu128 and Transformers 5.17.0.
Do not replace the Transformers version without retesting the layer signatures,
rotary embeddings, recurrence, gradients and inverse GGUF transformations.
The reference PyTorch DeltaNet implementation is usable without installing FLA;
optimized kernels must be independently validated before benchmarking them.

The launcher creates a user systemd scope with a 24 GiB RAM ceiling, no swap,
four CPU equivalents and reduced CPU/I/O priority. It does not stop other model
servers, modify GPU power limits, or change the operating system's memory policy.
CUDA allocation is capped per process while reserving roughly 4 GiB initially on
each device. Host available memory must stay above 12 GiB; runtime checks also
reserve GPU headroom. Admission is not a guarantee against other processes
allocating memory later. Existing checkpoints survive an interrupted later step.

The **8 GB deployment budget is separate from training**. BF16 backbone training
on two GPUs is not evidence that this model can run within 8 GB. A future native
inference implementation must quantize the new modules, reuse resident packed
backbone kernels, and measure the codec, backbone, context and buffers together.
No existing GGUF/Metal runtime can load `latest.pt` as an NCM model.

## Phase 1: acquire a bounded, traceable dataset

```bash
.venv-ncm/bin/python -m scripts.ncm.data \
  --output results/ncm-data --alpha-rows 128 --agent-files 16
```

Both previously selected datasets are used, at pinned revisions:

- `TeichAI/Ox-Alpha-10k@ae678ee78880e14251a553a0d0dc8ec6bd6dad6c`
- `TeichAI/DeepSeek-v4-Pro-Agent@abbf7b71b145633b60beb21ddc0d158582fd80dd`

These are not OxCoder's original training mixture and are not certified correct
OxCoder outputs. Data acquisition never executes tool calls or source code from
the conversations. It records file names, revisions and hashes of downloaded
prefixes. Alpha acquisition is bounded by row count. Agent acquisition reads a
bounded number of per-session JSONL files. The initial pilot takes at most 12
chunks from each conversation; this favors early response content and is not a
balanced professional coding corpus. Scale and rebalance before quality training.

Splits are assigned to complete conversation-content hashes before chunking.
Duplicate target chunks are excluded globally. `train`, `validation`, and `test`
remain distinct. The codec trainer selects checkpoints on validation only.
The test split must not become a development set during repeated tuning.

Each output chunk is at most 127 UTF-8 bytes plus EOS. This codec target is
intentionally small. Original instructions and prior assistant text are retained
as conditioning, but the pilot generator bounds prompt tokens explicitly. Long
prompt truncation can remove essential task details; it is not quality-equivalent
to using OxCoder's full context.

## Phase 2: train and gate the codec

```bash
bash scripts/ncm/run_limited.sh .venv-ncm/bin/python -m scripts.ncm.train codec \
  --data results/ncm-data --output results/ncm-codec \
  --devices cuda:1 --steps 1000 --eval-every 100 --batch 16

# Continue the same optimizer and RNG states, targeting 6000 TOTAL updates.
bash scripts/ncm/run_limited.sh .venv-ncm/bin/python -m scripts.ncm.train codec \
  --data results/ncm-data --output results/ncm-codec \
  --devices cuda:1 --steps 6000 --eval-every 500 --batch 16 --resume
```

Files:

- `latest.pt`: full codec, optimizer, RNG, data identity and completed step.
- `best.pt`: validation-selected codec; frozen during generator training.
- `gate.json`: checkpoint hash and reconstruction metrics.
- `training.json`: losses, gradients, validation history and finite-run status.

The gate requires at least 99.9% non-padding symbol accuracy, 95% exact chunks,
and 99% correct EOS positions. These are preliminary admission thresholds, not
proof of production coding precision. Byte reconstruction must be evaluated on
complete unseen programs, identifiers, numbers, whitespace and tool-call JSON
before release. A low reconstruction loss is insufficient.

If this gate fails, normal generator training refuses to proceed. Improving
the codec, data coverage, codebooks or architecture is necessary. Training a
generator against a lossy codec cannot recover distinctions the codec discards.

Normal generator training now also requires `--complete-gate` with a passing,
SHA-bound whole-response evaluation from `evaluate_documents.py`, using the
follow-up corpus/manifest. The old chunk-only pilot cannot qualify by itself.
Structural `--smoke` checks remain limited to three updates and make no quality claim.

Freeze the selected codec before launching generator jobs. Do not train against
a `best.pt` path that another process may replace:

```bash
cp results/ncm-codec/best.pt results/ncm-codec/selected.pt
# After development selection is finished, evaluate the test split once.
bash scripts/ncm/run_limited.sh .venv-ncm/bin/python -m scripts.ncm.evaluate_codec \
  --checkpoint results/ncm-codec/selected.pt --data results/ncm-data \
  --split test --device cuda:1 --output results/ncm-codec/test-evaluation.json
```

The evaluator rejects overwritten result files and reports strict UTF-8/EOS
validity as well as exact reconstruction. Use `--split validation` for development.

## Phase 3: audit and initialize the pretrained backbone

```bash
.venv-ncm/bin/python -m scripts.ncm.audit_source \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --output results/ncm-source-audit.json
```

Inspect every audit result. Do not continue a failed audit. Norm/decay and
quantized-matrix tolerances differ because the latter are already lossy.
Shapes alone do not catch a value-head permutation error.

The generator command loads all 32 layers, adds adapters and writes a tensor
transfer inventory. A structural smoke run is allowed even when the codec gate
fails, specifically to test the loader, autograd and checkpoint implementation:

```bash
bash scripts/ncm/run_limited.sh .venv-ncm/bin/python -m scripts.ncm.train generator \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-codec/selected.pt --data results/ncm-data \
  --output results/ncm-generator-smoke --devices cuda:0,cuda:1 \
  --steps 3 --lr 0.0001 --prompt-tokens 32 --rank 8 --smoke \
  --audit-reference scripts/kindi_q4/kindi/reference-oxcoder-q4.json
```

Smoke runs are hard-limited to three updates. They test the first, middle and
last levels and cannot be reported as quality training. Required checks: finite loss,
nonzero up-projection gradient in every layer adapter, an actual parameter
update, exact reload of trainable parameters, and completion of all concept
generation levels. Generated bytes may be meaningless or invalid at this stage.

`--audit-reference` first tests the unadapted backbone against the existing
llama.cpp fixtures. It uses the original vocabulary head in bounded chunks only
for the audit. All reference output positions are evaluated with teacher-forced
prefixes; a mismatch blocks adaptation. This tests all 32 layers on two short
prompts, not general behavior or long-context parity. The vocabulary head is
not used by NCM generation.

## Phase 4: gated generator training

Only after the codec admission gate passes:

```bash
bash scripts/ncm/run_limited.sh .venv-ncm/bin/python -m scripts.ncm.train generator \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-codec/selected.pt --data results/ncm-data \
  --output results/ncm-generator --devices cuda:0,cuda:1 \
  --steps 100 --lr 0.0001 --prompt-tokens 128 --rank 8 \
  --audit-reference scripts/kindi_q4/kindi/reference-oxcoder-q4.json
```

This is a finite initial training budget, not an estimate of sufficient training.
Each update conditions on one conversation and supervises every scale, with
equal per-level loss weighting. Inputs for a scale contain only earlier scales'
reconstructions; the current target code is never supplied as an input. The
codec is frozen. Adapters and the new representation interfaces receive the
gradient through the original frozen backbone, using layer checkpointing.

`latest.pt` stores only trainable generator weights, optimizer state, RNG and
identities. It depends on the original GGUF and exact codec checkpoint. It is
not a standalone model and does not duplicate the multi-gigabyte backbone.
`--resume` restores optimizer state and targets a total update count. It rejects
different dataset, codec, source, adapter rank or key hyperparameters.

The first version uses clean earlier-level teacher targets. The NCM paper uses
conditioning corruption to reduce train/generation mismatch; implementing and
ablating corruption and training on generated earlier levels is an outstanding
research phase. A decreasing teacher-forced loss does not establish good
free-running generations.

## Phase 5: required evaluation before deployment

The current pipeline does not certify a professional coding model. The following
work remains necessary before treating the candidate as a replacement:

1. Measure full-forward parity of the reconstructed pretrained backbone against
   a trusted implementation, separating BF16/quantization differences from bugs.
2. Evaluate frozen-codec reconstruction on complete unseen coding files and JSON.
3. Evaluate free-running conditional generation, not just concept cross-entropy.
4. Compare executable hidden-test pass rates and tool-call validity with unchanged
   OxCoder on identical prompts, including long dependencies and exact edits.
   Execute generated programs only inside a constrained sandbox.
5. Select checkpoints on development tasks; score final held-out tasks once.
6. Benchmark source loading, prompt processing, code generation per scale,
   reconstruction, full-answer latency, UTF-8 byte count and common-tokenizer
   output-token count separately. Count total codec plus generator cost.
7. Measure peak resident memory across all processes/devices, with the trained
   codec and generator together. Do not add discrete CPU and GPU results and
   call them a measured Mac unified-memory result.
8. Implement the new concept scheduler and codec in a native inference backend.
   Ordinary llama.cpp token sampling cannot run this architecture unchanged.
9. Re-evaluate after inference quantization and on the actual target hardware.

Training-speed optimizations such as fused loss kernels or alternative optimizers
can be benchmarked after correctness. Their published speedups cannot be
multiplied or assigned to this new architecture without measurement.

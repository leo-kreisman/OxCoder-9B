# Contextual codec with area downsampling

This stage replaces a measured structural weakness in the independent byte-codec
experiment. It is codec training, **not a usable converted OxCoder model**.
The preceding generator stages failed executable coding tests. Their checkpoint
and evaluation files remain unchanged, and the original OxCoder runtime is not
modified.

## Architectural change

The selected earlier codec had no contextual encoder blocks and linearly
interpolated a 128-position sequence down to one position. Its first coarse code
therefore depended only on the two middle byte positions. This was confirmed by
the [dependency audit](validation/v8-controls/codec-context.json). Accurate final
byte reconstruction did not establish a document-wide concept hierarchy.

The new codec has two bidirectional Transformer blocks in its encoder and two
in its decoder, width 128, four attention heads, 4,096 entries per codebook and
levels `[4,16,64,128]`. Downsampling uses area pooling; upsampling retains linear
interpolation. All earlier-level inputs, quantization targets and generation
inputs use the same checkpoint-declared resampling rule. The first level now
contains four codes rather than one. These changes require a new codec and new
generator targets; old generator weights are not claimed compatible.

Each training update retains full-resolution reconstruction and residual-VQ
loss. An additional reconstruction loss, weighted 0.05, uses a uniformly sampled
strict prefix of the concept levels. This encourages useful early-level
information while retaining full reconstruction on every update. It is an
auxiliary objective inspired by multiscale training, **not an exact copy of
upstream NCM's fine-level dropout implementation**. Its effect must be evaluated.

The auxiliary straight-through estimator supplies encoder gradients; the
ordinary VQ loss trains codebooks. Evaluation always uses all levels and does
not sample a prefix. No lossless byte side channel, answer lookup or
autoregressive reconstruction fallback is introduced.

## Compatibility and tests

`CodecConfig` records `downsample` and `coarse_loss_weight`. Older checkpoints
default to `linear` and `0.0`, preserving their numerical behavior. Resume identity
comparison normalizes absent legacy fields to those defaults. Changing either
field or the architecture is a new experiment; initialization rejects a
different configuration. A generator reads the frozen codec's configuration,
rather than accepting a runtime pooling override.

The 53-test suite covers dependency reach, area-target/inference consistency,
auxiliary encoder gradients, RNG replay, deterministic full-level evaluation,
legacy configuration defaults, rejection of incompatible codec initialization,
checkpoint restoration and the earlier source/prefix/generator controls. Passing
these tests does not establish text reconstruction or coding competence.

## Dataset and optimizer

Input is the existing checksummed
`results/ncm-muon-distillation-v5/data` corpus, containing 14,388 training,
7,118 validation and 8,638 test chunks. It includes both pinned creator dataset
subsets, `TeichAI/Ox-Alpha-10k` and `TeichAI/DeepSeek-v4-Pro-Agent`, and 458 chunks
from accepted OxCoder teacher responses. Conversation separation and
evaluation-first target deduplication remain in force. Only training chunks
enter optimization; the test split is not used for checkpoint selection.

Muon updates eligible hidden matrices; AdamW handles embeddings, codebooks and
the output head. The learning rate is 0.001, batch size 16, seed 17. This is a
fresh initialization, not a resume or an unnoticed change to a frozen codec.

## Run and inspect

Use the pinned environment described in [GENERATOR-V6.md](GENERATOR-V6.md).
From the repository root:

```bash
python -u -m scripts.ncm.train codec \
  --data results/ncm-muon-distillation-v5/data \
  --output results/ncm-contextual-codec-v9 --devices cuda:1 \
  --steps 20000 --eval-every 1000 --width 128 --layers 2 \
  --codes 4096 --levels 4,16,64,128 --downsample area \
  --coarse-loss-weight 0.05 --optimizer muon --lr 0.001 --batch 16
```

The local service is `ncm-contextual-codec-v9`; host memory is capped at 8GiB,
swap is disabled, CPU quota is two cores and runtime is capped at two hours.
GPU admission and periodic headroom checks remain enabled. Existing inference
services are not stopped. These limits are training safeguards, not an 8GB
OxCoder NCM inference measurement.

```bash
systemctl --user status ncm-contextual-codec-v9
journalctl --user -u ncm-contextual-codec-v9 -n 3 --no-pager
cat results/ncm-contextual-codec-v9/training.json
```

`best.pt` is selected by validation symbol accuracy; `latest.pt` includes full
optimizer/RNG state for `--resume`. After the finite run, audit the stable selected
checkpoint on complete responses and syntax/Unicode probes:

```bash
python -m scripts.ncm.evaluate_documents \
  --checkpoint results/ncm-contextual-codec-v9/best.pt \
  --data results/ncm-corpus-v2 --device cuda:1 \
  --output results/ncm-contextual-codec-v9/complete-gate.json

python -m scripts.ncm.audit_codec_context \
  --codec results/ncm-contextual-codec-v9/best.pt \
  --output results/ncm-contextual-codec-v9/dependency.json
```

The complete-response gate is stricter than the training loop's chunk gate.
A high symbol accuracy or a passing chunk gate cannot substitute for complete
code preservation. Do not launch quality-qualified generator training against a
failed complete-response gate. A passing codec would only permit the next
generator experiment; a usable model additionally requires fresh executable
coding tests, original-model comparison, serialization parity, tool-use tests
and measured deployment memory/latency. No artifact in this stage is a release.

## Intermediate controls

The immutable update-8,000 snapshot reached all 128 input positions in the
continuous-latent dependency probe, correcting the preceding two-position
limitation. This establishes computational dependence, not semantic quality.
It reconstructed 20/30 complete development responses exactly, with 99.9836%
non-padding symbol accuracy; four of six syntax/Unicode probes were exact.
The complete-response gate therefore still failed at that point. These are
codec reconstruction results, not generated functions or OxCoder coding scores.
The reports are retained under `validation/v9-controls/`.

## Coverage and decoder calibration

The 20,000-update base run selected update 20,000 by its aggregate chunk symbol
score. That checkpoint reconstructed only 14/30 complete responses and four of
six probes exactly. The immutable update-8,000 snapshot was better on complete
responses, at 20/30, and was therefore used to initialize an explicitly separate
coverage stage. Neither checkpoint passed the whole-response gate.

The coverage stage mixes natural training chunks with synthetic byte sequences.
Synthetic coverage teaches representation of symbols, not reasoning or coding
knowledge. Unused fine-level codebook entries can be reinitialized from observed
training residuals, with their optimizer moments cleared. No validation target
is added to training to repair a particular observed error.

`--select-complete` changes validation selection explicitly: prioritize a passing
whole-response/probe gate, then the number of exactly reconstructed development
documents plus probes, then natural validation symbol accuracy. These repeatedly
used documents are development data, not an independent qualification benchmark.
Checkpoint identity records this selection policy. Historical runs retain their
original policy and reports.

```bash
python -u -m scripts.ncm.calibrate_codec \
  --checkpoint results/ncm-contextual-codec-v9/midpoint.pt \
  --data results/ncm-muon-distillation-v5/data --corpus results/ncm-corpus-v2 \
  --output results/ncm-contextual-coverage-v9 --device cuda:1 \
  --steps 6000 --eval-every 1000 --batch 32 --lr 0.0001 \
  --revive-every 1000 --select-complete
```

The completed coverage stage selected update 6,000 and reconstructed 28/30
complete responses and all six probes exactly. Its complete gate still failed.

A decoder-only continuation freezes the learned encoder, positions and
codebooks. `--disable-coarse-loss` disables only the earlier training auxiliary
objective during this full-reconstruction calibration; it does not change the
encoding, quantization or inference resampling rules. This option is rejected
unless the encoder is frozen. A parameter digest verifies that the frozen
representation remains unchanged across the run. Global and data RNG states
are saved for reproducible stochastic-codec resumes.

```bash
python -u -m scripts.ncm.calibrate_codec \
  --checkpoint results/ncm-contextual-coverage-v9/best.pt \
  --data results/ncm-muon-distillation-v5/data --corpus results/ncm-corpus-v2 \
  --output results/ncm-contextual-decoder-v9 --device cuda:1 \
  --steps 10000 --eval-every 1000 --batch 32 --lr 0.0001 \
  --decoder-only --disable-coarse-loss --select-complete
```

The corresponding local units are `ncm-contextual-coverage-v9` and
`ncm-contextual-decoder-v9`. Inspect each `training.json` and `complete-gate.json`
to distinguish active training, successful process completion and actual gate
results. None of these states automatically qualifies or uploads a generator.

Decoder calibration completed 10,000 updates and selected update 6,000. It
reconstructed **29/30** complete development responses and **6/6** probes exactly,
with 99.999532% non-padding symbol accuracy on the complete responses. The
remaining response had invalid terminated text, so the broad gate remains
**failed**. The frozen-parameter digest confirmed that calibration did not alter
the encoder, positions or codebooks.

The selected codec SHA256 is
`47bef510d3e01f8823bb964bb855244c74aae4941d34ddbc8611480cffaef64c`.
The [version 10 generator experiment](GENERATOR-V10.md) uses the existing explicit
exact-target training curriculum to investigate this substantially more faithful
contextual representation. That research exception does not pass the broad gate
or remove failing validation examples. It does not authorize release promotion.

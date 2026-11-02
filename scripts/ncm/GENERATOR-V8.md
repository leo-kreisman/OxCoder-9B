# Nearest-code residual-head experiment

Recorded 2026-11-01. This stopped experiment is **not a usable NCM release**.
Both update-250 and update-500 checkpoints passed **0/12** function tests. The
last recorded update was 521; the retained checkpoint is update 500. The planned
3,000-update run was stopped after the coarse-representation dependency audit.
The [contextual codec stage](CONTEXTUAL-CODEC-V9.md) addresses that measured
limitation. The original OxCoder Q4 model and its working runtime are unchanged.

## Why the preceding head was rejected

The calibrated geometry head from [version 7](GENERATOR-V7.md) added learned
class-frequency biases to negative squared-distance logits. Its residual MSE
term simultaneously encouraged prediction of the actual residual vector. Those
objectives need not agree: even supplying the exact correct codebook vector can
select another code when its class bias is larger.

`audit_head_geometry.py` isolates this question by supplying each observed
training code's own vector directly to the head's scoring function. It does not
run the generator, use test targets, or measure coding quality. On the stopped
v7c update-875 checkpoint, the final-level head selected the supplied code ID
only 1.2615% of the time across 3,250 observed codes. Frequency-weighted accuracy
was 20.5157%. This is evidence of an objective conflict, not proof that the head
could never learn shifted vectors to classify correctly.

The new nearest-code head freezes class biases at zero. A linear projection with
a trainable offset predicts each residual. Logits remain scaled negative squared
Euclidean distances to the frozen codebook. Initialization uses admitted training
counts to set the projection offset to the weighted mean code vector and to set
the distance scale; counts no longer create class preferences in the logits.
The scale is positive, so it affects training probabilities but not nearest-code
argmax ordering.

On the update-100 v8 control, supplying exact code vectors selected the same code
ID in 100% of the first four levels and 99.8769% of the final level. The final
frequency-weighted result was 99.99943%. Four very close fine-level vectors still
failed the finite-precision identity check. This is not perfect numerical
identity and does not establish that generated code is correct.

## Training changes

This stage starts from v6c's shared adapters and interfaces, with a newly
initialized head. It does not continue the failed v7c head or its optimizer.
The recipe adds final-level decoder cross-entropy on the continuous predicted
residual, weighted 1.0, to the preceding code CE, normalized residual MSE,
expected-code decoder CE and response-ending losses. The frozen codec decoder
provides this training gradient. At inference, output still comes from selected
discrete concept codes; continuous residuals do not bypass quantization.

The first 100 updates train the heads only. Subsequent updates train all 32
rank-32 residual adapters, latent projection and other interfaces. Muon updates
hidden matrices; AdamW updates heads and the remaining parameters. Source
backbone matrices and the codec remain frozen. Prefix states are detached, so
target-side adapters receive gradients but prompt processing does not. The
failed level-batching optimization remains disabled.

Both requested source datasets remain in use:

- `TeichAI/Ox-Alpha-10k` at `ae678ee78880e14251a553a0d0dc8ec6bd6dad6c`.
- `TeichAI/DeepSeek-v4-Pro-Agent` at `abbf7b71b145633b60beb21ddc0d158582fd80dd`.

The run also includes the existing 33 accepted responses from the original
OxCoder Q4 teacher, with incomplete responses excluded. These are bounded
dataset subsets and response-level distillation; no new teacher service is used
during this run. The exact-codec training curriculum contains 12,361 chunks,
including 454 teacher chunks. Twelve nonexact training chunks are excluded and
their IDs recorded. Validation examples are not removed for reconstruction
failures. Sampling and full untruncated conditioning are unchanged from v7.

The code-count audit uses those same 12,361 admitted chunks. The earlier
all-eligible count file included twelve extra chunks and is not the admitted
count file. Their checksums differ deliberately.

## Reproduction

Use the pinned Linux/CUDA environment and prepared data described in versions
[6](GENERATOR-V6.md) and [7](GENERATOR-V7.md). The tested training machine has two
16GB NVIDIA GPUs; source weights are dequantized to resident BF16. This is not a
Mac runtime, an SSD-offloading result, or an 8GB inference measurement.

```bash
python -u -m scripts.ncm.train_geometry \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --init results/ncm-generator-v6c/latest.pt \
  --data results/ncm-geometry-data-v7 \
  --prefix-audit results/ncm-prefix-audit-v7b.json \
  --output results/ncm-generator-v8 --steps 3000 \
  --head-warmup 100 --eval-every 250 --save-every 25 \
  --head-input-norm --calibrate-head --nearest-code \
  --residual-byte-loss 1.0
```

The stopped local service was `ncm-nearest-v8`. It had a 24GiB host-memory maximum, zero
swap allowance, four-core CPU quota, twelve-hour runtime ceiling and periodic
GPU/host headroom checks. A completed service only means the configured run
ended, not that its model passed evaluation. To resume an interrupted run, use
the same command with `--resume`; never launch two writers for one output path.

```bash
systemctl --user status ncm-nearest-v8
journalctl --user -u ncm-nearest-v8 -n 5 --no-pager
python -m scripts.ncm.audit_head_geometry \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --checkpoint results/ncm-generator-v8/latest.pt \
  --counts results/ncm-geometry-data-v7/admitted-code-counts.pt \
  --output results/ncm-generator-v8/oracle-head-control.json
```

The audit hashes and loads the same checkpoint byte snapshot, even if the trainer
atomically replaces `latest.pt` concurrently. Generator loading requires a stable
checkpoint; copy it to an immutable filename before independent inference.
Checkpoint identities include `nearest_code=true` and the continuous-residual
loss weight. Frozen class biases are omitted from the trainable checkpoint and
reconstructed as zero on load; projection offsets are included.

## Reading results and qualification

`training.json` records the actual completed update count, loss components,
source/teacher provenance, five code accuracies and adapter gradient norms.
`coding-N.json` contains generated text, termination failures and isolated
executable tests for the checkpoint at update N. `chunks-N.json` measures free
generation separately from teacher-forced loss. Evaluations occur every 250
updates and at the final update. No model is automatically promoted or uploaded.

The twelve elementary function tasks have been reused for debugging and are
development tests, not an untouched benchmark. Any candidate that passes them
still requires fresh program tests, comparison to the original OxCoder model,
complete-response codec fidelity, tool-use evaluation, checkpoint reload parity
and deployment memory/latency measurements. The broader codec gate still fails
at 28/30 exact development responses. Its noncontextual encoder and coarse-code
semantics remain significant architectural limitations.

Every artifact remains `quality_ready=false`. Head identity, finite gradients,
lower loss and fast invalid output cannot qualify a coding model. Small control
reports are preserved in [`validation/v8-controls/`](validation/v8-controls/).

## Coarse-representation dependency control

`audit_codec_context.py` measures input-position influence on the first coarse
latent before discrete quantization. The selected codec has no contextual
encoder layers. Linear interpolation from 128 positions to one consequently
reads only positions 63 and 64, using zero-based indexing. A directional
derivative through the actual checkpoint confirmed exactly those two positions.

An area-pooling control reached all 128 positions, as did a contextual encoder
in the small unit control. Area pooling by itself does not establish semantic
concepts, invertibility or coding quality. It is **not** a checkpoint conversion:
changing pooling or introducing contextual encoder layers changes the target
code distribution and requires a newly trained and gated codec, newly encoded
generator targets, and new generator training. The existing checkpoints are not
silently relabeled or loaded under a changed quantization rule.

This structural result narrows a remaining weakness in the current independent
byte-codec variant. Its first concept cannot encode whole-response information
absent from those two input positions. Later generator levels still see the
prompt, so the result alone is not a mathematical impossibility proof for the
generator; it does show why byte reconstruction accuracy cannot establish a
useful semantic hierarchy. It is a substantive difference from upstream NCM's
contextual encoder and area downsampling.

```bash
python -m scripts.ncm.audit_codec_context \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --output results/ncm-codec-context-audit-v8.json
```

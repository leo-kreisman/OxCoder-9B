# Frozen OxCoder encoder with a native next-concept decoder

This is a separate architecture experiment, not a release-qualified model.
It retains all 32 original OxCoder layers as a frozen **prompt encoder** and
trains a new bidirectional decoder to predict multiscale concept codes. The
original vocabulary head does not generate answers or verify proposed tokens.
All answer bytes come from the frozen residual-VQ codec applied to the predicted
concept hierarchy.

This differs materially from the version 24 adapter generator: the original
9B layers are not themselves converted into the concept decoder. Deployment
still requires the source OxCoder model, the trained native decoder, tokenizer,
and frozen codec. Calling the new decoder alone “OxCoder9B NCM” would omit an
essential part of the model. Neither architecture has yet passed the coding
acceptance gate.

## Motivation and controls

The controlled version 24 learning-rate comparison retained the baseline rate.
Both arms saw the same 600 examples in 150 updates from the same checkpoint.
The higher rate did not improve fixed validation concept loss or free chunk
accuracy. Prompt replacement increased loss in both arms, establishing that
conditioning had an effect, while complete coding generation still failed.

The native decoder separates prompt encoding from generator training. Prompt
states can be computed once and reused across epochs; gradients need not pass
through all nine billion frozen source parameters for every update. This is
an architectural tradeoff, not a measured training-speed claim. It also discards
the source model's pretrained answer-decoder computation, so retaining the source
encoder does not guarantee its coding competence will transfer.

## Data and leakage boundaries

The first selection has 4,096 distinct training conversations and the same
64 fixed validation conversations used by the previous comparison:

| Training source | Selected conversations |
|---|---:|
| NVIDIA OpenCodeInstruct | 2,980 |
| TeichAI Ox-Alpha-10k | 749 |
| TeichAI DeepSeek-v4-Pro-Agent | 367 |

There are 196 distilled training targets and 2,387 first-response chunks.
These are selection counts **before** exact-codec training filtering.
Selection uses stable identity hashes and explicit source/coding/distillation
weights. It does not inspect model predictions or target quality scores.
It selects one chunk per conversation, with a preference for first chunks and
an available admitted distilled response. This is a bounded pilot rather than
the entire prepared corpus.

Prepared archive SHA256:
`76e473333c630f0e4969e459f1065f07598affcad71d466f5d41dfe263acba26`.
New selection SHA256:
`c8f2b77ce6059a5a38d584e31e89f66d8a0c3879d796cafbc4c12ad2d45238fa`.
Fixed validation selection SHA256:
`bddf25264de77c432db612ca72c73c96506dfeb66c50bd179a98a79a864e16ec`.

`context_features.py` passes only `condition.ids` into the source encoder.
For a first chunk these are the formatted conversation and assistant opening.
For a later chunk they also include previous gold response chunks, as in the
existing training pipeline. The **current target chunk** and future chunks
never enter that encoder call. Standalone generation replaces previous gold
chunks with the model's own generated chunks. Validation-chunk scores therefore
must not be presented as complete-response coding results.

Every selected row must appear in a feature shard or an explicit rejection
record. Only training rows with inexact non-padding codec reconstruction may be
rejected. All 64 validation rows remain. The merge checks source, codec, code,
selection and shard identities, file checksums, source coverage, distillation
presence, duplicate rows and split separation. The loader additionally checks
tensor shapes, finite values, code ranges and target-byte identity.

## Model and training

The source prompt features are BF16, width 4,096, at most 2,048 tokens. They are
projected to width 768 with normalization and sinusoidal prompt positions. The
native decoder has 12 blocks, 12 attention heads, and a four-times-width feed
forward sublayer. Each block contains self-attention over a concept level and
cross-attention to the source prompt features.

The frozen byte codec uses width 128, 4,096 entries per codebook, and levels
`[4, 16, 64, 128]`. Its checkpoint is
`47bef510d3e01f8823bb964bb855244c74aae4941d34ddbc8611480cffaef64c`.
All positions within a level are predicted together. During training, the four
levels share one forward using a block-diagonal self-attention mask; no level
can read another level's query features. Each level input contains only the
reconstructed previous levels. Tests compare this combined execution with four
independent level passes and check gradients with activation checkpointing.

The native decoder contains 129,306,113 trainable parameters in addition to
the frozen source encoder. The initial recipe uses:

- FP32 trainable parameters and optimizer state, BF16 autocast for matrix work.
- Muon for hidden matrices; AdamW for heads, embeddings, positions and norms.
- Learning rate 0.0003, 100-update warmup, cosine decay to 10% of the peak.
- Batch 32, audited microbatch 8, 5,000 planned updates, gradient norm clipping at 1.
- Mean cross entropy across four levels plus 0.1 times response-stop loss.
- Previous-code corruption rising from zero after warmup to 0.15 over 200 updates.
- Activation checkpointing in decoder blocks; prompt-length sorting inside an
  already selected batch to reduce padding without changing its examples.
- Checkpoint every 100 updates; all 64 validation chunks every 500 updates.

Codec reconstruction remains FP32 during generation to match its label audit.
The source encoder and codec are frozen throughout. A trainable decoder
checkpoint stores optimizer state, CPU/CUDA random state and shuffled-epoch
position. Resume rejects a changed model, data or implementation identity.
An immutable weights snapshot is published for each validation checkpoint.

Feature extraction completed with 4,094 training rows and all 64 validation
rows. Two training rows failed exact non-padding codec reconstruction and were
recorded as rejections. The two feature directories occupy approximately
11.1 GiB. The previous generator remains preserved at update 550 with SHA256
`68e618e87131963963a7d4dc6517306564b2f962933dd72cb4c94c248b96fd2d`.

An independent update-100 snapshot passed a 16-conversation cached-versus-fresh
conditioning audit. Every prompt state was bitwise equal after fresh source
encoding, and all predicted concept IDs, reconstructed bytes and stop
probabilities matched exactly. This includes features originally produced on
either GPU and prompts up to 1,851 tokens in that sample. The checkpoint SHA256
was `e18a1d11b3e0ff2326161d9c50681ec89c091ed43189b3e505489335ca3f18a8`.
The report is `results/ncm-context-runtime-audit-v25/report.json` and the command
is `python -m scripts.ncm.audit_context_runtime`. This verifies the feature
cache/inference boundary, not whether the generated answers are useful.

`probe_context_checkpoint.py` selects 16 fixed training conversations, balanced
across source names, and evaluates their conditional chunks without updating
weights. Its purpose is to distinguish underfitting from train/validation
divergence. Its scores cannot satisfy the twelve-task gate.

A full-shape synthetic CUDA audit compared microbatches 2, 4 and 8 on the same
eight examples at 2,048 prompt tokens. Median forward/backward times were
0.3002, 0.2081 and 0.1802 seconds respectively. The microbatch-8 gradient's
relative L2 difference from microbatch 2 was 0.003085, below the predetermined
0.02 bound; relative loss difference was 1.39e-7, below 0.01. Its peak PyTorch
allocation was 1,960,276,992 bytes and the first optimizer step took 0.1206 s.
These figures exclude source encoding, feature loading, checkpointing and
evaluation. They are **not** end-to-end training or generation speed. The audit
temporarily suspended only the GPU1 feature worker and resumed it in `finally`.
The other GPU continued extraction. Evidence is stored locally under
`results/ncm-context-batch-audit-v25b/report.json`.

The orchestrator checks the numerical report and decoder/trainer source hashes
before choosing microbatch 8. The standalone trainer's conservative default
remains 2; specify `--microbatch 8` to reproduce the audited run. A CPU synthetic
integration test also verifies that interrupting at a saved checkpoint and
resuming yields bitwise-identical model, optimizer, sampler and random states
to uninterrupted training. Neither synthetic test measures coding quality.

## Execution

Use the pinned Python environment documented by the earlier training stages.
This implementation targets Linux CUDA. It is not a Mac runtime or an 8 GB
whole-process deployment measurement. GPU allocator limits and host/application
reserves are enforced separately.

```bash
python -m scripts.ncm.context_features prepare \
  --data results/ncm-curated-training-v23 \
  --json-cache results/ncm-prepared-json-v24 \
  --validation results/ncm-validation-selection-v24 \
  --output results/ncm-context-selection-v25 --count 4096 --seed 25001

# Run both partitions, each on its own free device. Each performs the original
# source transfer audit before saving any prompt features.
python -m scripts.ncm.context_features extract \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --selection results/ncm-context-selection-v25 \
  --output results/ncm-context-features-v25-0 \
  --device cuda:0 --shard 0 --shards 2

python -m scripts.ncm.context_features extract \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --selection results/ncm-context-selection-v25 \
  --output results/ncm-context-features-v25-1 \
  --device cuda:1 --shard 1 --shards 2

python -m scripts.ncm.context_cache \
  --selection results/ncm-context-selection-v25 \
  --shards results/ncm-context-features-v25-0 results/ncm-context-features-v25-1 \
  --output results/ncm-context-pipeline-v25/features.json

python -m scripts.ncm.train_context_decoder \
  --cache results/ncm-context-pipeline-v25/features.json \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --output results/ncm-context-training-v25 --device cuda:1 --microbatch 8
```

The feature cache can require tens of gigabytes of disk. Its reader maintains a
bounded 512 MiB host LRU rather than loading the entire cache into RAM. This is
a training-data cache, not SSD expert offloading during model inference.

`run_context_stage.py` provides the workstation's explicit experiment handoff:
wait for the version 24 checkpoint boundary, stop only its named trainer and
monitor, record the saved weights/optimizer/RNG identity, extract both feature
partitions, merge them, then train and evaluate the new decoder. It does not
delete or overwrite the older model. Errors are recorded and must be inspected;
an incomplete extraction is not silently accepted.

## Acceptance and independent reload

```bash
python -m scripts.ncm.context_runtime \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint results/ncm-context-training-v25/evaluated-500.pt \
  --output results/new-context-coding-report.json --device cuda:0
```

This calls the existing `coding_eval.py` without changing its twelve prompts,
test cases, chunk budget, stop threshold or isolated execution checks. It loads
the original encoder and new decoder and performs complete free generation.
No teacher answer, reference implementation, retrieval answer or AR fallback
is available. A passing immutable checkpoint must score 12/12 again in a fresh
process with identical generated answers. The runner records that result only
after checking both reports and checkpoint identities. It requests that the
trainer stop at its next saved boundary when this gate passes.

Passing these elementary tasks would establish that specific acceptance result,
not professional coding competence or a deployment throughput claim. A lower
loss, successful feature extraction, working gradients, or a training-only
memorization result cannot substitute for it.

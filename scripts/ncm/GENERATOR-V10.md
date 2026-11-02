# Generator training with the contextual representation

This is an experimental OxCoder-initialized next-concept generator, **not a
qualified coding model**. The preceding generator stages scored 0/12 elementary
function tasks. Version 10 uses a new contextual codec and must be evaluated
independently; earlier timings or accuracy do not transfer to it.

The initialization passed all 33 source-reference checks, adapter-gradient checks
and checkpoint reload parity. The new prefix-cache audit **failed**. Consequently
the cached continuation stopped before any training update. The active follow-up
uses full-context computation with `--uncached`; it does not reuse prefix states
or relax the numerical tolerance. Its first executable evaluation, at update
250, scored **0/12**. All twelve generations ended with invalid codec output;
two emitted some malformed text before failing. The bounded training run
continues, but this checkpoint is not usable.

## Frozen representation and sources

The selected [contextual codec](CONTEXTUAL-CODEC-V9.md) has two encoder and two
decoder Transformer blocks, area downsampling, four concept levels
`[4,16,64,128]`, and 4,096 entries per codebook. Its first query can depend on all
128 input positions, unlike the earlier tokenwise representation. This is a
computational dependency result, not proof of semantic concepts.

The codec reconstructs 29/30 complete development responses and all six probes
exactly. Its broad gate therefore remains failed. Only training chunks that
reconstruct exactly are admitted to this research run. Rejected IDs and the
remaining source/teacher counts are recorded in `curriculum.json`; evaluation
examples are not removed for failing reconstruction. This is the same explicit
research exception introduced in version 6, not a lowered release threshold.

For the uncached run the filter retained 10,872 training chunks, including 430
teacher-response chunks, and rejected one reconstruction failure. Both sources
remain represented: 10,733 Alpha chunks and 139 Agent chunks. These are chunks,
not independent conversations or verified correct programs.

Data remains the pinned subsets of `TeichAI/Ox-Alpha-10k` and
`TeichAI/DeepSeek-v4-Pro-Agent`, plus accepted original-OxCoder teacher responses.
Prepared conditioning originates from the untruncated 4,096-token-limit corpus
described in version 7. The uncached stage selects complete prompts of at most
2,048 tokens, without truncating prompts or targets. Before the exact-codec
curriculum filter this leaves 10,873 training chunks (10,734 Alpha and 139 Agent),
5,221 validation chunks and 5,439 test chunks. It contains text and prompt IDs,
not cached old-codec targets; new concept targets are computed from the new codec.

The generator reuses all 32 original OxCoder text backbone layers. Backbone
weights remain frozen BF16 dequantizations of the original Q4 GGUF. Rank-32
residual adapters and new interfaces train. This is not a speculative drafter,
and the original autoregressive vocabulary head is used only for the source
transfer audit. It does not verify or repair generated answers.

## Initialization and required numerical checks

An initial three-update structural run uses the new codec, new four-level
interfaces and fresh adapters. It checks the original backbone against the
existing llama.cpp fixtures, requires gradients in every layer adapter, and
checks checkpoint reload parity. Those three updates are **not quality training**.

Example initialization, using the isolated FLA environment and memory limits
described in version 6:

```bash
python -u -m scripts.ncm.train generator \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --data results/ncm-muon-distillation-v5/data \
  --output results/ncm-generator-v10-init --devices cuda:0,cuda:1 \
  --steps 3 --eval-every 3 --save-every 1 --rank 32 --lr 0.0001 \
  --optimizer muon --smoke \
  --audit-reference scripts/kindi_q4/kindi/reference-oxcoder-q4.json \
  --complete-data results/ncm-corpus-v2 \
  --teacher-data results/ncm-muon-distillation-v5/teacher \
  --complete-gate results/ncm-contextual-decoder-v9/complete-gate.json \
  --prompt-tokens 1024 --parallel-attention --layer-split 12,20 \
  --kernel-backend fla --kernel-audit results/ncm-kernel-audit.json \
  --concept-input-norm 0.8 --exact-target-curriculum
```

`continue_generator.py` waits for the initialization unit to finish. It rejects
incomplete initialization, failed source/gradient/reload checks and a codec hash
mismatch. It copies stable codec and initialization snapshots into its run
directory and records their hashes. A new full-model prefix-cache audit must
pass before training begins; the old codec's numerical report is not reused.
The audit uses the actual first and last levels, including four-level codecs.

```bash
python -u -m scripts.ncm.continue_generator \
  --unit ncm-generator-v10-init \
  --initial-run results/ncm-generator-v10-init \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --data results/ncm-geometry-data-v7 \
  --output results/ncm-generator-v10 --steps 12000
```

For a direct initialization command rather than a systemd unit, invoke the
continuation only after initialization has exited and all required reports
exist. The supplied unit must not name an unrelated active process. Each
continuation output directory has one owner; the orchestrator refuses to reuse
an existing directory. Its status distinguishes waiting, auditing, training,
failure and process completion.

## Training and monitoring

### Failed cache path and full-context replacement

The new audit reported maximum relative forward error of 0.11014 and final-level
target-gradient error of 0.80149. The acceptance limits remain 0.02 and 0.05,
respectively. The old representation's passing audit does not qualify this one.
The failed report is retained at `results/ncm-generator-v10/prefix-audit.json`.
The `ncm-generator-v10` service exited with failure as intended; its proposed
12,000-update cached recipe above was never started.

Create the bounded full-prompt subset and run the replacement explicitly:

```bash
python -m scripts.ncm.restrict_conditioning \
  --data results/ncm-geometry-data-v7 \
  --output results/ncm-contextual-data-v10 --prompt-tokens 2048

python -u -m scripts.ncm.train_geometry \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-generator-v10/codec.pt \
  --init results/ncm-generator-v10/initial.pt \
  --data results/ncm-contextual-data-v10 \
  --prefix-audit results/ncm-generator-v10/prefix-audit.json \
  --output results/ncm-generator-v10/uncached \
  --steps 3000 --head-warmup 100 --eval-every 250 --save-every 25 \
  --head-input-norm --calibrate-head --nearest-code \
  --residual-byte-loss 1.0 --uncached
```

`--uncached` records the failed optimization audit but never prepares or reuses
cached prefix states. During adapter training gradients pass through the full
prompt and target computation. Level batching is rejected in this mode. This
costs more computation and cannot inherit the cached speed measurements.
The subset manifest binds the new data to its parent checksum and records every
source count. An existing output directory is not silently overwritten.

Training uses the nearest-code geometry heads from version 8, frozen zero class
biases, input feature normalization, residual supervision, decoded-byte losses
and a learned response-ending head. The first 100 updates train only the heads;
subsequent updates also train the adapters and interfaces. Muon updates hidden
matrices and AdamW handles heads and remaining parameters. Full-context adapter
gradients are enabled after warmup. Failed level batching remains disabled.

The replacement service is `ncm-generator-v10-uncached`. It is bounded to twelve
hours, 24GiB host memory, no swap and a four-core CPU quota. GPU/host headroom
checks remain enabled. This Linux/CUDA reference training path is not an 8GB
inference measurement or a Mac deployment.

```bash
systemctl --user status ncm-generator-v10-uncached
journalctl --user -u ncm-generator-v10-uncached -n 5 --no-pager
cat results/ncm-generator-v10/uncached/identity.json
```

The failed cached orchestrator's commands remain in `status.json`. The uncached
command is recorded in its systemd unit and above. Its files are under
`results/ncm-generator-v10/uncached/`:

- `training.json`: completed updates, loss components, source/teacher provenance,
  code accuracies and adapter gradients.
- `latest.pt`: resumable weights, all optimizers, RNG and identity.
- `coding-N.json`: generated responses and executable development tests every
  250 updates and at the final update.
- `chunks-N.json`: free-generation versus teacher-forced chunk diagnostics.
- `development-step-N.pt` and `best-development.json`: immutable snapshots when
  the number of passed development functions improves, including the first
  evaluated baseline. Checksums bind each snapshot to its evaluation.

To resume an interrupted training child, use its recorded command with `--resume`
and the same source, snapshots, data, audit and recipe. Do not start a second
writer or reset the optimizer silently. A timeout or failed numerical audit is
recorded as failure; the runner does not relax tolerances or continue regardless.

The twelve function tasks have been reused during debugging. Their score is a
development measure, not independent professional qualification. Every status
and checkpoint remains `quality_ready=false`. Even a completed 3,000-update run
must pass broader codec fidelity, fresh coding/tool-use tests, source-model
comparison and deployment memory/latency checks before a usable model can be
claimed. Nothing in this runner uploads or promotes a release automatically.

## External implementation check, 2026-11-01

The [MCL4-COE NCM project](https://github.com/mESmaC/mcl4-coe-ncm)
was inspected at `f70d90468f21b3dcfe30aa79c4ec70d379d9f3c0`. Its
[architecture mapping](https://github.com/mESmaC/mcl4-coe-ncm/blob/f70d90468f21b3dcfe30aa79c4ec70d379d9f3c0/docs/10_NCM_MCL4_COE_ARCHITECTURE_MAPPING.md)
describes donor-embedding anchoring and prompt cross-attention, but also reports
that its Stage-2 concept accuracy is insufficient for free-running coarse-to-fine
generation. Its classifier/decision results do not establish executable code
generation. No code or weights from that project were installed into this run.

An [upstream author comment](https://github.com/HazyResearch/ncm/issues/16)
also reports that codebook restarts improved reconstruction but harmed downstream
generation in their experiments. This matters because the present codec used
fine-code revival: its 29/30 reconstruction result alone cannot select a usable
generator. The v10 executable evaluations remain the deciding evidence.

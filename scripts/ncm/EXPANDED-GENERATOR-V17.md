# Expanded creator-data training: version 17

## Objective and current evidence

The first version 17 evaluation, at update 100, passed **0/12** coding tasks.
One of six development chunks had valid UTF-8/EOS structure; none matched its
target exactly. Training continued after this check. The evaluated checkpoint
was preserved with SHA256
`83a29ad651b8e70514ad374e830c48817ccac4bd4824ed984472557483d545e6`.
See [coding evidence](validation/v15-controls/expanded-coding-100.json) and
[chunk evidence](validation/v15-controls/expanded-chunks-100.json).

Across those 100 updates, 88 sampled the new expansion and four sampled cached
OxCoder teacher responses; 63 were Alpha-origin and 37 Agent-origin. Mean
update duration was 4.50 seconds on this workstation, excluding startup,
checkpoint writes and evaluations. This is training timing, not inference
tokens per second or an estimate for a Mac.

The objective is a useful OxCoder-derived next-concept coding generator. There
is no qualified NCM coding checkpoint yet. This stage reuses the actual OxCoder
text backbone, its trained adapters and independent concept classifiers, and
expands training data. It is not speculative decoding, an autoregressive
fallback, an MLX conversion, or a model-release command.

The original Q4_K_M OxCoder model passed all 12 elementary executable coding
development tasks. Version 16b's NCM candidate at update 100 passed none.
Both use the same task definitions and isolated checker. Their answer formats
and generation limits differ: the AR control allows 512 generated tokens;
the NCM evaluator uses its documented chunk limit. This is a capability
diagnostic, not a speed comparison or a general coding benchmark.

The AR control runs through `evaluate_ar_baseline.py`, records the source and
server-binary hashes, uses greedy decoding with thinking disabled, and retains
the finish reason and test outcomes for each response. Its answers are never
training inputs. Aggregate evidence is in
[ar-baseline.json](validation/v15-controls/ar-baseline.json).

## Immutable data inputs

The expansion descends from `ncm-contextual-data-v10/prepared.pt`, SHA256
`d110694ad0374b0ca30c72bb5e201c2f83d9d9775a55fe5220e797cb3f9996db`.
It preserves all 21,533 base rows, including the existing creator subsets,
cached OxCoder teacher responses, and evaluation partition.

The additional responses come from the two pinned creator datasets and the
contamination admission described in [CREATOR-EXPANSION-V15.md](CREATOR-EXPANSION-V15.md).
The admission manifest SHA256 is
`90320b8a463abcf2a8667a924de69b9e2cb1209948d42abf49702e7b567d80e6`.
Source answers are demonstrations, not independently verified solutions.

Preparation uses the original tokenizer and 128-byte UTF-8-safe response chunks.
Each chunk retains the complete preceding prompt and earlier response chunks.
Prompts exceeding 2,048 tokens are excluded rather than truncated. At most eight
evenly spaced eligible chunks are selected per additional response:

| Preparation measure | Result |
|---|---:|
| Admitted responses considered | 4,228 |
| Responses with selected chunks | 4,123 |
| Candidate chunks examined | 171,445 |
| Candidate chunks exceeding context limit | 27,551 |
| Selected Alpha chunks | 27,765 |
| Selected Agent chunks | 3,285 |
| Total added chunks | 31,050 |
| Base plus added rows | 52,583 |

The prepared file SHA256 is
`c9ef1955cc786bbf467d6fb028615e5d3a97bb02a304dbbf55b0285b642ae761`.
The unchanged evaluation fingerprint is
`cd6ce9b9a86447ab4250a2ffb405eebe99334729a370d537e68fcab9d2f4611d`.
Preparation counts precede the trainer's exact-reconstruction admission. Consult
the run's `curriculum.json` and `sampling.json` for actual eligible training rows;
do not describe all acquired or prepared examples as optimized examples.

The recorded admission retained **41,899 of 41,923 training chunks**, including
430 existing teacher-response chunks. Of those retained chunks, 38,478 come
from Alpha and 3,421 from Agent, counting their teacher derivatives. No evaluation
row was filtered. Per-step history confirms whether each selected row actually
participated in an update; eligibility alone is not exposure.

## Stage transition and training

Initialize from the saved version 16b update-100 checkpoint. Retain all trained
adapters/interfaces, independent classifier weights and the frozen version 9
byte codec. Reset optimizer state and RNG for an explicit new data stage.
This differs from `--continue-checkpoint`, which preserves optimizer/RNG state
and forbids changing data or learning settings.

The bounded version 17 schedule uses 2,000 updates, seed 17001, no repeated
head-only warmup, and the version 16 learning rates, losses, sampler and
conditioning corruption. Muon updates adapter/projection matrices; AdamW
updates classifiers and other interfaces. Prefix caching remains disabled
because the numerical audit failed. No decoded text is replaced with teacher
output during evaluation.

```bash
export PYTHON=/absolute/path/to/the/training/environment/bin/python
export PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m scripts.ncm.train_independent \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init results/ncm-independent-v16b/latest.pt \
  --data results/ncm-contextual-expansion-v17 \
  --output results/ncm-independent-v17 \
  --retain-heads --data-expansion --head-warmup 0 \
  --steps 2000 --seed 17001 --allocator-cap-gib 11.25 --eval-every 100
```

The data manifest and generator parent must match the recorded lineage. A
changed evaluation fingerprint, cross-split conversation, altered codec, altered
source model or mismatched parameter inventory stops loading. Resume using the
same command with `--resume`, keeping the initializer immutable. Do not modify
hashed trainer modules during the run. Checkpoints include trainable state,
optimizer state, RNG, history and identity, normally saved every ten updates.

The first invocation reached update 41, with update 40 saved, then stopped on
the 12 GiB host-memory reserve. An optional convolution-extension build had
selected a Python Ninja launcher that recursively invoked itself. Its entire
process group was terminated, including orphan descendants; the unrelated
applications were left running. The training command was resumed with unchanged
identity, restoring the saved model, optimizer and RNG at update 40.

Subsequent extension builds use an explicitly checked native Ninja executable
in a separate systemd unit with `MemoryMax=4G`, `TasksMax=64`, `CPUQuota=150%`
and a 900-second time limit. Extension files are installed to a separate target
directory absent from the training `PYTHONPATH`. Building an extension does
not authorize changing a running training environment. The operator audit in
`audit_causal_conv.py` compares outputs and input/weight gradients in FP32 and
BF16; a passing operator audit still requires a whole-backbone comparison and
an explicitly identified new training stage before adoption.

The optional extension subsequently built successfully after limiting the build
to this workstation's `sm_120` architecture and one NVCC thread. The CUDA kernel
sources were unchanged. The resulting local wheel SHA256 is
`49e54fb555182ba75f7747ce5b020dd9776656c06975e7bf6cc36f7908630df6`.
It is not a Mac artifact or a portable wheel for older CUDA architectures.

The isolated operator audit passed at sequence lengths 64, 512 and 2,048 in
FP32 and BF16, checking outputs and both input/weight gradients against the
unwrapped Transformers fallback. BF16 relative L2 differences remained below
0.0029. BF16 forward/backward speed ratios were approximately 1.04, 1.20 and
1.63 respectively, while the training job was also using the GPU. These are
contended operator timings, not end-to-end training or inference speedups.
The extension remains excluded from version 17's training environment.
See [operator evidence](validation/v15-controls/causal-convolution-audit.json)
and [build identity](validation/v15-controls/causal-convolution-build.json).

This is resident BF16 research training across two 16 GiB CUDA devices, with
12 backbone layers on the first and 20 on the second. The allocator cap is per
device, constrained further by free-memory admission and runtime reserves.
It does not measure the future 8 GiB inference target. Stop any separately
launched baseline server before admitting this training job.

## Monitoring and decisions

In the recorded workstation run, `ncm-independent-v17.service` owns the training
process; `results/ncm-independent-v17.log` is its stdout/stderr log. Service
availability is local and is not supplied by the repository checkout. Equivalent
foreground execution uses the command above.

Inspect these artifacts in the output directory:

1. `admission.json`: available memory and enforced allocator limits.
2. `curriculum.json`: codec reconstruction admission; failures remain visible.
3. `sampling.json`: eligible responses/chunks by source, coding and teacher origin.
4. `identity.json`: source, initializer, codec, data and implementation hashes.
5. `progress.json`: completed updates, losses, level accuracies, memory peaks,
   actual source exposure, and `expansion`/`distilled` markers.
6. `chunks-N.json`: teacher-forced code prediction and free-running decoded
   development chunks at each 100-update boundary.
7. `coding-N.json`: actual generated answers and sandboxed executable checks.

To retain each evaluated checkpoint while training continues, run this separate
stdlib-only observer from the repository root:

```bash
python3 -S -m scripts.ncm.preserve_evaluations \
  --directory results/ncm-independent-v17 \
  --unit ncm-independent-v17.service --seconds 14400
```

`-S` avoids unrelated site packages shadowing the repository's `scripts`
namespace. The observer reads the service status, never starts or restarts it,
and stops when the service exits or its duration expires. It copies a checkpoint
only after all 12 development tasks finish, verifies the copied bytes against
the evaluation's SHA256, and records `snapshot_preserved=false` if the latest
checkpoint advanced before copying. It never assigns an older score to newer
weights. `evaluated-N.pt` and `evaluated-N.json` are research artifacts; they do
not authorize deployment. Keep model binaries out of source control.

The version 17 recovery check replayed update 41 from saved update 40 with
identical example selection, loss and code accuracies. Timing and allocator
peaks are excluded from that comparison. This tests one recovery boundary,
not general cross-device reproducibility.

Training loss alone cannot authorize a release. If teacher-forced prediction
improves while free generation remains invalid, investigate accumulated
coarse-to-fine errors before increasing training blindly. If free text becomes
valid but code fails, inspect functional failure types and improve training
coverage using training-only data. If training examples cannot be learned,
repeat a bounded memorization/control test before spending a larger budget.

Passing these repeatedly inspected 12 elementary tasks is necessary evidence
of progress, but insufficient for deployment. A candidate must also pass
separate, previously unused coding and tool-use evaluations, preserve code
through its codec, terminate correctly, and meet memory and end-to-end speed
targets in the intended runtime. Measure cold/warm prompt processing, generated
output speed and whole-request latency separately. Neither 8/8 training-chunk
memorization nor the original AR model's 12/12 score transfers to the NCM model.

## Optional classifier initialization control

`initialize_independent.py` can expand a codebook-distance head into an
independent linear classifier without initially changing its logits. For code
vectors B, residual projection P and bias p, temperature multiplier a, and
class bias b, it sets W = 2aBP and bias = a(2Bp - ||B||²) + b. Input normalization
must match. This permits later independent updates; it does not convert an AR
model into NCM or invent concept knowledge.

An FP32 algebra control using the old actual checkpoint measured relative L2
logit errors below 2.3e-7 and identical argmax predictions on sampled hidden
features at every level. See
[head-expansion-algebra.json](validation/v15-controls/head-expansion-algebra.json).
Version 17 does not use this option: it retains the already independent version
16b classifiers. No held-out quality improvement is attributed to the helper.

## Related upstream evidence reviewed during this stage

The [NCM coding-agent discussion](https://github.com/HazyResearch/ncm/issues/17)
contains community training logs and ongoing experiments, not a demonstrated
OxCoder conversion. An independent author reports exact-string problems with
released codecs and recommends separately checking structured output. In a
[separate report](https://github.com/HazyResearch/ncm/issues/19), the same author
reports a small prefix-conditioned NCM losing to an equal-size AR model on one
classification task. Those findings concern different codecs, models and data;
their percentages are not measurements of this repository's model.

The practical implication is to retain executable-code and exact-reconstruction
gates even when loss or prose improves. Entity masking that requires known
correct spans cannot recover previously unknown identifiers in generated code.
It is not installed here as a hidden correction or AR fallback.

The authors also report in [issue 16](https://github.com/HazyResearch/ncm/issues/16)
that restarting unused codebook entries improved reconstruction but harmed
downstream generation in their experiments. Consequently a better codec metric
alone does not justify replacing the current codec. Every changed codebook
requires a separately identified generator stage and generation evaluation.

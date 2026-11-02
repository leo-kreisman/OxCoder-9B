# Independent-classifier OxCoder adaptation

## Evidence motivating the change

The [actual-backbone audit](validation/v15-controls/oxcoder-head-audit.json)
compared two output interfaces on eight training chunks. It used both creator
sources and existing OxCoder teacher responses, the version 9 byte codec, and
the version 10 uncached OxCoder-derived backbone. The codec reconstructed these
eight supplied targets exactly before classifier training.

| Output interface | Trainable head parameters | Exact freely generated training chunks after 500 head-only updates |
|---|---:|---:|
| Existing codebook-distance heads | 2,097,668 | 0/8 |
| Independent per-level classifiers | 67,125,248 | 8/8 |

The independent classifiers also reached 100% teacher-forced code accuracy at
all four levels. The geometry arm reached 100%, 99.2188%, 74.4141% and 9.8633%.
Both arms used the same fixed hidden features. The backbone and adapters did
not update during this audit. The independent arm starts with new weights and
has substantially more parameters; this is not an equal-capacity comparison.

This result removes one demonstrated learning bottleneck. It does not establish
generalization, executable-code correctness, professional coding quality, Mac
speed or 8 GiB inference memory. The audit exports no deployment checkpoint.

## Version 16 training recipe

`train_independent.py` loads the actual version 10 checkpoint, retains its text
backbone and adapter parameters, and replaces only the old concept heads with
fresh independent linear classifiers. The old checkpoint's frozen codec remains
unchanged; version 14's separately trained tokenizer and codebooks are incompatible
and are not substituted into this model.

The current data is `ncm-contextual-data-v10`, containing the original admitted
creator subsets and cached response-level OxCoder distillation. The separately
downloaded full creator reservoir is **not** silently added to this run. Any
expansion requires a separately hashed prepared dataset and a new training stage.

The recipe is:

- 32 frozen OxCoder text layers, resident BF16 research implementation across
  two CUDA devices; 12 layers on the first and 20 on the second.
- Existing rank-32 residual adapters, learned concept projection, level and
  positional inputs, end-of-response head, and fresh per-level classifiers.
- Four concept levels of lengths 4, 16, 64 and 128 with 4,096 entries each.
- Exact-reconstruction filtering on training chunks only. Selected validation
  failures remain in evaluation. The broader codec quality gate still fails.
- Full source chat template and complete previous assistant chunks; maximum
  2,048 prompt tokens, with over-limit examples excluded rather than truncated.
- 80% coding-response sampling; within each coding stratum, source/teacher
  buckets weighted by square root of response counts, then uniform response
  and chunk selection. This does not claim every source answer is correct.
- First 25 updates train classifiers and termination using detached features.
  Subsequent updates propagate gradients through the adapters and interfaces.
- AdamW at 0.0003 for classifiers and termination; AdamW at 0.00003 for remaining
  interfaces; PyTorch Muon at 0.00003 for adapter/projection matrices. Weight
  decay is 0.01. The planned bounded stage is 1,000 updates.
- Equal mean cross-entropy weight per level, plus 0.1 end-of-response BCE.
  No geometry residual MSE or continuous-residual decoder loss is applied.

After warmup, training corrupts earlier concept IDs by uniformly replacing them
within their corresponding codebook. Replacement probabilities across the four
levels are 0.5, one third, one sixth and zero. Labels remain the clean codes;
only conditioning changes. This is an explicit experimental robustness recipe,
not a claim to reproduce the paper's selected corruption rate or training scale.
Generation never receives clean target concepts or a teacher answer.

## Training preflight and resource behavior

The first run stopped deliberately after update 27, preserving its optimizer
and RNG states. It completed the 25 head-warmup updates and two full adapter
updates with finite, nonzero gradients in every adapter. Both creator datasets
and teacher responses were sampled.

Peak CUDA allocations through that preflight were 5,664,455,680 bytes on device
zero and 10,283,245,568 bytes on device one. These are allocator measurements
during training, not total system memory or deployment requirements. The process
reserves host/GPU headroom and caps its allocator independently of the user's
future inference budget.

The first resumption hit the 10 GiB allocator cap despite 5.16 GiB of physical
GPU memory remaining free. The next attempt uses
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation.
The allocator-only retry reached update 36 but also hit the 10 GiB cap on a
later example. Its last saved state was update 30. That state was copied to
`continuation-parent.pt` and is the immutable parent of `ncm-independent-v16b`.
The new stage requests an 11.25 GiB cap; admission still limits this by initial
free memory minus a 4 GiB reserve. Runtime headroom checks retain a 3 GiB reserve.

`--continue-checkpoint` restores model, optimizer and RNG state, preserving the
original update count and planned 1,000-update schedule. It permits a separately
identified implementation/resource profile, but rejects any change in data,
source model, codec, seed, loss, learning rates or other learning settings. This
is an explicit new stage, not a weakened exact-resume check.

## Commands and recovery

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
  --init results/ncm-generator-v10/uncached/latest.pt \
  --data results/ncm-contextual-data-v10 \
  --output results/ncm-independent-v16 \
  --steps 1000 --stop-after 27 --eval-every 100
```

The resource continuation uses a new directory:

```bash
"$PYTHON" -m scripts.ncm.train_independent \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init results/ncm-generator-v10/uncached/latest.pt \
  --data results/ncm-contextual-data-v10 \
  --output results/ncm-independent-v16b \
  --continue-checkpoint results/ncm-independent-v16/continuation-parent.pt \
  --allocator-cap-gib 11.25 --steps 1000 --stop-after 100 --eval-every 100
```

For resumption use the same command with `--resume` and either a later
`--stop-after` boundary or no interruption boundary. `--steps` remains 1,000;
changing it would change experiment identity. Each output directory has an
exclusive lock. For a continuation directory, retain its original `--continue-checkpoint`
argument on later resumes. Saved state includes trainable parameters, optimizer state,
Python/Torch/CUDA RNG states, update history and implementation hashes.

The loader rejects changed source weights, codec, prepared data, parameter
inventory or training implementation. Do not edit a running trainer or weaken
the resume identity check to force an incompatible checkpoint to load.

Checkpoints are atomic and normally saved every ten updates. A resource or
numerical failure exits without marking training complete. A checkpoint records
the last saved update, not any partially executed later update.

At each 100-update boundary, the trainer runs the existing held-out chunk
diagnostic and 12 executable coding development tasks. `quality_ready=false`
remains explicit even if those elementary tasks improve; they are insufficient
to qualify a professional coding release. A successful training process alone
does not pass this gate.

## Preparing the larger creator expansion

After [creator admission](CREATOR-EXPANSION-V15.md) finishes:

```bash
"$PYTHON" -m scripts.ncm.prepare_creator_expansion \
  --base results/ncm-contextual-data-v10 \
  --addition results/ncm-creator-admitted-v15b \
  --output results/ncm-contextual-expansion-v17
```

This preserves the original evaluation inventory byte-for-byte in a canonical
JSON fingerprint, appends only admitted training responses, and selects at most
eight evenly spaced eligible UTF-8 chunks per new response. It retains complete
prompt history and previous response chunks. The original OxCoder tokenizer is
used; no tokenizer is trained. Source responses remain available in full in the
reservoir even when context bounds prevent using later chunks.

Preparation does not train a model and does not establish codec reconstruction
quality. The trainer rejects an unannounced dataset change. The explicit
`--data-expansion` stage validates base lineage, rechecks the evaluation
fingerprint, requires training-only added rows, rejects conversation split
conflicts, and reapplies training-only codec admission. See the
[version 17 run and decision guide](EXPANDED-GENERATOR-V17.md).

Version 16b stopped deliberately at update 100 rather than completing its
planned 1,000 updates. Its coding result was **0/12**; none of its six selected
development chunks was generated exactly. The training-set head control did
not translate into useful coding. The original autoregressive OxCoder GGUF
subsequently passed **12/12** on the same elementary coding diagnostic.
These repeatedly inspected tasks are development checks, not a sealed final
benchmark. Their generated baseline answers are excluded from training.

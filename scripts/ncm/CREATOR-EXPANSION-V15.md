# Creator-corpus expansion and OxCoder head audit

## Purpose and boundaries

The small NCM baseline has not produced usable coding outputs. This stage
expands the original creator demonstrations and tests a specific classifier
change on the actual OxCoder-derived backbone. Dataset admission, training-set
memorization, held-out generation and deployment are separate gates.

The original model remains unchanged. No usable NCM checkpoint is published by
these commands. Existing teacher responses remain available in the earlier
training corpus; this acquisition does not generate new OxCoder distillation.

## Complete source acquisition

`acquire_creator_corpora.py` reads these immutable dataset revisions:

| Dataset | Revision | JSONL files | Source bytes |
|---|---|---:|---:|
| `TeichAI/Ox-Alpha-10k` | `ae678ee78880e14251a553a0d0dc8ec6bd6dad6c` | 1 | 130,363,129 |
| `TeichAI/DeepSeek-v4-Pro-Agent` | `abbf7b71b145633b60beb21ddc0d158582fd80dd` | 4,006 | 279,532,490 |

The downloader archives complete files with gzip and records both raw and
archive SHA256 checksums. It validates file lengths against pinned Hugging Face
metadata, bounds each download, and leaves a 1.5 GiB disk reserve. A partial
download cannot become a finalized source. Restarting revalidates cached files
and rebuilds the extracted reservoir in deterministic source order.

The extraction retains at most four spread-out assistant responses per
conversation, including their previous messages. Targets above 32,768 characters
and prompts above 262,144 characters are rejected explicitly, not truncated.
The full source remains in its archive. Exact duplicate targets are removed.
Source code and tool calls in demonstrations are never executed.

Transient HTTP 408, 429, 499 and selected 5xx errors retry at most six times,
with bounded backoff. Permanent failures such as 404 stop acquisition. Twelve
workers encountered a proxy 499 in the recorded run; the retry run uses eight.

```bash
"$PYTHON" -m scripts.ncm.acquire_creator_corpora \
  --output results/ncm-creator-reservoir-v15 --workers 8
```

The complete acquisition finished with 4,007 verified files and 23,508
retained responses: 9,935 Alpha and 13,573 Agent. Their original partitions are
18,795 training, 2,397 validation and 2,316 test responses. Size filtering
rejected 710 candidates. These counts precede contamination admission.
See [acquisition evidence](validation/v15-controls/creator-acquisition.json).

`progress.json` reports downloaded source counts. A final `manifest.json` with
`sources_complete=true` is required before admission. A source archive or an
extracted answer does not establish that the answer is correct.

## Additive training admission

```bash
"$PYTHON" -m scripts.ncm.admit_creator_corpora \
  --reservoir results/ncm-creator-reservoir-v15 \
  --frozen results/ncm-open-corpus-v11c \
  --output results/ncm-creator-admitted-v15b
```

The admission script verifies both corpus manifests and checksums. It downloads
the same pinned HumanEval and MBPP exclusion files used by the frozen corpus,
requiring their recorded sizes and checksums. It then:

1. Registers existing conversation/repository groups and splits without changes.
2. Registers the expanded reservoir's original validation and test groups before
   considering any training candidate.
3. Excludes all nontraining examples from the additive output.
4. Excludes existing exact targets and near duplicates using the original
   token-based SimHash filter, with Hamming distance at most three.
5. Rejects 32-token overlaps with protected benchmark or held-out material,
   checking both the proposed target and its prompt/history.
6. Rejects normalized copies of the local coding diagnostic's request text.

The first filtering attempt treated shared developer instructions as held-out
problem content and rejected Agent examples for the same 583 overlapping spans.
That incomplete pass is retained for diagnosis but is not used for training.
The revised pass inventories identical system/developer messages found in at
least 20 distinct training groups and five held-out groups. It ignores only
those exact messages in prompt-to-prompt holdout comparisons. Benchmark checks,
answer/target checks, user questions, assistant history and tool content stay
protected. `shared-instructions.json` records the hashes and group counts;
source examples themselves are not rewritten.

The output is a gzip JSONL **training-only addition**, not a resplit of the old
dataset. Rejection counts and provenance are retained. Original held-out groups
remain excluded even when a duplicate appears under another source name.

The completed revised admission retained **4,228** responses: 3,626 Alpha and
602 Agent, totaling 21,504,913 target UTF-8 bytes. It excluded 4,713 nontraining
responses, 751 existing exact duplicates, 3,538 near duplicates, and 10,278
benchmark/holdout-overlap candidates. These are recorded filter outcomes, not
independent judgments that every rejected answer would leak an evaluation.
The [version 17 preparation](EXPANDED-GENERATOR-V17.md) subsequently selected
31,050 context-eligible chunks from 4,123 of the admitted responses.

These conservative lexical filters can reject unrelated boilerplate and cannot
prove that all semantic contamination has been removed. They do not examine
model scores to choose training examples. `used_in_training=false` remains in
the immutable admission manifest; a later run must bind this manifest's hash
and report its own exposure counts before claiming to use the expansion.

Do not retrain the tokenizer or change codebooks under an existing generator.
Such changes invalidate stored concept targets and checkpoint compatibility.
Prepare a separately identified corpus using the intended codec's tokenizer;
keep the original evaluation partition fixed for comparisons.

## Actual OxCoder classifier control

The upstream small-model control showed that independent concept classifiers
could memorize eight training windows where the tied heads failed. This does
not demonstrate that the same intervention works with the OxCoder backbone.
`audit_oxcoder_heads.py` tests that question separately.

```bash
export PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m scripts.ncm.audit_oxcoder_heads \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint results/ncm-generator-v10/uncached/latest.pt \
  --data results/ncm-contextual-data-v10 \
  --output results/ncm-oxcoder-head-audit-v15 \
  --steps 500 --examples 8 --max-prompt 2048
```

The loader validates the original GGUF hash, codec identity and prepared data.
It reuses the 32-layer OxCoder text backbone with its existing residual adapters,
using the audited FLA kernels and full uncached computation. The previously
failed prefix-cache optimization remains disabled. Weights are resident BF16
for this research control; this is not 4-bit Mac inference or an 8 GiB test.

The audit chooses distinct training responses, stratified by source and teacher
origin, within an explicit prompt bound. It never truncates a prompt. For each
level it computes fixed hidden features using correct earlier concept levels.
Those features are detached. Only the classifier parameters update; neither the
backbone, adapters nor codec learns during this control.

Two arms receive the same feature tensors, labels, update count and AdamW
settings: learning rate 0.001, weight decay 0.01 and mean level cross-entropy.
The geometry arm retains the previously trained tied heads. The independent
arm starts fresh per-level linear classifiers. This is not equal-parameter or
equal-initialization comparison. `IndependentConceptHeads` deliberately returns
no continuous residual; geometry-specific residual losses cannot be applied to
its logits.

`report.json` separates:

- Oracle codec reconstruction of supplied targets.
- Teacher-forced classifier learning on fixed hidden features.
- Free hierarchical generation from the prompts with no supplied concept codes.

Free generation must recompute later-level features using predicted earlier
levels. Cached teacher-forced features cannot substitute for this check. The
eight-example control is not held-out coding evaluation, even if every example
is memorized. It exports an audit report, not a deployment model.

## Decision criteria

If the new classifiers cannot fit the fixed features, investigate feature
separability and optimization before broadening training. If teacher-forced
accuracy improves while free generation fails, examine conditioning mismatch
and cascading early-level errors. Do not infer success from decreasing CE.

A subsequent adaptation needs clean training-only demonstrations, correctly
bound concept targets, response-level distillation provenance, explicit
conditioning-corruption settings, resumable optimizer/RNG state, and executable
held-out coding tests. Promotion requires coding quality as well as reconstruction
and memory/latency measurements on the intended runtime. No control in this
document by itself meets those requirements.

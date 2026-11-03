# Verified short-program curriculum

This is an additive training-data preparation step. It does not modify the
running v31/v32 datasets, model, codec or evaluation. A completed archive is
not a trained checkpoint. Both creator datasets and their admitted distillation
remain in the base corpus for any later training stage.

## Motivation and fixed selection

The existing NVIDIA responses frequently spend an entire 127-byte window on a
code fence, signature and documentation before any implementation. A
training-only inventory found that AST normalization and docstring removal
reduce 10,000 responses from 8,302,141 to 4,120,308 UTF-8 bytes. It found 890
programs fitting one 127-byte window and 3,465 fitting two. The 188 previously
verified teacher responses similarly shrink from 170,695 to 79,498 bytes.
These are inventory counts before exclusions or fresh execution checks.

The selection rule considers every already admitted training response; it
does not select by benchmark function name, failure, expected answer or score.
The default maximum is 254 UTF-8 bytes, independently of what the twelve
coding tasks require. A supplement can expose complete implementations during
short-context training. Whether it improves generation remains to be measured.

## Transformation and verification

`compact_code_curriculum.py` parses a single Python block, removes only the
initial docstring statement of modules, classes and functions, then uses
`ast.unparse`. Comments and fences disappear. Identifiers, control flow,
expressions, runtime string literals, imports and annotations are retained;
the resulting AST must equal the transformed AST. A documented empty body
receives `pass`. Explicit source/documentation introspection is rejected.

Removing docstrings is an observable semantic change. The transformation is
not described as universally behavior preserving. Prompts mentioning
documentation, docstrings, comments or explanations are excluded. Eligible
training prompts receive an explicit code-only, no-documentation request.
Original rows and messages remain unchanged on disk.

The existing benchmark and frozen-holdout exclusion filter runs on each new
prompt/target pair. Group splits remain fixed. Benchmark material is used only
for exclusion, never as generated supervision. This lexical filter cannot prove
absence of all semantic overlap.

Each selected response is joined to the pinned publisher Parquet row. The
publisher's original implementation must pass every locally supported test
first. Then the untransformed response and compact response must pass the same
tests with at least one assertion actually executed per test. For teacher
responses, the teacher text is checked again. Invalid tests, unsupported
dependencies, failures and timeouts cause exclusion. Tests are not repaired to
accept a candidate.

Execution uses the existing isolated `bwrap` worker: no network or host workspace,
two CPU seconds, five wall-clock seconds and a 256 MiB address-space limit per
test. Passing publisher tests remains an incomplete check, not a proof of
general correctness. The archive retains `targets_are_verified: false` for
compatibility with that distinction and records the narrower executable proof
in `publisher_checks`.

## Reproduction

Use the pinned training environment and dependency paths. The command consumes
existing archives, not newly sampled teacher output:

```bash
"$PYTHON" -m scripts.ncm.compact_code_curriculum \
  --admitted results/ncm-opencode-v19 \
  --distilled results/ncm-distillation-opencode-curated-v23 \
  --frozen results/ncm-open-corpus-v11c \
  --output results/ncm-compact-code-v33 \
  --maximum-bytes 254
```

The output directory must be new. `progress.json` reports filtering and test
progress. `checks.jsonl` retains every selected program's local execution
outcome. `identity.json` pins input archives, publisher source, exclusion
corpus, implementation and length bound. Only complete successful preparation
renames the temporary gzip archive and writes an admitted `manifest.json`.
Partial output cannot be consumed as an admitted dataset.

Each transformed record keeps its source revision, original publisher ID,
conversation group and teacher checksum/flag. A new response ID distinguishes
it from its parent. Transformation metadata includes the parent response ID
and SHA256 of the original target. The output's license and attribution remain
those of the source dataset.

## Before training

The archive still needs tokenization, complete-response chunking, strict
frozen-codec reconstruction admission, exclusion against the exact frozen
validation inventory, and a separately identified corpus-expansion training
stage. Importing optimizer state must be explicit. No running trainer silently
changes its corpus or hashes. Compare subsequent checkpoints on the unchanged
validation set and twelve executable tasks; a candidate still requires fresh
process replay before acceptance.

The seven preparation tests cover protected splits, immutable source records,
teacher provenance, prompt/byte limits, retained runtime strings and real
sandbox rejection of a docstring-sensitive transformation.

## Conditioning and frozen-codec admission

After archive completion, `screen_compact_corpus.py` joins every admitted record
to its execution proof, checks that all assertions actually ran, and prefers a
compact teacher response over a compact original within the same conversation.
It appends candidates to the already admitted base without removing any base
training row. The exact 64-window validation inventory is hashed before and
after. New targets overlapping protected validation targets are excluded.

```bash
"$PYTHON" -m scripts.ncm.screen_compact_corpus \
  --base-selection results/ncm-complete-selection-v27 \
  --base-manifest results/ncm-complete-pipeline-v28/features.json \
  --addition results/ncm-compact-code-v33 \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --output results/ncm-compact-training-v33 --device cuda:1
```

This step loads the small frozen codec, not the 9B source model. It observes the
existing host/GPU reserve rules and requests at most 2 GiB of CUDA allocator
capacity. FP32 codec screening excludes the entire response if any chunk fails
exact nonpadding reconstruction, including its EOS. Overlong prompts or missing
chunks also exclude the entire response. The source tokenizer and complete
previous assistant chunks form conditioning; no prompt truncation occurs.

`selection.pt` holds texts and token IDs. `admission.json` is explicitly a
text-only admission format for the original-tower trainer, with no cached
encoder tensors. It must not be supplied to the separate native decoder's
`FeatureCache`. The common original-tower corpus loader validates the pending
manifest before it is renamed to the admitted path. This establishes input
eligibility only; a new optimizer qualification and explicitly identified
training continuation are still required. Three additional tests cover
complete-response boundaries, teacher preference and rejection of a purported
passing proof with zero executed assertions.

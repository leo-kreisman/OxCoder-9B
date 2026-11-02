# Combined experiment results — 2026-10-28

**No usable OxCoder-NCM checkpoint was produced.** Both requested datasets and
cached OxCoder responses were used in codec adaptation. Reconstruction still
fails, so generator quality training remains blocked. This does not establish
that architectural adaptation is impossible; it establishes that the tested
representations/recipes are not adequate for exact code reconstruction.

The [reproduction guide](FOLLOWUP.md) contains commands, pinned dependencies,
source revisions, data handling, optimizer differences and continuation criteria.
Raw evidence is in [validation/followup/](validation/followup/), indexed by a
[SHA256 manifest](validation/followup/manifest.json). The previous failed pilot
is preserved separately in [RESULTS.md](RESULTS.md).

## What ran

- Prepared 1,216 complete creator-source responses: 939 train / 122 validation /
  155 test, from 1,000 Alpha rows and 64 agent files.
- Queried unchanged OxCoder on sixteen distinct training conversations; retained
  eight complete text responses, three agent-source and five Alpha-source.
  Eight output-limit truncations were excluded. Responses remain unverified.
- Audited released XSum and WMT14 NCM codecs on complete responses and syntax
  probes. Both text pre-tokenizers round-tripped the screened inputs exactly;
  the learned codec introduced the errors.
- Compared AdamW and a portable ANVIL2-derived matrix optimizer with identical
  initialization/batch order. Also ran joint encoder/quantizer/decoder adaptation.
- Ran contextual-byte, tokenwise-byte and larger-codebook controls, all freshly
  initialized on the expanded creator-plus-teacher corpus. Total: **54,000 codec
  updates** across six runs. These are not 54,000 updates of the 9B backbone.
- Verified upstream residual scheduling with known concept codes, ran seventeen
  unit tests, and exercised the generator's failing admission gate.

## Complete-response reconstruction

All rows use the same thirty development responses (sixteen Alpha and fourteen
agent). Units differ: the first five rows count pre-tokenizer tokens; the final
three count UTF-8 byte symbols plus EOS. Do not directly compare those accuracy
columns as if the units were identical. Exact whole-response counts are directly
comparable. Six separate syntax/Unicode probes must also pass.

| Codec / adaptation | Total updates | Selected update | Symbol/token accuracy | Exact responses |
|---|---:|---:|---:|---:|
| Released XSum | 0 | — | 83.34% tokens | 0/30 |
| Released WMT14 | 0 | — | 75.27% tokens | 0/30 |
| XSum decoder, AdamW | 1,000 | 1,000 | 90.11% tokens | 0/30 |
| XSum decoder, ANVIL-derived | 1,000 | 1,000 | 90.39% tokens | 0/30 |
| XSum joint adaptation, AdamW | 2,000 | 500 | 89.24% tokens | 0/30 |
| Byte codec, contextual encoder/decoder | 10,000 | 9,000 | 99.6438% symbols | 1/30 |
| Byte codec, tokenwise, 512-entry books | 30,000 | 26,000 | 99.8919% symbols | 1/30 |
| Byte codec, tokenwise, 4,096-entry books | 10,000 | 6,000 | 99.8891% symbols | 3/30 |

Every row fails. The best exact-response result is **10%**, despite approximately
99.89% per-symbol accuracy. Both tokenwise variants reconstruct four of six
syntax probes exactly; neither passes the probe gate. This is why a low average
reconstruction loss is insufficient for coding.

Checkpoint selection used development token/symbol accuracy, not these final
whole-response scores. For official codecs, intermediate selection used eight
complete development responses; for byte codecs it used 7,118 validation chunks.
These are development measurements. No new test-split performance was measured.

The [byte error audit](validation/followup/byte-tokenwise-errors.json) identifies
ordinary spaces/letters and Unicode continuation bytes among the errors. The
problem is not confined to obscure Unicode or whitespace after termination.
Larger books helped exact-document count in this small sample, but do not show a
reliable quality solution. More updates did not monotonically improve exactness.

## Optimizer finding

On the same frozen-code decoder task:

| Optimizer arm | Synchronized training work, 1,000 updates | Complete-response token accuracy |
|---|---:|---:|
| AdamW | 14.07 s | 90.11% |
| ANVIL-derived + AdamW head | 18.85 s | 90.39% |

ANVIL-derived updates took approximately **1.34 times as long** for a 0.28
percentage-point accuracy increase. This single configuration does not justify
a training acceleration claim. It does not measure the upstream optimized
eight-H100 implementation. Neither arm passed reconstruction, so there is no
quality-qualified winner. AdamW remains the simpler baseline for subsequent work.

These times exclude download, loading, checkpoint writes and validation. The
byte trainer's `elapsed_seconds` instead measures elapsed time within each
invocation and resets on resume; do not sum its cumulative log entries or mix
that field with the official decoder's training-only counter. No inference
tokens/s, Mac speed or end-to-end conversion-time claim follows from either.

## Correctness and resource checks

- Seventeen CPU tests passed: gradients, shape/transformation rules, serialization,
  optimizer transpose/reference behavior, exact resume, zero/nonfinite gradients,
  complete-response gates, data grouping and preservation of recorded tool data.
- The official scheduler matched all nine known-code level inputs exactly;
  perturbing a code affected later inputs but not its own or earlier inputs.
  Oracle output matched upstream reconstruction. This is not learned generation.
- A normal generator launch failed at the codec gate before allocating/loading
  the backbone or creating a generator output directory. The complete-response
  gate also requires matching checkpoint and corpus hashes.
- The original 5,629,117,536-byte GGUF still hashes to
  `a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.
  Original inference services remained running; only the temporary teacher
  service was stopped after caching.
- Training used one RTX 5060 Ti for these codec runs. The official AdamW,
  ANVIL-derived and joint runs peaked at approximately 0.814, 0.820 and 1.110
  billion bytes of PyTorch GPU allocations respectively. These exclude CPU,
  driver and other processes, and are **codec-training measurements**, not a
  resident 9B model or an 8GB deployment demonstration.

## Artifacts and current implementation boundary

All checkpoint paths/hashes are in the evidence manifest. Principal local paths:

```text
results/ncm-corpus-v2/
results/ncm-distill-v2/
results/ncm-official/{xsum,wmt14}/
results/ncm-codec-adamw-v2/
results/ncm-codec-anvil-v2/
results/ncm-codec-joint-v2/
results/ncm-byte-data-v3/
results/ncm-byte-codec-v3/
results/ncm-byte-tokenwise-v3/
results/ncm-byte-widebook-v3/
```

These are ignored local research artifacts, not GitHub release files. There is
no new trained 9B generator, native 4-bit NCM inference engine or Mac/Metal NCM
checkpoint. The original structural generator prototype remains available, but
its old three updates do not constitute a coding model.

The immediate research problem is preserving code through the learned codec.
Increasing training speed alone does not solve it. A subsequent representation
change must first demonstrate near-lossless complete-program reconstruction,
then demonstrate actual free-running code quality after generator adaptation.
No promotion gate was weakened to make this experiment appear successful.

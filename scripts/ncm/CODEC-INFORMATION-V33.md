# Frozen codec information diagnostic

This read-only diagnostic measures the representation used by the active
training runs. It changes no model weights, training examples, generation rule,
or acceptance test. The [full report](validation/v33-controls/codec-information.json)
records the codec and corpus hashes, exact training IDs and implementation hash.

## Sampling and controls

Using seed 33103, sample 64 first-response chunks and 64 later-response chunks
from each of the three admitted training sources. This produces two disjoint
192-window diagnostic sets. No validation or executable-task target is used.
Encode each window separately in FP32 without TF32, matching the label-generation
precision policy. Decode in batches of eight for this diagnostic only.

Measure code usage, entropy, and agreement with the most frequent code at each
position. These modal codes are computed on the same diagnostic sample, so their
agreement is an in-sample baseline, not generalization accuracy. For each level,
replace that level with its position-wise modal codes while retaining all other
gold levels. Also decode strict gold prefixes. All replacement paths still have
privileged target-derived information; none is autonomous generation.

## Results

| Measurement | First chunks | Later chunks |
|---|---:|---:|
| Full gold reconstruction, exact windows | 192/192 | 192/192 |
| Four-code level usage / 4096 entries | 75 | 82 |
| Four-code level position-mode agreement | 14.19% | 12.11% |
| Final-level usage / 4096 entries | 2289 | 2512 |
| Final-level position-mode agreement | 6.54% | 8.04% |
| Gold first level, byte accuracy | 8.48% | 7.83% |
| Gold first two levels, byte accuracy | 16.23% | 16.35% |
| Gold first three levels, byte accuracy | 68.54% | 67.22% |
| Modal final level, other levels gold, byte accuracy | 45.39% | 57.83% |

Replacing the first level while retaining the other gold levels preserved
191/192 first chunks and 176/192 later chunks exactly. Replacing the second
preserved 192/192 and 187/192. Replacing the final level produced zero exact
windows in both sets. Thus the representation is not a single-code collapse,
but complete character reconstruction strongly depends on fine-level codes.
High coarse-label classification accuracy alone cannot establish useful text.

These measurements do not isolate why a generator fails or prove that a new
codec would improve it. Gold residual levels can compensate for missing coarse
information, and modal replacements differ from real model errors. The current
codec and active controls remain unchanged. Any future representation change
requires its own reconstruction and training qualification; it cannot silently
reuse incompatible generator labels.

```bash
"$PYTHON" -m scripts.ncm.diagnose_codec_information \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --output results/ncm-codec-information-v33.json --device cuda:1
```

The measured diagnostic allocated 29,828,096 bytes at its GPU peak. It loads only
the small frozen codec and the admitted text inventory, not the 9B source model.
The usual host/GPU reserves and a 2 GiB allocator cap remain enforced. This is
neither an inference-memory measurement for OxCoder nor a passed coding gate.

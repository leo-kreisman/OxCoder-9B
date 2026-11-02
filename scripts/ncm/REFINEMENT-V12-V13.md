# Reconstruction and generator learning controls

## Scope

These stages diagnose the failed open-data NCM baseline before another OxCoder
backbone adaptation. They use the existing 50M-token Python corpus, the two
creator-dataset subsets and the existing teacher responses. They do not create
new distillation examples. The original OxCoder GGUF remains unchanged.

No checkpoint in this experiment has qualified as a usable NCM coding model.
A codec reconstructs an answer supplied to it; a generator must produce an
answer from a prompt. Reconstruction and memorization results cannot establish
the latter capability.

## Completed baseline

Both version 11 arms completed their planned codec and generator updates.
The [final comparison](validation/v12-controls/v11-comparison.json) records:

| Arm | Exact reconstructed documents | Token reconstruction | Generated coding tasks |
|---|---:|---:|---:|
| Code-only codec | 0/1,100 | 94.6898% | 0/12 |
| Mixed codec | 0/1,100 | 95.2901% | 0/12 |

Both generators also produced zero exact windows on the 64-window development
continuation diagnostic. Successful pipeline completion did not pass quality.

## Precision and reconstruction diagnosis

`diagnose_open_codec.py` compares three paths on the same fixed samples:

1. Encoder latent directly through the decoder, bypassing quantization. This is
   an oracle control that receives the complete target, not an inference mode.
2. Normal discrete-code encoding and decoding in FP32.
3. Normal encoding and decoding under BF16 autocast.

The mixed version 11 codec, on 128 validation windows, measured:

| Path | Correct non-padding tokens | Exact windows |
|---|---:|---:|
| Unquantized oracle | 99.9937% | 127/128 |
| FP32 quantized | 95.3015% | 6/128 |
| BF16 quantized | 95.2763% | 5/128 |

FP32/BF16 code IDs agreed on 97.8269% of positions. The similar reconstruction
errors show that changing numerical precision alone does not solve the problem.
The control localizes the major loss to the quantized reconstruction path; it
does not prove that the missing information can be recovered by a better decoder.
The [recorded diagnostic](validation/v12-controls/before.json) includes train
and validation aggregates. Source-text excerpts are kept only in local results.

The original version 11 trainer surrounded the full codec forward call with
BF16 autocast. Its comment claiming float32 quantization was inaccurate:
`z.float()` inside the quantizer does not disable autocast of eligible matrix
operations. The new trainer exposes `--fp32`; refinement runs use FP32 throughout.

## Frozen decoder calibration: version 12

`calibrate_open_codec.py --scope decoder` loads the selected mixed codec and
trains only its decoder. It keeps the whole codec in evaluation mode, which
disables dropout and EMA updates without disabling decoder gradients.

- Full-resolution codes only; no randomly discarded fine levels.
- FP32, AdamW, peak learning rate 0.0001, 50-update warmup and cosine decay to 10%.
- Batch 16, accumulation 2, 4,000 updates, 80/20 code/assistant windows.
- Hash every non-decoder state tensor, including codebooks and EMA buffers,
  before training and at every save. Any mutation fails the run.
- Train on training documents only. Select by exact development documents first,
  token accuracy second, using the same 64 code plus 100 assistant documents.
- Evaluate the selected checkpoint on 1,000 code plus 100 assistant documents.

The completed run reached **3/1,100 exact documents**, **96.3472% token accuracy**
and 93 syntactically valid Python documents. The representation hash remained
unchanged. This improves reconstruction but still fails its gate. Evidence:
[selected checkpoint](validation/v12-controls/decoder-selected.json),
[recipe](validation/v12-controls/decoder-identity.json).

## Generator output-head control

`audit_open_generator.py` takes eight fixed training windows (seed 7129) and
tries to memorize their codes for 500 updates. It uses the same version 11
generator body, frozen original mixed codec, clean teacher-forced inputs, FP32,
no dropout, AdamW at 0.0003, and evaluates free generation at every level.
At generation time it supplies neither gold coarse codes nor an AR verifier.

Two head configurations were compared:

- Original trained codebook-distance heads: **0/8 exact concept windows** after
  500 updates; decoded token accuracy 80.1758%.
- `--untied-heads`: reinitialize only the output heads as independent learned
  classifiers. By update 300 it reaches **8/8 exact concept windows**, maintained
  through update 500. Decoded accuracy equals the codec oracle's **95.1172%**.

The classifier variant has more trainable parameters; this is not an equal-size
performance comparison or a general claim against tied heads. It establishes
that the same body and hierarchy can learn these training examples with a
different output interface. Held-out generation still needs a separate run.
The memorization checkpoint is not exported as a model.

Evidence: [original heads](validation/v12-controls/generator-tied-memorization.json),
[independent classifiers](validation/v12-controls/generator-untied-memorization.json).

`open_baseline.py generator --untied-heads` exposes this option for the next
training experiment. It accepts the original codec configuration or the reviewed
refinement with dropout disabled; unrelated architectural changes are rejected.
Changed codec codebooks require freshly encoded targets and a new generator run.

## Joint full-resolution refinement: version 13

The decoder-only failure motivates updating the representation as well.
`--scope joint` loads the selected version 12 codec into the same upstream
architecture with encoder/decoder dropout and nested fine-level dropout set to
zero. It optimizes the upstream full-resolution CE plus VQ loss in FP32.
Codebooks remain non-gradient parameters updated by the upstream EMA path.
Dead-code restarts remain disabled. There is no raw-token side channel.

The initial recipe is 6,000 updates, batch 16 with accumulation 2, seed 47,
the same 80/20 sampling and 0.0001 peak learning rate. This is a distinct recipe,
not a resumption of the original codec optimization. Removing nested dropout
could weaken coarse-level usefulness; reconstruction improvements therefore
must be followed by generation tests. It cannot automatically qualify a release.

Version 13 completed. Its selected update 5,000 reconstructs **22/1,100 documents
exactly**, with **98.3366% token accuracy** and 213 syntactically valid Python
documents. This remains a failed reconstruction gate. See
[selected evaluation](validation/v12-controls/joint-selected.json).

## Follow-up experiments

Two bounded follow-ups run separately, with immutable parent checkpoints:

1. **Untied-head generator** (`results/ncm-generator-untied-v13-b32`): train a
   fresh 29.68M-parameter upstream generator on the fixed version 13 codec,
   with independent classifiers, BF16 autocast, batch 32, accumulation one,
   6,000 updates and the existing 0.1 maximum conditioning-corruption schedule.
   Other architecture settings match version 11. The codec stays frozen.
   This is a downstream diagnostic despite the imperfect codec, not admission
   of the codec to a release. It is not a 9B model or another OxCoder adaptation.
2. **Token-coverage refinement** (`results/ncm-coverage-v14`): start from the same
   selected codec, train 12,000 updates in FP32 with batch 64, accumulation one.
   Each batch has 56 natural windows, sampled 80/20 code/assistant, plus eight
   synthetic token-identity windows. Those eight use uniform IDs from the
   non-control vocabulary; some contain repeated tokens, EOS and padding.
   They exercise rare symbols, not new coding knowledge. Synthetic sequences
   need not form valid UTF-8 text or programs and never enter generator training.
   Counts appear separately as `synthetic_token_coverage` in exposure reports.

The coverage run completed all 12,000 updates. Its selected update 11,000
reconstructed **397/1,100 complete documents exactly**, at **99.8324% token
accuracy**, with 732 syntactically valid Python documents. The long-document
bucket remains weak: only 9/246 documents longer than 2,048 tokens were exact.
The reconstruction gate remains failed. See
[selected evaluation](validation/v12-controls/coverage-selected.json).

The coverage experiment changes batch size as well as the sample mixture; any
gain cannot be attributed uniquely to either change. It may also worsen
predictability of concepts, which requires another downstream evaluation.
The generator used the fixed version 13 codec throughout; it did not load the evolving coverage checkpoint.

The untied generator also completed its 6,000 updates. The selected checkpoint
was update 2,000: teacher-forced concept accuracy 36.1544%, free concept accuracy
3.8319%, decoded nonpadding accuracy 2.2411%, and zero exact continuation windows
out of 64. It passed **0/12 coding development tasks**. Later training improved
teacher-forced accuracy but did not improve free generation. This arm is a
failed downstream experiment, not a promoted model. Evidence:
[selected continuation diagnostic](validation/v12-controls/untied-generator-selected-evaluation.json),
[coding diagnostic](validation/v12-controls/untied-generator-coding-development.json).

Both jobs checkpoint their optimizer/RNG states. The 20-update batch-32 generator
preflight peaked at 4,628,881,920 CUDA-allocated bytes; this is a training allocator
measurement, not full-process residency or 9B inference memory. The earlier
three-update batch-four preflight remains under `ncm-generator-untied-v13` and
was not continued. The batch-32 configuration keeps the effective batch of 32
while reducing accumulation overhead.

The suite passed **67 tests**, including decoder-gradient isolation, hashing of
frozen EMA buffers, synthetic control-token placement, and tensor-exact recovery
after interrupting and resuming a joint-training fixture. Test evidence is in
[unit-tests.txt](validation/v12-controls/unit-tests.txt). These establish
implementation controls, not model quality.

The source snapshots and parent/dataset hashes are mandatory for comparing
these results. Neither a decreasing training loss nor the eight-example
memorization control can substitute for independently executed coding tests.

## Reproduction and operation

Run from the repository root in an environment providing the pinned upstream
dependencies. Local development uses `results/ncm-deps` and
`results/ncm-data-deps` on `PYTHONPATH`; these are environment directories, not
shipped model dependencies. `PYTHON` below names that environment's interpreter.

```bash
export PYTHON=/absolute/path/to/python
export PYTHONPATH=results/ncm-deps:results/ncm-data-deps
export OMP_NUM_THREADS=2

"$PYTHON" -m scripts.ncm.diagnose_open_codec \
  --checkpoint results/ncm-open-baseline-v11/codec-mixed/best.pt \
  --data results/ncm-open-baseline-v11/data \
  --output results/codec-diagnostic/before.json --device cuda:1

"$PYTHON" -m scripts.ncm.calibrate_open_codec --scope decoder \
  --checkpoint results/ncm-open-baseline-v11/codec-mixed/best.pt \
  --data results/ncm-open-baseline-v11/data \
  --output results/decoder-calibration --steps 4000 --device cuda:1

"$PYTHON" -m scripts.ncm.calibrate_open_codec --scope joint \
  --checkpoint results/decoder-calibration/best.pt \
  --data results/ncm-open-baseline-v11/data \
  --output results/joint-refinement --steps 6000 --device cuda:1

"$PYTHON" -m scripts.ncm.audit_open_generator \
  --generator results/ncm-open-baseline-v11/generator-mixed/best.pt \
  --codec results/ncm-open-baseline-v11/codec-mixed/best.pt \
  --data results/ncm-open-baseline-v11/data --untied-heads \
  --output results/generator-control/untied.json --device cuda:0

"$PYTHON" -m scripts.ncm.open_baseline generator --untied-heads \
  --codec results/joint-refinement/best.pt \
  --data results/ncm-open-baseline-v11/data \
  --output results/untied-generator --steps 6000 \
  --batch 32 --accumulation 1 --device cuda:0

"$PYTHON" -m scripts.ncm.calibrate_open_codec --scope joint \
  --checkpoint results/joint-refinement/best.pt \
  --data results/ncm-open-baseline-v11/data \
  --output results/token-coverage --steps 12000 --batch 64 \
  --accumulation 1 --coverage-fraction .125 --device cuda:1
```

Omit `--untied-heads` for the original-head control. These controls always use
the original codec bound into the generator checkpoint, not the refined codec.

Each refinement directory has an exclusive process lock. Checkpoints include
optimizer state, global and sampler RNG state, source hashes, parent model and
dataset hashes, exposure counts and selection history. `--stop-after N` creates
an interruption checkpoint without changing the planned learning-rate schedule.
`--resume` rejects different source code or recipe settings. Earlier recipes can
only be resumed with their recorded implementation; do not bypass the guard.
Version 12 and version 13 preserve source snapshots under `implementation/`.

The process leaves 12 GiB host RAM and 3 GiB free GPU memory as runtime reserves,
caps its CUDA allocator at 8 GiB or less, and stops below 1 GiB free disk. These
are training controls, not a measured 8 GiB whole-process inference guarantee.

Read `progress.json` for live updates and `selected-evaluation.json` for the
selected checkpoint's broad reconstruction score. `training_complete=true`
only means the finite optimization run ended. `quality_ready` remains false.
The test partition is not used for any of these controls or selections.

## Research context

The [NCM paper](https://arxiv.org/html/2610.14162v1) evaluates learned multiscale
codecs and generation separately and documents residual reconstruction errors.
An [author's issue response](https://github.com/HazyResearch/ncm/issues/16)
warns that codebook restarts improved reconstruction/utilization while hurting
downstream generation. Neither evidence supports promoting a reconstruction
result alone into a claim of usable OxCoder conversion.

## Next controls

[Creator expansion and actual OxCoder classifier audit](CREATOR-EXPANSION-V15.md)
documents the next acquisition, admission and frozen-backbone learning controls.

# Initial OxCoder → next-concept adaptation results

For the larger creator-data + cached-teacher experiment, see
[the follow-up results](FOLLOWUP-RESULTS.md). This file preserves the initial pilot.

Recorded 2026-10-27. **A working structural prototype was tested; a usable NCM
coding model was not achieved.** The codec failed its fidelity gate, so normal
generator quality training was deliberately not started. Three structural
generator updates are not a trained coding checkpoint.

## What actually ran

| Stage | Measured result | Interpretation |
|---|---|---|
| Source inverse transforms | Six selected tensors passed comparison with original BF16 ranges | Checks norms, decays, and one quantized projection; not all tensors |
| Full backbone transfer | All 32 text blocks loaded; 6,919,565,824 block/final-norm parameters reused | Original packed token embeddings also provide prompt conditioning; original output head is replaced |
| Pre-adaptation AR audit | 33/33 teacher-forced greedy predictions matched two llama.cpp fixtures | Stronger than shape checks, but limited to two short prompts |
| Codec training | 20,000 updates, batch 16; best validation checkpoint at step 5,500 | Continued training did not improve the selected validation score |
| Generator structural test | Three updates at levels 0, 2 and 4 | Finite loss and nonzero up-projection gradients in all 32 adapters |
| Checkpoint reload | Identical fixed-input logits after deliberately perturbing and restoring the head | Serialization check, not coding-quality evidence |
| Concept generation | All five levels completed | Output was invalid/unusable, as expected from three test updates |
| CPU unit tests | 10 passed | Codec gradients, target isolation, transforms, UTF-8/EOS handling, data integrity, gate enforcement and serialization |

Raw evidence is in [validation/](validation/), including the
[backbone audit](validation/backbone-audit.json),
[source audit](validation/source-audit.json),
[generator run](validation/generator-training.json), and
[structural result](validation/structural-validation.json).

## Codec fidelity: the blocking result

The codec window contains at most 127 UTF-8 bytes plus EOS. Metrics count
non-padding symbols, including EOS. Exact-chunk accuracy requires every symbol
through the expected EOS to match; positions after that EOS do not matter.

| Metric | Validation, selected checkpoint | Untouched test split |
|---|---:|---:|
| Chunks | 147 | 230 |
| Non-padding symbol accuracy | 99.6950% | 99.4643% |
| Completely exact chunks | 71.4286% (105/147) | 61.7391% (142/230) |
| Correct EOS position | 100% | 100% |
| Valid UTF-8 with EOS and no preceding PAD | Not recorded in training history | 89.5652% |
| Admission gate | **Fail** | **Fail** |

The preliminary gate requires 99.9% symbol accuracy, 95% exact chunks and 99%
EOS accuracy. Even passing those thresholds would not establish professional
coding quality. The test result demonstrates why a high per-symbol score alone
is inadequate: a chunk with one changed identifier, quote or digit is wrong.

The test split was evaluated once after checkpoint selection. It must not be
used to tune the next experiment. The pilot contains only 1,304 training chunks;
it is neither a large corpus nor a representative coding benchmark. No program
pass rate has been measured. The codec's validation score plateau is a reason
to investigate its representation and data coverage, not to assume that simply
running more generator updates will solve reconstruction errors.

See [codec-training.json](validation/codec-training.json),
[codec-gate.json](validation/codec-gate.json), and
[codec-test.json](validation/codec-test.json).

## Data and checkpoints

Both requested source families were used: **Ox-Alpha-10k** and
**DeepSeek-v4-Pro-Agent**, with pinned revisions in the
[dataset manifest](validation/dataset-manifest.json).
Acquisition read 128 Alpha conversations and 16 agent session files. Complete
conversation hashes determine the split before chunking. Each conversation
contributes at most its first 12 unique chunks; this favors early response text.
The manifest records 1,304 train / 147 validation / 230 test chunks.

Local checkpoints, relative to the repository root:

```
results/ncm-codec-pilot/selected-step5500.pt
results/ncm-codec-pilot/latest.pt
results/ncm-generator-verified/latest.pt
```

The selected codec is frozen for the generator experiment. The codec `latest.pt`
contains the optimizer/RNG at step 20,000 and is not the best checkpoint. The
generator checkpoint contains only newly trained modules and optimizer state;
it requires the original source GGUF and selected codec. These files are ignored
research artifacts, **not uploaded release assets**. Hashes and sizes are in
[artifacts.json](validation/artifacts.json).

The initial `results/ncm-generator-smoke` run tested levels 0/1/2. The reported
authoritative run is `results/ncm-generator-verified`, which additionally audited
the original AR behavior, tested the largest level, and perturbed/restored the
head before checking serialization.

## Hardware, memory and speed scope

Training used Linux, two RTX 5060 Ti cards with 16 GB each, and 64 GB host RAM.
The source matrices were dequantized to resident BF16 across both GPUs; no
dense-weight SSD offloading occurred during generator forward/backward passes.
The source audit temporarily reads original output-head chunks only to compare
pre-adaptation predictions with the AR reference.

The verified run's peak PyTorch allocations were 7,077,963,776 bytes on GPU 0
and 7,111,639,040 bytes on GPU 1. These are training allocations, exclude other
processes and CPU/runtime overhead, and are **not an 8 GB deployment result**.
Existing inference services remained running. The launcher bounded host memory,
disabled swap for the job, and reserved GPU headroom.

No meaningful output-token/s claim is made. Generating invalid text quickly is
not an inference-quality speedup. Per-update timings in the raw log are short
training diagnostics, not end-to-end inference benchmarks. No Mac measurement
or Metal NCM implementation exists in this experiment.

## Next experiment and promotion criteria

1. Keep the original OxCoder runtime as the working baseline.
2. Diagnose codec errors by character class and byte position on development
   data. Expand representative code, whitespace, identifiers, numbers, Unicode
   and tool JSON coverage. Separate codec fidelity from language-model quality.
3. Compare codec widths, quantization hierarchies, codebook initialization and
   loss schedules on fixed validation data. Record each change as a new run;
   do not overwrite the failed pilot or relax its acceptance thresholds.
4. Require nearly lossless reconstruction of complete unseen programs before
   allocating a substantial generator training run. Chunk scores are insufficient.
5. Once admitted, train the new interfaces/adapters on all scales, then evaluate
   free-running output. Test conditioning corruption and generated earlier
   scales to address the difference from clean teacher-forced training inputs.
6. If adapters cannot learn the representation, compare staged backbone
   unfreezing/distillation with independent NCM training. Neither has been
   implemented or shown successful here.
7. Only after executable coding and tool-use quality passes, implement packed
   4-bit inference for the new scheduler/codec and measure total memory against
   the 8,000,000,000-byte target. A standard llama.cpp GGUF loader alone cannot
   execute this new architecture.

This outcome establishes a reproducible place to continue architectural
research. It does not establish that OxCoder's coding ability survives NCM
adaptation, or that the resulting model will be faster.

# OxCoder-9B — Q4_K_M GGUF

Unmodified text-generation GGUF from [prithivMLmods/OxCoder-9B-GGUF](https://huggingface.co/prithivMLmods/OxCoder-9B-GGUF/tree/d108751ef79e645b07d04030427b6b67b284d07e),
converted from [OrionLLM/OxCoder-9B](https://huggingface.co/OrionLLM/OxCoder-9B).
This release contains the Q4_K_M quantization, not the original full-precision checkpoint.
No training or re-quantization was performed for this mirror.

## Download and reconstruct on macOS or Linux

Download `download.sh` from this release into an empty folder and run:

```sh
bash download.sh
```

Requires `curl` and Python 3. The script downloads the six parts and supporting
files, then verifies and joins them. Alternatively download all six `.partNNN`
assets, `model-manifest.json` and `reassemble.py`, then run `python3 reassemble.py`.
Load the resulting `OxCoder-9B.Q4_K_M.gguf` into a compatible GGUF runtime.

The parts are raw byte splits, not independently loadable GGUF shards.
Five parts are 1 GiB each; the last holds the remainder. Keep about 11 GiB free
on disk for both parts and the reconstructed file. Interrupted downloads resume.
After successful verification, the parts may be removed manually.

Weights: 5,629,117,536 bytes (approximately 5.24 GiB). Runtime memory
also includes context state and working buffers; this release makes no measured
RAM, speed or quality claim for a Mac. No vision projector is included.

Quantized source revision: `d108751ef79e645b07d04030427b6b67b284d07e`.
Full-file SHA-256: `a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.

Model weights are Apache-2.0 according to the source model cards; see the LICENSE
and NOTICE assets in the model release. The native Kindi fork has a separate
AGPL-3.0 license at [scripts/kindi_q4/LICENSE](scripts/kindi_q4/LICENSE).
Original and converter model cards are included unmodified. This mirror is not
an official release by OrionLLM or prithivMLmods.

[Download release](https://github.com/leo-kreisman/OxCoder-9B/releases/tag/model-oxcoder-9b-q4-k-m)

## Runtime options

1. **GGUF / llama.cpp with Metal:** the existing deployment and comparison baseline.
   Approximately **17 output tokens/s** has been reported on an M3-family Mac;
   this is an operator observation, not a reproducible benchmark published by this repository.
2. **Native Kindi Q4 / CPU, CUDA and experimental Metal:** uses a losslessly
   repacked version of the same GGUF. The complete CPU/CUDA implementation,
   verification tests and recorded resident-memory results are included in this
   repository. Start with the [native runtime guide](scripts/kindi_q4/README.md).
   For Apple Silicon, follow the [detailed native Metal implementation plan](scripts/kindi_q4/METAL_IMPLEMENTATION.md);
   it distinguishes the existing experimental bridge from the remaining full-GPU
   implementation and defines correctness, memory and speed acceptance gates.
   Native Metal does not require MLX.
3. **MLX / Kapitsa:** an experimental second deployment using a separately downloaded
   native 4-bit OxCoder checkpoint. Validate plain decoding first, then compatible
   speculation and prefix caching. oMLX is a separate comparison engine.

The [Mac runtime evaluation and implementation guide](docs/MAC-RUNTIME-EVALUATION.md)
specifies installation, memory accounting, phase gates, context tests, coding and
tool-call tests, Harness² integration boundaries, and selection/rollback criteria.
Its target is an **18GB M3-family Mac with an 8GB inference allowance** while other
applications remain in use. The existing 128K configuration remains a separate
profile; populated-context capacity must be measured for each candidate.

The guide is an implementation specification. It does not claim that Kapitsa,
OxCoder speculative decoding, or Harness² integration has been tested on this Mac.
Use the [run manifest](docs/evaluation/run-manifest.example.json),
[result template](docs/evaluation/RESULTS-TEMPLATE.md), and
[phase checklist](docs/evaluation/IMPLEMENTATION-CHECKLIST.md) to record evidence.
The GGUF release and its checksums are unchanged; the release does not bundle MLX weights.

## Next-concept conversion research

The latest [expanded-data training stage](scripts/ncm/EXPANDED-GENERATOR-V17.md)
adds 31,050 prepared chunks from both creator datasets, preserves existing
OxCoder distillation and freezes evaluation membership. The original AR source
passed 12/12 elementary coding checks; the preceding NCM candidate passed 0/12.
The research pipeline is not a usable NCM model release.

The [OxCoder-initialized NCM experiment](scripts/ncm/README.md) implements a learned
text codec, coarse-to-fine concept prediction, and trainable adapters through all
32 pretrained text layers. It includes bounded dataset acquisition, inverse GGUF
weight transforms, resumable training, reconstruction gates and structural tests.
This is an architectural research path, separate from the runtime options above.
It is **not a validated replacement checkpoint**, and does not run in the existing
GGUF server. See its [recorded results and limitations](scripts/ncm/RESULTS.md)
before training or making speed/quality comparisons.
The [combined-data and distillation follow-up](scripts/ncm/FOLLOWUP.md) includes
official-codec audits and optimizer/representation controls; its
[measured results](scripts/ncm/FOLLOWUP-RESULTS.md) still fail reconstruction gates.
The [version 6 experiment](scripts/ncm/GENERATOR-V6.md) improves reconstruction
to 28/30 exact development responses and advances to real generator training
with both creator datasets and cached teacher responses. It also adds isolated
executable-code evaluation. These are development results, not a usable NCM
release or an 8GB deployment measurement.
Both version 6 generator runs scored **0/12** on the elementary function
diagnostic. The [version 7 investigation](scripts/ncm/GENERATOR-V7.md) documents
the failure analysis; its three executable evaluations also scored **0/12**.
The [version 8 experiment](scripts/ncm/GENERATOR-V8.md) removes a measured conflict
between residual supervision and biased code selection; its two coding evaluations
also failed. The [contextual codec stage](scripts/ncm/CONTEXTUAL-CODEC-V9.md)
addresses a measured limitation in the coarse representation, using both dataset
subsets and cached teacher responses. The [version 10 generator workflow](scripts/ncm/GENERATOR-V10.md)
uses that new representation with fresh initialization and numerical audits.
Its first executable evaluation also scored **0/12**; training remains experimental.
No usable NCM checkpoint has been released.

The [open-data baseline](scripts/ncm/OPEN-BASELINE-V11.md) uses the released NCM
modules to compare code-only and mixed-data codecs, followed by small-generator
diagnostics. It adds a bounded Python corpus and explicit conditioning corruption.
This prerequisite experiment is not a converted OxCoder model. The version 10
adapter job stopped at update 2,077 after its GPU memory reserve check failed.
Both version 11 arms subsequently completed with 0/12 coding passes. The
[refinement experiments](scripts/ncm/REFINEMENT-V12-V13.md) document the
reconstruction diagnosis, output-head control and subsequent training recipes.
The [actual OxCoder classifier audit and next training stage](scripts/ncm/INDEPENDENT-GENERATOR-V16.md)
follow an 8/8 training-chunk memorization control with independent heads;
this is not held-out coding success. The
[complete creator-corpus expansion](scripts/ncm/CREATOR-EXPANSION-V15.md)
documents source acquisition, contamination checks and frozen evaluation splits.

## Native Q4 implementation and recorded validation

The native runtime is in `scripts/kindi_q4/`, with tests in `tests/kindi_q4/`.
The [validation report](scripts/kindi_q4/VALIDATION.md) and
[raw measurement records](scripts/kindi_q4/validation/) include a resident CUDA
run of **25.66 output tokens/s over 64 output tokens**, with a sampled
**6,785,564,672-byte host-plus-GPU peak**. This is the recorded native runtime
result, not a benchmark of the existing llama.cpp deployment. Normal resident
generation does not stream dense weights from SSD. Historical forced-streaming
measurements are retained for comparison and are not the recommended OxCoder mode.

The native guide covers installation, conversion, integrity checks, CPU/CUDA
execution, API serving, prefix snapshots, memory limits and troubleshooting.
The Metal handoff specifies each implementation phase and its tests, including
same-machine comparison against llama.cpp before selecting a default runtime.
The independent MLX evaluation plan remains available as an optional alternative.

To obtain the runtime and documentation, clone or update this source repository:

```sh
git clone https://github.com/leo-kreisman/OxCoder-9B.git
cd OxCoder-9B
# For an existing clean checkout instead: git pull --ff-only
```

The existing model release contains weights and reconstruction utilities. Its
`download.sh` does not install this runtime or update a source checkout. Use the
model from that release with the conversion instructions in the native guide.

# Kindi Q4: native OxCoder9B inference

Apple Silicon implementation handoff: [Native Metal implementation and validation plan](METAL_IMPLEMENTATION.md). It maps the existing bridge to a complete GPU backend, specifies the 8GB memory budget, and defines phased correctness, profiling, cache and same-machine comparison gates. Planned components are explicitly distinguished from current commands.

This experimental fork adapts Kindi's **converter and complete inference engine** to the original mixed Q4_K/Q6_K weights of OxCoder9B. It is not a llama.cpp wrapper, a ternary conversion, an MLX implementation, or a model training recipe. The source GGUF remains unchanged. The converted pack is executed by native C/AVX2 or CUDA kernels with Kindi's hybrid recurrent/attention forward pass, tokenizer, server and prefix cache.

The initial supported profile is **dense Qwen3.5/OxCoder9B**, 32 layers, width 4096, FFN 12288, vocabulary 248320. Other architectures are rejected. **Edge0-35B-A3B MoE is not implemented by this fork.** Expert routing, expert residency and MoE-specific batch verification cannot be inferred from this dense implementation.

Upstream base: [ahtx/kindi](https://github.com/ahtx/kindi/tree/50fb5e6d67d5b9fd643e5a76ceda4f3d8ac19b5a), commit `50fb5e6d67d5b9fd643e5a76ceda4f3d8ac19b5a`. This modified source is AGPL-3.0-only; see LICENSE and NOTICE. The existing upstream ternary modules remain for provenance, but their benchmark claims do not describe this Q4 adaptation.

## Implementation and numerical contract

1. `kindi convert` reads GGUF and dispatches Q4_K/Q6_K/F32 models to `q4pack.py`. It checks the architecture, dimensions, convolution, RoPE, recurrent-layer schedule, norm epsilon, vocabulary size and tensor inventory before conversion.
2. Every Q4_K block (144 bytes per 256 weights) and Q6_K block (210 bytes) is preserved, including original scales and minima. This is a **lossless storage transformation**, not further quantization. Q4_K_M contains mixed precisions; it is not exactly four bits for every parameter.
3. Quantized matrices change from `[row][column_group][block_byte]` to `[row/8][column_group][row%8][block_byte]`. Each tensor starts at a 16KiB boundary. The converter processes bounded chunks and reverses each transformation to check byte identity. Small alpha/beta projections are decoded exactly into float32 for the recurrent gates. Conversion does not allocate a full float32 model.
4. `weights.bin`, `f32.bin`, `index.json`, tokenizer files and a manifest are written into a temporary directory. SHA256 checksums are recorded, then the directory is atomically renamed. Existing destinations are rejected.
5. `cpu4.py` binds the complete hybrid forward pass in `kcpu4.c`. `q4kernels.h` implements quantized block decoding and direct AVX2 products, with a portable scalar fallback. Bonsai-specific Hadamard transforms and recurrent-head reordering are removed for the original Qwen checkpoint.
6. `gpu4.py` executes the same model using Kindi's CUDA recurrent/attention kernels and the new row-tiled Q4/Q6 GEMV, embedding and normalization kernels in `q4.cu`. No external inference server is involved.
7. Full vocabulary scoring is used. Approximate shortlist heads and speculative decoding are deliberately rejected until implemented and validated for this format.

Exact block preservation does **not** imply bit-identical floating-point logits across backends. CPU reduction order, CUDA arithmetic and llama.cpp activation quantization can differ. The checked-in fixture establishes agreement on two specific greedy continuations (33 tokens), not universal equivalence or coding quality.

## Installation

Use Python 3.11 and an isolated environment. Run from the repository root:

```bash
python3 -m venv .venv-kindi-q4
.venv-kindi-q4/bin/python -m pip install -r scripts/kindi_q4/requirements.txt
export PYTHONPATH="$PWD/scripts/kindi_q4"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export OMP_WAIT_POLICY=PASSIVE
```

CPU: a C compiler and libm are required. Linux builds use OpenMP and `-march=native`; binaries are compiled locally and must not be copied between incompatible CPUs. macOS's portable CPU build does not require OpenMP.

CUDA: install the additional dependencies and a CUDA toolkit compatible with CuPy. The tested environment used CUDA 12.8, CuPy 13.6 and an RTX5060Ti. This is distinct from upstream's CUDA 10.2/Kepler instructions.

```bash
.venv-kindi-q4/bin/python -m pip install -r scripts/kindi_q4/requirements-cuda.txt
export CUDA_PATH=/usr/local/cuda-12.8  # adjust to the installed toolkit
export CUDA_VISIBLE_DEVICES=1       # select an otherwise idle physical GPU
export KINDI_DEVICE=gpu
```

No model download is performed automatically. Supply an authorized local GGUF. The validated source is `OxCoder-9B.Q4_K_M.gguf`, 5,629,117,536 bytes, SHA256:

```text
a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f
```

## Phase 1: conversion and integrity

```bash
export MODEL=/absolute/path/OxCoder-9B.Q4_K_M.gguf
export PACK=/absolute/path/oxcoder-kindi-q4
.venv-kindi-q4/bin/python -m kindi convert "$MODEL" --out "$PACK"
```

Allow approximately another 5.7GB on disk for the converted pack. The quantized weight file is 5,610,356,736 bytes for the validated source, plus float gates, metadata and tokenizer files. It has not magically become a 1–2GB model. The first conversion measured approximately 48 seconds on the test workstation; that is not a Mac estimate.

By default the engine validates all manifest checksums at startup. This reads the pack and affects startup time and filesystem cache state. After a full integrity check, controlled repeat benchmarks may set `KINDI_Q4_VERIFY=size`; size and profile validation still run, but same-size corruption will not be detected. Record this setting with results. Do not silently disable integrity checks in a distributed package.

Acceptance: conversion succeeds, block round trips pass, hashes validate, no full float model allocation, no unsupported profile accepted.

## Phase 2: independent correctness

```bash
export KINDI_DEVICE=cpu   # or gpu
.venv-kindi-q4/bin/python -m kindi --pack "$PACK" verify
PYTHONPATH="$PWD/scripts/kindi_q4" .venv-kindi-q4/bin/python -m unittest discover -s tests/kindi_q4 -v
```

`reference-oxcoder-q4.json` is tied to the source SHA256. Its outputs came from an independent llama.cpp build at commit `5e7f6271c06b9104862ab799278a1b7f1323a449`, CPU execution, greedy decoding and context 512. `tests/kindi_q4/reference.cpp` is the test-only reference generator; llama.cpp is not required by the runtime. Compile it against that checkout's public headers and libraries if regenerating fixtures. Never regenerate expected tokens using Kindi itself.

Set `GGML_QUANT_ORACLE=/absolute/path/libggml-base.so` when running unit tests to compare the block decoder directly with independent GGML dequantizers. Without this optional library, the tests compare native decoding with the separate NumPy implementation and check matrix products and malformed manifest rejection.

Acceptance: all fixture tokens match, quantized block decoding matches exactly, matrix products pass numerical tolerances, invalid manifests fail before native buffers are used. Stop performance evaluation on an output mismatch.

## Phase 3: memory-controlled Linux measurement

The budget is **8,000,000,000 bytes**, not eight GiB. Dedicated GPU memory must not be treated as free extra capacity when assessing a design intended for an 8GB unified-memory allowance.

`run_limited.py` starts one systemd user service with two CPU cores of quota, low CPU/I/O priority, a time limit and no swap. CPU tests get a hard 8GB host cgroup cap. CUDA tests get a hard 1.7GB host cap and count **cgroup memory plus physical memory used by the selected GPU** every 0.5 seconds, stopping the experiment above8GB. CUDA additionally checks host RSS plus its allocation pool before allocating tensors. GPU memory is sampled, not hard partitioned; instantaneous spikes between samples remain possible. The guard stops only its own experiment and never flushes the system cache or kills other workloads.

```bash
export KINDI_DEVICE=gpu
export KINDI_Q4_VERIFY=size
export KINDI_Q4_GPU_CACHE_BYTES=5800000000
export KINDI_Q4_STREAM=0
printf 'def add(a, b):\n    return' > /tmp/kindi-code.txt
.venv-kindi-q4/bin/python scripts/kindi_q4/run_limited.py \
  --gpu 1 --report results/kindi-resident-resources.json -- \
  "$PWD/.venv-kindi-q4/bin/python" "$PWD/scripts/kindi_q4/probe.py" \
  --pack "$PACK" --prompt-file /tmp/kindi-code.txt \
  --output "$PWD/results/kindi-resident.json" --tokens 64 --context 512
```

For CPU set `KINDI_DEVICE=cpu` and omit `--gpu`. It does not use the CUDA cache-size variable. `KINDI_Q4_STREAM=1` on CPU releases mapped layer pages after use; the default relies on normal mmap residency.

The probe stops at EOS, records input/output IDs, first-token logits, load time, time to first token, decode throughput, completion time, process RSS, cgroup events and I/O bytes. Decode throughput excludes prefill and the first token. Do not label the combined throughput printed by `kindi verify` as decode speed. The upstream `kindi bench` is disabled for Q4 because its context-position benchmarks are not suitable evidence for this adaptation.

Acceptance: no OOM, no swap, memory remains within budget, output correct, and the measured workload is identified. A configured context limit is not a populated context test. Repeat with real 4K and 16K input files only after the small test is stable; keep output limits and abort thresholds. Do not populate 128K on the strength of a short-context benchmark. This fork currently accepts 32..32768 context capacity; only 512/1024 capacity and short populated prompts have been tested. Existing 128K configurations of a different runtime are not changed.

## Phase 4: force real SSD streaming

Repeat **the same prompt and output count** with:

```bash
export KINDI_Q4_GPU_CACHE_BYTES=1600000000
export KINDI_Q4_STREAM=1
export KINDI_Q4_PIN_BOUNDARIES=1
```

The CUDA cache is bounded in bytes and evicts whole tensors by LRU. At 1.6GB or more it preferentially keeps the embedding and output tensors. `KINDI_Q4_PIN_BOUNDARIES=0` disables that preference for an ablation. The cache budget, rather than the STREAM flag alone, forces CUDA eviction. After uploads, the runtime advises Linux to release the corresponding host file-cache pages; this is advisory, not direct I/O. Upload staging is limited to 8MiB chunks. Physical `read_bytes`, logical weight bytes and cache misses must be reported separately.

OxCoder is **dense**: nearly every transformer weight is used on every token. Streaming a small subset of the model therefore causes almost a full reread per token. LRU cannot turn this into sparse MoE inference. If resident mode fits the allowance, use it. SSD streaming is a functional fallback with a severe bandwidth cost, not the preferred performance mode.

Acceptance: token IDs match resident mode and physical storage reads demonstrate actual streaming. Cache hits or successful mmap alone are not evidence of SSD-offloaded decode performance.

## Phase 5: API, streaming and prefix caching

```bash
export KINDI_Q4_GPU_CACHE_BYTES=5800000000
export KINDI_Q4_STREAM=0
export KINDI_PARK_MAX=0
export KINDI_CHECKPOINT_BUDGET_MB=192
.venv-kindi-q4/bin/python -m kindi --pack "$PACK" --ctx 512 \
  --state-dir /absolute/path/kindi-q4-state serve --port 8093
```

Use the same resource wrapper for a long-lived server, choosing an appropriate time limit; do not run a second model concurrently. `http://127.0.0.1:8093/v1` exposes `oxcoder9b-q4-kindi`. The baseline is one active request, one conversation and full output-head scoring. The server serializes inference; concurrent clients can still queue, so client retries and parallel agent requests must be disabled.

```bash
curl http://127.0.0.1:8093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"oxcoder9b-q4-kindi","messages":[{"role":"user","content":"Write a Python function that returns the square of a number. Code only."}],"temperature":0,"max_tokens":32}'
```

Kindi checkpoints recurrent state and caches prefixes. Keep system prompts, tools and conversation serialization stable. The memory budget for snapshots is explicit; parked conversations are disabled for the initial deployment. Snapshot fingerprints distinguish CPU, CUDA and Metal and include arithmetic source hashes. Disk snapshots do not increase model intelligence or decode throughput; they avoid repeated prompt work.

Run `tests/kindi_q4/server_smoke.py PACK REPORT.json` under the resource guard. It checks HTTP completion, exact repeat output, nonzero cached tokens, SSE completion and reuse of a disk opening after a server restart. It asserts the expected generated code without executing arbitrary model output. It owns localhost 18193 and terminates its own test server.

Acceptance: first and repeated answers agree, cached-token counts rise, SSE ends correctly, disk snapshots restore after restart, memory stays bounded. Tool schemas are inherited from Kindi, but professional coding-agent/tool-call reliability has not been established by this smoke test.

## Metal status and Mac validation gate

For the detailed engineering procedure, use [METAL_IMPLEMENTATION.md](METAL_IMPLEMENTATION.md). The steps below describe the current bridge only.

`KINDI_DEVICE=metal` selects an **experimental, unvalidated** native Objective-C++/Metal matrix path. It compiles `metal4.mm` with Apple Clang and `q4.metal` with the Metal API. Quantized tensors are wrapped as shared, page-aligned mmap buffers without a duplicate full weight array. The remainder of the hybrid forward pass stays on CPU. This is not the CUDA execution strategy ported in full: every matrix dispatch currently waits for completion, so synchronization overhead can dominate.

No Mac was available to compile or benchmark it. There is no claim that it exceeds an existing17tokens/s M3 implementation. `KINDI_Q4_STREAM=1` is rejected with Metal because mapped-resource eviction has not been validated. Metal allocation admission and process RSS checks are safeguards, not a hard 8GB macOS unified-memory limit. RSS and Metal allocations overlap; adding them blindly double-counts shared pages, while RSS alone may miss driver allocations.

On the Mac, install Xcode Command Line Tools and CPU requirements, then use `KINDI_DEVICE=metal`, `KINDI_Q4_STREAM=0`, context 512 and `kindi verify`. First establish successful compilation and reference output; monitor process physical footprint, system memory pressure and swap throughout. Stop on growing swap or interference with company applications. Never raise global wired-memory limits to make this test pass. Increase populated context only after measuring total footprint. Keep the existing llama.cpp server as the practical baseline and restore it if this experiment is slower.

Before treating Metal as deployment-ready, an implementer must: validate kernels against the CPU decoder; compare full model logits and reference continuations; eliminate unnecessary CPU/GPU barriers; move recurrent/attention operations into the command graph where beneficial; measure physical footprint and cold/warm prefill; and run identical coding tasks at the same sampling settings. No CUDA multiplier predicts these results.

## Results and selection rule

See [VALIDATION.md](VALIDATION.md) and the accompanying JSON evidence. Choose by **correctness, total task time and measured memory**, not a single peak decode number. The implemented fast path is native resident Q4 inference; forced SSD streaming trades speed for a smaller resident weight cache. Speculation, Edge0 MoE kernels, Metal graph-managed pools and packed expert batching remain separate work and are not advertised as working features of this dense fork.

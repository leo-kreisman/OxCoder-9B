# Native Metal implementation and validation plan for Kindi Q4

Status: implementation specification, inspected against repository source on 2026-10-26. The existing Metal bridge is experimental and has not been compiled or benchmarked on macOS in the recorded validation. Instructions below distinguish existing commands from components that still need implementation.

The objective is a correct, responsive, resident-memory OxCoder9B Q4_K_M runtime on Apple Silicon under an **8,000,000,000-byte inference allowance**. Success means an improvement over a measured llama.cpp Metal baseline on the same Mac, with comparable outputs, populated context and sampling. A successful build alone does not establish a speedup.

This document is an engineering handoff for maintainers and coding agents. It does not require MLX, model training, a different quantization, or SSD weight streaming. Model files and optional prefix snapshots can reside on storage; normal generation is intended to reuse weights already resident in unified memory. Operating-system paging must be observed rather than mistaken for intentional offloading.

## Contents

1. [Scope and current evidence](#1-scope-and-current-evidence)
2. [Source map and current execution](#2-source-map-and-current-execution)
3. [Architecture to implement](#3-architecture-to-implement)
4. [Memory budget and context accounting](#4-memory-budget-and-context-accounting)
5. [Phase 0: inventory and baseline](#5-phase-0-inventory-and-baseline)
6. [Phase 1: compile and validate the existing bridge](#6-phase-1-compile-and-validate-the-existing-bridge)
7. [Phase 2: instrumentation and allocation control](#7-phase-2-instrumentation-and-allocation-control)
8. [Phase 3: GPU-resident computation](#8-phase-3-gpu-resident-computation)
9. [Phase 4: command scheduling and kernel tuning](#9-phase-4-command-scheduling-and-kernel-tuning)
10. [Phase 5: prefill and prefix-state reuse](#10-phase-5-prefill-and-prefix-state-reuse)
11. [Phase 6: application integration](#11-phase-6-application-integration)
12. [Benchmark protocol and selection](#12-benchmark-protocol-and-selection)
13. [Failure diagnosis and rollback](#13-failure-diagnosis-and-rollback)
14. [Completion checklist](#14-completion-checklist)
15. [Primary references](#15-primary-references)

## 1. Scope and current evidence

### Supported model contract

The current pack implementation accepts the dense Qwen3.5/OxCoder9B profile:

| Property | Required value |
|---|---:|
| Transformer layers | 32 |
| Hidden width | 4096 |
| FFN intermediate width | 12288 |
| Query heads / KV heads | 16 / 4 |
| Attention head dimension | 256 |
| Full-attention layers | 8, every fourth layer |
| Recurrent Gated-DeltaNet layers | 24 |
| Recurrent key heads / value heads | 16 / 32 |
| Recurrent state dimension | 128 |
| Recurrent inner width | 4096 |
| Convolution kernel length | 4 |
| Rotary dimensions / base | 64 / 10,000,000 |
| Vocabulary entries | 248320 |
| Packed tensor types | Q4_K, Q6_K, F32 |

The source checkpoint identified in [VALIDATION.md](VALIDATION.md) remains the reference. A new checkpoint, model architecture or pack layout requires its own integrity checks and reference fixtures. Do not disable profile validation to make another model load.

This is a dense runtime. Expert routing, GPU expert pools, adaptive MoE residency and Edge0-35B-A3B execution are outside this implementation. Speculative decoding and approximate vocabulary shortlists are also outside the initial Metal target. They are not hidden features enabled by a device flag.

### Recorded evidence

Linux/CUDA native inference generated 64 tokens at 25.66 output tokens/s with a measured 6.79GB host-plus-GPU peak. CPU and CUDA passed two reference continuations totaling 33 tokens; HTTP streaming and persistent prefix restoration passed small integration tests. These establish the implementation's Linux behavior, not Mac throughput. See [validation/summary.json](validation/summary.json) and the full [validation report](VALIDATION.md).

No current result establishes that this Metal bridge exceeds llama.cpp Metal, that an M3 reaches the CUDA speed, or that 128K populated context fits the new runtime. Preserve any independently working long-context runtime as a separate baseline. Its context configuration is not automatically transferable to this backend.

### Status vocabulary

Use these labels in subsequent reports:

- **Implemented:** source exists and the specified path is wired into runtime selection.
- **Compiled:** that source built on the named macOS, SDK and chip.
- **Correctness checked:** the named fixtures and numerical tests passed.
- **Measured:** workload, timing definition, memory accounting and repetitions are recorded.
- **Recommended:** correctness and resource gates passed, and comparative testing supports choosing it over the baseline.

Do not upgrade one label to the next without the corresponding artifact.

## 2. Source map and current execution

| File / symbol | Existing responsibility | Required work or constraint |
|---|---|---|
| [q4pack.py](kindi/q4pack.py): `convert`, `verify_pack`, `PROFILE_DIMS` | GGUF validation, lossless quantized packing and checksums | Preserve as the pack contract; no requantization needed |
| [q4kernels.h](kindi/q4kernels.h) | CPU block decoding and direct quantized matrix products | Numerical oracle for Metal kernel tests |
| [cpu4.py](kindi/cpu4.py): `_lib`, `Engine` | C ABI, mmap ownership, native compilation and state allocation | Existing Metal selection uses this CPU engine; build/device metadata and lifecycle need strengthening |
| [kcpu4.c](kindi/kcpu4.c): `gemv`, `gdn`, `attn`, `ffn`, `kc_step` | Complete scalar/AVX2 model execution | Port semantics and dependencies, including residual ordering |
| [metal4.mm](kindi/metal4.mm): `kq_metal_init`, `kq_metal_matvec` | Metal initialization and one matrix product at a time | Replace global singleton ownership for a complete Metal backend; eliminate per-matrix host handoffs |
| [q4.metal](kindi/q4.metal): `q4_gemv` | Q4_K/Q6_K row-tiled GEMV | Validate directly, then benchmark layout and reduction variants |
| [gpu4.py](kindi/gpu4.py), [q4.cu](kindi/q4.cu) | Working CUDA adaptation | Useful semantic reference; CUDA resource calls are not portable to Metal |
| [engine.py](kindi/engine.py): `make_engine` | Dispatches Q4 CPU, CUDA and Metal bridge | Add a distinct complete Metal implementation only after its gates pass |
| [cache.py](kindi/cache.py): `Runner`, `Openings`, `engine_fingerprint` | Recurrent checkpoints, parked sessions and disk prefixes | Add backend-neutral snapshot ownership and synchronized restore for Metal-owned state |
| [sampling.py](kindi/sampling.py): `Sampler` | Greedy / stochastic sampling | Current GPU detection is CuPy-specific; make Metal sampling explicit |
| [server.py](kindi/server.py): `Kindi` | Chat template, serialization, HTTP and streaming | Preserve protocol; retain one active inference request |
| [probe.py](probe.py) | EOS-aware, bounded token probe | Extend Mac telemetry; keep its timing definitions explicit |
| [run_limited.py](run_limited.py) | Linux systemd/cgroup experiment guard | **Not usable as a macOS memory limiter** |
| [test_quant.py](../../tests/kindi_q4/test_quant.py) | Native CPU kernel comparison | Does not currently dispatch `q4_gemv` on Metal; add dedicated Metal coverage |
| [server_smoke.py](../../tests/kindi_q4/server_smoke.py) | HTTP, repeated output, SSE and restart test | Reuse after Metal state integration and memory monitoring |

### What the current bridge actually does

`KINDI_DEVICE=metal` still constructs `cpu4.Engine`. It compiles `kcpu4.c` with `KINDI_METAL`, links the Objective-C++ bridge, and redirects model `gemv()` calls to `kq_metal_matvec()`.

For each matrix, the bridge:

1. Finds or creates an `MTLBuffer` wrapping the mapped tensor.
2. Copies the CPU activation into a shared input buffer.
3. Creates a command buffer and compute encoder.
4. Dispatches the quantized matrix kernel.
5. Commits and calls `waitUntilCompleted`.
6. Copies the output back to a CPU array.
7. Continues normalization, recurrence, attention or activation work on CPU.

For one normal decoded token, there are approximately **201 matrix submissions and waits**: `24 × 3` recurrent projections, `8 × 4` attention projections, `32 × 3` FFN projections and one output head. This is a count from the current forward-pass structure, not a hardware measurement. Recurrent alpha/beta gates and embedding lookup additionally execute on CPU in this bridge.

### Why shared memory alone does not solve this

Unified memory allows shared physical storage, but dependent CPU and GPU operations still need ordering. The bridge's repeated submission, waiting and activation copying remain overhead. Removing the waits without changing ownership would cause the CPU to read unfinished output or overwrite inputs the GPU still needs.

Apple recommends limiting unnecessary command submissions while maintaining GPU utilization. That motivates profiling and grouping this runtime's work; it does not justify copying a graphics frame loop or triple-buffering entire models into an 8GB inference process. [Apple: command buffers](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html).

## 3. Architecture to implement

### Keep a reference implementation

Retain the existing CPU engine and synchronous Metal bridge as explicit development references. Introduce the complete backend in a new module, for example `kindi/metal_engine.py`, with resource ownership in an Objective-C++ instance rather than the current static globals. This filename is a proposed implementation target, not an existing module or launch option.

Do not change the default backend until the new one passes all release gates. Any new selection flag must be documented and validated in configuration; unknown values must fail rather than silently falling back to CPU.

### Desired execution boundary

```text
CPU: tokenizer, request handling, orchestration, logs
       |
       | token IDs, positions, bounded prefill metadata
       v
Metal: embedding -> 32 hybrid layers -> final norm -> full output head
       |              |
       |              +-> persistent recurrence and attention K/V
       v
Metal argmax, or explicit sampling/logits export boundary
       |
       v
CPU: token ID, stream response, next decode iteration
```

The target is one completed model-step submission per decoded token, or a small measured number of dependency-correct submissions if the single-buffer version performs worse or exceeds execution limits. Device barriers may still be necessary between kernels. A token-level host wait is acceptable because ordinary autoregressive decoding needs the selected token before the next step. The optimization removes **per-matrix** host waits, not mathematical dependencies between generated tokens.

No additional large model, teacher or drafter is required. Do not make speculative decoding a prerequisite for finishing this backend.

### Backend interface contract

Implement and test these operations; names below specify responsibilities rather than an already available ABI:

| Operation | Required behavior |
|---|---|
| `create(pack, context, budget)` | Validate profile and checksums; preflight memory; create pipelines and resources; return an owned handle |
| `prefill(token_ids, position, chunk_limit)` | Process an ordered token sequence; preserve recurrence and K/V; bounded allocation and cancellation points |
| `decode(token_id, position, sampling)` | Produce logits or selected token and leave state exactly after this input token |
| `snapshot(position)` | Capture a coherent recurrent state and required attention prefix after completed GPU writes |
| `restore(snapshot)` | Validate model/backend identity and capacity; restore state before further execution |
| `reset()` | Restore the same empty state as a fresh engine |
| `metrics()` | Return memory, dispatch, wait, prefill and decode counters without large allocations |
| `destroy()` | Complete/cancel owned work safely, release buffers and pipelines, then unmap owned storage |

Use explicit error returns across the C ABI. An encoding, allocation or command error must not be converted into zero logits followed by continued generation. On failed or partially executed state updates, invalidate the active state and restore a known checkpoint or reset; never cache the failed state.

Preserve single-request execution and one loaded model. Protect against two engine instances aliasing the current global Metal input/output buffers. Instance ownership is required before reloads or multiple in-process model lifetimes are advertised.

## 4. Memory budget and context accounting

### Accounting policy

The inference allowance is 8,000,000,000 decimal bytes. It includes model weights, state, K/V, scratch, tokenizer/runtime allocations, checkpoints, driver-associated process allocations and transient loading buffers. It is not a GPU-only budget. Total installed machine RAM is not the amount available to the inference process.

Implement an allocation ledger keyed by unique backing allocation or VM range. Distinguish aliases from copies. Wrapping a mmap region as an `MTLBuffer` must not count the same backing pages twice in the ledger; allocating a private copy of those weights is a real second allocation and must be admitted as such.

Collect at least:

- Process `phys_footprint` via `task_info(TASK_VM_INFO)` and process resident size.
- Metal `currentAllocatedSize`, buffer allocation sizes and explicit owned-allocation totals.
- File-backed mappings and any private copies of model tensors.
- System swap/compression deltas and memory-pressure state during the experiment.
- Temporary conversion, pipeline compilation, snapshot and restoration peaks.

Apple's `currentAllocatedSize` describes Metal resources, not the entire Python process. Its recommended working-set size is a performance recommendation, not an application memory cap. Neither should be substituted for the 8GB allowance. [Allocated resources](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize), [recommended working set](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).

`phys_footprint` is useful process telemetry, but do not assume it includes every clean file-backed weight page in the way the model-residency budget requires. Reconcile it with mappings and the allocation ledger; disclose unresolved attribution. Do not calculate unified-memory use by blindly adding process RSS to Metal allocation bytes. Apple's task-info tests illustrate the footprint interface and revision checks. [Apple task-info tests](https://github.com/apple/darwin-xnu/blob/main/tests/task_info.c).

### Initial operating thresholds

Use the following project policy until measurements justify a stricter one:

| Threshold | Action |
|---|---|
| Projected owned allocation plus measured runtime overhead above 7.5GB | Refuse additional capacity or configuration before allocating |
| Measured/accounted footprint reaches 7.5GB | Stop issuing new work, record diagnostics and release the experiment safely |
| 8GB allowance | Report violation; the run cannot qualify as fitting |
| Sustained memory-pressure warning or growing swap during steady decode | Abort performance qualification even if nominal allocations fit |
| Monitoring unavailable or ambiguous allocation accounting | Mark memory compliance unverified; do not claim an 8GB success |

The 0.5GB gap is an initial reserve, not a proof that all driver transients fit. Adjust downward when measured non-model overhead is larger. A macOS watchdog and admission checks are cooperative safeguards, not the Linux cgroup hard limit. Do not use `ulimit` on virtual address space as a substitute for unified-memory accounting, raise global wired-memory limits, purge the system cache, or terminate unrelated applications.

### Static and context-dependent storage

The checked source produces a quantized weight file of 5,610,356,736 bytes. Add `f32.bin`, tokenizer objects and all non-weight buffers. Use the manifest's actual file sizes rather than assuming every parameter is exactly four bits.

For the current eight full-attention layers with FP16 K/V:

```text
KV bytes per token = 8 layers × 2 (K,V) × 4 KV heads × 256 dimensions × 2 bytes
                   = 32,768 bytes
KV allocation      = capacity_tokens × 32,768
```

| Configured K/V capacity | FP16 K/V allocation alone |
|---:|---:|
| 512 | 16,777,216 bytes |
| 4,096 | 134,217,728 bytes |
| 8,192 | 268,435,456 bytes |
| 16,384 | 536,870,912 bytes |
| 32,768 | 1,073,741,824 bytes |
| 131,072, hypothetical extension | 4,294,967,296 bytes |

The last row is arithmetic, not an implemented context configuration. This fork currently rejects capacity above 32768. A different runtime may use a different state representation, K/V precision or allocation policy. These numbers do not disprove another runtime's reported 128K configuration.

The recurrent matrices require `24 × 32 × 128 × 128 × 4 = 50,331,648` bytes. Convolution history adds `24 × 3 × 8192 × 4 = 2,359,296` bytes. One complete recurrent checkpoint is therefore about 52.69MB before metadata. Checkpoints accumulate separately from the live state. Attention-prefix copies, parked conversations and in-flight disk snapshot staging add more.

Start with one conversation, `KINDI_PARK_MAX=0`, a 192MiB checkpoint allowance and capacity512. Increase actual populated context and configured capacity separately. The checkpoint budget does not include all other runtime allocations and is not the global memory guard.

### Weight residency

The existing pack uses 16KiB-aligned tensor offsets. At runtime query the host page size and validate actual pointer alignment, padded lengths, VM-region boundaries and `maxBufferLength`. Apple's no-copy API requires page-aligned storage within a single VM region. Keep the mapping alive until every wrapping buffer and pending command is released. Do not attach a deallocator that frees memory owned by NumPy's mmap. [Apple: no-copy buffers](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29?language=objc).

If file-backed wrapping fails on the target OS, stop with diagnostics. A fallback copy must be budgeted, loaded in bounded chunks and proven to release the original backing residency appropriately; silently duplicating the entire model is unacceptable. Shared/private storage experiments are optional ablations after the no-copy baseline is correct.

A one-time model read from SSD is normal startup. Repeated large weight reads or growing swap during warmed decode mean the resident-memory goal has not been achieved. Record that condition as memory pressure or paging, not a successful SSD acceleration feature.

## 5. Phase 0: inventory and baseline

**Deliverable:** hardware/runtime manifest, verified model identity and a same-machine llama.cpp baseline. No backend modification is needed for this phase.

Execution order: collect the inventory and baseline configuration first, then perform Phase1's small compile/kernel checks, implement Phase2's Mac monitor, and return to the full-model baseline and Phase1 reference checks. Proceed to Phase3 only after those gates pass. If adequate memory telemetry already exists, it may satisfy Phase2's monitoring prerequisite; record its limitations. Do not launch an unmonitored full model merely to keep the numbered sections strictly sequential.

### Environment inventory

Run from the repository root. The commands below only collect local environment information and create an experiment directory:

```bash
export EXPERIMENT="$PWD/results/kindi-metal-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPERIMENT"
uname -m > "$EXPERIMENT/architecture.txt"
sw_vers > "$EXPERIMENT/macos.txt"
system_profiler SPHardwareDataType SPDisplaysDataType > "$EXPERIMENT/hardware.txt"
sysctl hw.memsize hw.pagesize > "$EXPERIMENT/memory-hardware.txt"
sysctl vm.swapusage > "$EXPERIMENT/swap-before.txt"
vm_stat > "$EXPERIMENT/vm-before.txt"
xcode-select -p > "$EXPERIMENT/developer-directory.txt"
xcrun --find clang > "$EXPERIMENT/compiler-path.txt"
xcrun clang --version > "$EXPERIMENT/compiler-version.txt"
git rev-parse HEAD > "$EXPERIMENT/repository-head.txt"
git status --short > "$EXPERIMENT/repository-status.txt"
```

Hardware inventories may contain serial numbers; remove identifiers before publishing artifacts. Record dirty source hashes because a Git commit alone does not identify uncommitted code. Full Xcode is needed for Instruments/GPU capture; the basic bridge uses Apple Clang plus the SDK. Verify installed tool availability instead of installing or upgrading system-wide tools automatically.

Record device name, GPU core configuration, OS/SDK, power source, power mode, display workload, thermal conditions and normal competing applications. Do not compare a cold idle machine with a warmed busy one and attribute the difference entirely to the backend.

### Establish the comparison

Keep the installed llama.cpp deployment configuration available. Record its binary version/commit, exact GGUF SHA256, template, context capacity, actual prompt token IDs where available, K/V precision, offload settings and sampling parameters. Use its installed `--help` to establish valid flags; this guide does not assume all releases share CLI options.

Capture cold-start, warm-prompt, warm-decode and multi-turn measurements separately. Run one model process at a time. For independent raw-token oracle tests, bypass chat templates in both engines; for agent-facing tests, require identical rendered messages and tool definitions. Do not compare a plain completion against a large agent prompt.

Use greedy output first for parity. Also evaluate the intended production sampling configuration later, without changing it between backends. A historical tokens/s counter without input/output lengths and memory measurements is context, not an accepted baseline.

**Gate:** the source model is known, baseline runs are reproducible, and a representative baseline fits the allowance without sustained paging. If the baseline itself exceeds the budget, record the limitation and reduce the test capacity explicitly for both engines.

## 6. Phase 1: compile and validate the existing bridge

**Deliverable:** a reproducible Mac build and direct Metal kernel tests. This phase does not imply that the bridge is fast.

### Set up the isolated runtime

```bash
python3.11 -m venv .venv-kindi-q4
.venv-kindi-q4/bin/python -m pip install -r scripts/kindi_q4/requirements.txt
export PYTHONPATH="$PWD/scripts/kindi_q4"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export KINDI_CPU_CACHE="$EXPERIMENT/native-cache"
export KINDI_STATE_DIR="$EXPERIMENT/state"
export KINDI_SPEC=0
export KINDI_Q4_STREAM=0
export KINDI_PARK_MAX=0
export KINDI_CHECKPOINT_BUDGET_MB=192
export KINDI_Q4_VERIFY=full
export PACK=/absolute/path/to/oxcoder-kindi-q4
```

The macOS compiler path currently does not enable OpenMP; setting more OpenMP threads will not make that portable CPU portion parallel. Do not install the CUDA requirements on macOS. If a converted pack is not available, follow the [conversion instructions](README.md#phase-1-conversion-and-integrity) using a new destination and sufficient free disk space. An already converted compatible pack can be copied with all files and then checksum-verified.

### Compile and inspect the bridge

The existing `cpu4._lib()` compiles C with `-DKINDI_METAL`, links Objective-C++ with Foundation/Metal and compiles the shader at initialization. Run the following small check before loading model weights:

```bash
KINDI_DEVICE=metal .venv-kindi-q4/bin/python - <<'PY'
from kindi.cpu4 import _lib
library = _lib()
print("Metal bridge initialized; shader compilation succeeded")
print("Metal allocated bytes:", library.kq_metal_allocated())
PY
```

Capture compiler and shader diagnostics. The cached library has a `.so` filename even on macOS; inspect the binary architecture rather than assuming the suffix means Linux. Include backend selection, compiler/SDK version and compile options in future build-cache keys; current keys principally hash source and machine architecture. Never reuse an x86 or CUDA library as an Apple Silicon artifact.

The current Metal shader requires a 32-lane execution width and a 128-thread group. Query these capabilities and fail clearly on mismatch. This is an implementation requirement, not a claim that every Metal device has those properties.

### Required test additions

Create dedicated Metal tests in `tests/kindi_q4/`, with these properties:

1. Allocate page-aligned backing storage with the same row-tiled layout as the pack. Ordinary NumPy allocations are not guaranteed to satisfy no-copy Metal alignment.
2. Invoke **`kq_metal_matvec` itself**, not just `kc_matvec`. The existing `test_quant.py` calls `kc_matvec`, which still computes CPU products even when the library was built with `KINDI_METAL`. Passing that test is not Metal validation.
3. Exercise Q4_K and Q6_K independently: signed values, nonzero minima, extreme valid scales, rows spanning tile boundaries, representative matrix widths and bounded portions of large matrices.
4. Compare with the separate NumPy block decoder and CPU products. Use an FP64 accumulation oracle for small fixtures. Start with the existing small-fixture tolerance (`atol=1.5e-4`, `rtol=1e-5`); report errors instead of widening tolerances merely to pass a new reduction scheme.
5. Test invalid shapes, pointer alignment, lengths, allocation failure, command failure and teardown. Ensure errors propagate without fabricated output.
6. Verify repeated kernel calls with different input vectors and tensors do not read stale input or reuse the wrong buffer.

For full model verification, after memory monitoring is active:

```bash
KINDI_DEVICE=metal .venv-kindi-q4/bin/python -m kindi --pack "$PACK" verify
```

`verify` currently constructs capacity1024 regardless of a global `--ctx` value. Budget that capacity. The checked fixture contains 16 raw code tokens and 17 chat tokens through EOS for one source SHA256. An unknown source needs independent reference generation, not edited expected tokens.

Use layer-boundary dumps when parity fails: embedding, pre-normalization residual, projections, recurrence update, attention result, FFN result, final norm and logits. Compare both absolute/relative error and the top-two logit margin. Small floating-point differences near tied logits need diagnosis; large errors or different state order are implementation defects. Do not silently accept changed outputs in an optimization advertised as preserving the model.

**Gate:** bridge compilation, actual Metal kernel invocation, bounded allocation checks and existing full-model fixtures pass. Otherwise stop before performance claims or default changes.

The full-model part of this gate depends on Phase2 monitoring. Passing the small kernel checks allows work on that monitor; it does not authorize advancing to GPU-forward-pass optimization without completing the full-model checks.

## 7. Phase 2: instrumentation and allocation control

**Deliverable:** a measurement harness that can distinguish model work, synchronization, prefill and memory pressure.

Extend or add a Mac-specific monitor before long or populated-context tests. `run_limited.py` uses systemd and Linux cgroups and must not be invoked as a Mac limiter. `probe.py` currently records macOS peak RSS but not complete unified-memory accounting. Treat missing monitoring as a failed qualification gate.

### Runtime counters

Record per request and, in diagnostic mode, per token:

| Counter | Purpose |
|---|---|
| CPU encoding time | Detect Python/Objective-C command-building overhead |
| Command buffers submitted | Confirm that per-matrix scheduling is actually removed |
| Explicit CPU wait count and wall time | Quantify synchronization overhead |
| GPU execution intervals | Distinguish GPU compute from CPU waiting and scheduling |
| Per-stage kernel time in diagnostic captures | Locate matrix, recurrence, attention and head bottlenecks |
| Activation bytes copied CPU↔GPU | Verify that internal activations remain device-owned |
| Weight-buffer creations and allocations | Confirm no repeated tensor copies during warm decode |
| Recurrent/KV/scratch/checkpoint allocation bytes | Explain total footprint and capacity changes |
| Actual prompt tokens, reused tokens and new tokens | Explain prefill work and cache effectiveness |
| Generated tokens, EOS, final finish reason | Prevent post-EOS speed inflation or incomplete answers |

Read command-buffer GPU timestamps only after completion. Use `GPUEndTime - GPUStartTime` for that buffer's execution interval; overlapping buffers cannot be summed into end-to-end latency. Preserve a monotonic CPU wall clock as the request-time authority. [Apple: GPU timing](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime).

Use Instruments Metal System Trace/GPU capture to confirm scheduling and dependencies. Diagnostic captures and validation layers can change performance; run final timing with them disabled, retaining separate correctness captures. [Apple: Metal analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/).

### Monitoring and termination

Sample process/Metal/accounting metrics about every100–250ms during initial experiments, with admission checks before large allocations. Report sampling frequency and unobserved transient risk. Read system pressure and swap at baseline, during requests and after completion. Existing unrelated swap is not automatically a failure; sustained growth correlated with the experiment invalidates a resident-memory performance result.

The monitor owns only the test child/process group. On a stop condition, cancel further inference, allow bounded cleanup and terminate that child if necessary. Do not run stress allocators to discover the memory limit. Do not infer that low Python RSS means Metal weights are absent from the budget.

### Timing definitions

```text
load_s       = engine_ready - process/engine_start       # define boundary explicitly
TTFT_s       = first_emitted_token - request_start
prefill_s    = final_prompt_step_complete - prefill_start
warm_decode  = (output_count - 1) / (last_token_time - first_token_time)
completion_s = request_complete - request_start
new_prefill_tps = newly_processed_prompt_tokens / prefill_s
```

`probe.py` currently excludes initialization from its TTFT field and includes first-step lazy weight loading in that field. Preserve that distinction in exported reports. `kindi verify` reports combined prompt/output processing, not pure decode. Streaming HTTP TTFT also includes template/tokenization/server work. Do not mix these definitions in one speed comparison.

**Gate:** counters show the current bridge's actual submission/wait behavior, memory is attributable, and output correctness is unchanged by instrumentation.

## 8. Phase 3: GPU-resident computation

**Deliverable:** a complete model forward pass whose intermediate activations do not require CPU processing between matrix operations.

Implement in small stages, comparing each stage with `kcpu4.c` before replacing the next. Keep the original mixed quantized blocks. Weight decoding should happen in registers or bounded tiles, never by materializing the full model as FP16/FP32.

### 3A. Owned resources and backend binding

Introduce an engine-owned Metal context with one queue, cached pipeline states, immutable tensor descriptors, scratch buffers, recurrent states and K/V buffers. Allocate during initialization or explicit capacity changes. Avoid creating pipeline objects or large buffers per token. Apple's command-structure guidance supports reusing expensive device objects; each committed command buffer itself is single-use. [Apple: command structure](https://developer.apple.com/documentation/Metal/setting-up-a-command-structure).

Add complete ABI checks for sizes, types, ownership and error returns. Validate the manifest before interpreting C/Metal tensor descriptors. Separate diagnostic logits export from the fast token-ID result. Keep buffer destruction and mmap ownership deterministic.

For each stateful allocation, document shape, dtype, alignment, producer, consumer and lifetime. Include transient export buffers in admission calculations. A second backend should not keep a complete CPU shadow model or shadow K/V cache just for convenience.

### 3B. Embedding, residuals, norms and FFN

Port and test:

- Quantized embedding row lookup from the row-tiled matrix.
- Residual additions and RMS normalization with the original epsilon and norm parameters.
- FFN gate/up projections, `silu(gate) * up`, down projection and residual placement.
- Final normalization and full output-head projection.

The native code sometimes delays residual addition using the previous block's output. Preserve that mathematical ordering while changing buffer ownership. Do not apply Bonsai Hadamard transforms, fold factors or head permutations to this original Qwen pack.

Validate projection outputs separately before fusing. Use FP32 accumulation and the existing state precision initially. Any FP16 arithmetic or fast-math change becomes its own numerical/performance ablation, not an invisible side effect of moving to GPU.

### 3C. Recurrent Gated-DeltaNet layers

Port the entire recurrent sequence, including the small F32 alpha/beta projections. Leaving those on CPU would retain a per-layer synchronization dependency.

Preserve:

1. Input normalization and qkv/gate projections.
2. Alpha/beta gating, stable softplus/exponential behavior and per-head parameters.
3. Causal convolution: three stored history entries plus the current input; update history exactly once per input token.
4. Query/key L2 normalization and key/value head mapping (`value_head % key_heads` in this profile).
5. FP32 delta-rule state update in `[value_head][128][128]` order.
6. Per-head output normalization, SiLU gate and output projection.

The pack's projection rows already use the original value-head order. A previous prototype failed because it retained a ternary-model permutation. Add a regression fixture that exposes head-order errors rather than relying only on human-readable text.

Tests must cover zero state, a sequence of at least several dozen tokens, reset, snapshot, resume, divergent prefix and restored continuation. Matching only the first token cannot validate a recurrent model.

### 3D. Full-attention layers

Port query/key/value projections, per-head normalization, partial RoPE, attention K/V writes, stable causal softmax, weighted values and query-output gating.

Preserve the actual layout: each query head's projection carries both its query values and its gate values; key/value heads use grouped-query mapping. The scale in this profile is `1/sqrt(256) = 0.0625`. Do not rotate the gate half or all256 dimensions when the configured rotary dimension is64.

Write each input token's K/V at the correct position once. Initially retain FP16 K/V and FP32 accumulation. Avoid allocating a full context-by-context attention matrix. A simple bounded per-head decode implementation is adequate before optimizing segmented reductions. Chunked prefill requires its own causal-mask and position tests.

Test positions near capacity boundaries, multiple query heads sharing one KV head, RoPE boundaries and attention to old tokens. Include a populated prefix, not just a configured empty K/V allocation.

### 3E. Output selection and sampling

For greedy mode, add a deterministic full-vocabulary argmax with a defined tie rule consistent with the reference. Return one token ID after completion. Do not score a shortlist or omit EOS entries to improve timing.

For stochastic mode, first preserve the existing sampler by explicitly exporting the completed logits once per token. This costs about0.99MB for248320 FP32 logits and must be included in timing. Later, implement device top-k/top-p with a documented RNG contract and dedicated statistical tests if profiling justifies it. Moving sampling is not required to demonstrate correct greedy acceleration.

The existing `Sampler` recognizes CuPy-style engines via attributes such as `k` and `aout`. Do not spoof those attributes on Metal. Add a backend capability interface or keep an explicit CPU-logits sampling path. Record seed, temperature, top-p, top-k and repetition settings for both engines.

**Gate:** complete forward execution matches reference fixtures; extended recurrent/attention tests pass; internal activation readback is eliminated except deliberately retained diagnostics/sampling; memory remains inside the admitted budget.

## 9. Phase 4: command scheduling and kernel tuning

**Deliverable:** fewer submissions and waits with measured benefit and unchanged output semantics.

### Scheduling sequence

Start with one command buffer per layer while establishing device dependencies, then evaluate one per token. Allocate immutable per-dispatch parameters or use safe immediate parameter copies; do not overwrite a single shared parameter buffer while earlier GPU work still reads it. Ensure activation scratch reuse follows proven producer/consumer lifetimes.

Use explicit dependency handling appropriate to the installed Metal API, encoder type and resource hazard mode. Queue order alone is not a reason to omit required memory barriers. Retain tracked resources initially; untracked resources/heaps require separate hazard tests. Apple documents synchronization scopes and access conflicts. [Apple: resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization).

Do not merely remove `waitUntilCompleted` from the existing bridge. That implementation copies results back to CPU immediately and is therefore dependent on the wait. First move the consuming operations onto the GPU and encode the necessary ordering.

For a normal decode step, perform one host completion wait at the token-result boundary. Cache snapshots, debug exports and cancellation may add explicitly counted boundaries. Record those separately; a claim of one wait per token must specify the operational mode.

Long prefill chunks may need smaller submissions for responsiveness or driver execution limits. Test bounded alternatives instead of assuming the fewest possible command buffers always wins. No special Metal4-only feature is required for the initial design; advanced APIs must be feature-gated against the actual SDK/device. [Apple: feature tables](https://developer.apple.com/metal/capabilities/).

### Quantized kernel experiments

Run controlled, reversible ablations in this order:

| Experiment | Expected mechanism | Required evidence |
|---|---|---|
| Reuse input activation across gate/up or q/k/v projections | Reduce duplicated reads and dispatch overhead | Correct outputs, kernel time and total layer time |
| Tune rows per threadgroup and lane reductions | Improve occupancy/register use | Device capabilities, no out-of-bounds access, representative matrix sizes |
| Fuse local elementwise operations | Reduce activation traffic and small launches | Layer-state parity and measured end-to-end gain |
| Use bounded dequantized tiles for multi-token prefill | Amortize weight reads | Scratch budget, batch-causal correctness and prefill timing |
| Change row layout or tensor placement | Improve coalescing/reuse | New versioned pack format, lossless round trips and independent runtime validation |
| Change numerical precision or fast math | Possibly increase compute throughput | Explicit quality/parity review; not part of the default lossless optimization claim |

Keep decode GEMV and prefill GEMM results separate. A kernel that helps a batch of32 prompt tokens may hurt batch-one decode. Test actual large output-head matrices as well as small layer matrices. Kernel microbenchmarks must warm pipelines, include enough repetitions and account for synchronization correctly.

Estimate a bandwidth plausibility ceiling using the quantized bytes actually consumed per step, plus state/KV traffic. Do not blindly multiply the entire checkpoint size by tokens/s: dense embedding lookup reads one row, whereas the output head and transformer matrices have different access patterns. Measured effective bandwidth is a diagnostic, not a guarantee of attainable speed.

**Gate:** profiler counters demonstrate the intended scheduling change; at least five paired warm repetitions improve the relevant wall-clock metric; correctness and memory gates remain passed. Revert optimizations that merely move time outside the measured interval.

## 10. Phase 5: prefill and prefix-state reuse

**Deliverable:** efficient subsequent turns without corrupting recurrent state or duplicating large buffers.

### Bounded prefill

Implement a sequential GPU prefill baseline first. Then introduce chunk sizes1,4,8,16 and32 as measured alternatives. Chunk size is a backend memory/performance parameter, not the number of user requests.

Within a recurrent layer, states for tokens in a chunk remain causally dependent. Projection batching is possible, but each token's recurrence must consume the state after its predecessor. Full-attention prefill also requires correct causal masking. Do not parallelize state updates as if all tokens were independent.

Materialize only current-layer/current-chunk activations, using a scratch ledger. Keep checkpoint boundaries and cancellation points defined between completed chunks. If a chunk fails after partially mutating state, invalidate it and restore a committed checkpoint; a failed chunk cannot be treated as completed prefill.

### Prefix snapshots and cache identity

The existing cache stores recurrent matrices, convolution history and relevant attention-prefix state. Adapt it to explicit backend snapshot/restore operations rather than relying on NumPy/CuPy `.set()` conventions for Metal-owned buffers.

Required cache-key fields include model/pack identity, tokenizer/template identity where relevant, backend and arithmetic version, K/V and recurrent dtypes, layout, RoPE settings and compatible capacity. Source hashes currently contribute to `engine_fingerprint`; extend the list for every new Metal arithmetic source and numerical configuration. Never resume CUDA/CPU/Metal states interchangeably without explicit compatibility validation.

Snapshot after completed GPU writes. For asynchronous disk serialization, give the writer an immutable bounded snapshot; do not let the next token mutate the same storage. Account for live state, snapshot staging, serialization buffers and cached copies simultaneously. Cap retained disk entries and reject malformed/incompatible metadata.

Required integration cases:

- Identical prompt repeated: same output, nonzero reused tokens.
- Append-only conversation: correct suffix processing.
- Prefix divergence: restore at or before the first differing token and reprocess the suffix.
- A changed system prompt or tool schema: no stale prefix reuse.
- Process restart with a saved opening: correct restore and continuation.
- Reset/uncached execution: same result as restored execution.
- Budget-driven checkpoint eviction: correctness preserved even when speed regresses.
- Cancellation and client disconnect: no invalid state saved as a reusable checkpoint.

Persistent prefix loading does not make cold model weights resident automatically. Report cached prompt tokens, prefill time and weight startup time separately. A cache hit after restart may still have long TTFT from weight loading.

**Gate:** cache-enabled and cache-disabled outputs agree on the same backend; repeated-turn latency improves; checkpoint activity stays within the total allowance. This phase improves prompt processing and does not itself establish faster token generation.

## 11. Phase 6: application integration

**Deliverable:** a stable single-user OpenAI-compatible endpoint with truthful backend and memory reporting.

After the earlier gates pass, the existing bridge can be launched using current commands:

```bash
export KINDI_DEVICE=metal
export KINDI_Q4_STREAM=0
export KINDI_PARK_MAX=0
export KINDI_CHECKPOINT_BUDGET_MB=192
.venv-kindi-q4/bin/python -m kindi --pack "$PACK" --ctx 512 \
  --state-dir "$EXPERIMENT/state" serve --host 127.0.0.1 --port 8093
```

This command selects the **current bridge**, not the proposed complete backend. Do not invent a launch command for the complete backend before implementing its dispatcher and documenting its configuration. Run any server under the Mac monitor added in Phase2. There is no built-in hard macOS8GB cap in this command.

The client uses base URL `http://127.0.0.1:8093/v1` and model ID `oxcoder9b-q4-kindi`. Set one active request and disable automatic retry storms. Use a timeout appropriate for measured cold prefill; keep cancellation responsive. Advertise the tested capacity through server metadata rather than the theoretical model context limit.

Extend health/metrics output with actual backend identifier, chip, capacity, current populated context, memory-accounting status, prefix-cache counts and any CPU fallback. A fallback must be visible in logs and benchmark artifacts. Do not present a CPU fallback result as Metal throughput.

Reuse the existing smoke test after monitoring is ready:

```bash
.venv-kindi-q4/bin/python tests/kindi_q4/server_smoke.py \
  "$PACK" "$EXPERIMENT/server-smoke.json"
```

It starts its own server on localhost18193, tests repeated completion and SSE, restarts the server and checks disk-prefix reuse. Ensure that port is free and no other model process is resident. The test does not establish tool-call quality, large-context performance or memory compliance by itself.

Add controlled agent tests: a small file edit, a tool-call/return cycle, a multi-file change with hidden tests, a repair after a command error and a longer conversation that changes retrieved files. Record pass/fail, tool-call schema validity, number of steps, total generated tokens, full task time and memory. Execute generated code only in the project's existing sandbox/test isolation, not by blindly evaluating model output in the inference process.

## 12. Benchmark protocol and selection

### Variant matrix

Keep each variant identifiable and change one factor at a time:

| ID | Variant | Purpose |
|---|---|---|
| L0 | Installed llama.cpp Metal, same GGUF | Practical same-machine baseline |
| C0 | Kindi native CPU | Numerical/reference diagnostic, not a Mac speed target |
| M0 | Existing synchronous Metal bridge | Establish cost of per-matrix CPU/GPU handoffs |
| M1 | Complete Metal computation, conservative scheduling | Isolate moving CPU model operations onto GPU |
| M2 | M1 with grouped command submission | Isolate scheduling improvement |
| M3 | M2 with one accepted kernel/fusion change | Isolate kernel optimization |
| P1 | Best decode variant with bounded chunked prefill | Measure prompt processing independently |
| S1 | P1 with recurrent/prefix snapshots | Measure repeated-turn latency and snapshot overhead |

M1–S1 are planned variants, not implementations currently present. Do not label a variant M2 just because an environment variable was set; record the source changes and profiler evidence.

### Workloads

Run the smallest case first; advance only within the admitted memory and runtime envelope.

| Test | Actual prompt workload | Output limit / purpose |
|---|---|---|
| Reference fixtures | Checked-in raw token sequences | Exact short continuations through EOS |
| Short decode | 128–512 populated tokens | 128 outputs; stop normally at EOS |
| Medium prompt | About4K actual tokens | 128 outputs; prefill and warm decode |
| Larger prompt | About16K actual tokens, only after admission | Same output limit; prefill/memory effects |
| Typical coding task | Real instructions, tool schemas and code | Task completion, not a fixed throughput score |
| Multi-turn | At least10 representative tool/chat turns | Cache reuse, prompt growth and leaks |
| Sustained session | Several minutes after short cases pass | Thermal behavior, memory growth and responsiveness |

The current probe caps output requests at128. Do not quietly change that cap for a sustained test; build a bounded multi-request harness or an explicitly guarded long-run mode. Exact4K/16K inputs require capacity for prompt plus output and template overhead; a4096-token prompt plus128 outputs does not fit capacity4096.

Use tokenizer counts, not approximate word counts. At least five measured warm repetitions per selected case are required after one warmup. Alternate baseline and candidate runs to reduce thermal/order bias, unloading each model before the next. Report all observations, median and range; a five-sample p95 is not a robust tail-latency estimate. Use a larger sample when making p95 claims.

A “cold process” may still have OS-cached file pages. Label weight/cache state and measured I/O where possible. Do not globally purge caches to manufacture a cold result on an actively used workstation.

### Minimum result artifact

The following is an example schema with placeholders, **not measured results**:

```json
{
  "status": "example-schema-not-a-result",
  "variant": "M2",
  "source_revision": null,
  "source_file_hashes": {},
  "model_sha256": null,
  "pack_sha256": null,
  "machine": {"chip": null, "gpu_cores": null, "ram_bytes": null},
  "platform": {"macos": null, "sdk": null, "compiler": null},
  "runtime": {
    "backend": null,
    "context_capacity": null,
    "populated_prompt_tokens": null,
    "reused_prompt_tokens": null,
    "kv_dtype": "f16",
    "recurrent_dtype": "f32",
    "prefill_chunk": null,
    "temperature": 0,
    "top_p": null,
    "top_k": null,
    "seed": null
  },
  "timing": {
    "load_s": null,
    "request_ttft_s": null,
    "prefill_s": null,
    "decode_tokens_per_s": null,
    "completion_s": null
  },
  "output": {"tokens": null, "finish_reason": null, "reference_pass": null},
  "scheduling": {"commands_per_token": null, "host_waits_per_token": null},
  "memory": {
    "allowance_bytes": 8000000000,
    "peak_phys_footprint": null,
    "peak_rss": null,
    "peak_metal_allocated": null,
    "unique_owned_bytes": null,
    "file_backed_residency_accounting": null,
    "snapshot_peak_bytes": null,
    "swap_delta_bytes": null,
    "pressure_state": null,
    "sampling_interval_ms": null,
    "compliance_verified": false
  },
  "limitations": []
}
```

### Selection rule

Treat these as project acceptance criteria, not externally established performance laws:

1. Reject any candidate with unexplained reference failures, corrupted snapshots, unsupported fallback, memory violation or sustained paging.
2. Require the selected candidate to improve median warm decode by at least10% over L0 on repeated comparable workloads, with a gain larger than ordinary run-to-run variation, before recommending it as a generation-speed upgrade.
3. Also inspect cold TTFT, warm TTFT and complete coding-task time. A decode winner that worsens typical task time should not become the default. If prefill is the main benefit, describe it as a prefill improvement instead.
4. Reject a configuration that meets speed goals only by consuming memory needed by normal applications, reducing tested context silently or changing model/sampling quality.
5. If no candidate wins reliably, keep llama.cpp as the recommended runtime and retain the experimental backend and measurements for further work.

No fixed tokens/s target is promised. The decision is comparative and machine-specific.

## 13. Failure diagnosis and rollback

| Symptom | First checks | Response |
|---|---|---|
| Compile or shader creation failure | SDK, ARM64 interpreter/compiler, function name, type sizes, pipeline limits | Save diagnostics; fix build without weakening tensor validation |
| No-copy buffer creation returns nil | Page alignment, length, single VM region, lifetime, `maxBufferLength` | Fail explicitly; any copy fallback needs a fresh memory admission |
| Correct first token, later divergence | Recurrent update order, convolution history, K/V position, reset semantics | Compare layer states across a multi-token sequence |
| Repeated nonsensical text | Norm/projection mapping, GDN head ordering, omitted residual, invalid logits | Stop speed tests; restore reference execution |
| CPU tests pass but Metal is wrong | Confirm test actually calls Metal, then decode/type/layout/reduction checks | Add direct Metal tests; do not relabel CPU validation |
| GPU appears intermittently idle | Per-matrix host waits, CPU gates/attention, Python dispatch overhead | Inspect command timeline; complete device ownership before deleting waits |
| Faster kernel, unchanged request latency | Prefill, tokenizer, cache miss, sampling/readback or CPU bottleneck | Measure full path and separate stages |
| High warm tokens/s, poor multi-turn latency | Prefix mismatch, incompatible snapshots, changing templates | Inspect reused-token counts and serialized prefix identity |
| Repeated disk reads in warm decode | Weight duplication, eviction, pressure or insufficient budget | Reduce capacity/overhead; disqualify resident-memory claim |
| Memory climbs after each request | Buffer retain cycles, snapshot staging, model handles, autorelease lifetime | Bound caches and add repeated-reset/reload leak tests |
| Only long-context cases fail | Attention indexing, chunk masks, scratch sizing and K/V budget | Stop capacity growth; reproduce at the earliest failing position |
| Good initial speed, poor sustained speed | Thermal/power state and competing GPU work | Report sustained values; do not publish the cold peak alone |

Keep changes phase-scoped and preserve a known-good checkpoint of source/configuration after each gate. Record rejected approaches and why they failed in the experiment report, not as enabled production options. Rollback means stopping the owned experiment and restoring the prior runtime/configuration; it does not mean changing global system memory policy.

## 14. Completion checklist

A maintainer or coding agent should deliver:

- A complete Metal backend with owned resources, explicit dispatch and teardown.
- Validated Q4_K/Q6_K kernels plus recurrent, attention, FFN and sampling tests.
- Direct Metal tests that cannot silently execute the CPU path.
- Allocation admission, Mac telemetry, bounded cancellation and clear memory-accounting limitations.
- Reference outputs, layer comparisons for investigated mismatches and immutable model/source hashes.
- Profiling evidence that per-matrix host waits were removed correctly.
- Bounded prefill and recurrent/prefix snapshot integration with restart tests.
- Same-machine llama.cpp comparison across short, populated-context and multi-turn workloads.
- A technical result report stating accepted and rejected configurations, including cases where no speedup was obtained.
- Updated README selection commands and dependency/platform requirements for the implementation actually shipped.
- Updated `SOURCE-SHA256.txt` and a separately versioned source bundle if distributing a new archive; historical archives remain historical.

The existing AGPL license and provenance notices apply to derivative implementation files. Do not package model weights without preserving their separate publisher terms. This plan itself does not alter model files, train a model, or activate SSD offloading.

## 15. Primary references

The plan's runtime structure and numerical details come from the linked repository files. External platform behavior was checked against these primary sources on 2026-10-26:

- [Apple Metal command-buffer best practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html): submission frequency and CPU/GPU coordination.
- [Apple command structure](https://developer.apple.com/documentation/Metal/setting-up-a-command-structure): object reuse, encoding and command-buffer lifetime.
- [Apple no-copy buffer creation](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29?language=objc): backing-memory requirements.
- [Apple current allocated size](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize) and [recommended working set](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize): Metal allocation telemetry and its scope.
- [Apple resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization): GPU resource dependency handling.
- [Apple Metal performance analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/) and [GPU start time](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime): profiling and timing.
- [Apple task-info tests](https://github.com/apple/darwin-xnu/blob/main/tests/task_info.c): process VM/footprint query behavior.
- [Apple Metal capability tables](https://developer.apple.com/metal/capabilities/): feature gates for optional API extensions.

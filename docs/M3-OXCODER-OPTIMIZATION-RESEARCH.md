# OxCoder9B on M3: acceleration research and implementation procedure

Research date: 2026-11-04. Status: external evidence and an implementation
specification; no OxCoder M3 benchmark has been performed by this repository.

## Objective and acceptance boundary

Serve the actual OxCoder9B coding model on an M3-family Mac with 18 GB unified
memory, while company applications remain running. The inference deployment has
an allowance of **8,000,000,000 bytes**, including the server and its helpers.
The reported existing GGUF deployment generates approximately 18 output tokens/s.
That observation needs a controlled local baseline before ratios are claimed.

The target tiers are at least 30, 40, and then higher output tokens/s. A tier
qualifies only on a declared workload, with correct results and memory within
the allowance. A single short response is insufficient evidence of sustained
coding-agent throughput. The existing configured 128K context remains available
as its own profile; configured capacity and populated context are different.

This guide supplements [the full runtime evaluation specification](MAC-RUNTIME-EVALUATION.md).
Its phases apply to GGUF and native Metal alternatives. No model training is
required. Keep source weights, tokenizer and chat template identifiable across
all trials. A catalog Qwen benchmark is a reference control, never an OxCoder
result. The native Kindi Q4 path has its own
[Metal handoff](../scripts/kindi_q4/METAL_IMPLEMENTATION.md).

## Directly relevant external evidence

### Kapitsa's M3 Pro comparison

The engine authors published an M3 Pro comparison on a 36 GB machine with
18 GPU cores, AC power, macOS 14.1 and a controlled display/background load.
Their protocol takes the median of five 512-token coding answers.

| Qwen3.5-9B engine | Decode tokens/s | Greedy tokens/s |
|---|---:|---:|
| Kapitsa | 77.5 | 84.2 |
| mlx-lm | 26.7 | 27.4 |
| llama.cpp, unsloth Q4_K_M | 23.4 | 23.4 |

The separate prompt-length sweep reports Kapitsa at 64.5 tokens/s at 2K,
60.5 at 4K, 50.4 at 16K and 46.9 at 32K. These are author measurements on base
Qwen, different engine configurations and potentially different weight layouts;
they do not isolate a single kernel's effect. Tool follow-up with thinking on
failed in the cited Kapitsa record, despite valid throughput phases.

Sources: [protocol, results and limitations][m3-results],
[rival versions and commands][rivals]. Pin the relevant source revision when
reproducing; moving default configurations may change decoding methods.

### Memory interpretation: a correction to the rounded table

The [raw record][m3-raw] reports a decode-phase peak physical footprint of
8,541,440,192 bytes. That is about 8.54 decimal GB or 7.96 GiB. Its tool phase
peaks at 8,554,088,064 bytes. Both exceed this repository's **8.00 decimal GB**
allowance. The table's rounded “8.0 GB” must not be used as proof of compliance.

The record reports no additional swapouts during measured phases. Existing
system swap usage was nonzero, which alone does not prove inference was swapping.
The memory sample and a system with more total RAM do not establish equivalent
behavior alongside company applications on an 18 GB Mac. Download size is also
not peak physical footprint. Conversion, loading and sustained operation each
need measurement.

### Why Kapitsa deserves an OxCoder trial

[Kapitsa's architecture][architecture] is native C++20 with Metal backends. It
uses neither the MLX nor llama.cpp runtime. Reading an MLX-format checkpoint does
not make its execution MLX-based. Its generation path supports draft verification
and recurrent-state handling; `--spec 0` provides the plain-decoding control.
Selected Neural Engine work applies to prefill, rather than a promise of faster
output generation. These are engine capabilities, not verified OxCoder support.

### Other candidates and counterevidence

- [Rapid-MLX][rapid] reports approximately 1.5x typical and up to 4x decode
  acceleration for Qwen3.5-9B on an M4 Pro. Its M5 figures cannot be transplanted
  to an M3. Treat it as a separate MLX runtime candidate.
- [DFlash's Apple implementation][dflash] includes small-batch verification
  kernels and quantized drafters. Verify both implementation compatibility and
  actual acceptance with OxCoder. Base-Qwen compatibility alone is insufficient.
- A [llama.cpp Metal MTP report][mtp-regression] measured slower decoding for
  Qwen3.5-9B on an older build. A [small-batch Metal investigation][metal-small]
  identified verification overhead. Together they justify testing current
  versions; neither establishes a current OxCoder speedup.
- [Strata][strata] focuses on MoE residency and movement. OxCoder9B is dense.
  Its expert cache has no direct role here; keeping the dense weights resident
  is the relevant starting point.

## Hardware interpretation

Confirm the exact chip with `system_profiler SPHardwareDataType`. An 18 GB
configuration suggests M3 Pro, but record the machine's actual identity rather
than inferring it. Apple lists 150 GB/s bandwidth for M3 Pro in its
[technical specifications][apple]. M3, M3 Pro and M3 Max have different limits.

For intuition only, dividing 150 GB/s by roughly 5–6 GB of weights gives a
25–30-step/s idealized bandwidth estimate for a full weight read per serial
step. It ignores cache behavior, repeated reads, quantization overhead and
other work; it is not a rigorous bound or an expected benchmark score.
Speculative verification can amortize a weight read over several accepted
output tokens, so output tokens/s can exceed serial steps/s. Acceptance,
verification cost and additional memory determine whether it helps.

## Phase 0: preserve a reproducible baseline

Create a new run directory. Record hardware, OS build, power state, engine commit,
launch command, model SHA-256, tokenizer/template revision, process IDs, context
settings and cache types. Do not silently update the working deployment first.

```bash
mkdir -p runs/oxcoder-m3/baseline
system_profiler SPHardwareDataType > runs/oxcoder-m3/baseline/hardware.txt
sw_vers > runs/oxcoder-m3/baseline/os.txt
pmset -g > runs/oxcoder-m3/baseline/power.txt
shasum -a 256 /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  > runs/oxcoder-m3/baseline/model.sha256
/path/to/llama-server --version > runs/oxcoder-m3/baseline/engine.txt 2>&1
/path/to/llama-server --help > runs/oxcoder-m3/baseline/help.txt 2>&1
```

Expected existing mirror SHA-256:
`a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.
Record AC/battery mode, thermals and ordinary application load. Benchmark one
inference server at a time. Stop only experiment-owned processes between arms.

## Phase 1: current GGUF and Metal

Build or install a pinned current llama.cpp with Metal support. Save build
configuration and backend logs. Confirm actual GPU layer placement and selected
kernels; a binary called llama-server does not establish GPU execution.

Use a short-context comparison profile before the long-context profile. Suggested
initial settings are one slot, 4096 context, physical batch 64 or 128, logical
batch 256, all supported layers on GPU, and Flash Attention on versus off. Check
that each flag exists in that exact binary; hardware/backend support can differ.
Do not copy the Linux sweep's CUDA device selection or its load flags blindly.

Change one factor per arm: backend revision; Flash Attention; physical batch;
then supported cache precision. Preserve model and requests. Quantized caches
may save memory, but must pass output checks. If an option is unsupported,
record rejection and continue with the last valid profile. Never mark a failed
launch as a zero-speed result in an average.

Exit condition: a reproducible baseline with complete output, measured memory,
and generation/prefill timings. If it already reaches a target tier, run the
sustained and quality phases before adopting it.

## Phase 2: inspect Kapitsa import before conversion

Read and pin [Kapitsa's usage guide][usage], installation source, version and
license. Save local help for `import`, `serve` and `doctor`. Use a separate port
and conversion directory/name. First inspect the GGUF:

```bash
kapitsa import /absolute/path/OxCoder-9B.Q4_K_M.gguf --dry-run
```

Check architecture acceptance, tensor formats, tokenizer, template and predicted
size. The documented importer handles Q4_K tensors differently from Q6_K and
Q8_0 tensors. Q4_K_M is mixed precision; therefore **importing the file does not
by itself prove bit-exact weights or numerically equivalent outputs**. Audit the
conversion log and actual resulting formats. Preserve the original GGUF.

Then use the installed version's documented import command with a unique model
name. Record output paths, sizes and checksums. Verify exported configuration
against OxCoder's layer count, hidden dimensions, vocabulary and special tokens.
A successful import is only structural admission.

The published generic commands include `kapitsa doctor --model MODEL --json`
and `kapitsa serve --model MODEL`. Use the printed imported model name or its
verified path, rather than `9b`, which resolves a catalog model. Consult local
help for budget, context, slots, prefill chunks and draft selection before launch.

## Phase 3: plain native Metal control

Serve imported OxCoder with speculation disabled (`--spec 0`) and explicitly
prevent automatic loading of an unrelated drafter (the usage guide documents
`--dflash none`). Inspect logs to confirm the actual execution mode. Use one
conversation and a small context first.

Set an engine memory budget conservatively below the deployment limit, then
measure whole-process physical footprint separately. An engine budget can govern
only GPU allocations or planned caches; it is not an OS-enforced cap for all
server/helper memory. The precise semantics must be checked for the pinned build.

Run the same prompts and output limits as the GGUF baseline. Different backends
may diverge numerically at near ties; distinguish such divergence from broken
conversion. Record token IDs if exposed, rendered text, stop reasons and executable
results. Test tokenizer/template round trips and tool message formatting before
running a coding agent.

## Phase 4: speculation, admitted separately

Try an existing compatible quantized drafter only after the target works plainly.
Inventory the existing OxCoder file for MTP tensors before enabling MTP; never
assume they were retained by its publisher. For DFlash, verify target dimensions,
layer taps, vocabulary/tokenizer agreement and supported recurrent architecture.
Measure accepted tokens per verification cycle and time spent drafting/verifying.

Compare plain versus speculative execution at temperature zero and at the normal
serving settings. Start with the backend's supported short draft length; test a
small fixed sweep instead of assuming longer drafts are faster. Include ordinary
new code, copying/editing existing code, prose and tool calls. Speculation may win
only on some workloads. Prompt lookup needs no extra weight file and deserves
its own edit/copy arm; do not infer its performance from a fresh-code prompt.

Reject configurations that exceed memory, corrupt output, lose required stop/tool
semantics, or are slower on the intended workload. Freeze the winning settings
before final evaluation. Keep the plain profile as a fallback.

## Phase 5: cache and context qualification

Measure prompt processing and decode separately. Prefix caching can reduce time
to first token on repeated agent turns without changing output tokens/s. Include:

1. Cold system prompt plus tools.
2. Identical prefix with a small appended user/tool message.
3. An intentional earlier-prefix edit.
4. Alternating conversations.
5. Restart and restore, if disk snapshots are enabled.

Hybrid recurrent models need valid recurrent-state checkpoints as well as
attention caches. Verify restored and cold execution on the same inputs. Report
actual reused tokens, uncached tokens and first-token latency. Any SSD prefix
snapshot feature stores conversation state; it is separate from SSD weight
streaming. This OxCoder work targets resident weights.

Test populated context at 1K, 4K, 16K, then 32K and 128K only if memory permits.
Include output headroom. Preserve all over-budget outcomes rather than silently
reducing context and calling the result a successful 128K run.

## Phase 6: measurement and selection contract

For every arm, collect at least five measured requests after a declared warmup,
then a sustained multi-minute run. Freeze prompt files and seeds. Keep cold and
warm measurements separate. Record median, range and slow tail; do not select
one unusually favorable answer.

Record bytes rather than rounded “GB” labels. Monitor loading, prefill, decode,
tool turns and idle retained caches. Include child processes and simultaneous
resident helper models. Prefer macOS physical footprint sources; report RSS if
that is all available, and identify the limitation. Record system swap counters
before and after, memory pressure, and any compression growth. Existing swap
usage alone is not evidence of current paging. A sampled monitor may miss short
peaks, so record its interval and avoid claims of a hard cap unless enforced.

The manifest should contain these fields:

```json
{
  "model": "OxCoder9B",
  "engine_commit": "PINNED_REVISION",
  "model_sha256": "CHECKSUM",
  "conversion_identity": "SOURCE_AND_FORMAT_DETAILS",
  "chip": "ACTUAL_CHIP",
  "budget_bytes": 8000000000,
  "configured_context": 4096,
  "actual_prompt_tokens": 0,
  "generated_tokens": 0,
  "decoding_mode": "plain_OR_SPEC_METHOD",
  "median_decode_tps": null,
  "prefill_tps": null,
  "ttft_seconds": null,
  "peak_deployment_bytes": null,
  "memory_measurement_method": "REQUIRED",
  "swapouts_delta": null,
  "functional_checks_passed": null,
  "complete_response": null,
  "qualified": false
}
```

Executable checks must cover normal and boundary cases, edits preserving existing
behavior, and required tool-call round trips. Speed from a truncated answer is
usable as a throughput sample only; it is not task completion. Capture extra
prose and formatting failures even when an extracted function passes.

Decision order: model identity and correctness, memory, completed-task latency,
then sustained generation speed. Claim 30/40 tokens/s only for the workload and
context that actually passed. If all configurations fail a tier, publish the
measured winner and its limits; do not substitute a smaller model silently.

## Deterministic execution and handoff

An implementing code agent should maintain a finite, versioned experiment matrix.
Each row specifies the binary hash, model/conversion hash, command, prompts,
seeds, context and timeout. A runner launches one arm, waits for readiness,
monitors resources, sends the fixed requests, writes results atomically, stops
only its own server and advances. Resume only completed rows whose identities
match; partial results require a new attempt ID.

Failure decisions should be explicit: unsupported option → reject that arm;
memory breach → stop and reduce cache/prefill allocation in a separately named
arm; low acceptance → shorten/disable drafting; correctness failure → reject
conversion or speculative mode; cache miss → inspect prefix/state handling.
Never change multiple factors without identifying a new recipe. Record the
reason and parent configuration for every change. This reduces repeated manual
or model-mediated monitoring while retaining an auditable experiment history.

The existing [Linux sweep](OXCODER-8GB-SPEED.md) demonstrated approximately 68
output tokens/s on an RTX 5060 Ti under a sampled combined memory budget. It is
not a Mac runner: replace CUDA process accounting, GPU selection and systemd
supervision with macOS implementations before using it here.

## Sources

[m3-results]: https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-10-22-m3pro-36gb/README.md
[m3-raw]: https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-10-22-m3pro-36gb/results/sw-q35-9b-kapitsa-20261022-231956.json
[rivals]: https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-10-22-m3pro-36gb/RIVALS.md
[architecture]: https://github.com/nace-ai/kapitsa/blob/main/docs/architecture.md
[usage]: https://github.com/nace-ai/kapitsa/blob/main/docs/usage.md
[rapid]: https://github.com/raullenchai/Rapid-MLX
[dflash]: https://github.com/z-lab/dflash-mlx-fork
[mtp-regression]: https://github.com/ggml-org/llama.cpp/issues/23752
[metal-small]: https://github.com/ggml-org/llama.cpp/issues/25250
[strata]: https://github.com/ro99/strata
[apple]: https://support.apple.com/en-ie/117737

External projects change rapidly. Resolve and save the exact revisions before
implementation; these links identify the research sources, not immutable builds.

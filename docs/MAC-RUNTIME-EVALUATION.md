# OxCoder-9B: Mac runtime implementation and evaluation

Specification date: **2026-10-23**. Status: **implementation plan; target-Mac results pending**.

## 1. Objective and deployment boundary

Evaluate a second inference implementation for OxCoder-9B without replacing the
working GGUF deployment until the alternative passes memory, correctness, latency,
and coding-workflow tests. The primary candidate is a native MLX 4-bit checkpoint
served by Kapitsa. Test speculative decoding only after plain decoding works.
Evaluate oMLX separately where persistent prefix caching or its kernels provide
a better result. These are alternative server processes, not packages to stack
inside one inference request.

The delivery target is an M3-family Mac with 18GB unified memory. Existing company
applications consume approximately 6.5–8GB, and available memory varies. Allocate
at most **8,000,000,000 bytes** to the inference deployment, including its supporting
processes. Do not use a Linux workstation's discrete GPU or larger RAM capacity
as evidence that a configuration meets the Mac requirement.

No model training is part of this implementation. Use existing checkpoints and
compatible existing drafters. Do not train a drafter, silently change the target
model, or substitute a generic Qwen checkpoint for OxCoder to obtain a faster score.

The existing deployment reports approximately **17 output tokens/s** and supports
a configured **128K context window**. Treat those as operator-reported properties.
The backend version, cache precision, actual occupied context, and measurement
method must be captured before this becomes the controlled baseline. A configured
128K limit does not establish that 128K tokens were populated during that speed test.

### Deployment profiles

| Profile | Purpose | Initial status |
| --- | --- | --- |
| `gguf-existing` | Preserve current binary, model, launch settings and 128K configuration | Working according to operator; controlled measurements pending |
| `gguf-evaluation` | Same GGUF with explicit short-context settings; optionally newer Metal backend | Pending |
| `mlx-kapitsa-plain` | Native MLX OxCoder, speculation disabled | Pending |
| `mlx-kapitsa-spec` | Same MLX target plus a verified compatible drafter or MTP head | Pending |
| `mlx-omlx` | Same MLX target on a separate serving/cache implementation | Optional, pending |
| `daily-agent` | Winning engine, fixed harness, measured task completion | Pending |
| `long-context` | Largest separately qualified populated context; preserve original 128K fallback | Pending for new engines |

## 2. What changes, and what does not

OxCoder is a **dense Qwen3.5-family model with hybrid attention/recurrent layers**.
It is not the earlier 35B MoE deployment. Packed expert reads, expert residency,
and GPU-managed expert pools are not directly applicable to this dense model.
Loading this checkpoint through an MoE offloading design does not create experts.

The useful optimization axes are:

1. Efficient native quantized matrix and recurrent-state kernels.
2. Keeping the active weights and state resident within the memory allowance.
3. Processing several accepted tokens per target verification pass, if speculation pays.
4. Reusing unchanged prompt prefixes and restoring the complete hybrid model state.
5. Reducing unnecessary agent turns without weakening task validation.

SSD storage is useful for checkpoint files and persisted prefix states. Reading
dense weights from SSD repeatedly during decode is not a target operating mode.
OS swap is not an inference optimization. Neither a fast database nor SSD cache
removes the need to hold the active tensors required by a forward pass.

Harness² and a remote Opus improver operate above the inference engine. They can
change procedures, prompt length and attempt count. They do not accelerate the
local model's matrix operations. Record their contribution using successful-task
latency and cost, separately from local output throughput.

## 3. Checkpoints and reproducibility

### Existing GGUF

- File: `OxCoder-9B.Q4_K_M.gguf`.
- Size: `5629117536` bytes, approximately 5.24 GiB.
- SHA-256: `a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.
- Converter source: `prithivMLmods/OxCoder-9B-GGUF` at
  `d108751ef79e645b07d04030427b6b67b284d07e`.
- Download/reassembly: retain the [repository release procedure](../README.md).

### Native MLX candidate

Use [OmAr-Kader-DeV/OxCoder-9B-mlx-4Bit](https://huggingface.co/OmAr-Kader-DeV/OxCoder-9B-mlx-4Bit/tree/ae7575d8655be764ddf8478e5f04caeea74ec4ce),
pinned to `ae7575d8655be764ddf8478e5f04caeea74ec4ce`.
The downloaded configuration identifies `qwen3_5` and affine quantization with
`bits=4`, `group_size=64`. Repository files are approximately 5.04GB; this is not
the runtime footprint. Save the original model card and configuration with results.

Download directly on the destination machine, or transfer the exact downloaded
directory. Do not convert a full-precision 9B model within the 8GB allowance.
Install the Hugging Face CLI in an isolated environment if it is not available;
record the installed version. The following assumes `hf` is already on PATH:

```bash
OX_WORK="$HOME/oxcoder-evaluation"
OX_MODEL_DIR="$OX_WORK/models/OxCoder-9B-mlx-4Bit"
mkdir -p "$OX_WORK/models" "$OX_WORK/runs" "$OX_WORK/profiles"
hf download OmAr-Kader-DeV/OxCoder-9B-mlx-4Bit \
  --revision ae7575d8655be764ddf8478e5f04caeea74ec4ce \
  --local-dir "$OX_MODEL_DIR"
```

Verify configuration without loading model tensors:

```bash
python3 - "$OX_MODEL_DIR" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cfg = json.loads((root / "config.json").read_text())
assert cfg["model_type"] == "qwen3_5", cfg.get("model_type")
q = cfg["quantization"]
assert (q["bits"], q["group_size"], q["mode"]) == (4, 64, "affine"), q
rows = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or ".cache" in path.relative_to(root).parts:
        continue
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    rows.append({"file": str(path.relative_to(root)),
                 "bytes": path.stat().st_size, "sha256": h.hexdigest()})
(root.parent / "oxcoder-mlx-file-manifest.json").write_text(
    json.dumps(rows, indent=2) + "\n")
print("Validated quantization; wrote file manifest. No model was loaded.")
PY
```

The MLX and GGUF conversions use different quantization schemes. They may also
have different conversion provenance. Establish the original checkpoint revision
where possible; otherwise record it as unknown. A model-card revision alone does
not prove which original weight revision was converted. Cross-format evaluation
measures a deployment combination, not a pure engine-only speed difference.

The configuration's `mtp_num_hidden_layers` field does not prove usable MTP weights
are present. Inspect the tensor index or safetensors headers and the loader's
mapping before enabling MTP. Do not deserialize full tensors merely to list keys.
Also verify whether vision tensors are loaded; this is a text-only evaluation,
and unused vision allocations must still be counted if the runtime retains them.

## 4. Memory contract and context accounting

### Units and scope

Use bytes in all machine-readable results. This specification's 8GB ceiling is
**8.0 decimal GB**, approximately **7.45 GiB**. It is not an 8 GiB allowance.
Record upstream units explicitly; do not relabel a vendor's “8GB” result without
knowing how that measurement was collected.

Include target weights, draft weights, attention KV state, recurrent state,
verification/rollback state, prefill workspaces, allocator caches, tokenization,
server overhead, helper processes and local agent/monitor overhead. Do not add
Metal allocation bytes to process footprint indiscriminately: unified-memory
allocations can appear in both. Report the categories separately alongside a
process-tree footprint measurement.

Physical footprint can omit some reclaimable, clean file-backed resident pages.
For a memory-mapped GGUF, also inspect resident model-file mappings with a verified
VM-region collector. Add resident pages excluded from footprint to the accounted
deployment total, without counting the same Metal-backed pages twice. Record
physical footprint and this corrected total separately. Virtual mapping size is
not resident memory. If the collector cannot establish this accounting, the strict
8GB qualification remains unverified even when the footprint counter is low.

Kapitsa's `--mem-budget` limits its GPU allocation budget. It does not establish a
whole-process ceiling. oMLX memory guards also require independent observation.
Neither is equivalent to a Linux cgroup memory limit. A macOS sampled watchdog
cannot guarantee that no transient allocation ever crosses 8GB.

### Required monitor, implemented before loading candidates

Implement a supervisor with the following contract, or use an existing tool that
meets it. Do not label a run memory-qualified until this evidence exists.

1. Start sampling before server launch, through model loading, generation, cache
   writes, idle retention and shutdown. Sample every 250ms; record actual intervals.
2. Track only the launched process tree using PID plus process start time. Include
   helper processes and the local harness. Do not discover candidates using broad
   process-name matches and then terminate unrelated applications.
3. Obtain macOS physical footprint through supported process accounting, such as
   `proc_pid_rusage` with a versioned `rusage_info` structure, or an equivalently
   verified native collector. RSS alone is insufficient, particularly under
   compression. Verify ABI/field definitions against the installed macOS SDK.
4. Record each PID's footprint and RSS, engine allocation counters if available,
   system memory pressure, compressed memory and swap-in/swap-out deltas. A sum of
   per-process footprints can conservatively double-count shared allocations;
   disclose the accounting method rather than subtracting unverified estimates.
5. Apply limits to the accounted deployment total, including resident model pages
   omitted from footprint. Warning threshold: 7.2 billion bytes. Abort threshold:
   7.6 billion bytes, leaving
   response margin below 8 billion. These are operational thresholds, not OS caps.
   A candidate that cannot run with this headroom is not qualified for this profile.
6. Abort for sustained yellow/red pressure, an attributable sustained rise in swap
   activity, or workstation responsiveness loss. Record competing application load.
   Pre-existing swap usage alone is not evidence of new model swapping.
7. On abort, cancel the owned request, terminate the owned server gracefully, then
   terminate remaining owned children if necessary. Preserve logs and mark the run
   `aborted`; never turn an aborted short response into a throughput success.
8. Test the supervisor with a small bounded dummy process and a deliberately low
   test threshold before supervising the model. Verify child tracking, cleanup,
   sample gaps and that unrelated processes are untouched. Do not stress the whole Mac.

If footprint collection is unavailable, classify memory status as **unverified**.
Report “observed below the operational threshold” when that is the evidence;
reserve “hard bounded” for an implementation that actually prevents allocations
above the bound. Do not change system-wide Metal memory limits, disable company
applications, or run cache-purge utilities to manufacture a passing result.
Track attributable allocations in external XPC/accelerator services where used;
they may not be descendants of the server PID. Do not kill shared macOS services.
The initial Kapitsa profile disables Neural Engine offload to reduce this ambiguity;
enable it only as a separate measured candidate with adequate accounting.

### Why populated context must be measured

The pinned MLX configuration has 32 text layers, with 8 full-attention layers,
4 KV heads and head dimension 256. An illustrative unquantized FP16 attention
KV estimate is:

```text
bytes/token = full_attention_layers × 2(K,V) × KV_heads × head_dim × bytes/element
            = 8 × 2 × 4 × 256 × 2 = 32,768 bytes/token
```

| Populated tokens | FP16 attention KV alone |
| ---: | ---: |
| 4,096 | 134,217,728 bytes |
| 16,384 | 536,870,912 bytes |
| 32,768 | 1,073,741,824 bytes |
| 65,536 | 2,147,483,648 bytes |
| 131,072 | 4,294,967,296 bytes |

This excludes recurrent state, allocator padding, snapshots, draft KV and all
weights. It is a structural estimate, not a measured engine allocation. At 128K,
approximately 5.04GB of weights plus this FP16 KV estimate already exceeds 8GB.
A runtime with supported KV quantization or different state storage can behave
differently. Four-bit **weights** do not imply four-bit **KV**.

Preserve the existing 128K profile and inspect its real settings. Qualify a new
128K profile only after actual prompt occupation, generation and follow-up pass
the memory/correctness gates. If Kapitsa cannot meet it, retain a faster short-
context profile and the existing long-context profile as distinct choices.
Never silently lower an advertised context limit or truncate the prompt to pass.

## 5. Phase 0 — Capture and freeze the working deployment

**Inputs:** destination Mac, current GGUF server/client, existing model file.

1. Record exact chip name, GPU core count, physical RAM, macOS build, power mode,
   plugged-in status and thermal conditions. “M3” does not identify M3 versus
   M3 Pro/Max or the available memory bandwidth.
2. Identify the serving process and executable. Confirm whether the deployment is
   standalone llama.cpp, an application-bundled build, or a proxy to another server.
   Capture version/build hash, launch arguments and model path locally; redact
   credentials before publishing artifacts.
3. Capture configured context, parallel slots, batch/microbatch sizes, GPU layers,
   Flash Attention, K/V precision, recurrent-state precision, sampling controls,
   reasoning behavior, tokenizer/chat template and any draft settings.
4. Save the client configuration and endpoint. Record whether generated reasoning
   is retained in subsequent history and whether tool schemas or system prefixes
   change between turns. Preserve original bytes, not reformatted approximations.
5. Hash the GGUF and templates. Store the rollback launch command in
   `profiles/gguf-existing/`. Do not overwrite the running application's binary.
6. Capture idle system pressure and swap counters with company applications in
   their normal state. Plan serial tests; two loaded inference servers are prohibited.

On macOS, these are useful inventory commands, not sufficient memory instrumentation:

```bash
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
system_profiler SPDisplaysDataType
vm_stat
sysctl vm.swapusage
```

**Deliverables:** filled hardware/runtime fields in the
[run manifest](evaluation/run-manifest.example.json), original configuration backup,
model checksums, monitor self-test record.

**Gate G0:** original deployment can be restored; candidate model identity is pinned;
monitor works; no inference process is competing with the next run. Otherwise stop
before loading another model and document the missing prerequisite.

## 6. Phase 1 — Establish controlled GGUF measurements

First measure `gguf-existing` with its actual settings. Then create an explicit
evaluation profile; do not retroactively assign its results to the original setup.
Use exactly one request and one conversation slot.

Record server decode speed, time to first visible token, prompt-processing time,
total generated tokens including reasoning, visible answer tokens, total response
time, correctness and memory. A display showing 17 tokens/s may omit prefill and
reasoning wait. Do not reinterpret it as end-to-end speed.

For a standalone recent llama.cpp build, the following is a **candidate evaluation
command**, not a replacement for the saved original command. Confirm every option
in that exact binary's `--help` before running. Set absolute paths first:

```bash
OX_LLAMA_BIN="/absolute/path/to/llama-server"
OX_GGUF="/absolute/path/to/OxCoder-9B.Q4_K_M.gguf"
"$OX_LLAMA_BIN" --version
"$OX_LLAMA_BIN" --help > "$OX_WORK/runs/llama-help.txt"

# Run through the Phase 0 supervisor, not alongside the production server.
"$OX_LLAMA_BIN" \
  --model "$OX_GGUF" \
  --host 127.0.0.1 --port 8090 \
  --ctx-size 16384 --parallel 1 \
  --n-gpu-layers all --flash-attn on \
  --batch-size 256 --ubatch-size 256 \
  --cache-ram 0 --spec-type none --jinja
```

Retain any required existing template override and sampling configuration. If an
option is unsupported, inspect that version's documentation and record the mapping;
do not silently drop it. `--cache-ram 0` prevents an additional large prompt-state
store in this comparison; it does not prove all active-slot prefix reuse is off.
Use server restarts for application-cold tests and inspect cache counters.

Check startup logs for Metal offload of the intended layers. CPU fallback, an
unexpected vision load, or a different model makes the run a different candidate.
Do not introduce KV quantization during this first controlled comparison. If the
existing setup already uses it, preserve and record it; add an explicit matching
setting to the evaluation profile after checking support.

**Optional backend update:** install/build a pinned newer llama.cpp beside the
existing executable. Release b11404 includes Metal small-row matrix multiplication
work relevant to speculative verification. It is a reason to test newer builds,
not a guarantee of an OxCoder speedup. Test old/new binaries with the same GGUF,
template, context and sampling. Keep compilation separate from timing runs.

**Gate G1:** five valid repetitions of the short-context tests, no unexplained
errors, accounted memory below the operational threshold, and a recorded baseline.
Keep the reported 17 tokens/s row separate if the controlled result differs.

## 7. Phase 2 — Install and qualify plain MLX/Kapitsa

Kapitsa documents support for custom Qwen3.5-family MLX checkpoints. This makes
OxCoder a candidate; successful import and generation still need verification.
Use the pinned OxCoder directory, not Kapitsa's `9b` alias, which selects base Qwen.
Here, MLX identifies the checkpoint format; Kapitsa supplies its own optimized
runtime. Do not assume that installing Python `mlx-lm` changes Kapitsa's kernels.

Install on Apple Silicon using the upstream package method:

```bash
brew install nace-ai/tap/kapitsa
kapitsa --version > "$OX_WORK/runs/kapitsa-version.txt"
kapitsa serve --help > "$OX_WORK/runs/kapitsa-serve-help.txt"
kapitsa doctor --help > "$OX_WORK/runs/kapitsa-doctor-help.txt"
```

Record the formula/package version and executable checksum. The Homebrew command
installs the current package; it does not pin a historical version. Preserve the
installed artifact or reproducible package reference before tests and do not upgrade
mid-series. Documentation source pins are listed in [Section 15](#15-source-register).
Kapitsa requires Apple Silicon and macOS 14 or newer at the researched revision.

Use a modest initial GPU budget, one slot and 256-token prefill chunks. A budget
of `6` is a conservative starting input to the GPU allocator, not a process cap;
record the exact byte interpretation from the installed planner. If loading fails,
inspect the breakdown rather than increasing the budget automatically.

```bash
NI_ANE=0 kapitsa doctor --model "$OX_MODEL_DIR" \
  --context 16384 --slots 1 --prefill-chunk 256 \
  --dflash none --spec 0 --mem-budget 6 --json \
  > "$OX_WORK/runs/kapitsa-plain-doctor.json"

# Launch under the verified supervisor after stopping the GGUF server.
NI_ANE=0 kapitsa serve --model "$OX_MODEL_DIR" \
  --model-id oxcoder9b-local \
  --private --local --host 127.0.0.1 --port 8091 \
  --context 16384 --slots 1 --prefill-chunk 256 \
  --dflash none --spec 0 --mem-budget 6 --disk-cache 0
```

`--dflash none` alone can leave MTP enabled. **Both** `--dflash none` and
`--spec 0` are required for the intended plain baseline. Validate effective mode
in startup logs. Do not infer it solely from the command or a client model label.
Kapitsa can accept a client model name without using that name to select weights;
the actual loaded directory/checksum is authoritative.
`NI_ANE=0` keeps prefill on the GPU. Preserve it in off/on speculation comparisons.
An optional later Neural Engine comparison must include any service allocations,
new numerical behavior and changed prefill performance in its evidence.

The following request tests connectivity only; it is too short for a speed claim:

```bash
curl --fail-with-body --max-time 120 \
  http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"oxcoder9b-local","messages":[{"role":"user","content":"Return exactly READY."}],"temperature":0,"max_tokens":32,"stream":false}'
```

Then test a short code completion, JSON output, multi-turn continuation, and a
round trip with a harmless tool. Verify EOS/stop handling, absence of unintended
template control tokens, and bounded cancellation. A plausible English reply is
not enough to qualify a coding-agent endpoint.

If the custom checkpoint fails import, preserve the error and tensor/config
metadata. Do not reshape incompatible tensors, borrow a base-model chat template
without testing, or substitute base-Qwen outputs. Mark the candidate unsupported
and continue with the working GGUF runtime or the optional oMLX path.

**Gate G2:** correct model and quantization loaded, plain decoding confirmed,
protocol smoke tests pass, memory evidence covers loading and inference. Proceed
to the matched benchmark before adding speculation or persistent cache.

## 8. Phase 3 — Tune plain decoding and prefill independently

Hold model, template, sampling, context and one-slot concurrency constant. Vary
only prefill chunk size: `256`, then `512`, then `1024` if prior memory gates pass.
Do not test 2048 merely because it is an upstream default. Record prefill latency
and peak memory at 4K and 16K occupied context. Select the smallest-memory setting
within 5% of the best median prefill latency; this 5% rule is a project policy.

Keep engine/kernel compilation outside steady-state measurements, but report
startup and first-request latency separately. First load, first kernel compilation,
warm decode and persisted-cache restore are distinct measurements.

If performance is unexpectedly poor, inspect in order:

1. Correct local model and native architecture path; no remote inference.
2. Metal/native kernel activation and CPU fallback messages.
3. Process footprint, pressure, swapping and unintended second model residency.
4. Reasoning/output lengths and actual populated prompt size.
5. Prefix hit counters and time spent in prefill versus decode.
6. Power mode and sustained thermal behavior.

Only tune after locating the expensive phase. A shorter answer or disabled reasoning
may improve task latency but must be reported as a different workload configuration.

**Gate G3:** matched plain-runtime results exist. A plain MLX candidate may advance
even if not faster yet, because speculation is an independent hypothesis; it may
not be promoted over GGUF without a measured advantage on the selected workload.

## 9. Phase 4 — Add one compatible speculative mechanism

### Compatibility checklist

Before loading a drafter, record its immutable repository revision, file hashes,
quantization, tensor dimensions, target-family compatibility, vocabulary, token IDs,
hidden-feature interface and runtime support. Confirm that the implementation can
verify candidates against the actual OxCoder target, including hybrid-state rollback.
Matching parameter count or a “Qwen” name is insufficient.

Kapitsa documents DFlash support and uses `z-lab/Qwen3.5-9B-DFlash` for base Qwen9B.
Treat that as a **candidate source**, not a verified OxCoder pairing. A fine-tuned
target can accept fewer draft tokens even when the tensor interface is compatible.
Resolve and pin the drafter revision before downloading. Verify that a quantized
drafter is supported by the selected loader; a repository name alone does not prove
its on-device precision or runtime size. Prefer an existing supported quantized
artifact over loading full precision and hoping it fits.

A native MTP head is a separate candidate only when its tensors and target mapping
have been confirmed. Do not silently borrow a base-model MTP head. If no supported
drafter/head fits, keep the plain MLX result; training is out of scope.

### Controlled sequence

1. Re-run the winning plain Kapitsa profile to establish the same-build baseline.
2. Estimate target + drafter + both states + verification workspace. Run `doctor`
   with the explicit drafter directory. Abort the phase if it cannot fit the budget.
3. Copy the plain launch profile; change only `--dflash none` to
   `--dflash /absolute/path/to/pinned-compatible-drafter`. Keep `--disk-cache 0`,
   one slot and the established prefill chunk. Verify that the drafter is active.
4. Run deterministic correctness checks before throughput tests. Begin with 64
   output tokens, then 256, then the standard 512-token throughput cap.
5. Test the application's normal sampling settings separately. Temperature-zero
   gains cannot be advertised as gains at temperature 1 or another production setting.
6. Collect accepted/proposed draft tokens, target verification calls, accepted
   tokens per verification, draft/verify time and rollback/fallback events where
   exposed. Define denominator conventions; never count a target bonus token as
   an accepted draft token without stating that convention.
7. If a runtime exposes multiple draft depths/tree policies, vary one at a time
   after the default passes. Save all relevant environment variables and options.
   `--spec N` configures MTP when no external drafter is used; do not describe it
   as a universal DFlash depth control.
8. If testing native MTP, return to `--dflash none`, enable the verified MTP head
   with its supported depth, and record a separate candidate. Do not combine two
   speculative methods without explicit runtime support and another correctness test.

### Correctness requirements

For the **same target checkpoint, engine and template**, compare speculation off/on
at greedy decoding. Use at least 20 fixed prompts including code, Unicode, JSON,
long common prefixes, early EOS, tool calls and stop sequences. Save output token
IDs when available. Investigate every divergence; batching and floating-point
differences can affect near-ties, but a mismatch is not automatically acceptable.
Do not waive a regression because a response remains grammatical.

At nonzero temperature, identical seeds do not necessarily produce identical
outputs because the implementation may consume randomness differently. Confirm
the verifier's documented sampling semantics and compare task outcomes across
multiple seeds. Do not claim that a small empirical test proves exact distribution
preservation. A greedy-only verification path must be labeled greedy-only.

Compare GGUF versus MLX using task correctness and latency, not mandatory identical
tokens: their quantized weights differ. Keep KV precision unchanged inside each
speculation off/on pair.

**Gate G4:** no unresolved correctness defect; no silent drafter deactivation;
accounted memory passes; accepted-token telemetry or equivalent evidence proves
the requested method ran. Promote speculation only when end-to-end benefits exceed
noise and no target workflow regresses materially. High acceptance alone is not a win.

## 10. Phase 5 — Prefix caching and optional oMLX

### Prefix-cache experiment

Keep weights resident. Measure these distinct cases for each surviving engine:

| Case | Procedure | Expected evidence |
| --- | --- | --- |
| Application-cold | Restart server with its experiment disk cache disabled; first request | Zero reusable prefix or documented minimal template reuse |
| Exact repeat | Repeat the same request bytes | Prefix reuse; not a stored-answer substitution |
| Append-only conversation | Include previous messages and response, then add a turn | Reuse of unchanged serialized prefix |
| Late divergence | Change a controlled token near the end | Partial reuse or a documented hybrid checkpoint fallback |
| Early divergence | Change an early system-prefix token | Corresponding prefix invalidation |
| Restart restore | Enable bounded SSD state cache, save, restart, repeat | Restore latency, bytes read and correct resumed state |

Application-cold does not mean cold OS file cache. Do not flush the Mac's caches.
Tag warm OS-cache runs accurately. Persisted prefix caching must restore all required
recurrent state as well as attention KV; arbitrary rewinds are not necessarily
supported by a hybrid model. Compare resumed output with recomputation.

For Kapitsa, test a copied profile with `--disk-cache 2G` after the no-disk-cache
baseline. Its active slot may already reuse prefixes while disk caching is off.
Report those as separate mechanisms. Keep stable system/tool prefixes; append
changing task material later where correct. Do not remove required history or
change semantics simply to manufacture a cache hit.

A new harness revision, model hash, tokenizer/template version or state format
requires a new cache namespace or verified engine invalidation. Ensure incorrect
state cannot be restored after those changes. Cache files may contain source code
and conversation data; store them in the same access-controlled local area as the
task workspace. Use an explicit disk limit and inspect actual directory growth.

### oMLX as a separate candidate

Use oMLX only after the preceding baseline exists. At the researched revision it
requires macOS 15+, supports RAM/SSD prefix storage, and supplies optimized kernels
in the official app. Source/Homebrew custom-kernel builds require the documented
Metal compiler toolchain; a successful Python install is not proof those kernels exist.

Install one upstream distribution and capture its version/build. Run `omlx serve
--help` in that installation. For a Python-managed installation, the documented
kernel diagnostic is:

```bash
python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
```

Run it inside the oMLX environment, not an unrelated system Python. For the packaged
app, use its logs/diagnostics if the CLI shim does not expose that interpreter.

Create an oMLX model root containing only the pinned OxCoder MLX directory. Configure
one model, one concurrent request, localhost binding, explicit context/output
limits, a bounded memory guard and a fixed SSD cache size. Relevant documented CLI
settings include `--model-dir`, `--max-concurrent-requests 1`, `--memory-guard-gb`,
`--paged-ssd-cache-dir` and `--paged-ssd-cache-max-size`. Verify units and the model's
context controls in the installed version before creating the launch profile.
Do not accept automatic memory/disk budgets derived from all 18GB of physical RAM.

Require the implementing agent to export the actual resolved settings, including
RAM hot-cache bytes and any auxiliary server processes. Begin with SSD caching off
using the installed version's documented control; then enable a bounded 2GB cache
for the paired experiment. If there is no verified way to disable a cache, record
that limitation and use a fresh isolated cache namespace for cold tests.

Test Qwen3.5 tool parsing, template fidelity, cancellation, and resumed hybrid state.
If evaluating quantized KV/state, create a separate candidate with an additional
quality test; do not attribute its memory reduction to weight quantization.
Do not run oMLX and Kapitsa simultaneously or assume oMLX cache files are portable
to Kapitsa. Share checkpoint files only where both loaders support the exact format.

**Gate G5:** prefix reuse is demonstrated by counters and timing; no stale-state
or resume correctness failures; SSD and RAM caches remain bounded. Keep a
no-persistent-cache fallback and document when a cold read remains unavoidable.

## 11. Phase 6 — Benchmark protocol and context qualification

### Experiment matrix

Run screening first, then qualify finalists. Do not exhaustively combine every
option on a workstation being used for other work.

| ID | Engine | Target | Speculation | Prefix state | Purpose |
| --- | --- | --- | --- | --- | --- |
| A0 | Existing runtime | Existing GGUF | Existing setting recorded | Existing setting | Real deployment reference |
| A1 | Pinned llama.cpp | Same GGUF | Off | Cold / active-slot warm | Controlled GGUF baseline |
| A2 | Newer pinned llama.cpp | Same GGUF | Off, then supported method separately | Matched to A1 | Backend improvement |
| B0 | Kapitsa | Pinned MLX | Off | Cold / active-slot warm | Native MLX baseline |
| B1 | Kapitsa | Same MLX | Explicit compatible drafter | Matched to B0 | Speculative gain |
| B2 | Kapitsa | Same MLX | Winning off/on setting | Bounded SSD restore | Persistent reuse |
| C0 | oMLX | Same MLX | Recorded off/default verified | Cold / warm / SSD restore | Alternative kernel/cache combination |
| C1 | Qualifying engine | Same target | Winning setting | Explicit KV quantization if supported | Optional longer-context capacity |

For a 4K prompt test reserve output space in the configured context. An example
is an 8K configured window with 4,096 input tokens and at most 512 generated tokens.
For a 16K prompt use a configured window large enough for 16,384 + output, such as
32K, **only if its allocation plan passes memory checks**. Alternatively use a
16K total-context test with a smaller prompt and label the actual token count.
Never call a truncated 15K prompt a full 16K input test.

### Workloads

1. **Decode-focused:** fixed short coding prompt, maximum 512 generated tokens.
   Collect sufficiently long completions; an early EOS is a legitimate task result
   but not comparable to a 512-token throughput sample. Do not suppress EOS in
   coding-quality tests. Record actual length and compare length-matched runs.
2. **Prefill-focused:** exact saved source-code prompts around 4K and 16K tokens,
   output capped at 128. Record actual template-inclusive token count.
3. **Warm agent turn:** stable tool/system prefix, append real tool output and a
   follow-up request. Measure reused tokens and new-token prefill.
4. **Real task:** bounded repository edit, tool calls and tests; time to a verified
   patch. Re-run in a fresh worktree for every attempt.
5. **Context ladder:** 4K, 16K, 32K, 64K, then the existing 128K profile's actual
   capacity. Stop increasing on the first memory/correctness failure. Use realistic
   code with distributed identifiers and dependencies, not only repeated filler.
6. **Sustained use:** finalists run a representative 15-minute serial session to
   detect thermal slowdown, retained allocations, cache growth and responsiveness
   problems. Abort immediately on the monitor's stop conditions.

Context tests must verify no silent prompt truncation, correct references to
information near the beginning/middle/end, and at least one follow-up after a tool
result. Retrieval of a single planted string is insufficient evidence of coding
quality at that context length. Record the largest **tested populated context**
and the separately **configured maximum**.

### Repetitions and ordering

Use five repetitions per screening cell, reporting median, minimum, maximum and
all raw observations. Do not publish a meaningful p95 from five samples. For
finalists use at least 20 repetitions for tail-latency reporting and disclose sample
count. Alternate candidate order across sessions to reduce thermal/order bias,
while loading only one server at a time. Keep idle cooldown and background workload
consistent; include sustained-use results rather than reporting only cool-start peaks.

Use one greedy suite and one production-sampling suite. Fix reasoning mode, token
budget, stop conditions and tool schema. Keep seed where supported and record
unsupported controls. A temperature of zero is a comparison setting, not a general
claim that coding is more correct at zero.

### Metrics and definitions

Use monotonic timestamps in the benchmark client. Preserve raw response/SSE data
and raw server timing fields, with sensitive task contents stored locally.

| Metric | Definition |
| --- | --- |
| Queue delay | Server admission delay where exposed; separate from prefill |
| TTFT-any | Send to first generated reasoning/content token; exclude empty role chunks |
| TTFT-visible | Send to first user-visible answer content |
| Total request latency | Send to completed response, including reasoning and prefill |
| Server decode tokens/s | Target-emitted token count divided by server decode duration, with server semantics documented |
| Visible end-to-end tokens/s | Visible answer tokens divided by total request latency |
| Prefill tokens/s | Newly processed prompt tokens divided by measured prefill duration |
| Prefix reuse ratio | Reused prompt tokens / total serialized prompt tokens |
| Speculation gain | Matched plain decode duration / speculative decode duration, alongside complete-request comparison |
| Coding completion | Verified tasks solved / tasks attempted, including failures and timeouts |
| Task latency | User request to independently validated patch, including tools and remote improver calls |
| Memory | Peak footprint and corrected accounted deployment total, measurement method and maximum sampling gap |

Never count SSE chunks as tokens: speculation can deliver several tokens in one
chunk. Use server token counts/token IDs where possible. If re-tokenizing visible
text, identify that as a separate visible-text metric; it may not reconstruct the
original generated token stream. Never count proposed or rejected draft tokens as
completed output. Include hidden reasoning in total-generation accounting if the
server exposes it, and mark unavailable fields `null` rather than zero.

Recent llama.cpp responses expose `timings` fields such as `cache_n`, `prompt_n`,
`prompt_ms`, `predicted_n`, `predicted_ms`, and `predicted_per_second`. Other engines
may use `usage.prompt_tokens_details.cached_tokens` and proprietary timing fields.
Implement explicit versioned adapters, preserve originals, and fail validation if
a required field is absent. Do not assume the same JSON field has identical meaning
across engines. Client timing remains necessary even when server timing exists.

### Benchmark-runner implementation contract

An implementing code agent should deliver a small runner with these components:

- **Manifest loader:** validates absolute model paths, model hashes, engine version,
  byte budget, one-request setting, prompt fixture hashes and explicit sampling.
- **Supervisor:** meets Section 4, starts one owned server, waits for readiness,
  records load memory, cancels timed-out requests, cleans up children.
- **Request driver:** talks directly to the local endpoint first; supports streaming,
  records monotonic event times, handles reasoning/tool deltas and final usage.
- **Engine adapters:** map actual runtime telemetry into the common result record;
  missing/unsupported values remain explicit. Save effective fallback settings.
- **Fixture builder:** uses the target tokenizer/template to count serialized input
  and verifies prompt hashes. Detects rather than hides context overflow.
- **Correctness runner:** isolated worktree and bounded commands; tests created
  independently of candidate output; distinguishes compile, test and protocol errors.
- **Reporter:** writes one JSONL record per attempt and aggregate tables from those
  records. Excludes warm-up from steady-state summaries but retains it in raw data.

Before live inference, test the driver against a mock SSE endpoint that returns
empty deltas, multi-token chunks, reasoning, a tool call, an error, a timeout and
missing usage. Verify that each case produces the correct status and that no
fabricated token speed is reported. This is test infrastructure to implement;
the documentation does not claim a benchmark runner is already bundled.

**Gate G6:** all metrics are traceable to raw evidence; finalists pass short-context
and realistic-task tests. Long-context qualification is a separate result, not a
prerequisite for retaining a useful short-context profile.

## 12. Phase 7 — Coding-agent and Harness² integration

### Qualify the inference API first

Connect the actual agent client only after direct endpoint tests pass. Use an
explicit local provider/model configuration and verify requests reach the intended
PID/model. A model name displayed by a client is not evidence of which weights ran.

Protocol tests must cover:

1. Text streaming with finish reason and usage.
2. A single harmless function call with structured arguments and a matching tool result.
3. Multi-turn continuation preserving message order and tool-call IDs.
4. Invalid arguments followed by a bounded correction, without executing malformed actions.
5. Cancel, timeout, context overflow and server restart; no orphan generation loop.
6. Stable system/tool serialization across turns, including the selected reasoning-history policy.

Disable client parallel subagents initially. One API conversation slot does not
prevent a client from launching concurrent tools or separate processes; supervise
and budget those explicitly. Cache correctness takes precedence over cache hit rate.

### Harness² deployment boundary

[Google Research Harness²](https://github.com/google-research/harness2) separates
executor and improver configuration. Its published benchmark setup targets Linux
and includes benchmark-specific dependencies and cloud-provider routes. It is not
a verified turnkey Mac + OxCoder + Opus integration.

Implement two distinguishable deliverables:

- **Mac daily runtime:** a fixed, tested procedural harness around the local OxCoder
  endpoint. Store its prompt/tool configuration as a versioned artifact. This can
  benefit from Harness²-derived procedures without running the entire refinement
  benchmark on the Mac during every coding task.
- **Harness² refinement/evaluation adapter:** a separately validated mapping from
  the chosen substrate/provider to the local solver and an optional remote Opus
  improver. Confirm model routing, tool serialization, credentials and judge paths
  using the pinned upstream code. Do not label a handwritten retry loop “Harness²”.

For upstream configuration, `HARNESS2_SOLVER_MODEL` and
`HARNESS2_IMPROVER_MODEL` use the substrate's provider/model format. A local
OpenAI-compatible endpoint still needs a provider adapter/configuration. Setting
those environment variables alone does not implement that adapter. The upstream
judge path is separate; do not assume it follows the local solver setting.

If using Opus 4.8, obtain the actual available model ID from the configured provider
and record it. Do not invent an ID from the display name. Cloud credentials stay
outside committed manifests. Record which source files, failure traces and tool
outputs are transmitted; a remote improver means the overall workflow is no longer
entirely local. Cloud model loading does not consume local weight memory, but local
clients and tool processes still count toward the deployment allowance.

### Integration and evaluation steps

1. Start with a frozen harness: concise tool contracts, repository constraints,
   inspect-before-edit, bounded execution, test requirements and repeat-loop detection.
2. Execute 10 representative development tasks to diagnose protocol and procedure
   failures. Keep these separate from the final held-out task set.
3. If enabling refinement, submit only the selected failure evidence to the improver,
   store its proposed harness change, and evaluate it before promoting the artifact.
   Version and hash each harness revision; do not change prompts midway through
   an inference-engine comparison.
4. Freeze the resulting harness for daily execution and held-out evaluation. Reuse
   stable prefixes. Changes to early harness text intentionally invalidate cache.
5. Compare local OxCoder with the initial harness, local OxCoder with the improved
   frozen harness, and local OxCoder with live remote refinement if that is a required
   operating mode. Use the same engine/profile and task budgets.
6. Include the remote calls, network delay, extra tokens, failed attempts and all
   verification time in the live-refinement results. Record currency/token costs.
   A faster decoder can coexist with a slower total workflow.

For the held-out coding suite, use at least 30 tasks: bug fixes, small features,
refactors, regression-test additions, API integration and multi-file changes. Cover
the languages actually used. Store task repositories/commits, instructions, initial
state and independent test commands. Give each candidate the same attempt limit,
tool permissions, timeout and output budget. Reset the worktree between attempts.

Define success using independent tests and patch review. Model self-approval or an
LLM judge alone does not establish correctness. Record baseline passes lost and
new passes gained individually. Preserve failures involving invalid tools,
unrelated edits, skipped tests, leaked secrets or fabricated results; do not hide
them behind an average pass percentage.

**Gate G7:** protocol correctness and held-out coding results pass; no new critical
regression; all live-improver costs are included. A professional-use claim requires
evidence for the actual task distribution and review process, not merely a model
size, a framework name or a high decode counter.

## 13. Phase 8 — Decide, document and roll back

Apply hard gates before ranking speed:

1. Actual target is OxCoder with recorded immutable weights and quantization.
2. Observed accounted deployment memory stays below the operational threshold, with no
   unresolved pressure/swap/responsiveness problem.
3. No unresolved speculation/state-restore/protocol correctness defect.
4. No critical held-out coding regression; smaller changes are reported task by task.
5. Required context is qualified or explicitly provided by a separate retained profile.

Among passing candidates, select according to the real workload:

| Workload | Primary ranking | Secondary checks |
| --- | --- | --- |
| Long code generation | Median decode throughput at matched output lengths | Correctness, total latency, sustained speed |
| Iterative coding agent | Time to independently verified patch | Pass rate, warm TTFT, retries, cost |
| Large repository context | Prefill + follow-up latency at occupied context | Memory, state restore, early-context correctness |
| Mixed daily work | Weighted representative task latency, weights fixed in advance | Worst-case latency, memory headroom, operational complexity |

Promotion policy: require at least a **10% median improvement in the chosen primary
metric**, larger than observed variation, without material regressions in required
secondary metrics. This is a project decision threshold, not an upstream claim.
If the controlled decode baseline is exactly 17 tokens/s, 10% is 18.7 tokens/s and
20% is 20.4 tokens/s. Those numbers are thresholds, not forecasts.

A candidate with unchanged decode speed can win an agent workload through reliable
prefix reuse. A candidate with faster decode can lose if prefill, draft loading,
failures or repeated retries increase completed-task time. If no candidate clearly
wins, retain the working GGUF profile and document the negative result.

Publish a final profile containing engine/package pin, executable checksum, target
and drafter hashes, launch arguments, cache limits, actual memory observations,
sampling/template configuration, tested context, client/provider settings and known
limitations. Keep the original binary/configuration and GGUF available.

Rollback procedure:

1. Cancel the candidate request and stop its owned server/helpers.
2. Restore the client endpoint and original profile files.
3. Start the saved GGUF command; verify its loaded model hash and effective settings.
4. Run a short code/tool smoke test. Do not delete the model or modify release assets.
5. Isolate the failed candidate's cache and logs for diagnosis. Do not import its
   serialized states into the restored runtime.

## 14. Result status and troubleshooting

No target-Mac measurements were produced while writing this specification. The only
performance observation supplied for this deployment is approximately 17 output
tokens/s. Do not populate benchmark tables from vendor numbers.

| Symptom | First evidence to inspect | Next controlled action |
| --- | --- | --- |
| Fast decode, long wait | TTFT, uncached prompt tokens, reasoning length | Fix prefix serialization; compare cold/warm separately |
| Drafter makes generation slower | Acceptance, draft/verification time, footprint | Disable drafter; reduce supported draft work in a separate run |
| Works briefly, then slows | Thermal state, pressure, swap, retained cache | Stop on pressure; qualify sustained behavior with lower resource use |
| 128K accepted but answer is wrong | Actual serialized input and truncation logs | Verify beginning/middle/end coverage and effective context |
| Plain MLX refuses startup | Planner allocation breakdown and loaded tensors | Reduce allowed buffers/state; keep GGUF if the target cannot fit |
| “Spec on” matches plain timing | Effective mode and fallback logs | Confirm head/drafter loaded and used before reporting a result |
| Tool calls fail after engine swap | Template, tool parser, IDs, streamed argument assembly | Fix protocol adapter before testing task quality |
| High cache hit but wrong continuation | Cache identity and recurrent-state restore | Disable reuse; compare recomputation; block promotion |
| oMLX unexpectedly slow | Native kernel diagnostic, architecture path | Fix build/runtime compatibility; rerun same workload |
| Fast local engine but slow Harness² | Number of solver/improver/judge/tool calls | Compare frozen harness versus live refinement at equal quality |

## 15. Source register

Research checked on 2026-10-23. Repository source revisions below identify the
documentation consulted; they do not assert that a Homebrew/app installation uses
the same build. The model is independently pinned above. Record installed versions
and preserve their actual help output before executing any recipe.

| Source | Consulted revision / reference | Role |
| --- | --- | --- |
| [Kapitsa usage](https://github.com/nace-ai/kapitsa/blob/b1a5aa74bfb7d055e2bdb42ed518c8153e4fe36a/docs/usage.md) | `b1a5aa74bfb7d055e2bdb42ed518c8153e4fe36a` | Custom model loading, serving options, memory and prefix state |
| [Kapitsa KFlash](https://github.com/nace-ai/kapitsa/blob/b1a5aa74bfb7d055e2bdb42ed518c8153e4fe36a/docs/kflash.md) | Same source revision | Speculative implementation and benchmark context |
| [Kapitsa benchmarks](https://github.com/nace-ai/kapitsa/blob/b1a5aa74bfb7d055e2bdb42ed518c8153e4fe36a/docs/benchmarks.md) | Same source revision | Developer-reported results; not target-Mac/OxCoder measurements |
| [Kapitsa comparison tables](https://github.com/nace-ai/kapitsa/blob/b1a5aa74bfb7d055e2bdb42ed518c8153e4fe36a/README.md) | Same source revision | Published M3 Pro comparison below |
| [MLX OxCoder config](https://huggingface.co/OmAr-Kader-DeV/OxCoder-9B-mlx-4Bit/blob/ae7575d8655be764ddf8478e5f04caeea74ec4ce/config.json) | `ae7575d8655be764ddf8478e5f04caeea74ec4ce` | Architecture, quantization, context estimate |
| [Original OxCoder](https://huggingface.co/OrionLLM/OxCoder-9B) | Upstream model card; resolve weight provenance independently | Target identity |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/4231b23dc2c46661d01157aac5245bdcf03ce4b1/tools/server/README.md) | `4231b23dc2c46661d01157aac5245bdcf03ce4b1` | Current CLI and timing interfaces |
| [llama.cpp Metal release](https://github.com/ggml-org/llama.cpp/releases/tag/b11404) | `b11404` | Small-row Metal kernels relevant to verification |
| [oMLX](https://github.com/jundot/omlx/blob/3df94603c047ec77f5381d11d1d6daff75794870/README.md) | `3df94603c047ec77f5381d11d1d6daff75794870` | Alternative native kernels, memory controls and prefix storage |
| [Harness² setup](https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/docs/SETUP.md) | `1cfe254e8d25b15f37048c3730134686abe60335` | Upstream platform and executor/improver configuration |

### Published comparison: motivation, not a target-Mac result

Kapitsa's pinned README reports the following for **base Qwen3.5-9B**, on a
**36GB M3 Pro**, macOS 14.1, on AC power:

| Runtime in the developer's table | Prompt processing, tokens/s | Generation, tokens/s |
| --- | ---: | ---: |
| Kapitsa with KFlash | 381 | 77 |
| mlx-lm | 353 | 27 |
| llama.cpp b11804 | 307 | 23 |

These are the developer's runs, not independent reproductions. The README describes
this comparison as one run per cell and notes that other engines were not retuned
for that machine. Its catalog download figure includes companions; it is not a
measured 8GB process footprint. The comparison also changes speculation and runtime,
so it does not isolate one kernel's contribution.

The results justify an experiment. They are not an OxCoder forecast: chip variant,
RAM, context, draft acceptance, sampling and quantization can differ. The decisive
comparison is the same destination Mac, the same saved tasks, the same memory
allowance, and independently checked outputs.

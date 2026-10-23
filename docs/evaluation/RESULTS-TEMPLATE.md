# OxCoder runtime evaluation report

Status: **template — no measurements recorded**.

Copy this document to a dated results directory. Populate it from raw records,
not from displayed peak counters or upstream benchmarks. Keep private prompts,
repository content and credentials out of published artifacts.

## Provenance

- Evaluation dates:
- Exact Mac chip / GPU cores / physical memory / macOS build:
- Engine version and executable checksum for each candidate:
- Target checkpoint revision, quantization and file manifest:
- Drafter revision, quantization and compatibility evidence, if used:
- Tokenizer/template hashes and reasoning-history policy:
- Workload suite revision and independently authored test revision:
- Original deployment configuration backup:
- Raw artifact directory and manifest:

## Existing observation versus controlled baseline

| Observation | Context occupancy | Decode tokens/s | Measurement status |
| --- | --- | ---: | --- |
| Existing M3-family deployment | Not supplied | Approximately 17 | Operator-reported; backend/settings not yet captured |
| Controlled existing deployment | Pending | Pending | Pending |
| Controlled GGUF evaluation profile | Pending | Pending | Pending |

Do not replace the operator-reported row with an unrelated model or benchmark.
Explain any difference between the controlled baseline and the observation.

## Memory qualification

| Candidate | Collector/accounting method | Peak observed footprint, bytes | Maximum sample gap, ms | Pressure / swap behavior | Gate |
| --- | --- | ---: | ---: | --- | --- |
| A0 existing | Pending | — | — | Pending | Pending |
| A1/A2 GGUF | Pending | — | — | Pending | Pending |
| B0 plain MLX | Pending | — | — | Pending | Pending |
| B1 speculative MLX | Pending | — | — | Pending | Pending |
| B2 persistent prefix | Pending | — | — | Pending | Pending |
| C0 oMLX | Pending | — | — | Pending | Pending |

Record loading, prefill, decode, verification, cache serialization and retained
idle peaks separately. Explain shared-allocation accounting and measurement gaps.
Add the corrected accounted deployment peak, including resident model pages omitted
from physical footprint and attributable external helper allocations. Gate on that
total; do not use a low physical-footprint counter alone to qualify an mmap runtime.
Use `unverified` if only RSS or engine allocation counters are available.
State explicitly whether the bound is sampled observation or a hard allocation cap.

## Inference results

One row per candidate, occupied context, sampling policy and cache case. Add rows
rather than averaging incompatible workloads.

| Candidate | Actual input / output tokens | Cache case | Temperature / reasoning | N | Median decode tok/s | Median TTFT-visible, s | Median total, s | Correctness |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Pending | — | — | — | — | — | — | — | Pending |

- Screening: report min/median/max and all five repetitions.
- Finalists: report at least 20 observations when presenting p95.
- Include total generated tokens and visible tokens separately.
- Describe warm-up exclusions; retain warm-up observations in raw artifacts.
- Record early EOS, timeout, API error, cancellation and memory abort counts.
- Include sustained 15-minute results and first-versus-last segment throughput.

## Speculative decoding

| Candidate | Effective mechanism | Proposed tokens | Accepted tokens | Accepted per verification | Matched plain/spec gain | Greedy correctness |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Pending | Pending | — | — | — | — | Pending |

Define acceptance denominator and target bonus-token treatment. State unsupported
telemetry explicitly. Describe output divergences, their diagnosis, and resolution.
Record any runtime fallback that disabled speculation. Such runs cannot populate
the active-speculation row.

## Prefix reuse and context

| Candidate | Configured limit | Actual populated tokens | KV/state precision | Reused/new prompt tokens | Restore/prefill latency | Correct resumed result |
| --- | ---: | ---: | --- | --- | --- | --- |
| Pending | — | — | — | — | — | Pending |

Include application-cold, exact repeat, append-only, late divergence, early
divergence and restart restore. Report SSD read/write bytes and cache directory
size. State the largest tested populated context independently of the configured
maximum. Document whether the original 128K profile remains the long-context choice.

## Coding workflow and Harness²

| Engine + harness combination | Tasks solved / attempted | Baseline passes lost | Median verified-task time | Local calls | Remote calls / cost | Critical defects |
| --- | --- | --- | --- | --- | --- | --- |
| Initial frozen harness | Pending | Pending | — | — | — | Pending |
| Improved frozen harness | Pending | Pending | — | — | — | Pending |
| Live remote improvement, if tested | Pending | Pending | — | — | — | Pending |

Attach task-by-task outcomes, independent test results, attempts and timeouts.
Distinguish a Harness² adapter from a fixed harness inspired by its procedures.
Identify the actual remote provider model ID. Include all solver, improver, judge
and tool latency/cost in live-refinement results.

## Decision

- Required workload and predeclared primary metric:
- Hard gates passed/failed/unverified:
- Winning candidate, or decision to retain baseline:
- Improvement over controlled baseline and observed variation:
- Secondary metric and coding-quality tradeoffs:
- Short-context daily profile:
- Long-context profile and largest qualified occupied context:
- Known limits and unsupported combinations:
- Installation/configuration pins and rollback smoke-test outcome:

Do not declare a winner if memory is unverified, important correctness regressions
remain unresolved, or the comparison changes target model or reasoning budget
without explicitly identifying that difference.

## Per-attempt JSONL record contract

Each physical line must be one JSON object. The example below is formatted for
readability, with intentionally missing values. Preserve unknowns as `null`.
The implementing runner must validate required fields before accepting a completed
measurement; this template is not a JSON Schema or an implemented validator.

```json
{
  "schema_version": 1,
  "run_id": null,
  "attempt_id": null,
  "manifest_sha256": null,
  "candidate_id": null,
  "fixture_id": null,
  "cache_case": null,
  "status": "pending",
  "is_warmup": false,
  "request_start_monotonic_ns": null,
  "first_generated_token_monotonic_ns": null,
  "first_visible_token_monotonic_ns": null,
  "response_end_monotonic_ns": null,
  "prompt_tokens_total": null,
  "prompt_tokens_reused": null,
  "prompt_tokens_processed": null,
  "generated_tokens_total": null,
  "generated_tokens_visible_retokenized": null,
  "generated_tokens_reasoning": null,
  "prefill_ms": null,
  "decode_ms": null,
  "server_decode_tokens_per_second": null,
  "speculation_effective_mode": null,
  "draft_tokens_proposed": null,
  "draft_tokens_accepted": null,
  "verification_calls": null,
  "process_tree_peak_footprint_bytes": null,
  "peak_accounted_deployment_bytes": null,
  "resident_mapping_accounting_complete": null,
  "monitor_max_sample_gap_ms": null,
  "memory_gate": "pending",
  "correctness_status": "pending",
  "finish_reason": null,
  "error_category": null,
  "raw_server_timing_artifact": null,
  "raw_events_artifact": null,
  "memory_samples_artifact": null,
  "test_report_artifact": null
}
```

Allowed attempt statuses: `pending`, `completed`, `timeout`, `cancelled`,
`api_error`, `memory_abort`, `protocol_error`, `unsupported`.
Completion alone is not a passing quality or memory result. Aggregates must include
failure counts, not only the fastest successful rows.

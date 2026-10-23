# Implementation checklist for a code agent

Read the [technical specification](../MAC-RUNTIME-EVALUATION.md) before implementation.
This checklist indexes its gates; it does not replace their requirements.

## Scope

- [ ] Work on OxCoder-9B, not the earlier Edge0/MoE model.
- [ ] Target the M3-family Mac and an 8,000,000,000-byte inference allowance.
- [ ] Preserve the existing GGUF release, executable, client configuration and 128K profile.
- [ ] Treat approximately 17 output tokens/s as reported until controlled measurement exists.
- [ ] Use existing weights/drafters only; do not train or silently change target models.
- [ ] Never load two candidate inference servers at once.

## G0 — Inventory and supervision

- [ ] Identify actual current backend, model path, checksum and launch configuration.
- [ ] Record exact Mac variant, OS, power mode, background load and available headroom.
- [ ] Pin native MLX checkpoint and validate affine 4-bit/group-64 configuration.
- [ ] Inspect file provenance and tensor metadata without loading full-precision weights.
- [ ] Implement and test process-tree footprint collection and targeted cancellation.
- [ ] Account for resident file mappings and external helpers without double-counting Metal allocations.
- [ ] Verify warning/abort thresholds and sample-gap reporting using a small dummy process.
- [ ] Fill the run manifest; preserve credentials outside artifacts.
- [ ] Verify rollback files before loading a candidate.

## G1 — Controlled existing/GGUF baseline

- [ ] Measure the current setup as it is before changing settings.
- [ ] Create a separate single-slot controlled profile.
- [ ] Confirm actual Metal layers, context and cache precision.
- [ ] Collect five matched repetitions, cold/warm distinction and raw server timings.
- [ ] Test an updated llama.cpp binary separately if needed; preserve the old binary.
- [ ] Record results even if they do not match the reported 17 tokens/s.

## G2 — Plain native MLX

- [ ] Install/version-pin Kapitsa and archive its actual CLI help.
- [ ] Use the exact OxCoder directory; reject generic `9b` substitution.
- [ ] Plan one slot, 256-token prefill chunk and conservative GPU budget with `doctor`.
- [ ] Start under the supervisor with both DFlash and MTP disabled.
- [ ] Confirm effective mode and loaded model identity from logs.
- [ ] Pass text, code, JSON, stop/EOS, tool-round-trip and cancellation tests.
- [ ] Mark unsupported import paths explicitly; do not disguise fallback as success.

## G3 — Plain runtime tuning

- [ ] Compare 256/512/1024 prefill chunks one variable at a time where memory permits.
- [ ] Preserve exact target, sampling and templates across comparisons.
- [ ] Separate kernel compilation/startup from warm inference.
- [ ] Identify prefill, decode, memory or thermal bottleneck from measurements.

## G4 — Speculation

- [ ] Pin compatible drafter/head and verify tensor/vocabulary/loader interface.
- [ ] Verify actual quantization and total runtime memory, not just download size.
- [ ] Check native MTP tensor presence; config metadata alone is insufficient.
- [ ] Run greedy same-target off/on correctness suite before timing.
- [ ] Test production sampling separately and document verifier semantics.
- [ ] Record effective mechanism, acceptance, verification and fallback behavior.
- [ ] Keep speculation off if incompatible, over budget, incorrect or slower.

## G5 — Caching and optional oMLX

- [ ] Test exact repeat, append-only, early/late divergence and restart restore.
- [ ] Verify complete hybrid-state restoration against recomputation.
- [ ] Keep cache identity tied to model/template/runtime/harness versions.
- [ ] Bound RAM and disk state storage and include transient serialization memory.
- [ ] If testing oMLX, verify native kernels and stop Kapitsa first.
- [ ] Preserve the same MLX target; export effective oMLX settings.
- [ ] Treat KV quantization as a separate candidate with correctness testing.

## G6 — Benchmark runner and context

- [ ] Implement versioned telemetry adapters; leave unavailable metrics null.
- [ ] Mock-test SSE handling, multi-token chunks, reasoning, tools, errors and timeouts.
- [ ] Count actual tokens; never treat chunks or draft proposals as emitted tokens.
- [ ] Include output reserve in context sizing; prohibit silent truncation.
- [ ] Screen at 4K/16K occupied input, then qualify realistic larger contexts sequentially.
- [ ] Stop the ladder at a failed resource gate; preserve original long-context profile.
- [ ] Collect at least 20 finalist observations before reporting tail latency.
- [ ] Run bounded sustained-use test and record thermal/pressure effects.

## G7 — Agent quality and Harness²

- [ ] Validate the actual client/provider protocol against the local server.
- [ ] Keep local inference serial and account for tools/client memory.
- [ ] Implement any Harness² provider adapter explicitly; do not assume upstream Mac support.
- [ ] Verify remote improver model ID and separate judge routing if used.
- [ ] Distinguish fixed-harness evaluation from live refinement.
- [ ] Freeze harness revision before comparing engines.
- [ ] Use held-out coding tasks with independent tests and fresh worktrees.
- [ ] Include remote/tool time, costs, failed attempts and regressions.

## Selection and handoff

- [ ] Complete the result template with links to raw evidence.
- [ ] Apply memory/correctness gates before speed ranking.
- [ ] Select by predeclared workload metric and measured improvement above noise.
- [ ] Publish separate short/long-context profiles if no single profile meets both needs.
- [ ] Verify rollback to GGUF and document exact installation/configuration pins.
- [ ] Leave every unexecuted test `pending`, `unsupported` or `skipped` with a reason.
- [ ] Do not claim a Mac benchmark based on research, syntax validation or Linux execution.

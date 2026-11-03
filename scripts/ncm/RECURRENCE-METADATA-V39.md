# Reusing immutable recurrence metadata

The original ragged recurrent path constructs identical cumulative sequence
boundaries and uploads them from CPU to GPU in every recurrent layer. Backward
layer recomputation repeats that work. These are small index tensors, not
model weights or learned state. Code inspection motivates removing the repeated
construction and upload; the earlier CPU-only profiler does not quantify how
much GPU time this can save.

`cached_recurrence_metadata.py` preserves the same convolution, projections,
normalization, FLA recurrence and output projection. Its bounded cache holds
only the CPU/GPU boundary tensors, keyed by batch, ordered sequence lengths
and explicit device. Batch ordering remains unchanged. Repeated calls share
the same tensors; mutation counters are checked on reuse. At most eight keys
remain cached. No source or adapter weight is cached differently, no activation
checkpoint policy changes, and no SSD offloading is introduced.

The override is process-local and must span the gradient-bearing forward and
backward. Nested/conflicting overrides are rejected, and exception cleanup
restores the original recurrence. CPU execution delegates to the original
reference recurrence. Tests cover cumulative boundaries, reuse, bounded
eviction, mutation detection, invalid shapes, scope cleanup and unchanged CPU
objective gradients. The complete suite passed 300 tests in 13.827 seconds.
That does not qualify the new CUDA path by itself.

## Full-model qualification

`audit_cached_recurrence_runtime.py` waits for the v34 worker to save and finish
evaluating update 3500. The v34 supervisor is paused during the audit. The v38
training worker is also paused briefly during measurement to avoid changing
competing CPU load; its process, model state and optimizer remain intact.
Cleanup resumes both experiment-owned units, including after exceptions. A
systemd cleanup command also resumes them if the audit process is terminated.
Unrelated applications remain running.

Both arms restore the same immutable parent weights and complete optimizer
state, then perform four identical real updates with the same examples and
random seeds. The reference uses the resident v34 runtime; the candidate adds
only recurrence-boundary reuse. Both retain interleaved prefix construction,
layer checkpointing and resident head moments. Prompts of 64, 69, 512 and
2048 tokens exercise admitted shapes. Synthetic extensions/truncations are
marked as resource probes. Every update is discarded.

Qualification retains the established thresholds: final parameter relative
L2 below 1e-4, maximum absolute difference below 2e-4, frozen source, no codec
gradients, and warm median throughput ratio greater than 1.05. The last three
updates determine each median. The report records cache hits and misses as
well as synchronized wall time and allocated GPU peak. Memory limits and
application reserves remain enforced.

The audit runs as `ncm-cached-recurrence-audit-v39.service`, writing
`results/ncm-cached-recurrence-audit-v39/report.json`. It resumes the existing
trainer regardless of the optimization result. No training path automatically
adopts this experiment, and no speed gain is claimed before the full audit
completes. Any later adoption must record the qualified execution identity and
preserve the model, corpus, optimizer, random streams and acceptance evaluator.

## Completed result: retain the existing runtime

The [full audit](validation/v39-controls/full-runtime.json) passed numerical
qualification. Final parameter relative L2 was 2.296e-5 and maximum absolute
difference was 0.00012207. Frozen source and codec-gradient checks passed.

| Runtime | Warm median update time | Peak allocated bytes |
| --- | ---: | ---: |
| Existing resident checkpointed path | 2.66040 s | 6,974,623,744 |
| Cached recurrence boundaries | 2.59991 s | 6,974,623,744 |

The candidate recorded one metadata miss and 191 hits across four updates,
confirming that boundary reuse occurred. The measured throughput ratio was
1.02327, below the predeclared greater-than-1.05 adoption threshold. Therefore
this optimization was **not enabled** in training. Both original experiment
units resumed after the audit. The measurement supports only a modest effect
on this workload; it does not justify attributing the overall training time
to repeated boundary uploads.

The paused v38 worker retained its in-flight update 3130. That update's
wall timer includes the pause and records 91.37 seconds. Its
[timing annotation](validation/v39-controls/student-pause.json) excludes it
from ordinary training-step throughput estimates while preserving it in total
elapsed job time. The original training log is unmodified.

# Unpadded concept execution for original-backbone NCM training

This optimization continues the v30 original-backbone experiment. It preserves
the source weights, residual-VQ codec, objective, optimizer and sampling state.
It changes the execution of independent concept levels during the gradient
pass. It is not a qualified coding model, an inference speed claim, or a Mac
memory measurement. Acceptance still requires the unchanged twelve executable
tasks and identical fresh-process replay.

## Packed sequence layout

The four concept levels have lengths 4, 16, 64 and 128. The preceding execution
padded each to 128 positions and treated levels as separate batch members:
512 projected positions per example. The new path concatenates the actual
levels along the sequence dimension: 212 positions per example. Projection,
normalization, feed-forward and residual-adapter work uses that shorter tensor.

The levels remain independent. Full attention uses a block-diagonal mask over
concept positions, plus each example's valid frozen prompt keys. Rotary
positions restart at the actual prompt length for each level. Right padding in
the merged prompt keys is masked. Prompt prefill itself remains unpadded and
bypasses adapters.

GatedDeltaNet projections run once over the concatenated tensor. The short
depthwise convolution runs separately for each level, starting from that
example's frozen prompt convolution state. FLA's variable-length recurrence
receives cumulative level boundaries and a separate copy of the prompt's
initial recurrent state for each segment. No level consumes the recurrent
state of another level. The CPU reference executes separate recurrences.
Activation-checkpoint recomputation recreates attention cache forks instead
of mutating saved prefix states.

`frozen_ragged_objective.py` retains the v30 trajectory function for clean and
student-conditioned rollouts. Only the combined differentiable pass uses the
unpadded implementation. Inference likewise retains the qualified v30 path.
The codec encoder, adaptive residual labels, per-example conditioning flags,
equal-level loss and termination loss are unchanged. Oracle target bytes never
enter autonomous generation inputs.

## Numerical and timing evidence

The [full-model audit](validation/v31-controls/unpadded-execution.json) compares
all 78 trainable parameter gradients against uncached original-backbone
execution. The numerical thresholds remain relative gradient L2 below 0.05,
cosine above 0.99 for every parameter group, and logit relative L2 below 0.02.
Both shape bundles pass; the largest subgroup relative L2 is 0.01748.

| Prompt lengths, batch four | Padded v30 seconds | Unpadded seconds | Speed ratio | Unpadded allocated GPU peak |
|---|---:|---:|---:|---:|
| 64, 69, 128, 256 | 3.367 | 1.927 | 1.747× | 6.021 GB |
| 64, 512, 1,024, 2,048 | 4.692 | 3.189 | 1.471× | 6.362 GB |

These are one measured iteration after a warmup, including prompt preparation
and backward but excluding optimizer work. The longest two prompt lengths are
explicit synthetic shape controls. This is a 32–43% reduction in the measured
pass time, not a demonstrated reduction of the same size in an entire training
run. Host memory, driver allocations and other processes are excluded.

The [optimizer smoke](validation/v31-controls/unpadded-optimizer.json) uses the
immutable v30 update-400 checkpoint. It performs one clean and one
student-conditioned update, then discards their weights. Both updated adapters,
interfaces and classifiers with finite values while leaving source weights
unchanged and the codec gradient-free. Times were 10.08 and 5.82 seconds,
including parameter comparisons and initial overhead. Peak allocated GPU memory
was 6,437,857,792 bytes. These two updates are not a sustained throughput test.

The 232-test CPU suite passed before continuation. New controls cover mixed
teacher/student loss and gradients, level isolation, frozen parameters,
qualification rejection, and restoration of the next optimizer update and
random draws from a v30 checkpoint.

## Qualification and reproducible continuation

Use the pinned environment, source GGUF, codec and admitted complete-response
corpus described in [v29](FROZEN-TOWER-V29.md) and [v30](MULTI-PROMPT-V30.md).
Stop a training supervisor at a completed checkpoint boundary before importing
its checkpoint. Copy the parent to an immutable path; do not initialize from a
`latest.pt` that another process can replace.

```bash
"$PYTHON" -m scripts.ncm.audit_frozen_ragged \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-parallel-v24/latest.pt \
  --selection results/ncm-complete-selection-v27 \
  --output results/ncm-frozen-ragged-audit-v31 --device cuda:1

"$PYTHON" -m scripts.ncm.audit_frozen_ragged_optimizer \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-frozen-multi-training-v30/probe-400.pt \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --execution-audit results/ncm-frozen-ragged-audit-v31/report.json \
  --output results/ncm-frozen-ragged-optimizer-v31 --device cuda:1

"$PYTHON" -m scripts.ncm.train_frozen_ragged \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --init results/ncm-frozen-multi-training-v30/probe-400.pt \
  --selection results/ncm-complete-selection-v27 \
  --manifest results/ncm-complete-pipeline-v28/features.json \
  --execution-audit results/ncm-frozen-ragged-audit-v31/report.json \
  --optimizer-audit results/ncm-frozen-ragged-optimizer-v31/report.json \
  --output results/ncm-frozen-ragged-training-v31 \
  --device cuda:1 --session-steps 100 --eval-every 250
```

Run full-model controls without competing full-model workers. Output directories
must be new for audits. Append `--resume` only for the original training output,
immutable parent and unchanged configuration. Import preserves FP32 optimizer
masters and moments, global update, schedule, Python sampler and CPU/CUDA RNG.
It rejects changed parent code, corpus, batch size, loss scale, precision or
schedule. Execution qualification pins code hashes and the numerical reports.

`run_frozen_ragged_stage.py` provides the experiment's Linux systemd supervision.
Its paths, unit names and device are concrete deployment choices, not portable
defaults. It replaces the stopped v30 supervisor and starts from update 400.
The first session reaches the next quality gate before draining the older v28
evaluation backlog; it still waits for any active evaluator. Subsequent
sessions give that backlog time to finish. Independent native-decoder training
can continue within its separately checked memory allowance. Every child exit
resumes the older coordinator. No unrelated process is stopped.

Neither the CPU tests nor the execution/optimizer audits establish coding
quality. The twelve-task result must be produced by autonomous NCM generation,
with no original-language-model answer fallback. A passing checkpoint is only
a candidate until another process reloads the exact bytes and reproduces all
answers and checks. Deployment export and memory/latency measurements follow
that gate rather than substituting for it.

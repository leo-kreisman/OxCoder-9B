# Paired learning-rate continuation

This experiment compares the body learning rates `0.00003` and `0.0001` after
the FP32-master optimizer and CPU head-state offload have been qualified.
It does not change the architecture, quantization, datasets, corruption,
classifier learning rate, or executable acceptance tests. No speed or quality
improvement is assumed in advance.

## Controlled training arms

Both arms begin at the same immutable update-350 state. `change_body_lr.py`
changes only the declared body/interface learning rate and its optimizer
groups. It preserves the learned parameters, master weights, momentum,
AdamW moments, step counters, and sampling RNG. Both use the CPU AdamW head
optimizer qualified in [the optimizer offload guide](OPTIMIZER-OFFLOAD-V24.md).

Each arm runs updates 351 through 500. The coordinator compares the actual
logged example identities and worker assignments for all 150 updates, not
just the configured seed. A missing, duplicated, or different update fails
the comparison. Their resulting checkpoints must be paired with completed
12-task evaluations and verified checksums.

The baseline's existing acceptance monitor retains ownership if the baseline
passes 12/12. Otherwise, the comparison coordinator stops only that monitor
and its named training service after the update-500 evaluation is preserved.
The candidate then runs from update 350 on both GPUs. The comparison does not
start another model while the two-replica trainer occupies the devices.

## Frozen validation selection

`validation_probe.py prepare` selects 64 distinct validation conversations,
using a deterministic hash of each row identity and a round-robin selection
across available source buckets. Selection precedes model scoring. It rejects
cross-split conversations and duplicate row identities, and never filters by
reconstruction or prediction quality. The data and selection are checksummed.

The recorded selection contains 61 Ox-Alpha conversations and all three
eligible DeepSeek-Agent validation conversations, with prompts from 54 to
1,985 tokens. NVIDIA examples in this prepared corpus are training-only;
this diagnostic does not establish NVIDIA out-of-distribution performance.
The source imbalance is reported rather than hidden through repeated samples.

```bash
python -m scripts.ncm.validation_probe prepare \
  --data results/ncm-curated-training-v23 \
  --output results/ncm-validation-selection-v24
```

For each checkpoint, `validation_probe.py run` records:

- Cross entropy and code accuracy at each level under gold earlier concepts.
- The same metrics over concept positions overlapping nonpadding bytes.
- Exact reconstruction through the frozen codec, including failures.
- Free-running chunk validity, exactness, and nonpadding byte accuracy.
- The termination probability and expected final-chunk flag.

Gold-conditioned metrics diagnose prediction; they are not generation scores.
Free-running chunk generation receives only the existing prompt. Some prompts
include preceding target chunks from the dataset, so this is not a complete
response generation benchmark. The independent 12-task evaluator remains the
executable acceptance test.

```bash
python -m scripts.ncm.validation_probe run \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint /path/to/evaluated-checkpoint.pt \
  --selection results/ncm-validation-selection-v24 \
  --output results/new-validation-probe --device cuda:1
```

Use the pinned Python/dependency environment from the compact training guide.
The probe uses compact frozen weights and never updates model parameters.
Runtime admission reserves memory for other applications. It is an evaluation
on the training workstation, not an 8 GB Mac deployment measurement.

## Continuation policy

The higher-rate arm is selected only when all of these conditions hold:

1. Its executable-task score does not decrease.
2. Its exact and valid generated-window counts do not decrease.
3. Its mean generated nonpadding byte accuracy does not decrease.
4. Its mean level cross entropy improves by more than one percent.
5. The upper end of a descriptive paired 95% normal interval for its per-row
   cross-entropy change is below zero.

This conservative rule selects a continuation; it does not establish coding
usability. The interval is a development diagnostic with a small, imbalanced
sample, not a claim of generalization. Inconclusive comparisons retain the
baseline. Invalid or incomplete evidence fails explicitly.

The selected trainer resumes its original experiment with `--resume`, including
the same initialization argument and planned 5,000-update schedule. It restores
optimizer/master/RNG states from update 500. It does not reset the optimizer
or relabel the checkpoint's dataset identity. The coordinator continues to
preserve evaluations and requires a separate cold replay of any 12/12 candidate.

`run_lr_comparison.py` coordinates these recorded workstation paths and named
services. Its `progress.json`, per-arm probe reports, `comparison.json`, and
launch records remain in `results/ncm-lr-comparison-v24`. A running service or
a selected learning rate is not a completed training result.

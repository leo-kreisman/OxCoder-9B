# Prompt-conditioning control

`audit_prompt_conditioning.py` checks whether a frozen NCM checkpoint predicts
concepts better with the correct request than with an unrelated request.
Low training loss alone cannot establish that the model uses its conditioning.

The control selects 16 distinct conversations from the existing frozen
validation selection, using an identity hash rather than prediction quality.
It orders them by prompt length and rotates their prompts by one position.
Every replacement comes from a different conversation. The cyclic wrap can
produce a large length difference, so the report includes both lengths;
this control does not perfectly isolate semantic effects from positional ones.

Both arms receive the same clean earlier concept levels and predict the same
target codes. Only the prompt changes. For each level, the report contains:

- Correct-prompt and replacement-prompt cross entropy and code accuracy.
- Agreement between the two argmax predictions.
- KL divergence from the correct-prompt distribution to the replacement one.
- Mean replacement cross-entropy increase, with a positive value indicating
  that the correct prompt helped on these observations.

No gradients or optimizer updates are performed. Later levels receive gold
earlier concepts, so their scores are conditional diagnostics. The coarsest
level receives no gold earlier codes. None of these metrics replaces complete
free generation, executable tests, or checkpoint-reload evaluation.

```bash
python -m scripts.ncm.audit_prompt_conditioning \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint /path/to/immutable-evaluated-checkpoint.pt \
  --selection results/ncm-validation-selection-v24 \
  --output results/new-prompt-control --device cuda:0 --count 16
```

Use the pinned training Python environment. The selection, checkpoint, codec,
and source identities are checked. The report is written incrementally and
becomes complete only after every selected pair is evaluated. Nonfinite metrics
fail the run. Run on a device with enough memory for the compact frozen model
and the existing application reserve. This command is a quality diagnostic;
it does not measure inference throughput or an 8 GB deployment footprint.

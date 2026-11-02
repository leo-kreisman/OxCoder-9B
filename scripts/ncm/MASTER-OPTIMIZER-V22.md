# FP32-master Muon continuation

## Measured failure mechanism

The compact student keeps its residual adapter parameters in BF16 for forward
computation. Direct `torch.optim.Muon` updates those resident parameters in
their own dtype. The installed implementation also creates its momentum buffer
in that dtype. An update smaller than the local BF16 spacing can therefore
disappear on every step instead of accumulating.

`audit_optimizer_rounding.py` uses the real version 20 update-50 checkpoint
(`81d1e6d036a75bc694773559030b6607493ae882056d9d00bee882d045e28eac`).
For each Muon matrix it computes a direction from the saved momentum using the
installed Newton–Schulz implementation. It then applies that same direction
32 times, either directly to the resident tensor or to an FP32 master rounded
only for comparison. Learning rates and weight decay match the saved optimizer.

| Matrix family | Changed entries, direct resident update | Changed entries, accumulated FP32 then rounded |
|---|---:|---:|
| Adapter down projections, mean over 32 matrices | 7.91% | 88.24% |
| Adapter up projections, mean over 32 matrices | 22.09% | 94.64% |
| Existing FP32 concept projection | 99.97% | 99.97% |

Relative to the accumulated FP32 update, the mean adapter down-projection
update error was 0.910 for direct BF16 accumulation versus 0.086 after rounding
the accumulated master. For up projections it was 0.792 versus 0.045.
These are fixed-direction numerical controls, not future-gradient simulations
or coding scores. They do not mean every previous update was lost: 22.4% of
down-projection entries had changed over the actual first 50 updates.

## Implementation and checkpoint migration

`FP32MasterMuon` retains FP32 copies of eligible matrix parameters and delegates
the actual optimizer calculation to `torch.optim.Muon`. On each update it copies
the clipped model gradient to the corresponding master, updates FP32 momentum
and master weights, then copies rounded weights back to the forward parameters.
The source GGUF weights remain frozen. AdamW classifiers/interfaces and canonical
gradient-clipping groups remain unchanged. Newton–Schulz still uses PyTorch's
internal BF16 calculation; this change addresses accumulation precision.

The wrapper serializes masters explicitly because an ordinary optimizer state
dictionary does not contain its parameter values. Loading checks master dtype,
shape and finiteness, restores the FP32 optimizer state, and regenerates the
forward weights. Tests verify uninterrupted-versus-resumed updates, legacy
momentum migration, sub-ULP accumulation, and gradient clearing.

Legacy migration is explicit. It retains AdamW states, converts the saved Muon
momentum to FP32, initializes masters from the current rounded model, and
preserves the RNG and completed update count. It cannot recover low bits already
discarded by previous updates. This is a new numerical training stage, not a
bit-exact resumption of the old optimizer.

The version 20 trainer and coordinator were stopped before changing their
implementation. Its update-100 checkpoint was copied to the immutable
`results/ncm-master-parent-v22.pt`, SHA256
`f1f909b904c756288e000e47a4ddfb83f880b9e0a962723187240cc56c39968b`.
Version 22 first tests two continued updates, 101 and 102. Its full stage starts
from the same immutable update-100 parent after that check, retaining the
original 5,000-update budget. Smoke-test updates are not counted twice.

The relevant trainer flags are:

```text
--continue-optimizer --fp32-master-muon --migrate-muon
```

Use `--migrate-muon` only for a legacy Muon parent. Continuation from a checkpoint
already using FP32 masters loads the wrapper state directly. Exact resumption
within the same output directory still requires matching implementation and
experiment identity. Do not edit the active trainer.

`run_master_stage` coordinates the smoke check, continued training, immutable
evaluation snapshots, and cold replay of a complete 12/12 result. The test set,
grading, source model, codec and training corpus are unchanged. Fixing update
precision alone does not establish a usable NCM checkpoint.

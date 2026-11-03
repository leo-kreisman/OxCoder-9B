# Frozen source context with the original concept-generation backbone

This experiment retains all 32 original OxCoder layers in **both** prompt
processing and concept generation. The immutable source weights are shared;
prompt processing bypasses trainable adapters. Each concept level reads a
separate copy of that frozen hybrid prefix state, then uses the source layers
and trainable adapters to predict its codebook labels. The existing frozen
residual-VQ codec produces answer bytes. No autoregressive answer head or
verification fallback is added.

The new execution path is `frozen_tower.py`. It differs from the earlier v24
generator, which applies adapters to prompt positions too. It also differs
from v25/v28, whose small concept decoder starts independently of the source's
decoder layers. Reusing a v24 parameter checkpoint is initialization of a new
architecture, not a numerically equivalent training resume.

## Motivation and limits of the supporting research

The [context-tower conversion study](https://arxiv.org/html/2610.02657v1)
uses a second copy of the parent as its denoiser, attached to layer-aligned
frozen context. Its smaller-donor experiments retain substantially less coding
performance. That result motivates testing reuse of the source layers instead
of assuming a small new decoder inherits the source's knowledge. The published
method concerns token diffusion and softmax-attention models. Applying frozen
context to OxCoder's hybrid recurrence and a learned concept vocabulary is a
separate experiment; the paper does not establish that this implementation will
work or that it needs the same training budget.

## Cache semantics

`prepare` computes source states without invoking adapters or accepting target
bytes. Cached states record the owning backbone, prompt IDs, and source tensor
versions. Adapter updates leave the frozen context valid; source changes or use
with another backbone invalidate it. Each level and checkpoint recomputation
forks its own layer state. Full-attention queries can see other inputs within
their own level, while GatedDeltaNet retains its causal within-pass recurrence.
Current-level target labels never enter either path.

The uncached control recomputes the whole prompt for every level and applies
adapters only to concept positions. Comparing with the old adapted-prompt
reference would compare different architectures, not establish cache fidelity.

## Initial numerical audit: not admitted

Four CPU tests cover nonzero adapters, target and adapter gradients, checkpoint
replay, immutable cache forks, updates to adapters versus source weights, and
foreign-cache rejection. The complete suite passed 208 tests. These tests do
not establish full-model numerical fidelity or generated-code quality.

The first full OxCoder audit used the preserved v24 checkpoint, original
Q4_K_M source, and four training-only prompts. Timings include one frozen
prefill and four forward/backward concept levels; they exclude optimizer work.

| Actual prompt tokens | Uncached seconds | Cached seconds | Ratio | Numerical gate |
|---|---:|---:|---:|---|
| 64 | 1.944 | 1.629 | 1.19× | Passed |
| 69 | 1.805 | 1.637 | 1.10× | Failed |
| 512 | 3.545 | 1.755 | 2.02× | Passed |
| 1,976 | 11.880 | 2.476 | 4.80× | Passed |

At 69 tokens, interface-gradient relative L2 error was 0.1211 against the
unchanged maximum of 0.05. Overall gradient error was 0.1088. The cached path
is therefore **not admitted for training** by this audit. Its largest tested
prompt was 1,976, despite requesting the nearest example to 2,048; it does not
qualify the 2,048-token limit. Cached PyTorch allocation peaked at about 7.40 GB,
without optimizer state. These are local CUDA training measurements, not Mac
inference speeds or an 8 GB whole-process deployment result.

The complete original report, including implementation hashes and gradient
subgroups, is preserved in
[frozen-tower-initial.json](validation/v29-controls/frozen-tower-initial.json).
No coding acceptance result follows from these execution measurements.

## Aligned recurrent-state fork

The next control caches only a multiple of 64 prompt tokens, then replays the
remaining prompt tail with every concept level. Tail positions stay causal and
bypass adapters; concept positions use adapters and within-level attention.
This preserves the fused recurrent kernel's chunk boundaries. Prompts shorter
than 64 tokens replay in full. Cache ownership and frozen-source checks remain
unchanged.

The 69-token interface-gradient error fell from 0.1211 to 0.00795. However,
the 512-token case measured 0.05195, just outside the same 0.05 limit. No
threshold was relaxed and the aligned BF16 path remains unqualified. A real
2,048-token numerical case was also tested by extending the selected
1,976-token training prompt with repeated training-prompt tokens; it is marked
synthetic in the report, rather than presented as a natural quality example.
That case passed, at 4.96× the uncached control's training-pass speed.

See [frozen-tower-aligned.json](validation/v29-controls/frozen-tower-aligned.json).
The optional `frozen_tower_batch.py` additionally batches independent concept
levels. Its CPU tests check gradient agreement and padding isolation. It has
not yet been admitted by a full-model GPU audit.

## Explicit precision controls

The first FP32 execution control exhausted its allocator allowance during a
cached backward pass. It produced no completed numerical records and is not
admitted. The original failure is retained in
[frozen-tower-fp32-memory.json](validation/v29-controls/frozen-tower-fp32-memory.json).

FP16 is a separate execution policy with more mantissa precision and a smaller
exponent range than BF16. With reduced-precision reductions and TF32 disabled,
it passed all four unchanged logit and gradient gates, including the exact
2,048-token synthetic shape. Worst interface-gradient relative L2 was 0.01738,
below 0.05; worst logit relative L2 was 0.000412, below 0.02. This comparison is
FP16 cached versus FP16 uncached, not a claim of identical BF16 predictions.

| Prompt tokens | Uncached seconds | Cached seconds | Ratio |
|---|---:|---:|---:|
| 64 | 1.499 | 1.619 | 0.93× |
| 69 | 1.651 | 1.694 | 0.97× |
| 512 | 3.398 | 1.791 | 1.90× |
| 2,048 | 12.014 | 2.429 | 4.95× |

Peak cached PyTorch allocation was 7.34 GB, excluding optimizer state. The
short-prompt cases show cache overhead rather than acceleration. Full results:
[frozen-tower-fp16.json](validation/v29-controls/frozen-tower-fp16.json).
This qualifies only the hashed execution implementation. Optimizer memory and
actual generation remain separate gates; it is not a usable-model claim.

## Cache lifetime and independent level batching

The original cache fork stores a bound replacement method on its own cache
layer. That forms a reference cycle. Transient recurrent and convolution
states can retain training graphs until cyclic garbage collection runs.
`frozen_tower_cache.py` replaces that receiver with a weak reference. A CPU
regression disables cyclic GC, demonstrates the old retained layer, checks
that the new layer is released immediately, and checks independent state.
The shared historical cache implementation is preserved for earlier runs.

The new full-model control combines that lifetime fix with four independent
concept levels in one batch. It compares against uncached sequential levels,
including every trainable gradient and each parameter subgroup. All four
shapes pass the same numerical thresholds. Maximum interface relative L2 is
0.01288. Measured training-pass times are 0.888, 0.950, 1.054 and 1.748 seconds
at 64, 69, 512 and 2,048 prompt tokens, respectively. The last is 6.72× the
uncached control. Peak cached allocation is 5.78 GB. These changes were tested
together, so the timing difference does not isolate either change's effect.
The report is [frozen-tower-fp16-batch.json](validation/v29-controls/frozen-tower-fp16-batch.json).

The training objective in `frozen_tower_objective.py` alternates clean
teacher-forced trajectories and student-generated earlier levels. Target
bytes define residual labels only; tests verify that changing target bytes
cannot change an on-policy trajectory's inputs or decisions. Inference
accepts no target bytes and uses no original token-vocabulary answer head.

FP16 training uses a fixed loss scale with FP32 Muon masters. Unscaling and
clipping happen in FP32 before updating those masters. Head AdamW state stays
in host RAM. `frozen_tower_optimizer.py` has exact CPU resume and nonfinite
gradient rejection controls. The full-model optimizer smoke must also pass
before the trainer starts. Optimizer offloading here is unrelated to SSD
inference offloading.

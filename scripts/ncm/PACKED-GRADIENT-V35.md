# Packed-only backward state experiment

Standard `F.linear` with a frozen dequantized matrix still needs that matrix to
compute its input gradient. Retaining every dense matrix across all 32 source
layers would exceed the training memory budget. The current runtime therefore
checkpoints entire layers: it runs the layer once, recomputes its forward pass
during backward, and then computes input gradients.

`packed_gradient.py` tests a different tradeoff. Its custom autograd function
retains only the immutable packed GGUF byte tensor. Forward performs the same
unpack and `F.linear`; backward unpacks the same bytes again and computes
`gradient @ weight`. It does not compute or estimate weight gradients. This
removes the need to retain dense source weights and may allow ordinary training
activations to remain resident without recomputing every layer. Activation
memory increases relative to layer checkpointing and requires measurement.

This is an execution experiment. It neither changes quantization nor moves
weights to SSD. It is not automatically enabled in the v34 training run.

## Numerical evidence

The CUDA matrix audit tested nine representative real OxCoder source matrices,
including Q4_K/Q6_K weights, recurrent value-head permutations, attention and
MLP projections. Each was tested in FP16 and BF16 at four and 212 positions:
36 cases in total. All forward values and input gradients were bit-exact with
the existing implementation; packed tensors remained unchanged. Autograd saved
only the packed bytes. The allocator peak was 180,289,536 bytes for this
isolated audit, not for the full model.

The machine-readable evidence is
[packed-gradient.json](validation/v35-controls/packed-gradient.json). A separate
CPU hybrid-model control verifies unchanged full-objective gradients when
layer recomputation is disabled. The complete NCM unit suite passed 272 tests.

## Whole-model admission

`audit_packed_runtime.py` waits for a saved v34 session and pauses only that
experiment's supervisor. The running worker finishes normally. The audit loads
the immutable update-2500 checkpoint, restores identical full optimizer state
and compares four real updates using the resident checkpointed runtime against
four updates with packed-only autograd and no layer recomputation.

Both arms use the same training examples, RNG seed, canonical labels, corruption
and optimizer. Prompts of lengths 64, 69, 512 and 2048 exercise the qualified
shapes; the report marks synthetic extension/truncation explicitly. All updates
are discarded. Qualification requires finite unchanged-source execution, no
codec gradients, final trainable-parameter relative L2 below 1e-4, maximum
absolute difference below 2e-4 and more than 5% median speed improvement over
the last three updates. A 12 GiB allocator cap and host/GPU reserves constrain
this training experiment. They do not establish 8 GiB deployment suitability.

An optional additional profiled update records operator costs; it is excluded
from the timing comparison. Any exception marks the audit incomplete and
failed. Cleanup resumes the original supervisor, including when qualification
fails. A passing matrix audit alone cannot admit the optimization to training.

The custom backward supports first-order input gradients only. Scope changes
are process-local, serialized and restored on exceptions. Enter the packed
gradient context outside the prefix cache context and retain it through
backward. The existing no-gradient prefix cache may temporarily replace and
restore individual forwards while constructing prefix states.

No quality claim follows from these checks. The twelve autonomous executable
tasks and identical fresh-process replay remain required for acceptance.

## Qualified selection serialization

`selection_reader.py` adapts the existing prepared-data JSON-cache strategy to
the narrower `selection.pt` envelope. One-time conversion checks source hashes
before and after `torch.load(weights_only=True)`, then verifies all types,
values and ordering after JSON round-trip. It does not deserialize model or
optimizer state. The cached corpus loader repeats the original admission
checks, including complete responses, unchanged validation, both creator data
sources and distillation. A whole-corpus comparison must pass before adoption.

For the 49,648 training chunks and 64 held-out chunks in the compact corpus,
the original admitted read took 38.26 seconds. Three warm checksummed cached
reads took 3.00, 2.65 and 2.80 seconds. All returned types, values and ordering
were identical, as were 1,000 seeded sampler draws. The median is 2.80 seconds,
approximately 13.6x faster startup deserialization. This is not a training-step
or inference speed measurement. The [complete reader report](validation/v35-controls/selection-reader.json)
records the data and implementation hashes.

## Conditional training adoption

`run_packed_training_stage.py` waits for the completed v34 update-2500 session,
then retires its paused supervisor while the whole-model audit owns GPU1.
It requires both the full-update runtime audit and the complete serialization
audit. If either fails, it restarts the original v34 supervisor from saved
state. No failed optimization is allowed to strand the parent training run.

`train_packed_tower.py` restores the entire v34 optimizer and RNG state. It keeps
the same batch size, sampled corpus, loss, corruption, learning-rate schedule,
frozen source and codec, and unchanged evaluation functions. The runtime uses
packed-only autograd, interleaved prefix construction and resident FP32 head
moments. Data loading uses the separately qualified exact JSON serialization.
One thousand updates per session reduce repeated startup work; checkpoints
still save every 100 updates and quality evaluations still run every 250.
The final partial session also records its saved boundary.

Adoption is conditional. A completed matrix or serialization audit does not
by itself prove that whole-model training fits or runs faster. No generated
checkpoint is accepted until the existing twelve-task gate and fresh-process
replay pass.

## Exact precision on cold reload

The historical backbone constructor creates BF16 adapters. Loading FP16 saved
adapters into that constructor and only then switching execution to FP16 loses
low bits. Training's full optimizer-state restoration already recopies the
saved model tensors into the final execution dtype, so its in-process coding
reports are unaffected. A cold evaluation that omitted that restoration could
evaluate rounded adapter weights instead of the exact saved checkpoint.

`exact_checkpoint.restore_trainable` fixes the cold path after execution dtype
selection. It validates the complete trainable inventory, shapes, dtypes and
finiteness before copying anything, restores the original saved values, checks
bit-exact equality, and verifies the checkpoint hash before and after loading.
It uses read-only memory-mapped, weights-only checkpoint loading. Optimizer
state is not modified by this helper. A regression test deliberately creates
FP16 values that change through BF16 and verifies their exact restoration.

The v35 trainer applies this restoration before either training or evaluation.
`evaluate_exact_tower.py` provides the same corrected fresh-process gate for
v32–v35 checkpoints. It refuses to replace an existing report. This fixes
checkpoint fidelity; it does not turn a failed in-process quality gate into a
passing result. The concept-error diagnostic also uses exact restoration.

After the runtime policy, data-cache and reload fixes, the complete NCM unit
suite passed 284 tests. The whole-model CUDA audit remains a separate admission
requirement.

## Full-model memory result and fallback

The full no-recompute audit at the v34 update-2500 checkpoint did not qualify.
The checkpointed reference completed four updates, with warm median 2.6958 s
and peak allocation 6,974,623,744 bytes. The no-recompute arm exceeded its
11.18 GiB allocator admission during its first full update. Approximately
4.02 GiB remained physically free; this was the deliberate application reserve,
not permission to expand into other workloads' memory.

The [full report](validation/v35-controls/full-runtime.json) is incomplete and
failed. No no-recompute timing, numerical comparison, profiling result or coding
quality gain is claimed. The coordinator did not start the v35 trainer. It
restarted the qualified v34 supervisor from the saved update-2500 state. The
representative matrix audit and exact serialization cache remain separately
valid but cannot override a failed full-model admission.

`selective_packed_gradient.py` is the next **separate runtime experiment**. It
keeps packed-only input gradients and recomputes alternating layers instead of
retaining all layer activations. The dispatch checks the original layer index,
rejects unknown checkpoint call shapes, and restores every override on exit.
Full numerical and memory qualification is required before adoption. Its
`audit_selective_packed_runtime.py` uses the same four-update comparison and
fixed tolerance as v35, on the next immutable v34 checkpoint. CPU tests verify
all/no/partial checkpoint patterns and exception cleanup. Passing these tests
alone does not enable the experiment in a training worker.

After adding selective-recomputation controls, the complete NCM unit suite passed 291 tests in 15.732 seconds. Full GPU qualification remains separate.

### Conditional selective-runtime continuation

`run_selective_training_stage.py` waits for the immutable v34 update-3000
checkpoint and completed evaluation while `audit_selective_packed_runtime.py`
owns the device. It requires four reference and four candidate updates, relative
parameter L2 below 1e-4, maximum absolute difference below 2e-4, unchanged source,
no codec gradients and a warm median speed ratio above 1.05. Only that complete
report and the previously qualified exact JSON corpus reader admit
`train_selective_tower.py` (v37).

The continuation preserves model weights, all optimizer moments, random streams,
batch four, the compact mixture and the global 10000-update schedule. It changes
execution only: alternating source layers are recomputed; the other layers
retain their activation graphs with packed-only weight saves. The actual
no-recompute v35 trainer remains unqualified and is not used. Sessions yield
every 1000 updates, save every 100, and evaluate every 250. Failed admission
restarts the original v34 supervisor at its saved state.

The separate v36 student-conditioning experiment changes the objective on the
original corpus. Its numerical audit and quality results cannot qualify v37,
and v37's runtime audit cannot establish v36's learning benefit. Both preserve
the same twelve-task executable evaluator and require fresh-process replay.

Services are `ncm-selective-runtime-audit-v37.service`,
`ncm-selective-pipeline-v37.service`, and
`ncm-selective-tower-worker-v37.service`. Results use corresponding directories
under `results/`; logs and `progress.json` distinguish waiting, auditing and
actual training. The complete unit suite passed 294 tests before conditional
continuation was launched. The full CUDA audit remains authoritative for
adoption; there is no speed or memory claim until it completes.

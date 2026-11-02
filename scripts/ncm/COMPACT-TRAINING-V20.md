# Compact frozen-weight training

This experiment changes the storage and scheduling of the frozen OxCoder
backbone during NCM adapter training. It does not change the original GGUF
quantization, make an autoregressive draft model, or establish coding quality.
The generator still predicts four levels of learned discrete concepts, and the
frozen byte codec reconstructs each generated chunk. No teacher runs during
student generation.

## Weight storage and backward computation

`gguf_training_weights.py` reads the original Q4_K and Q6_K blocks into immutable
GPU byte buffers. Immediately before each frozen linear operation, a Triton
kernel unpacks the matrix to BF16. The kernel also applies the existing
GGUF-to-Transformers head permutation. `torch.nn.functional.linear` computes the
forward operation and the gradient with respect to its input. There is no
straight-through estimator for weights: they remain frozen. Layer activation
checkpointing releases temporary matrices and recomputes them during backward.

The packed linear module intentionally has no trainable weight parameter.
The trainable inventory remains the residual adapters, concept projection,
position/level embeddings, independent concept classifiers, and termination
head. Checkpoints contain those learned parameters and optimizer state. They
can be loaded by the original resident-BF16 backend for independent evaluation.

Physical placement and clipping semantics are separate. The original model
divides 32 layers into groups of 12 and 20 on two GPUs and clips each device's
trainable gradients separately. Moving all layers to one GPU must not silently
replace that with one global norm. `logical_gradient_groups` preserves the two
canonical groups, including the placement of input and output interfaces.

## Qualification sequence

Use an immutable initialization checkpoint, source GGUF, codec checkpoint, and
prepared training manifest. Run GPU audits sequentially so their memory and
timing measurements cannot be contaminated by competing experiment jobs.

1. `audit_gguf_training_weights` compares unpacked matrices, linear outputs, and
   input gradients against `Source.read` and the original `to_hf` conversion.
   The nine representative Q4_K/Q6_K matrix cases passed bit-for-bit. Admission
   additionally requires `--all-layers`: all 248 frozen linear matrices across
   all 32 blocks must pass. The representative test alone cannot qualify a
   complete model or an optimizer update.
2. `audit_training_runtime` compares reference and shared-prefix execution on
   the expanded corpus at 64, 69, 512, and 2,048 prompt tokens. It checks every
   trainable gradient and adapter/classifier/interface subgroups separately.
3. `audit_gguf_training_runtime` loads the resident and compact backends in
   separate phases. It compares complete outputs and gradients at the same
   four context lengths and records per-device memory peaks and timings.
   Require bit-exact complete outputs and the unchanged numerical gradient
   gates. An initial requirement for bit-exact whole-model gradients proved
   invalid: two independent loads of the unchanged BF16 reference itself differ
   in backward accumulation. `audit_training_reproducibility` records that
   control. `storage_policy` accepts numerical gradients only with this control,
   all-matrix exactness, exact full outputs, and every subgroup below the
   original thresholds. It does not claim bit-exact training. If the reference
   control is deterministic, the strict gradient-equality gate still applies.
   Test storage on the same two-GPU placement first, then separately qualify
   moving the compact model to GPU 1. These are distinct interventions.
4. `audit_packed_training --compact-weights --devices cuda:1` compares a packed
   multi-example pass against sequential shared-prefix passes. Prompts,
   recurrent histories, rotary positions, and target levels stay isolated.
   The loss is an equal mean over examples and levels, not a token-weighted
   mean. Packing remains optional and requires both numerical qualification
   and a measured speed advantage.
5. `train_compact --steps 2 --eval-every 2` executes actual optimizer updates,
   saves a checkpoint, and runs generation. This detects optimizer-state
   memory failures that forward/backward audits do not cover.

The expanded-data shared-prefix audit passed at 64, 69, and 512 tokens. At
2,048 tokens, interface-gradient relative L2 was 0.0513896, exceeding the fixed
0.05 gate. `prefix_policy` therefore limits sharing to 64–512 prompt tokens.
Other lengths use independent reference forwards/backwards for each level.
The gate is not relaxed. A separately successful future audit is necessary to
extend the qualified range. Packed groups likewise cannot contain a prompt
outside this range. The 512–1,024 packing diagnostic does not override it.

Recorded controls are in [validation/v20-controls](validation/v20-controls).
All 248 matrices passed exact checks. Full-model outputs were identical at all
four audited lengths in both placement configurations. The single-GPU path
peaked at approximately 5.4–6.5 GB of allocated GPU memory before optimizer
state; this is not a whole-process deployment-memory measurement. Sequential
compact execution was slower than the resident reference in these tests, so
storage compression by itself is not reported as a training speedup. Its
benefit is memory headroom for packing and freeing one device. The throughput
decision must include the packing and actual-optimizer measurements.

The completed packing comparison measured 1.3613× for 64/69-token prompts,
1.2964× for 64/69/128/256-token prompts, and 0.9695× for 512/1,024-token
prompts. All three passed the numerical checks. Because the final case was
slower, the current coordinator leaves independent-example packing disabled.
These are training-pass measurements, not generation throughput or a coding
score. A future context-specific packing policy also needs a composed
packed-versus-independent-level comparison before deployment.

The training curriculum rechecks complete offline teacher responses in the
immutable addition archives. Two responses contained textual tool-call tags
without an executed tool result, despite a normal API stop reason. All their
training chunks are excluded and recorded in `teacher-curriculum.json`.
Original creator traces and all validation/test rows remain unchanged. New
distillation admission rejects this textual form as well as structured calls.
This is an explicit learning-data filter, separate from the storage controls.

Code checksums and model/data identity are revalidated by the trainer. Updating
an audited implementation invalidates its qualification. Runtime measurements
are local CUDA training measurements, not Mac inference measurements.

## Reproduction commands

Run from the repository root with its training environment activated. Set the
following shell variables to existing, checksum-verified artifacts:

```bash
SOURCE=/absolute/path/to/OxCoder-9B.Q4_K_M.gguf
CODEC=results/ncm-contextual-decoder-v9/best.pt
INIT=results/ncm-pipeline-v19/initialization.pt
DATA=results/ncm-expanded-training-v19
```

Then run the independent audits, using fresh output directories:

```bash
python -m scripts.ncm.audit_gguf_training_weights \
  --source "$SOURCE" --device cuda:1 --all-layers --output results/weight-matrices.json

python -m scripts.ncm.audit_gguf_training_runtime \
  --source "$SOURCE" --codec "$CODEC" --checkpoint "$INIT" \
  --data "$DATA" --device cuda:0,cuda:1 --output results/compact-storage-audit

python -m scripts.ncm.audit_gguf_training_runtime \
  --source "$SOURCE" --codec "$CODEC" --checkpoint "$INIT" \
  --data "$DATA" --device cuda:1 --output results/compact-placement-audit

python -m scripts.ncm.audit_training_reproducibility \
  --first results/compact-storage-audit --second results/compact-placement-audit \
  --output results/reference-reproducibility.json

python -m scripts.ncm.audit_packed_training \
  --source "$SOURCE" --codec "$CODEC" --checkpoint "$INIT" \
  --data "$DATA" --compact-weights --devices cuda:1 \
  --output results/packing-audit.json

python -m scripts.ncm.train_compact \
  --source "$SOURCE" --codec "$CODEC" --init "$INIT" --data "$DATA" \
  --data-expansion --runtime-audit results/ncm-runtime-audit-v19.json \
  --weight-audit results/compact-storage-audit/report.json \
  --placement-audit results/compact-placement-audit/report.json \
  --matrix-audit results/weight-matrices.json \
  --reproducibility-audit results/reference-reproducibility.json \
  --output results/compact-smoke --steps 2 --eval-every 2
```

Add `--packed-audit results/packing-audit.json` only after it passes. The default
1,900-token packing budget counts each prompt once plus all concept positions.
Longer examples run singly. The maximum inference context is unchanged.

`--continue-optimizer` is a distinct operation from a fresh learning stage: it
requires matching accumulation, learning rates, corruption, and seed, retains
the optimizer/RNG/history, and interprets `--steps` as the final total update
count. `--resume` requires the same complete experiment identity and resumes
the output directory's latest checkpoint. Do not combine independent runs in
one directory.

`run_compact_stage.py` provides the durable stage coordinator for the recorded
artifact paths. It rejects active conflicting audit/training units, runs the
optimizer smoke, and launches the learning stage only after the smoke succeeds.
The previous baseline coordinator must be held or stopped before this command;
it must not be allowed to launch a competing trainer. Supply the repository,
training interpreter, and source paths explicitly. Unlike the standard-library
baseline coordinator, this module imports numerical qualification code and must
use the training environment, not `python -S`.

The coordinator preserves evaluated checkpoint hashes. A complete 12/12 result
causes it to stop only its own training service and reload the immutable
checkpoint through the original BF16 backend. Acceptance requires that replay
also pass all 12 unchanged tasks. Training loss, matrix equality, generation
speed, and a completed optimizer run cannot substitute for that result.

## Relevant conversion research

[BYOD](https://arxiv.org/abs/2610.41574) reports LoRA-only AR-to-diffusion
conversion with frozen parent weights, up to 102.4M training tokens, and
25,000 updates. Its output-quality limitations remain relevant. It preserves
token-level interfaces, so it is not evidence that the same schedule converts
a hybrid model into a newly learned concept vocabulary.

[Representation Alignment](https://arxiv.org/abs/2605.06885) aligns layer-wise
student and frozen teacher states using cosine similarity. Its reported
acceleration concerns AR-to-diffusion conversion with matching token positions.
Using it here would require an explicit, tested mapping from token positions
to multi-resolution concept positions; it is not enabled by naming an optimizer.

[Context-tower conversion](https://arxiv.org/abs/2610.02657) reports substantially
better coding generation under its frozen-tower configuration than its matched
in-place conversion. Its trainable denoiser, attention interface, and data budget
differ from this experiment. A separate frozen context encoder is a potential
research control, not a qualified replacement for the current generator.

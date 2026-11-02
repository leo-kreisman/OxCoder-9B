# NCM training acceleration and continuation

## Scope

This work accelerates the experimental OxCoder-derived next-concept generator.
It does not convert standard supervised fine-tuning into NCM by changing a
trainer flag. The model retains the 32 frozen OxCoder text blocks, residual
adapters, four independent concept classifiers, and a frozen hierarchical byte
codec. Training uses the admitted creator datasets and cached OxCoder teacher
responses described in [version 17](EXPANDED-GENERATOR-V17.md).

The training host has two RTX 5060 Ti cards with 16 GiB each. The backbone is
resident BF16 reconstructed from the validated Q4_K_M source, with a 12/20 layer
split. These training measurements do not establish Mac throughput or an 8 GiB
inference footprint. A faster training pass also does not establish usable code
generation. Version 17's update-200 development evaluation remained 0/12.

## Research and applicability

Primary sources checked on 2026-11-02:

| Project | Relevant mechanism | Applicability to this experiment |
|---|---|---|
| [Unsloth Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | Specialized hybrid-model kernels and memory management; vendor reports 1.5× training speed and 50% less VRAM against its FA2 setup | Relevant family, but its ordinary SFT trainer does not implement our hierarchical codes. The separate ~12× MoE result does not apply to dense OxCoder9B. |
| [Unsloth retained checkpoint layers](https://github.com/unslothai/unsloth/pull/14404) and [PyTorch checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/) | Spend available activation memory to avoid some backward recomputation | Implemented as an opt-in, per-device retained-layer policy; requires a whole-model gradient and memory audit. No Unsloth trainer is installed. |
| [VeOmni Qwen3.5 shared-prefix training](https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html#shared-prefix-training) | Compute a repeated prompt once and branch recurrent state/KV while preserving gradients | Directly relevant to four NCM levels sharing one prompt. Adapted below for continuous concept inputs and bidirectional within-level attention. |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | Fused normalization, MLP and loss kernels | Candidate component optimization, not an NCM trainer. This generator scores 4,096 concepts per level, not the original 248,320-token vocabulary; large-vocabulary loss savings cannot be assumed. Not adopted here. |
| [Chronicals](https://github.com/Ajwebdevs/Chronicals) | Fusion, packing, LoRA optimization | Its advertised 3.51× full-fine-tuning comparison uses Qwen2.5-0.5B on A100-40GB. That result does not measure a Qwen3.5 hybrid NCM conversion on these cards. Not installed. |
| [NCM upstream](https://github.com/HazyResearch/ncm) | Multiscale residual-VQ representations and parallel level prediction | Defines the research target. It does not supply a verified OxCoder architecture-conversion recipe. The [training-code issue](https://github.com/HazyResearch/ncm/issues/10) must not be mistaken for a released end-to-end trainer. |

No source above establishes a hundredfold reduction in time to a useful
OxCoder NCM model. Framework throughput, optimization convergence, data quality,
and free-running generation quality are separate measurements.

## Shared-prefix implementation

Ordinary training computes four independent sequences:

```text
prompt + 4 concept positions
prompt + 16 concept positions
prompt + 64 concept positions
prompt + 128 concept positions
```

The shared representation contains `P + 212` positions instead of
`4P + 212`. Token-local projections, normalization, MLPs and adapters process
this compact representation. Each level receives exactly the same earlier-code
conditioning as in the original objective. The first level is not conditioned
on target labels, and labels are never inserted into the shared prompt.

`shared_training.py` supplies three constraints:

1. Full-attention layers use a block mask. Prompt queries attend causally to
   the prompt only. A concept level attends to the prompt and all positions of
   its own level. Other concept levels are masked out. Position IDs restart
   at `P` for every branch, preserving the original RoPE positions.
2. Causal convolution computes the prefix once. Every suffix receives only
   the final three prefix rows as look-back context for the width-four kernel.
   Its temporary context outputs are discarded. It never reads another level.
3. GatedDeltaNet exports state at a multiple-of-64 prefix boundary. The
   remaining prefix tail is replayed before each suffix. This preserves the
   recurrent kernel's chunk partition instead of arbitrarily splitting a
   chunk. State indexing and reuse retain autograd links, so all suffix
   gradients accumulate into their shared prompt state.

The GDN index plan is vendored unchanged from VeOmni revision
`c6099b48c4b5ee336971d85398d792f2f3118ec5`; its Apache-2.0 license and attribution
are under [vendor](vendor/README.md). The entire VeOmni framework is not a
runtime dependency. Full attention continues using the existing SDPA backend.

All four level losses are combined before a single backward pass. The original
mean-of-level CE and final-level termination BCE weights remain unchanged.
Prompt gradients are retained; no detached inference cache is used. Prompts
shorter than 64 tokens use the original four-pass path. Head-only warmup also
uses the original path. Inference and development generation remain on the
original uncached path, so evaluation does not silently switch algorithms.

The mixer substitutions are scoped to a single process and remain active during
checkpoint replay. Do not share this training process with concurrent inference
threads. Other processes and inference servers are unaffected.

## Numerical and performance qualification

`audit_training_runtime.py` loads one immutable generator checkpoint and uses
real training examples nearest 64, 69, 512 and 2,048 prompt tokens. The 69-token
case exercises a nonempty GDN tail. Every variant receives identical weights,
labels and corrupted earlier-level inputs. No optimizer update occurs.

The audit checks all trainable gradients and separately checks adapters,
classifiers, and projection/position/level interfaces. Predetermined acceptance
thresholds are:

* Finite outputs and gradients, with no missing trainable gradient.
* Output-logit relative L2 error below 0.02.
* Gradient relative L2 error below 0.05 and cosine above 0.99, globally and in
  each parameter group. These are BF16 numerical tolerances, not bit identity.
* Every required prompt case passes; an out-of-memory variant fails.

CUDA synchronization surrounds the timed four-level forward/backward. Warmup
is excluded and individual repetitions, median latency, and per-device peak
allocated memory are recorded. The audit does not include optimizer stepping,
checkpoint I/O, evaluation, or resident optimizer buffers. A live continuation
must separately pass those memory and gradient checks before adoption is
considered operationally successful.

The convolution baseline explicitly unwraps HF's PyTorch fallback. Merely
adding an extension to `PYTHONPATH` can otherwise change decorated dispatch and
make a supposed reference-versus-extension comparison compare the same backend.
The preliminary `ncm-runtime-audit-v18` run is not the qualification report.

### Qualified measurement

The [complete qualification report](validation/v18-controls/runtime-audit.json)
uses the immutable version-17 update-260 checkpoint, SHA256
`65fad6588112487f4201e79fa1c93c4385b0e945c6b55d4a5dad846bb91a70c3`.
All eight reference/shared cases passed, including separate gradient checks
for adapters, classifiers and interfaces. Three timed repeats follow warmup.

| Actual prompt tokens | Four-pass reference | Shared prefix | Training-pass speedup |
|---:|---:|---:|---:|
| 64 | 1.1988 s | 0.4903 s | 2.45× |
| 69 | 1.2544 s | 0.5049 s | 2.48× |
| 512 | 2.8555 s | 0.9760 s | 2.93× |
| 2,048 | 11.6136 s | 3.2376 s | 3.59× |

At 2,048 tokens, peak allocated CUDA memory was 5.747 GiB on GPU0 and
9.623 GiB on GPU1 for shared training, versus 5.719 and 9.607 GiB for the
reference. These figures exclude resident optimizer state and CUDA allocations
outside PyTorch's allocator; they are not total-machine or inference memory.
Logit relative L2 errors were about 0.28–0.35%; global gradient errors were
about 1.1–1.3%. The largest individual parameter-group error was 4.35% for
interfaces at 69 prompt tokens, below the pre-set 5% limit. Results are not
bit-exact, and the audit is finite rather than exhaustive.

Only shared-prefix training with the reference convolution and normal layer
checkpointing is selected. The isolated convolution speedup did not justify
enabling it in the combined runtime. Keeping extra activations exceeded the
allocator cap in the long-prompt pilot and is not selected. Subsequent pilot
variants after that OOM are not clean memory comparisons; the final shared-only
qualification used a fresh process.

The live continuation restored update 260 and performed subsequent Muon/AdamW
updates with resident optimizer state. Updates 261–266 selected the same six
examples as the interrupted original run. Update 261, a 54-token prompt on the
unchanged fallback path, reproduced the original loss exactly. Across all six
replayed updates, including that first slower startup update, total update time
improved **2.51×**. The five shared-prefix updates individually improved
**2.64–3.27×**. This short paired check includes optimizer work but excludes
checkpoint/evaluation I/O; rounding allows parameter trajectories to diverge
after the first changed update. See
[resume and live timing](validation/v18-controls/resume-and-live-timing.json).

The live service is `ncm-independent-v18.service`, with progress and checkpoints
under `results/ncm-independent-v18`. It continues the original 2,000-update
schedule. A separate `ncm-evaluation-preservation-v18.service` retains
hash-matching evaluated checkpoints. Check live status rather than treating
this document as a live progress counter:

```bash
systemctl --user status ncm-independent-v18.service
tail -n 10 results/ncm-independent-v18.log
cat results/ncm-independent-v18/progress.json
```

The [CPU test log](validation/v18-controls/cpu-tests.txt) records 89 passing
tests. Run CPU tests without `results/ncm-kernels` on `PYTHONPATH`; installing
the GPU FLA dispatch into a CPU-only test process sends CPU tensors to Triton.
That environment error is separate from the GPU audit, which uses FLA.

```bash
export PYTHON=/absolute/path/to/the/training/environment/bin/python
export PYTHONPATH=results/ncm-causal-kernels:results/ncm-kernels:results/ncm-deps:results/ncm-data-deps
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m scripts.ncm.audit_training_runtime \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint /absolute/path/to/immutable-generator-checkpoint.pt \
  --data results/ncm-contextual-expansion-v17 \
  --output results/ncm-qualified-runtime-audit.json \
  --shared --keep 0 --lengths 64,69,512,2048 --repeats 3
```

The optional convolution package is not required for `--shared` with the
reference convolution. Add `--convolution` only when testing its separate
variant. `--keep 2,4` tests retained-checkpoint-layer variants. Independent
passing variants do not authorize combining them without testing that exact
combination.

## Resume without discarding learning

Freeze an existing saved checkpoint before altering trainer source. Record its
hash and update number. Start a new output directory with
`--continue-checkpoint`, the same original initializer, and all the original
learning/data arguments. The validator rejects changed learning rates, seed,
codec, source, dataset, loss, corruption or schedule. Model, optimizer and RNG
are restored. Execution changes require a complete passing audit bound to the
model/data and exact runtime source hashes.

For the version-17 expanded-data lineage, retain `--data-expansion` as well as
`--continue-checkpoint`: the former verifies the original expansion against its
original initializer; the latter restores existing optimizer/RNG state. Their
combination does not reset training or permit a different expansion.

```bash
"$PYTHON" -m scripts.ncm.train_independent \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init results/ncm-independent-v16b/latest.pt \
  --data results/ncm-contextual-expansion-v17 \
  --output results/ncm-independent-v18 \
  --retain-heads --data-expansion --head-warmup 0 \
  --steps 2000 --seed 17001 --allocator-cap-gib 11.25 --eval-every 100 \
  --continue-checkpoint /absolute/path/to/immutable-version-17-checkpoint.pt \
  --runtime-audit results/ncm-qualified-runtime-audit.json \
  --shared-prefix-training
```

The default remains the original runtime. An incomplete or failing audit cannot
enable acceleration. Resume the new directory with the same arguments plus
`--resume`; do not edit hashed implementation files during training. A resource
guard or failed numerical gate must retain the last checkpoint, never label it
a usable release, and never replace generated answers with teacher answers.

## Quality and release decisions

Track actual source/teacher examples sampled, development concept loss,
free-running valid UTF-8 and termination, executable-code results, and full
response reconstruction separately. Cached teacher examples remain part of the
training data; a speed audit does not generate additional teacher responses.

A passing speed audit authorizes only an execution optimization. A usable model
requires successful free-running code on unseen tasks, a broader held-out test,
and separate inference-memory/latency qualification. Repeatedly checking the
same twelve development tasks is not independent final validation.

The continuation's first evaluation, at total update 300, still passed **0/12**
development coding tasks and **0/6** free-running chunk exactness/valid-UTF8-and-EOS
checks. Training continued after this measurement. The evaluated checkpoint
was preserved, and its hash is recorded in
[evaluated-300.json](validation/v18-controls/evaluated-300.json). See the
[coding report](validation/v18-controls/coding-300.json) and
[chunk report](validation/v18-controls/chunks-300.json). The acceleration is
demonstrated; a usable NCM model is not yet demonstrated.

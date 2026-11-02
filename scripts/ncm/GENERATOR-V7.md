# Residual-geometry generator adaptation

This is a Linux/CUDA research stage. It does not provide a usable OxCoder NCM
release, an 8GB inference result, or a Mac/Metal runtime. The original Q4 GGUF
and its ordinary autoregressive runtime are unchanged. No autoregressive answer
generator or repair fallback is used by this experiment.

## Completed failure analysis

The version 6b generator completed 300 updates and scored 0/12 elementary Python
function tasks. Version 6c completed 1,000 further updates, also scoring 0/12.
Its final loss was 3.319997. These measurements invalidate any inference that
lower concept loss alone demonstrates retained coding ability.

`diagnose_generator.py` supplies successively more correct concept levels while
asking the model to predict the remaining levels. On a training code chunk and
two development chunks, supplying the first four correct levels still yielded
only approximately 12–15% correct non-padding bytes. Supplying all five levels
reconstructed the selected chunks exactly. Thus the generator, and not solely
the selected codec reconstruction, is failing.

A separate control froze the backbone and fit the original linear output head
to one training chunk's cached hidden features. After 51 AdamW updates it reached
100% teacher-forced code accuracy and generated that exact chunk without oracle
inputs. This establishes that the path can memorize an example; it is **not** a
generalization result, a complete function, or a deployable model. The probe
weights are saved separately and are never substituted for the coding candidate.

## Cached hybrid prompt states

`prefix.py` prefills the complete prompt in 512-token chunks and stores each
layer's attention/recurrent state. Each concept level receives an independent
fork of those states. Target-side attention remains bidirectional within its
level; prefix attention is causal. DeltaNet remains causal. A one-position
target gets a masked dummy position to select FLA's differentiable chunk path.
Real positions cannot attend to that dummy key.

During training, prefix states are detached. Gradients reach target-side
adapters but **do not** traverse prompt processing. This is a deliberate
truncated-gradient approximation, not equivalence to full-context training.
Adapter parameter-version checks invalidate states after each optimizer update.
Checkpoint recomputation creates fresh state forks, avoiding mutation of tensors
needed by backward. No weights are streamed from SSD during computation.

The CPU hybrid-model tests check forward equivalence, state isolation, stale
state rejection and target-input gradients. The full model audit separately
compares cached and uncached execution on synthetic token-ID prompts of length
64, 512 and 2,048. Forward relative-L2 tolerance is 0.02; target-input gradient
tolerance is 0.05. The first BF16 accumulation configuration failed the gradient
gate with a relative error of 0.072. That failed report is retained.

Disabling reduced-precision BF16/FP16 matmul accumulation passed the declared
gates: maximum measured forward relative error 0.01067, target-gradient errors
0.02057 and 0.03286, gradient cosine similarities 0.99994 and 0.99949. Discrete
code argmax agreement was not universally exact. This is approximate numerical
agreement, not bitwise equivalence. At the 2,048-token synthetic prompt, all five
levels took 4.395 seconds uncached versus 1.013 seconds with prefix preparation
included. These timings are **not coding throughput or output tokens/s**.

## Geometry-based output prediction

The previous shared unrestricted 4,096-class head is replaced by five separate
projections into the frozen 128-dimensional residual-codebook spaces. For level
`l`, let `r = W_l h` and let `E_l` be its frozen codebook. Its logits are:

```
(2 r E_l^T - ||E_l||²) / mean(||E_l||²) * exp(temperature_l) + bias_l
```

This distance-scoring design follows upstream NCM's `_TiedCodebookHead`, pinned
at `c70274273f489abc73832b2b98584a8483b6b203`; the MIT notice is included in
`NCM-UPSTREAM-LICENSE.txt`. The experiment additionally supervises the predicted
continuous residual. Initial biases use Laplace-smoothed frequencies computed
only from admitted training codes. Rare or unseen codes are not masked out.

The first geometry-head warmup, `ncm-generator-v7`, was stopped because of
persistently large losses. The `v7b` control added `--head-input-norm`, applying
parameter-free RMS normalization before each residual projection. This alone
did not correct the loss. Both stopped runs retain checkpoints and measurements.

A codebook probe identified a substantial imbalance: the first codebook's mean
squared norm is 1.872 across all entries, while some active vectors have norms
near 16. Its many low-norm unused entries are strongly preferred by uncalibrated
distance scoring at initialization. This is distinct from input-feature scaling.

Stage `v7c` additionally uses `--calibrate-head`. For each level, let `s` be the
training-frequency-weighted mean squared code norm and `m` the all-code mean.
Initialize the temperature multiplier to `m/s`, the residual projection to zero,
and the bias to `log(smoothed_training_frequency) + ||E||²/s`. The initial logits
then equal the smoothed empirical log-priors, without masking any codes. The
temperature is bounded; the compensation uses its actual bounded value. The
calibration test checks this equality even with severely unequal code norms.
Loss components are logged separately to distinguish classification, residual,
decoded-byte and termination failures.

The calibrated run starts again from v6c, not from either unstable head. Feature
normalization and calibration are recorded in checkpoint identity. This is a
new stage, not a hidden optimizer resume.

The loss is mean code cross-entropy across the five levels, plus 0.2 times mean
normalized residual MSE, plus final-level decoded-byte CE weighted 0.2 and
response-end BCE weighted 0.1. Target bytes appear only in losses and codec target
construction; current-level answers are never fed into that level's generator.

For the first 100 updates only the new heads and response-ending head train.
Thereafter Muon updates hidden adapter/projection matrices, with AdamW for heads,
embeddings and other parameters. Every adapter must receive a finite nonzero
gradient. The pretrained backbone matrices remain frozen, dequantized BF16.

## Conditioning and data mixture

The prompt limit increases from 1,024 to 4,096 tokens. Oversized examples are
rejected explicitly; neither source messages nor earlier response chunks are
truncated. This is an experimental training limit, not a model context claim.

The prepared candidates contain 12,373 training chunks: 12,033 Alpha and 340
Agent. Of those, 3,543 belong to responses tagged as containing code or tool-call
syntax. The tag is not a correctness verifier. Exact-codec curriculum admission
is applied afterward and its retained/rejected counts are recorded separately.
Validation and test examples are never admitted into training or filtered to
hide reconstruction errors.

Sources remain the bounded pinned subsets of `TeichAI/Ox-Alpha-10k` and
`TeichAI/DeepSeek-v4-Pro-Agent`, plus the existing 33 accepted OxCoder teacher
responses. This is response-level distillation, not hidden-state/logit matching,
and not training on the entirety of either dataset.

Sampling allocates 80% probability to coding-tagged responses when available.
Within each category it selects source/teacher-provenance buckets with weights
proportional to the square root of response count, then a response uniformly,
then one of that response's chunks uniformly. It does not give tiny final-chunk
strata equal probability to large source strata. `sampling.json` records counts.

## Reproduce and inspect

Use the environment documented in `GENERATOR-V6.md`, including the isolated,
audited FLA backend. From the repository root:

```bash
python -m scripts.ncm.prepare_geometry \
  --data results/ncm-muon-distillation-v5/data \
  --corpus results/ncm-corpus-v2 \
  --teacher results/ncm-muon-distillation-v5/teacher \
  --output results/ncm-geometry-data-v7 --prompt-tokens 4096

python -m scripts.ncm.audit_prefix \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --checkpoint results/ncm-generator-v6c/latest.pt \
  --output results/ncm-prefix-audit-v7b.json --full-precision-reductions

python -u -m scripts.ncm.train_geometry \
  --source /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-coverage-decoder-v6/best.pt \
  --init results/ncm-generator-v6c/latest.pt \
  --data results/ncm-geometry-data-v7 \
  --prefix-audit results/ncm-prefix-audit-v7b.json \
  --output results/ncm-generator-v7c --steps 2000 \
  --head-warmup 100 --eval-every 250 --save-every 25 \
  --head-input-norm --calibrate-head
```

Only one training process may own the output directory. `--resume` restores
weights, all optimizers and RNG state; source, codec, prepared data and recipe
identities must match. Increase `--steps` only after evaluating prior results.
The local launch uses a systemd user service with 24GiB host memory maximum,
20GiB high watermark, zero swap allowance, four-core CPU quota, low IO priority
and a twelve-hour runtime ceiling. GPU admission and periodic headroom checks
preserve room for other applications.

```bash
systemctl --user status ncm-geometry-v7c
journalctl --user -u ncm-geometry-v7c -n 10 --no-pager
cat results/ncm-generator-v7c/training.json
cat results/ncm-generator-v7c/curriculum.json
cat results/ncm-generator-v7c/coding-250.json
```

Every 250 updates, training saves a checkpoint and runs free-generation chunk
diagnostics and the twelve isolated executable function tasks. Checkpoint hashes
bind those reports to the exact evaluated parameters. These reused tasks are
development diagnostics, not independent professional qualification.

## Qualification still required

The complete codec gate remains failed at 28/30 exact development responses.
Even if elementary generation improves, the candidate still requires fresh
held-out coding tests, multiple complete responses, a source-model comparison,
serialization parity, and measured deployment memory/latency. The custom codec's
coarse-code semantic meaning is unproven. The presence of all 32 source layers
does not establish retention of source coding competence.

All artifacts remain explicitly `quality_ready=false`. A completed training
process does not change this flag or publish a release.

## Measured progress and rejected batching optimization

The calibrated stage completed its head warmup and saved through update 875 with
finite nonzero gradients through all 32 adapters. Update-250, update-500 and
update-750 executable diagnostics all scored **0/12**. Its final-level predictions
remained poor. Calibration fixes initialization; it did not establish coding
competence. The stage was stopped after a residual-scoring control identified a
conflicting objective, described in [the version 8 experiment](GENERATOR-V8.md).

An optional `forward_levels` path evaluates concept levels as independent padded
batch members sharing a detached prefix. This can reduce repeated weight reads.
The CPU tests check output/gradient agreement, padding isolation and detached
warmup. Full-model CUDA tests used synthetic 64- and 1,024-token prompts and
predeclared relative-L2 tolerances of 0.02 for logits and 0.05 for gradients.

Batching all five levels failed the gradient tolerance (maximum 0.0753). Keeping
the first level separate and batching the remaining four also failed (maximum
0.0741). Logit errors were small, and the warmed 1,024-token test reduced the
measured forward/backward interval from approximately 1.42s to 0.82s, but this
does not override the failed numerical gate. These are synthetic training timings,
not inference tokens/s. **Neither batching variant is enabled in the current
training run.** Reports are retained, without relaxing the threshold.

The trainer rejects `--batch-levels` unless supplied a passing `--batch-audit`
bound to the initialization/continuation checkpoint. No such passing full-model
report currently exists. `--continue-checkpoint` is an explicit new-stage path
retaining compatible geometry weights, optimizers and RNG, with lineage recorded;
it is not used to bypass an audit or reset a failed quality result.

Sequential training resumed from the immutable update-500 checkpoint using the
same recipe and `--resume`; that service was `ncm-geometry-v7c-resume`. Both v7c
services are now stopped. The recorded checkpoints and failed evaluations remain
available; version 8 uses an explicitly different head objective and output path.

The small control reports, including the failed audit, are preserved under
[`validation/v7-controls/`](validation/v7-controls/). The then-current 43 unit tests passed,
including geometry-head gradients, scale normalization, checkpoint restoration,
training-only sampling and oracle-input isolation. Unit tests establish the
implementation's behavior; they are not generated-code qualification.

# Muon integration for the OxCoder NCM work

The [training run guide](TRAINING-RUN.md) documents the supervised job combining
both source datasets, expanded OxCoder response distillation and codec continuation.

Muon is selectable for training eligible hidden matrices. It is not a model
conversion algorithm or an inference accelerator. The acceptance target remains
a usable OxCoder-derived NCM model, evaluated on executable generated code and
the deployment memory budget. Optimizer integration does not satisfy that target.

## Evidence and choice of implementation

The [Moonshot paper](https://arxiv.org/html/2502.16982v1) reports approximately 52%
of AdamW's training FLOPs for comparable performance in its compute-optimal
pretraining experiments. This is not a prediction of elapsed time for an existing
9B model's architectural adaptation. Its Qwen2.5-7B fine-tuning comparison did
not establish a Muon advantage, and it discusses pretraining/fine-tuning optimizer
mismatch. Changing the optimizer can therefore help, do little, or hurt depending
on the training stage. The retained AdamW baseline is necessary.

The installed, pinned PyTorch `2.11.0+cu128` already includes
`torch.optim.Muon`; the algorithm comes from that implementation. The later
[FP32-master continuation](MASTER-OPTIMIZER-V22.md) adds a precision and
checkpoint wrapper around it for BF16 adapters, without replacing its algorithm.
Use the [PyTorch API documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
and [author's repository](https://github.com/KellerJordan/Muon) for the algorithm.
The local code uses five Newton–Schulz steps, momentum 0.95, Nesterov momentum,
and `adjust_lr_fn="match_rms_adamw"`. This permits a useful initial comparison
using the AdamW learning rate, but does not establish that it is optimal.

## Parameter assignment

`anvil.make_optimizers` now accepts `muon` separately from the ANVIL-derived
implementation. Prior ANVIL results are not treated as Muon results.

- Muon: trainable 2D hidden matrices, including the NCM input projection and
  residual adapter matrices when training the generator.
- AdamW: vocabulary/output heads, all `nn.Embedding` tables (including VQ books),
  learned positional/level embeddings, biases, norms and other non-2D parameters.
- Frozen source weights: no optimizer allocation or updates.

The factory rejects a Muon request if no eligible hidden matrix exists. In
particular, the tokenwise codec control has only embedding/position/output
parameters and cannot benefit from switching its optimizer to Muon under this
partition. Reporting an AdamW-only run as a Muon experiment would be misleading.

Every matrix optimized by Muon resides wholly on one device. The current
layer-wise model partition does not split a matrix across devices, so there is
no distributed matrix reconstruction step here. This does not imply that full
9B fine-tuning would fit on the same hardware; frozen-backbone adapter training
and full-parameter optimization have different memory requirements.

## Commands

The official-codec adapter trainer accepts a third comparison arm:

```bash
bash scripts/ncm/run_limited.sh python -m scripts.ncm.adapt_codec train \
  --artifacts results/ncm-official/xsum --data results/ncm-corpus-v2 \
  --cache results/ncm-official/xsum/coding-cache.pt \
  --output results/ncm-codec-muon-v4 --steps 1000 --eval-every 250 --optimizer muon
```

`--muon-lr` optionally sets the matrix learning rate. Without it, Muon uses
`--lr` with RMS matching; other parameters use AdamW at `--lr`. Both arms in this
decoder comparison use zero weight decay, matching the recorded AdamW control.

The byte-codec and generator trainer also accepts `--optimizer muon` and
`--muon-lr`. It retains its existing weight decay of 0.01 for both parameter
groups. A generator command must still supply the required passing codec and
complete-response gates. Choosing Muon neither bypasses those gates nor relaxes
the three-update limit on structural tests. Do not run a large job merely to
reproduce the short optimizer comparison.

`OptimizerBundle` saves/restores the disjoint optimizers together, verifies
group type/count and refuses duplicate parameter ownership. Checkpoint identity
records the optimizer, PyTorch version, learning rate and scaling. Muon cannot
silently resume from an AdamW or ANVIL checkpoint. Historical default AdamW
checkpoints retain their original identity format and update configuration.

## Local verification

The integration test suite passes nineteen tests, including a new partition
check and exact continuation after checkpoint restoration for mixed Muon/AdamW.
A CUDA decoder run completed 1,000 updates with finite losses. Its outcome and
timing are stored separately under `validation/muon/`; this checks operability
and the controlled comparison, not NCM coding-model usability.

To select an optimizer for a later representation that passes reconstruction,
compare held-out executable-code pass rate versus total training time with the
same examples, initialization, sampling and evaluation budget. Include encoder,
teacher, validation, saving and compilation costs in end-to-end accounting.
Preserve the original OxCoder baseline throughout. The actual usable-model
criterion takes precedence over either an optimizer's name or its speed claim.

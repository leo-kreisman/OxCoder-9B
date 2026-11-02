# Adaptive residual supervision for the native NCM decoder

This experiment continues the frozen-OxCoder-encoder/native-concept-decoder
architecture documented in [version 25](CONTEXT-DECODER-V25.md). It changes the
training policy; the source encoder, codec, decoder architecture and standalone
inference algorithm remain the same. It is not a usable-model release.

## Diagnostic motivating the change

The version 25 update-500 checkpoint passed 0/12 executable coding tasks. Its
fixed 64-window validation mean concept cross entropy was 4.07253 and its free
non-padding byte accuracy was 0.077303. Sixteen fixed training windows had
cross entropy 2.48808 and free byte accuracy 0.159879. Conditional label
prediction was improving without producing complete correct programs.

`audit_residual_recovery.py` checks whether the frozen representation can repair
an erroneous early prediction. It first predicts all concept levels normally,
then preserves a selected prefix of those predictions and computes the later
residual codes using the **known target's encoder latents**. This is an oracle
diagnostic. Its output is never supplied to the executable evaluator and is
not an inference mode.

| Preserved predicted levels | Exact oracle training windows / 16 | Exact oracle validation windows / 64 |
|---|---:|---:|
| 0 | 16 | 64 |
| 1 | 16 | 64 |
| 2 | 15 | 62 |
| 3 | 7 | 23 |

Thus, the later codebooks can repair some early mistakes if supplied the correct
residual labels. This does not show that a generator can infer those labels
from a prompt. The experiment tests that separate question.

Evidence: `results/ncm-residual-recovery-v25/report.json`. Diagnostic checkpoint
SHA256: `2635052a085f45780e1fd5e17066b974d5c3584a45fdd719ee1d5abe02c5abb8`.
The first two version 25 executable checkpoints, updates 500 and 1,000, both
scored 0/12. These results must not be replaced with oracle reconstruction scores.

## Objective and information boundary

For target latent sequence `h`, previous reconstruction `z`, level resampling
operator `R_l`, and frozen codebook `E_l`, the correction label is:

```
y_l = argmin_k ||R_l(h - z) - E_l[k]||²
input_l = R_l(z)
z_next = z + upsample(E_l[selected_l])
```

A Bernoulli choice is made once per example, not independently at every level:

- Clean examples use `selected_l = y_l`. Their labels and inputs exactly match
  the original frozen codec's quantization trajectory.
- Self-conditioned examples use `selected_l = argmax model(input_l, prompt)`.
  Every input depends only on the prompt and the model's earlier predictions.
  Target latents define supervised labels but never construct the model inputs
  in this branch.

The mixture probability is 0.5. Target quantization and accumulated residuals
use FP32. Student proposals use the same BF16 policy as standalone inference.
The rollout is detached; the combined block-diagonal training forward computes
gradients against its resulting labels. No teacher or reference answer runs at
inference. There is no gradient through discrete sampling.

This replaces the version 25 random-code corruption with self-conditioned
trajectories **and** adaptive labels. It changes two related training choices;
results cannot isolate their separate causal effects. Adaptive training loss
also has different labels, so comparing its raw value directly with version 25
training loss is invalid. Both runs retain the original canonical-label
validation and unchanged complete-program tests.

## Parent, data and optimizer continuation

The immutable full parent checkpoint is version 25 update 1,000:

```
results/ncm-context-residual-parent-v26.pt
21a7612ef4710fb8d62ab1a54ff19cb49e7c0ba8e0e789036c5d004f4b07dfd3
```

The branch restores model weights, optimizer moments, shuffled-epoch position,
Python sampler state, and CPU/CUDA random states. It retains the original
5,000-update learning-rate schedule, global batch 32, microbatch 8, Muon/AdamW
split, clipping, precision and fixed corpus. The original version 25 run
continues independently. Data includes both creator sources and admitted,
publisher-test-checked offline distillation as documented in version 25.

Resume requires unchanged source/data/codec/configuration and implementation
hashes. A CPU integration test checks exact model, optimizer, sampler and RNG
continuation after a saved interruption. Additional tests prove clean-label
parity and that changing the target cannot change fully self-conditioned inputs.

## Commands

Run in the pinned Linux CUDA environment. The branch uses a separate directory:

```bash
python -m scripts.ncm.train_context_residual \
  --cache results/ncm-context-pipeline-v25/features.json \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init results/ncm-context-residual-parent-v26.pt \
  --output results/ncm-context-training-v26 \
  --device cuda:0 --microbatch 8 --on-policy 0.5
```

To resume, use the identical command with `--resume`; do not change the parent
file or implementation. The trainer writes a full `latest.pt` every 100 updates
and immutable `evaluated-N.pt` snapshots every 500 updates. A
`STOP_AFTER_CHECKPOINT` marker requests a clean stop at the next saved boundary.

`monitor_context_comparison.py` serializes full source-model evaluations of both
runs on one selected GPU. This avoids concurrent full-source evaluators competing
with the branch trainer on GPU0. Both native trainers use roughly 2.7 GB of
PyTorch peak allocation in early observed updates; these are training allocations,
not whole-process inference measurements or an 8 GB deployment qualification.

```bash
python -m scripts.ncm.monitor_context_comparison \
  --root /path/to/OxCoder-9B \
  --output results/ncm-context-comparison-v26 \
  --python /path/to/pinned/python \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --folders results/ncm-context-training-v25 results/ncm-context-training-v26 \
  --training-units ncm-context-training-v25.service ncm-context-training-v26.service \
  --device cuda:1
```

The monitor requires a user systemd session. Service names are explicit; it does
not terminate unrelated applications. Existing completed reports can be reused
only when their checkpoint checksum, twelve task records and counts agree.
A 12/12 result must repeat in a fresh process with the same checkpoint and
identical generated answers before the monitor writes `acceptance.json`.
Training completion and a running service never satisfy this gate.

## Research boundary

The [official NCM implementation](https://github.com/HazyResearch/ncm) provides
multiscale residual tokenization and parallel generation within each level. This
adaptive-residual training extension is an experimental implementation here;
it is not a published conversion recipe from the NCM authors. Their released
XSum/WMT14 codecs were already screened locally and did not reconstruct coding
text reliably. The OpenWebText checkpoint directory currently contains a TODO
rather than downloadable weights at release revision
`d342ce8aaea58830eef3b8ad473664673e41f692`. Reusing those releases cannot currently
be treated as a verified shortcut to a coding-capable conversion.

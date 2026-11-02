# Training-only representation alignment control

This control tests whether the original OxCoder body provides more useful
features for predicting the existing NCM concept labels than the transferred
student interface does. It does not change the active version 20 trainer and
does not supply teacher text or teacher hidden states during student inference.

## Research motivation and difference from the published method

[Repr-Align](https://arxiv.org/abs/2605.06885) aligns corresponding hidden
positions in autoregressive and masked-diffusion models with the same token
inventory. Its reported acceleration is for that conversion, not for NCM.
The present experiment needs a different position mapping because the student
predicts four scales of byte-window concepts. It initially tests the final
hidden layer; it does not reproduce the paper's all-layer objective.

[NCM](https://arxiv.org/abs/2610.14162) uses a learned multiscale codec and
predicts codes within each level in parallel. The existing codec and student
remain unchanged here. A useful representation control would motivate a
separately identified alignment-training stage; it would not establish coding
quality or justify rebranding a teacher-assisted system as an NCM student.

## Exact correspondence and data boundaries

`representation_alignment.py` reconstructs the original prompt from its token
IDs and rejects any failed encode/decode round trip. It tokenizes prompt plus
training target jointly, including tokens that cross the chunk boundary.
Character offsets are converted to UTF-8 byte intervals. For target token `t`,
the feature comes from causal hidden position `t - 1`, which predicts `t`.
Overlapping byte-fallback offsets share each byte's unit weight equally.

The map covers actual text bytes plus the codec's end-of-chunk position. It
excludes PAD positions. End-of-chunk uses the final input hidden state; it is
an explicit alignment convention, not a claim that the teacher predicted an
end-of-response token. Byte features are averaged over each fixed concept
interval at lengths 4, 16, 64, and 128. Empty intervals are masked from alignment
loss. The cosine-loss helper detaches teacher tensors and preserves gradients
only to the student.

`collect_alignment_features` selects distinct conversations independently in
the existing train and validation splits, balancing source buckets. The default
control selects up to 512 training and 64 validation chunks with prompts no
longer than 512 tokens. It never loads the 12 executable evaluation tasks.
Mapping failures are recorded explicitly rather than truncated or replaced.
The mapping-only run passed all 576 selected records.

The original causal body uses the already qualified compact GGUF storage.
Before collecting features it must pass the original AR token fixtures with
zero residual adapters. The optional paired student restores its learned
adapters only for its own feature passes, with gold earlier-level concepts.
Teacher passes always reset adapter output weights to zero. The checkpoint,
codec, source weights, tokenizer, implementation, and each feature file are
checksummed. Validation reconstruction failures remain in the inventory.

## Paired readout experiment

`audit_alignment_readout` trains separate, identically seeded linear concept
classifiers on frozen student features and frozen teacher features. Both arms
use the same selected chunks, update sequence, batch size, optimizer, and
nonpadding concept cross entropy. Only training rows with exact codec
reconstruction receive updates. Validation rows are never optimized or filtered.

The teacher arm has teacher-forced target text. Its score is therefore an
upper-bound representation diagnostic, not a free-generation score. Compare
validation code accuracy and cross entropy per level, not training loss alone.
If teacher features do not improve held-out predictability, do not add an
alignment loss merely because a different paper reported a speedup.

```bash
"$PYTHON" -m scripts.ncm.collect_alignment_features \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --data results/ncm-expanded-training-v19 \
  --student-checkpoint results/ncm-pipeline-v19/initialization.pt \
  --output results/ncm-alignment-features-v21

"$PYTHON" -m scripts.ncm.audit_alignment_readout \
  --features results/ncm-alignment-features-v21 \
  --output results/ncm-alignment-readout-v21
```

Use the established training environment and dependency paths from
[the compact training guide](COMPACT-TRAINING-V20.md). Feature collection needs
its own available GPU allocation; it must not compete with a full model teacher
on the same device. A completed control is not an accepted checkpoint. Actual
student generation and cold checkpoint replay must still pass the unchanged
12 executable tasks without teacher fallback.

## Completed readout control and decision

Feature collection completed for 512 training and 64 validation conversations,
with no offset-mapping exclusions. The original teacher passed its token
fixtures. The 512 selected training chunks reconstructed exactly through the
codec. Under the 512-token prompt bound, the eligible selection contained
Ox-Alpha and NVIDIA examples; it did not cover the longer Agent-source prompts.
The main training corpus continues to include both creator datasets.

After 2,000 updates of each identically initialized readout:

| Concept level | Student-feature accuracy | Teacher-feature accuracy | Student CE | Teacher CE |
|---|---:|---:|---:|---:|
| 4 positions | 7.42% | 15.23% | 5.455 | 5.198 |
| 16 positions | 1.95% | 6.45% | 7.371 | 8.574 |
| 64 positions | 5.27% | 11.04% | 5.932 | 6.373 |
| 128 positions | 35.53% | 4.26% | 3.948 | 9.634 |

These are validation concept predictions with gold conditioning, not generated
functions. Neither arm produced an exact full code-level window in this
validation control. Teacher features improve coarse top-1 accuracy but worsen
cross entropy at two intermediate levels and substantially impair the finest
residual readout. Consequently, full-level representation alignment is **not
enabled** in the active trainer. Coarsest-level-only alignment remains a
separate hypothesis requiring a controlled generation experiment.

The complete results and original-teacher parity report are preserved in
[validation/v21-controls](validation/v21-controls). Offline response collection
resumed after the bounded feature/readout experiment.

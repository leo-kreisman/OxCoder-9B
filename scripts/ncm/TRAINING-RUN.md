# Muon continuation with dataset and response distillation

This job prepares new teacher responses and trains the learned NCM codec.
It does not yet train the 9B generator: previous codecs corrupted complete
responses, so their reconstruction gate still blocks that stage. The objective
remains a usable coding model, not a passing codec metric alone.

## Data and supervision

The pinned corpus `results/ncm-corpus-v2` contains bounded subsets of
`TeichAI/Ox-Alpha-10k` and `TeichAI/DeepSeek-v4-Pro-Agent`. It has 939 training,
122 validation and 155 test responses. See [the corpus guide](FOLLOWUP.md) for
source revisions and selection details. Neither entire upstream dataset is
represented by this run.

The original OxCoder-9B Q4_K_M model serves as the teacher, unchanged. The job
requests up to 32 distinct training conversations per source, using temperature
zero, disabled reasoning, a 2,400-token prompt limit and a 1,536-token response
limit. Only nonempty, normally terminated text responses without tool calls are
accepted. No dataset command or tool call is executed. Accepted responses are
not automatically correct; this is response distillation, not teacher-logit
distillation or a verifiable-reward training run.

Complete responses and source-group splits remain recorded. Byte preparation
retains at most 16 spread training chunks per response and gives evaluation
chunks priority during duplicate removal. The resulting manifest records the
actual number of distilled chunks. Teacher text currently supervises codec
reconstruction; conditional teacher-to-generator distillation is a later stage.

## Optimization

`start_training.py` first collects teacher responses, stops its own teacher
server, prepares chunks, then invokes `train.py codec`:

- Initialize weights from `results/ncm-byte-codec-v3/best.pt`, the existing
  two-layer contextual codec. Architecture mismatches are rejected.
- Start fresh optimizer state and update count; this is not an AdamW checkpoint
  resumed as Muon. The initialization hash and source step enter run identity.
- Use Muon for eligible hidden matrices and AdamW for other trainable parameters,
  as documented in [MUON.md](MUON.md).
- Learning rate 0.0001, batch 16, seed 17, 12,000 additional updates.
- Evaluate validation chunks and atomically save checkpoints every 500 updates.
- Select by validation symbol accuracy, then test complete development responses
  and syntax/Unicode probes. Test-split scores are not used for model selection.

Final `status.json` distinguishes a finished codec run from a passing codec gate.
It always leaves `quality_ready=false` because this command does not produce a
validated coding generator. Even a passing reconstruction gate requires further
generator training and executable-code evaluation.

## Launch and supervision

Run from the repository root with the documented Python environment. The entry
point requires explicit paths to the teacher and llama.cpp server:

```bash
python -u -m scripts.ncm.start_training \
  --output results/ncm-muon-distillation-v5 \
  --teacher-file /absolute/path/OxCoder-9B.Q4_K_M.gguf \
  --server /absolute/path/llama-server
```

This runner currently targets CUDA GPU index 1. It reserves localhost port 18119
for its teacher and refuses to start if occupied. A new output directory is
required. It terminates only child processes it launches and records failures.
Do not launch it a second time against the same output directory.

The workstation launch uses a systemd user service named
`ncm-muon-distillation-v5`, a four-hour runtime limit, a 24 GiB host-memory limit,
no additional swap, four CPU cores' worth of scheduling quota and low CPU/I/O
priority. These are training limits, not proof of an 8GB inference footprint.
Per-process CUDA allocation limits and host/GPU headroom checks remain active in
the codec trainer.

```bash
systemctl --user status ncm-muon-distillation-v5
journalctl --user -u ncm-muon-distillation-v5 -n 30 --no-pager
cat results/ncm-muon-distillation-v5/status.json
tail -f results/ncm-muon-distillation-v5/distillation.log
tail -f results/ncm-muon-distillation-v5/codec_training.log
```

The second log appears when gradient training starts. `teacher/manifest.json`
records accepted/rejected teacher responses; `data/manifest.json` records actual
training composition; `codec/training.json` and `codec/latest.pt` record progress.
`complete-gate.json` appears after final evaluation. Check service exit status as
well as the JSON status: a forced resource-limit termination may prevent a final
status write.

To stop this job and all of its processes:

```bash
systemctl --user stop ncm-muon-distillation-v5
```

To resume interrupted gradient training, use `train.py codec --resume` with the
same data, output, optimizer, initialization checkpoint and hyperparameters as
the recorded command in `status.json`; increase `--steps` only if desired. This
restores both optimizers and RNG state from `codec/latest.pt`. Teacher response
caching can be resumed separately with the recorded distillation command.

## Verification before launch

Twenty unit tests passed, including strict weights-only initialization,
optimizer partitioning, exact optimizer checkpoint continuation and refusal of
failed complete-response gates. A separate two-update CUDA warm-start check is
recorded under `results/ncm-muon-warmstart-check-v5`; it tests the training path,
not model usability. All new checkpoints are local research artifacts.

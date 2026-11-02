# Open-data NCM coding baseline

## Purpose and status

This experiment tests the representation and learning pipeline before another
OxCoder backbone adaptation. It trains the released NCM modules directly with
an independently implemented trainer. It does not convert OxCoder's weights,
produce a 9B model, or qualify a deployment artifact. The original GGUF and
inference services are unchanged.

The previous version 10 adapter job stopped at update 2,077 after the GPU memory
reserve check failed. Checkpoints remain in its original directory. Its failed
coding results are not replaced by this baseline's reconstruction measurements.

Version 11 uses two codec arms: source code only, and 80/20 code/assistant windows.
Each codec is frozen and used to train an identically configured small concept
generator on the mixed corpus. The second comparison is necessary: a codec can
reconstruct text accurately while creating concepts that are difficult to predict.
No result automatically triggers another 9B training job or a release.

Both arms have now completed and failed: each reconstructed **0/1,100** documents
exactly and generated **0/12** passing coding tasks. The
[reconstruction and output-head investigation](REFINEMENT-V12-V13.md) records
the follow-up controls and their measured limits.

## Data acquisition

`open_corpus.py` downloads pinned public files, never executes their contents,
and retains provenance and SHA256 checksums. The initial target is 50 million
accepted source-code tokens counted with `results/tokenizer/tokenizer.json`,
whose checksum is recorded. This count excludes assistant responses, padding
and repeated training exposure. It is a bounded experiment, not a sufficient
training budget established by the NCM paper.

Sources:

| Source | Revision | Role |
|---|---|---|
| [The Stack v3 train](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train) | `1f61b735bc0a5698345ce2196730f24bfa467f33` | Bounded repository sample; Python content inline |
| [CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | `35a59fb025bc0a102f7d96eac09d145b896d487b` | Python-specific supplement to reach the token budget efficiently |
| [Ox-Alpha-10k](https://huggingface.co/datasets/TeichAI/Ox-Alpha-10k) | `ae678ee78880e14251a553a0d0dc8ec6bd6dad6c` | Previously acquired complete assistant responses |
| [DeepSeek-v4-Pro-Agent](https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent) | `abbf7b71b145633b60beb21ddc0d158582fd80dd` | Previously acquired agent responses |
| Original OxCoder teacher responses | Local checksummed manifest | Existing bounded distillation, not newly generated or correctness-verified |

The creator datasets are the existing bounded subsets, not their entire releases.
Exact post-filter source and teacher counts appear in the prepared manifest.
Distillation is included as data; this stage does not perform online teacher
logit matching or generate a new large teacher corpus.

The completed acquisition accepted **50,000,625 source-code tokens in 39,924
files** and **1,617,968 assistant tokens in 1,001 responses**. Its source counts
are 192 Stack files, 39,732 CodeParrot files, 818 Ox-Alpha responses and 183 Agent
responses. Thirty of the assistant responses are existing OxCoder distillations,
all in the training split. There are 36,649 training documents, 2,145 validation
documents and 2,131 test documents. The filters rejected 73,076 duplicates and
40 benchmark-overlap candidates. The corpus SHA256 and counts are recorded in
[the acquisition report](validation/v11-controls/acquisition.json).

Whole-repository Stack shards have low usable Python yield. One initial shard
also exceeded the 600 MiB download limit. The active acquisition uses one bounded
Stack shard and up to twelve CodeParrot shards, shuffled deterministically with
seed 37. Downloaded shards are deleted after processing. Failed acquisition
attempts `ncm-open-corpus-v11` and `v11b` are historical; `v11c` is the replacement.
Only incomplete temporary download files from those failed attempts were removed.

Filters:

1. Keep Python source, 128–32,768 UTF-8 bytes, accepted by `ast.parse` without execution.
2. Stack files must have `license_type=permissive` and not be vendor/generated files.
   CodeParrot uses an explicit MIT/Apache/BSD/ISC/Unlicense/CC0 allowlist.
3. Keep at most eight accepted files per repository. Code repositories split by
   stable lower-case repository-name hash, 90/5/5. Existing assistant conversation
   splits remain unchanged. Splitting precedes window creation.
4. Apply global lexical exact deduplication and a conservative SimHash filter
   with Hamming distance at most three. Text is never normalized in the training
   payload. This filter may reject distinct but similar programs; it is not a
   complete semantic duplicate detector.
5. Reject 32-lexeme overlaps with pinned HumanEval and the full MBPP collection,
   including their training partitions. Exclude recognizable benchmark paths.
   This is bounded contamination screening, not a proof of no benchmark leakage.
6. Require at least 1,000 documents in each split and the requested source-token
   budget before the pipeline proceeds. Exhausting download limits is reported
   as a failed acquisition, not silently called a complete 50M-token corpus.

Stack's ODC-By database license does not replace each file's license. Source
licenses and provenance remain recorded. No dataset files are added to the model
release by this workflow. Magicoder and TinyStories were research candidates;
they are not included in this specific coding ablation.

## Tokenization and conditioning

`open_baseline.py prepare` trains an 8,192-entry byte-level BPE tokenizer using
training target documents only. Both arms share this tokenizer, including its
assistant training text; the ablation changes codec batches only. There is no
text normalization. All accepted
documents must round-trip exactly before training can begin. Reserved PAD/BOS/EOS
literal collisions fail preparation instead of being silently reinterpreted.

The compact token corpus uses uint16 memory maps and checksummed metadata. It
keeps complete source documents, complete assistant responses and original
assistant prompts. The prepared manifest reports counts using the new tokenizer;
these differ from acquisition counts made with OxCoder's tokenizer.

Each codec target is a 128-token window, with EOS on the final window and PAD
after EOS. A generator sees only the preceding 128 tokens, left-padded where
necessary. Assistant prefixes include their original prompts and then previously
written response tokens; source-code prefixes are earlier file tokens. Long
conditioning uses an explicit sliding window. **This is not full-context agent
training and does not preserve a 128K inference window.** Complete-document
reconstruction concatenates windows before decoding text, preserving Unicode
whose byte-level pieces span window boundaries.

The mixture is 80/20 sampled windows, approximately tokens. Padding changes the
actual ratio. Progress reports record cumulative non-padding token exposure
separately for code and assistant material, including EOS. Repeated examples
count as exposure, not additional unique data.

## Released architecture and explicit differences

The code imports a clean checkout of [HazyResearch/ncm](https://github.com/HazyResearch/ncm)
at `c70274273f489abc73832b2b98584a8483b6b203`. A different revision or dirty checkout
is rejected. The upstream MIT license is in `NCM-UPSTREAM-LICENSE.txt`.

Codec settings:

- Released `VQVAE` and multiscale residual quantizer, width 256, four attention
  heads, embedding encoder and one Transformer decoder block.
- Twelve levels `[1,8,16,24,32,40,52,64,76,92,108,128]`, with 4,096 codes each.
- Learned cascaded coarse down/up-sampling; per-level residual convolutions
  with blend ratio 0.5; EMA decay 0.95; commitment cost 0.25.
- Upstream nested fine-level dropout, probability 0.5, minimum one retained level.
- Initialize with upstream k-means on 1,024 training-code windows, at most 8,192
  residual vectors per level and five iterations. Both arms use the same sample.
  EMA sums/counts are initialized consistently; dead-code restarts remain disabled.
- AdamW, peak learning rate 0.0003, 100-update warmup, cosine decay to 10%, gradient
  norm clipping at one. Microbatch eight with eight accumulation steps; 6,000 updates.
  EMA runs per microbatch, not per accumulated optimizer update.

The small generator uses the released `NextConceptTransformer`: six layers,
width 384, six heads, block-diagonal target attention, learned level-zero queries,
RoPE and codebook-tied distance heads. It has approximately 11.93M trainable
parameters. It trains for 3,000 updates with microbatch four and accumulation eight.

Conditioning corruption replaces earlier input codes with same-codebook random
codes, with probability decreasing linearly from 0.1 at the first level to zero
at the last. Cross-entropy targets remain clean. Loss is averaged equally across
levels. Generation starts from learned queries, predicts every level and decodes
the resulting codes; no gold coarse code, autoregressive verifier or answer
lookup is supplied. Prefix caching is disabled for this baseline.

This is not a paper reproduction: vocabulary, corpus, width, generator depth,
training budget, optimizer schedule and explicit k-means bootstrap differ. It is
also not the previous four-level independent byte codec. Passing infrastructure
checks does not establish usable code generation.

## Evaluation and selection

Both arms use the same development samples:

- Periodic codec selection: 64 source documents and up to 100 assistant responses.
- Final selected-codec evaluation: 1,000 source documents plus up to 100 assistant
  responses. Report exact documents, non-padding token accuracy, EOS/PAD fidelity,
  Python syntax, length buckets and unique codes at every level.
- A codec's broad reconstruction gate requires every evaluated document and its
  padding to be exact. High average token accuracy cannot replace this gate.
- Train a small diagnostic generator even if reconstruction is imperfect; this
  is explicitly experimental. Skip it if the finest codebook collapses to one code.
- Generator selection uses free-running concept accuracy within that codec arm;
  also report teacher-forced accuracy, decoded token accuracy and exact windows
  on 64 held-out continuations. Concept scores are not directly comparable across codecs.
- Evaluate 12 reused elementary function tasks with an isolated `bwrap` worker,
  eight 128-token generation windows maximum and predicted EOS required. These
  are development diagnostics, not independent professional qualification.
- Compare arms by executable development passes, exact continuation windows,
  then decoded non-padding token accuracy. A relative winner is not a qualified
  model. `quality_ready` remains false, including when a process exits successfully.

The test partition is preserved and never used for optimization or checkpoint
selection. It is reserved for a later independently specified qualification run.
The current stage does not claim evaluation on every modern coding benchmark.

## Commands

Run from the repository root in the existing PyTorch environment. The measured
development environment is Python 3.10, PyTorch 2.11.0+cu128 and Transformers
5.17.0, with Hydra 1.3.2 and OmegaConf 2.3.0. The base requirements are in
`requirements.txt`; `tokenizers` is also installed through Transformers.
The existing upstream checkout is `results/ncm-upstream`; a fresh environment
must clone `HazyResearch/ncm` and check out the pinned revision listed above.
The assistant corpus paths refer to complete, checksummed outputs of the earlier
`corpus.py` and `distill.py` workflows; they are not bundled downloads.
Additional
data packages can be isolated rather than replacing inference dependencies:

```bash
python -m pip install --target results/ncm-data-deps --no-deps \
  pyarrow==23.0.1 fsspec==2026.2.0
export PYTHONPATH="$PWD/results/ncm-deps:$PWD/results/ncm-data-deps"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

python -m unittest discover -s tests/ncm -v

python -u -m scripts.ncm.open_corpus \
  --output results/ncm-open-corpus-v11c \
  --assistant-corpus results/ncm-corpus-v2 \
  --assistant-corpus results/ncm-muon-distillation-v5/teacher \
  --max-shards 1 --tokens 50000000

python -u -m scripts.ncm.run_open_baseline \
  --corpus results/ncm-open-corpus-v11c \
  --output results/ncm-open-baseline-v11
```

Use fresh acquisition/output paths when rerunning. The pipeline waits for the
named acquisition service only while the corpus manifest is absent. If acquisition
has already finished, no systemd service is required. It runs each phase in its
own subprocess and logs the exact command and output. `--resume` resumes an
existing pipeline; individual phases resume only from an identical recipe and
checkpoint. Do not launch two writers against the same output directory.
Nonblocking file locks reject simultaneous pipeline or checkpoint writers.
The pipeline snapshots and hashes its source files; source changes between
phases fail the run rather than silently changing the experiment.

For manual execution after preparation:

```bash
python -m scripts.ncm.open_baseline prepare \
  --corpus results/ncm-open-corpus-v11c --output results/ncm-open-baseline-v11/data

python -u -m scripts.ncm.open_baseline codec \
  --data results/ncm-open-baseline-v11/data \
  --output results/ncm-open-baseline-v11/codec-code --assistant-fraction 0

python -u -m scripts.ncm.open_baseline generator \
  --data results/ncm-open-baseline-v11/data \
  --codec results/ncm-open-baseline-v11/codec-code/best.pt \
  --output results/ncm-open-baseline-v11/generator-code \
  --steps 3000 --batch 4 --accumulation 8
```

The pipeline also runs the `codec-mixed` arm with `--assistant-fraction 0.2` and
its own generator. Both generators use the same 0.2 mixture. Use either the
pipeline or manual commands, never both against the same directories.

## Resource limits and inspection

The acquisition service is `ncm-open-corpus-v11c`, limited to 6GiB host memory,
two CPU cores and four hours. The training pipeline service is
`ncm-open-baseline-v11`, limited to 10GiB host memory, two CPU cores and 24 hours.
Both disable swap and use low CPU/I/O priorities. Training uses GPU 1 with an
8GiB PyTorch allocator cap and checks external memory headroom. Existing servers
are not stopped. These are training limits, not an 8GB deployment measurement.

```bash
systemctl --user status ncm-open-corpus-v11c ncm-open-baseline-v11
journalctl --user -u ncm-open-corpus-v11c -n 5 --no-pager
cat results/ncm-open-baseline-v11/pipeline.json
tail -n 3 results/ncm-open-baseline-v11/codec-code.log
cat results/ncm-open-baseline-v11/codec-code/progress.json
```

`latest.pt` includes model, optimizer, RNG, sampling state and exposure counts.
`best.pt` is the development-selected checkpoint. `selected-evaluation.json`
binds its checksum. `comparison.json` appears only after both arms finish or
have an explicit collapse skip. `failure.json` reports a stopped pipeline.
Do not interpret a stale progress file as a running process.

Infrastructure validation includes source filters, split/window boundaries,
pre-tokenizer round trips, corpus mutation rejection, clean corruption targets,
absence of finest-target leakage and equality of teacher-forced versus sequential
level logits with forced codes. A three-update GPU interruption/resume control
reproduced all 106 codec state tensors exactly. This control used synthetic test
fixtures and says nothing about the quality of the eventual trained model.

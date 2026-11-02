# Publisher checks for offline distillation

`verify_distillation.py` checks newly generated training answers against the
tests distributed with `nvidia/OpenCodeInstruct`. It executes the publisher's
original answer first. A teacher answer is eligible only when both answers
pass every supported test, with at least one executed assertion per test.

These are training-data checks. They do not replace the unchanged 12-task NCM
generation evaluation, and they do not establish that a teacher or student is
generally correct. The original OxCoder model is used only for offline data
generation; NCM inference has no teacher fallback.

## Inputs and isolation

The verifier reads a SHA256-checked admission archive and its pinned source
Parquet file. It takes a snapshot of accepted teacher-cache records, joins them
to the publisher's training examples by their original IDs, and records each
cache file's SHA256. It never reads the 12 task prompts or their answers.

The source Parquet column `unit_tests` contains JSON lists of Python snippets.
Some snippets have literal escaped newlines between statements. Parsing accepts
valid Python unchanged; otherwise the decoder normalizes these separators
outside quoted strings. It preserves Unicode and Python string escapes. An
unparseable snippet or a snippet without an assertion is rejected.

Each candidate/test pair runs in a separate `bubblewrap` namespace with no
network or host workspace access. The process has a read-only Python runtime,
temporary scratch storage, a two-second CPU limit, a five-second wall limit,
a 256 MiB address-space limit, and process/file limits. Assertions are
instrumented to distinguish execution from a function definition that was
never called. Candidate output is suppressed. A crash, timeout, import failure,
or malformed worker response does not pass.

Only a single Python code block or directly parseable Python is supported.
Multiple code blocks are not heuristically combined or ranked by test outcome.
An unsupported answer remains excluded; it is not necessarily incorrect.
The test runner is not an adversarial reinforcement-learning reward oracle.

## Run verification

Run from the repository root in the Python environment used for data
preparation. `pyarrow` and the existing data tooling must be available; the
candidate worker uses the system Python standard library. Install `bubblewrap`
through the system package manager if it is absent.

```bash
python -m scripts.ncm.verify_distillation \
  --admitted results/ncm-opencode-v19 \
  --cache results/ncm-distillation-opencode-v20 \
  --output results/ncm-verified-distillation-v22
```

The output directory must be new. `identity.json` freezes the snapshot and
provenance; `report.json` is written atomically as checks finish. A running
teacher collector can continue writing new cache files: those new answers do
not enter an already captured snapshot. To verify later answers, run again
with a new output directory after collection completes.

The completed report distinguishes passing answers, teacher test failures,
unsupported code/tests, and failures of the original-answer control. The latter
can reflect environment limitations or publisher test defects and cannot be
used to condemn the teacher answer.

## Materialize a curated training archive

After both the teacher collection and verification snapshot are complete:

```bash
python -m scripts.ncm.curate_distillation \
  --source results/ncm-distillation-opencode-v20 \
  --verification results/ncm-verified-distillation-v22 \
  --output results/ncm-distillation-opencode-curated-v22
```

The curator requires an exact record-to-snapshot join, passing original and
teacher checks, executed assertions, and unchanged cache hashes. It compares
the checked teacher response with the final admitted archive byte for byte.
It retains the original answer text, prompt, conversation group, split, and ID.
It rejects a missing or changed checked answer instead of silently substituting
another response. The new archive includes the source license and attribution,
verification hashes, and a per-response check record.

`targets_are_verified` and `quality_ready` remain false: passing a finite set of
publisher tests is deliberately recorded as narrower evidence. The archive is
compatible with the existing training-expansion preparer. Its creation does
not change an active trainer's immutable data. A later training stage must
explicitly select it, preserve all frozen evaluation rows, retain the two
creator-dataset sources, and record a new prepared-data hash.

The current compact FP32-master run uses its existing immutable v19 prepared
data. No verification outcome changes its sampler or targets in place.

The first completed snapshot contains 249 admitted teacher answers: 107 pass
the local publisher checks, 39 fail after a passing original-answer control,
97 have unsupported code/test structure, and six fail the original-answer
control. This snapshot predates completion of the 512-request collection and
must not be presented as its final result. The source and result hashes are in
[`validation/v22-controls/publisher-test-snapshot.json`](validation/v22-controls/publisher-test-snapshot.json).

## Validation

`test_distillation_verifier.py` checks newline handling, Unicode/string escape
preservation, ambiguous code-block rejection, successful and failing sandbox
execution, and rejection of assertions that are never executed.
`test_distillation_curation.py` checks complete snapshot coverage, duplicate
and holdout rejection, executed evidence on both answers, exact preservation
of archived answers/prompts, and rejection of cache tampering.

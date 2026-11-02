# Prepared-data JSON cache

`prepared_reader.py` reduces repeated startup time for the internal prepared
corpus. It changes serialization only. It does not accelerate a training update,
change the model, or establish generation quality.

## Conversion and identity

Convert an existing, trusted prepared archive once:

```bash
python -m scripts.ncm.prepared_reader \
  --source results/ncm-curated-training-v23/prepared.pt \
  --sha256 76e473333c630f0e4969e459f1065f07598affcad71d466f5d41dfe263acba26 \
  --output results/new-prepared-json-cache
```

The converter checks the source checksum before and after
`torch.load(weights_only=True)`. It serializes the prepared envelope to JSON,
restores the outer row/condition tuples, and recursively checks exact types,
values, and dictionary ordering against the source. Unsupported objects and
nonfinite numbers fail conversion. The destination must not already exist.

The manifest identifies the source, JSON payload, converter implementation,
row count, and completed round-trip comparison. It accompanies `data.json`.
Subsequent loading uses JSON rather than pickle:

```python
from scripts.ncm.prepared_reader import load

data = load("results/new-prepared-json-cache", expected_source_sha256)
prepared = data["prepared"]
```

`expected_source_sha256` must come from the caller's already validated corpus
identity. The loader verifies format, qualification flag, source identity,
payload size, payload checksum, and row count. Checksums detect changes; they
do not authenticate an untrusted publisher. This utility handles prepared data
only, never model weights or optimizer checkpoints.

## Measured startup effect

On the training workstation, the version-23 corpus contains 113,889 rows.
The JSON payload is 350,940,029 bytes with SHA256
`29c28b62174cd35f64a66aa6b936178979942a60272e9c41bac4add91fbcbf56`.
The measured source load took 111.51 seconds. Three warm checksummed JSON reads
took 7.736, 7.675, and 7.566 seconds, a median of 7.675 seconds. The ratio of
that single source load to the median warm JSON read is 14.53. This is a
startup serialization comparison, not a cold-disk benchmark or a training
speedup. Maximum RSS for the JSON benchmark was 2,885,936 KiB.

The complete local measurement is retained in
`results/ncm-prepared-json-v24/benchmark.json`. Unit tests exercise unchanged
types and ordering, corruption, size limits, and source-identity rejection.

## Integration boundary

The ongoing version-24 learning-rate arms continue using their original reader
and recorded implementation identities. They do not silently switch to this
cache. An explicitly recorded later training stage can opt into the reader
after checking that source filtering, curriculum membership, and sampled row
identities remain unchanged. Optimizer state and experiment identities must
still follow the normal continuation checks. A faster data loader does not
replace executable evaluation and fresh-checkpoint replay.

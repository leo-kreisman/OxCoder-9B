"""Tokenize and codec-screen the compact supplement for original-tower training.

Produces a text/conditioning admission manifest, not cached encoder features.
The already admitted base and all 64 validation windows are retained verbatim.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .open_corpus import file_sha, shingles
from .acquire_creator_corpora import archived_digest
from .admit_creator_corpora import read_rows


def verified_rows(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    archive = folder / 'examples.jsonl.gz'
    if (not manifest.get('admission_complete') or manifest.get('training_started')
            or file_sha(archive) != manifest['examples_archive_sha256']
            or archived_digest(archive)[0] != manifest['examples_uncompressed_sha256']
            or file_sha(folder / 'checks.jsonl') != manifest['checks_sha256']
            or file_sha(folder / 'identity.json') != manifest['identity_sha256']):
        raise ValueError('Incomplete or changed compact archive')
    proofs = {}
    for proof in read_rows(folder / 'checks.jsonl'):
        if proof.get('passed') is True:
            if proof['id'] in proofs:
                raise ValueError('Duplicate proof')
            proofs[proof['id']] = proof
    rows = list(read_rows(archive))
    if len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} != set(proofs):
        raise ValueError('Proofs do not cover the exact admitted archive')
    for row in rows:
        proof = proofs[row['id']]
        checks = [proof.get(k, []) for k in ('original_checks', 'parent_checks', 'compact_checks')]
        if (row['split'] != 'train' or proof['split'] != 'train'
                or proof['parent_id'] != row['transformation']['parent_id']
                or proof['tests_sha256'] != row['publisher_checks']['tests_sha256']
                or not checks[0] or len({len(c) for c in checks}) != 1
                or not all(c.get('passed') is True and c.get('assertions_executed', 0) > 0
                           for group in checks for c in group)):
            raise ValueError('Invalid execution proof')
    return rows, manifest


def prefer_teacher(rows):
    groups = {r['group'] for r in rows if r.get('distilled')}
    return [r for r in rows if r.get('distilled') or r['group'] not in groups]


def complete_chunks(rows):
    return (bool(rows) and [c['chunk_index'] for _, c in rows] == list(range(len(rows)))
            and rows[-1][1]['is_final_chunk'] and not any(c['is_final_chunk'] for _, c in rows[:-1]))


def run(a):
    import torch
    from transformers import AutoTokenizer
    from .codec import Codec, CodecConfig, encode, PAD
    from .frozen_tower_training import load_corpus
    from .prepare_creator_expansion import add_response, frozen_evaluation_hash
    from .train import admission, headroom, save, write_json
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    base_manifest = json.loads(Path(a.base_manifest).read_text())
    identity = base_manifest['identity']
    codec_sha = file_sha(a.codec)
    if codec_sha != identity['codec_sha256']:
        raise ValueError('Codec differs from admitted base')
    training, validation, provenance = load_corpus(a.base_selection, a.base_manifest,
                                                  identity['source_sha256'], codec_sha)
    base = training + validation
    frozen_sha = frozen_evaluation_hash(base)
    old_records = {r['id']: r for r in base_manifest['records']}
    groups = {r['group']: r['split'] for r, _ in base}
    known = {r['id'] for r, _ in base}
    protected_exact = {r['target'] for r, _ in validation}
    protected_shingles = set().union(*(shingles(r['target']) for r, _ in validation))
    raw, archive_manifest = verified_rows(a.addition)
    rows = prefer_teacher(raw)
    counts = Counter(considered=len(raw), teacher_preference_excluded=len(raw) - len(rows))
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    write_json(out / 'resources.json', admission([a.device], 2))
    ck = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model']); codec.eval().requires_grad_(False)
    del ck
    selected, added, rejected = list(base), [], []
    with torch.no_grad():
        for number, original in enumerate(rows):
            headroom([a.device])
            if groups.get(original['group'], 'train') != 'train':
                reason, chunks = 'protected_group', []
            else:
                chunks, stats = add_response(original, groups, tokenizer, codec.config.length, 2048, 2)
                reason = None if not stats['over_context'] and complete_chunks(chunks) else 'incomplete_or_over_context'
            if reason is None and any(r['target'] in protected_exact or
                    bool(shingles(r['target']) & protected_shingles) for r, _ in chunks):
                reason = 'frozen_validation_target_overlap'
            if reason is None:
                for row, condition in chunks:
                    target = torch.tensor([encode(row['target'], codec.config.length)], device=a.device)
                    codes, _ = codec.targets(target)
                    output = codec.reconstruct_codes(codes).argmax(-1)
                    if not bool(((target == output) | (target == PAD)).all()):
                        reason = 'codec_reconstruction'; break
            if reason is not None:
                counts['rejected:' + reason] += 1
                rejected.append({'id': original['id'], 'reason': reason})
            else:
                for row, condition in chunks:
                    if row['id'] in known:
                        raise ValueError('Duplicate prepared identity')
                    known.add(row['id']); groups[row['group']] = 'train'
                    row.update(distilled=bool(original.get('distilled')), curriculum='compact-code-v33')
                    for key in ('source_revision', 'original_id', 'teacher_sha256', 'transformation', 'publisher_checks'):
                        if key in original:
                            row[key] = original[key]
                    selected.append((row, condition)); added.append(row['id'])
                counts['admitted_responses'] += 1
                counts['admitted_chunks'] += len(chunks)
                counts['distilled_responses'] += bool(original.get('distilled'))
                counts['one_window_responses'] += len(chunks) == 1
            if number % 100 == 0:
                write_json(out / 'progress.json', {'processed': number + 1, 'total': len(rows),
                                                   'counts': dict(counts), 'complete': False})
    if not added or frozen_evaluation_hash(selected) != frozen_sha:
        raise ValueError('No additions or changed validation')
    save(out / 'selection.pt', {'prepared': selected})
    records = [{'index': i, 'id': r['id'], 'group': r['group'], 'split': r['split'],
        'source': r['source'], 'distilled': bool(r.get('distilled')), 'prompt_tokens': len(c['ids']),
        'codec_exact': old_records[r['id']]['codec_exact'] if r['id'] in old_records else True}
        for i, (r, c) in enumerate(selected)]
    report = {'format': 'frozen-tower-text-admission-v33', 'complete': True,
        'identity': {'source_sha256': identity['source_sha256'], 'codec_sha256': codec_sha,
                     'selection_sha256': file_sha(out / 'selection.pt'), 'complete_training_responses_only': True},
        'base_corpus': provenance, 'addition_manifest_sha256': file_sha(Path(a.addition) / 'manifest.json'),
        'addition_archive_sha256': archive_manifest['examples_archive_sha256'],
        'source_tokenizer_sha256': file_sha(Path(a.tokenizer) / 'tokenizer.json'),
        'frozen_evaluation_sha256': frozen_sha, 'validation_unchanged': True,
        'counts': dict(counts), 'records': records, 'rejected': rejected,
        'codec_screen_device': a.device, 'codec_screen_precision': 'FP32; no TF32',
        'implementation_sha256': {n: file_sha(Path(__file__).with_name(n)) for n in
            ('screen_compact_corpus.py', 'prepare_creator_expansion.py', 'conditioning.py', 'codec.py')},
        'scope': 'Text-only original-tower training admission; no cached encoder tensors',
        'quality_ready': False, 'training_started': False}
    write_json(out / 'admission.pending.json', report)
    _, _, checked = load_corpus(out, out / 'admission.pending.json', identity['source_sha256'], codec_sha)
    (out / 'admission.pending.json').replace(out / 'admission.json')
    write_json(out / 'manifest.json', {**checked, 'counts': dict(counts),
        'frozen_evaluation_sha256': frozen_sha, 'validation_unchanged': True, 'training_started': False})
    write_json(out / 'progress.json', {'counts': dict(counts), 'complete': True})
    print(json.dumps({'counts': dict(counts), 'corpus': checked}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-selection', 'base-manifest', 'addition', 'codec', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--tokenizer', default='results/tokenizer')
    parser.add_argument('--device', default='cuda:1')
    run(parser.parse_args())

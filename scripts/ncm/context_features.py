"""Cache prompt-only OxCoder states for an encoder-decoder NCM experiment.

Selection is fixed before feature extraction. Labels never enter the frozen
encoder; only the existing conditioning token IDs are passed to its forward.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def unit_hash(value):
    return (int(hashlib.sha256(value.encode()).hexdigest(), 16) + 1) / (2**256 + 1)


def select_training(prepared, count, seed):
    if count < 1:
        raise ValueError('Positive training selection count required')
    groups, splits = defaultdict(list), {}
    identities = set()
    for row, condition in prepared:
        if row['id'] in identities:
            raise ValueError('Duplicate prepared row identity')
        identities.add(row['id'])
        if splits.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError('Conversation crosses splits')
        if row['split'] == 'train':
            if not 0 < len(condition['ids']) <= 2048:
                raise ValueError('Training context outside qualified range')
            groups[row['group']].append((row, condition))
    candidates = []
    for group, rows in groups.items():
        # Keep a distilled response in preference to its original target when
        # both exist. Distillation provenance remains in the prepared archive.
        teacher = [item for item in rows if item[0].get('distilled')]
        choices = teacher or rows
        response_ids = sorted({condition['response_id'] for _, condition in choices})
        response_id = min(response_ids, key=lambda x: unit_hash(f'{seed}:response:{x}'))
        choices = [item for item in choices if item[1]['response_id'] == response_id]
        first = [item for item in choices if item[1]['chunk_index'] == 0]
        if first and unit_hash(f'{seed}:first:{group}') < .5:
            chosen = min(first, key=lambda item: item[0]['id'])
        else:
            chosen = min(choices, key=lambda item: unit_hash(f"{seed}:chunk:{item[0]['id']}"))
        candidates.append(chosen)
    sources = Counter(row['source'] for row, _ in candidates)
    def priority(item):
        row, condition = item
        weight = (4 if condition['coding_response'] else 1) * (4 if row.get('distilled') else 1)
        weight /= math.sqrt(sources[row['source']])
        return -math.log(unit_hash(f"{seed}:select:{row['id']}")) / weight
    selected = sorted(candidates, key=lambda item: (priority(item), item[0]['id']))[:count]
    if len(selected) != count:
        raise ValueError('Insufficient distinct training conversations')
    if {row['source'] for row, _ in selected} != set(sources):
        raise ValueError('Selection dropped an entire training source')
    if not any(row.get('distilled') for row, _ in selected):
        raise ValueError('Selection contains no distillation')
    return selected


def prepare(a):
    import torch
    from .backbone import digest
    from .distillation_curriculum import filter_teacher_actions
    from .prepared_reader import load
    from .train import save, write_json
    folder = Path(a.data)
    manifest = json.loads((folder / 'manifest.json').read_text())
    data = load(a.json_cache, manifest['prepared_sha256'])
    filtered, teacher_filter = filter_teacher_actions(data['prepared'], manifest)
    training = select_training(filtered, a.count, a.seed)
    validation_folder = Path(a.validation)
    validation_manifest = json.loads((validation_folder / 'manifest.json').read_text())
    if (validation_manifest['prepared_sha256'] != manifest['prepared_sha256'] or
            digest(validation_folder / 'selection.pt') != validation_manifest['selection_sha256']):
        raise ValueError('Frozen validation selection differs')
    validation = torch.load(validation_folder / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
    if (len(validation) != validation_manifest['count'] or any(row['split'] != 'validation' for row, _ in validation)
            or {r['group'] for r, _ in training} & {r['group'] for r, _ in validation}):
        raise ValueError('Invalid or leaking validation selection')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    # Interleave a fixed hash order: no generation-quality or reconstruction
    # score is used to choose or order examples.
    selected = sorted(training + validation, key=lambda item: unit_hash(f"{a.seed}:order:{item[0]['id']}"))
    save(out / 'selection.pt', {'prepared': selected})
    report = {'prepared_sha256': manifest['prepared_sha256'],
        'validation_selection_sha256': validation_manifest['selection_sha256'],
        'selection_sha256': digest(out / 'selection.pt'), 'seed': a.seed,
        'counts': dict(Counter(row['split'] for row, _ in selected)),
        'training_sources': dict(Counter(row['source'] for row, _ in training)),
        'training_distilled': sum(bool(row.get('distilled')) for row, _ in training),
        'training_first_chunks': sum(c['chunk_index'] == 0 for _, c in training),
        'teacher_filter': teacher_filter, 'implementation_sha256': digest(__file__),
        'scope': 'New encoder-decoder experiment, weighted distinct-conversation curriculum; not continuation of v24',
        'quality_ready': False}
    write_json(out / 'manifest.json', report)
    print(json.dumps(report), flush=True)


def extract(a):
    import torch
    from .audit_backbone import audit
    from .backbone import Source, digest
    from .codec import Codec, CodecConfig, encode, PAD
    from .gguf_training_backbone import PackedOxBackbone
    from .train import admission, headroom, save, write_json
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if not 0 <= a.shard < a.shards:
        raise ValueError('Invalid extraction shard')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'admission.json', admission([a.device], 8))
    folder = Path(a.selection)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(folder / 'selection.pt') != manifest['selection_sha256']:
        raise ValueError('Feature selection changed')
    prepared = torch.load(folder / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
    codec_sha = digest(a.codec)
    ck = torch.load(a.codec, weights_only=True, map_location='cpu')
    if digest(a.codec) != codec_sha:
        raise ValueError('Codec changed during loading')
    codec = Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model']); codec.eval().requires_grad_(False)
    source = Source(a.source)
    encoder = PackedOxBackbone(source, [a.device], rank=1).eval().requires_grad_(False)
    if any(torch.count_nonzero(adapter.up.weight) for adapter in encoder.adapters):
        raise ValueError('Prompt encoder adapters are not zero')
    parity = audit(encoder, source, a.reference)
    write_json(out / 'source-parity.json', parity)
    if not parity['passed']:
        raise RuntimeError('Original OxCoder prompt encoder failed transfer audit')
    identity = {'architecture': 'frozen-oxcoder-prompt-encoder-native-ncm-decoder-v25',
        'source_sha256': source.sha, 'codec_sha256': codec_sha,
        'selection_sha256': manifest['selection_sha256'], 'prepared_sha256': manifest['prepared_sha256'],
        'encoder': 'all 32 original frozen layers, zero residual adapters, no AR answer generation',
        'encoder_inputs': 'condition.ids only; never current target bytes, target tokens, or target codes',
        'feature_dtype': 'bfloat16', 'maximum_prompt_tokens': 2048,
        'implementation_sha256': {name: digest(Path(__file__).parent / name) for name in
            ('context_features.py', 'backbone.py', 'gguf_training_backbone.py', 'gguf_training_weights.py', 'codec.py')},
        'shard': a.shard, 'shards': a.shards, 'quality_ready': False}
    write_json(out / 'identity.json', identity)
    records, rejected = [], []
    with torch.no_grad():
        for index, (row, condition) in enumerate(prepared):
            if index % a.shards != a.shard:
                continue
            headroom([a.device])
            byte_ids = torch.tensor([encode(row['target'], codec.config.length)], device=a.device)
            codes, _ = codec.targets(byte_ids)
            oracle = codec.reconstruct_codes(codes).argmax(-1)
            exact = bool(((oracle == byte_ids) | (byte_ids == PAD)).all())
            if row['split'] == 'train' and not exact:
                rejected.append({'index': index, 'id': row['id'], 'split': 'train',
                                 'reason': 'codec reconstruction not exact'})
            else:
                ids = torch.tensor([condition['ids']], dtype=torch.long)
                hidden = encoder(encoder.embed_prompt(ids))[0].to(torch.bfloat16).cpu()
                if not torch.isfinite(hidden).all():
                    raise FloatingPointError('Nonfinite prompt features')
                path = out / f'{index:06d}.pt'
                save(path, {'row': row, 'condition': condition, 'hidden': hidden,
                            'codes': [x[0].cpu() for x in codes], 'bytes': byte_ids[0].cpu(), 'codec_exact': exact})
                records.append({'index': index, 'id': row['id'], 'group': row['group'], 'split': row['split'],
                    'source': row['source'], 'distilled': bool(row.get('distilled')), 'path': path.name,
                    'sha256': digest(path), 'prompt_tokens': len(condition['ids']), 'codec_exact': exact})
                del hidden
            write_json(out / 'progress.json', {'records': len(records), 'rejected': len(rejected), 'quality_ready': False})
            print(json.dumps({'index': index, 'saved': len(records), 'rejected': len(rejected)}), flush=True)
    report = {'identity_sha256': digest(out / 'identity.json'), 'records': records, 'rejected': rejected,
        'counts': dict(Counter(r['split'] for r in records)), 'complete': True, 'quality_ready': False}
    write_json(out / 'manifest.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    for name in ('data', 'json-cache', 'validation', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--count', type=int, default=4096)
    p.add_argument('--seed', type=int, default=25001)
    p = subs.add_parser('extract')
    for name in ('source', 'codec', 'selection', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--reference', default='scripts/kindi_q4/kindi/reference-oxcoder-q4.json')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else extract)(args)

"""Frozen, conversation-disjoint diagnostics for comparing NCM checkpoints.

Selection uses data identities only, before either model is evaluated. Gold
conditioning is a prediction diagnostic; free generation receives only the
existing prompt. Neither result replaces the executable coding acceptance gate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def select(prepared, count=64, seed=24001, max_prompt=2048):
    if count < 1 or max_prompt < 1:
        raise ValueError('Positive selection bounds required')
    splits, ids, buckets = {}, set(), defaultdict(list)
    for row, condition in prepared:
        group, split = row['group'], row['split']
        if group in splits and splits[group] != split:
            raise ValueError('Conversation crosses data splits')
        splits[group] = split
        if row['id'] in ids:
            raise ValueError('Duplicate prepared row identity')
        ids.add(row['id'])
        if split != 'validation' or not 0 < len(condition['ids']) <= max_prompt:
            continue
        priority = hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()
        buckets[row['source']].append((priority, row, condition))
    for items in buckets.values():
        items.sort(reverse=True, key=lambda x: x[0])
    chosen, groups = [], set()
    while len(chosen) < count:
        before = len(chosen)
        for source in sorted(buckets):
            while buckets[source]:
                _, row, condition = buckets[source].pop()
                if row['group'] not in groups:
                    groups.add(row['group'])
                    chosen.append((row, condition))
                    break
            if len(chosen) == count:
                break
        if len(chosen) == before:
            break
    if len(chosen) != count:
        raise ValueError(f'Only {len(chosen)} distinct validation conversations; requested {count}')
    return chosen


def prepare(a):
    import torch
    from .backbone import digest
    from .train import save, write_json
    directory = Path(a.data)
    manifest = json.loads((directory / 'manifest.json').read_text())
    data_sha = digest(directory / 'prepared.pt')
    if manifest['prepared_sha256'] != data_sha:
        raise ValueError('Prepared data checksum differs')
    data = torch.load(directory / 'prepared.pt', weights_only=True, map_location='cpu')
    selected = select(data['prepared'], a.count, a.seed, a.max_prompt)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'selection.pt', {'prepared': selected})
    write_json(out / 'manifest.json', {
        'prepared_sha256': data_sha, 'selection_sha256': digest(out / 'selection.pt'),
        'implementation_sha256': digest(__file__), 'seed': a.seed,
        'count': a.count, 'max_prompt': a.max_prompt,
        'sources': dict(Counter(row['source'] for row, _ in selected)),
        'rows': [{'id': r['id'], 'group': r['group'], 'source': r['source'],
                  'prompt_tokens': len(c['ids']), 'final_chunk': c['is_final_chunk']}
                 for r, c in selected],
        'scope': 'Validation only, one chunk per conversation; no reconstruction or model-dependent filtering'})


def run(a):
    import torch
    from torch.nn import functional as F
    from .backbone import digest
    from .codec import encode, decode, PAD
    from .gguf_training_backbone import load_candidate
    from .train import admission, headroom, write_json
    torch.set_num_threads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'admission.json', admission([a.device], 8))
    directory = Path(a.selection)
    manifest = json.loads((directory / 'manifest.json').read_text())
    selection_sha = digest(directory / 'selection.pt')
    if manifest['selection_sha256'] != selection_sha:
        raise ValueError('Frozen selection changed')
    selected = torch.load(directory / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
    if len(selected) != manifest['count'] or any(r['split'] != 'validation' for r, _ in selected):
        raise ValueError('Invalid validation selection')
    model, codec, identity, checkpoint_sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
    if identity['prepared_sha256'] != manifest['prepared_sha256']:
        raise ValueError('Checkpoint trained with a different prepared data identity')
    if max(len(c['ids']) for _, c in selected) > identity['prompt_tokens']:
        raise ValueError('Validation exceeds checkpoint context')
    records = []
    with torch.no_grad():
        for row, condition in selected:
            headroom([a.device])
            target = torch.tensor([encode(row['target'], codec.config.length)], device=a.device)
            codes, inputs = codec.targets(target)
            prompt = torch.tensor([condition['ids']], dtype=torch.long)
            losses, accuracies, content_losses, content_accuracies = [], [], [], []
            for level, inp in enumerate(inputs):
                logits = model(inp, prompt, level)
                loss = F.cross_entropy(logits.flatten(0, 1), codes[level].flatten(), reduction='none')
                correct = (logits.argmax(-1) == codes[level]).flatten().float()
                # Local overlap, not a claim that the contextual encoder's
                # representation depends exclusively on these bytes.
                overlap = F.interpolate((target != PAD).float().unsqueeze(1),
                    size=codec.config.levels[level], mode='area').flatten() > 0
                losses.append(float(loss.mean())); accuracies.append(float(correct.mean()))
                content_losses.append(float(loss[overlap].mean()))
                content_accuracies.append(float(correct[overlap].mean()))
            oracle = codec.reconstruct_codes(codes).argmax(-1)
            output, _, stop = model.generate_chunk(codec, prompt)
            try:
                text = decode(output[0].tolist())
            except (ValueError, UnicodeError):
                text = None
            nonpad = target != PAD
            records.append({'id': row['id'], 'source': row['source'],
                'teacher_forced_ce': losses, 'teacher_forced_accuracy': accuracies,
                'overlapping_nonpad_ce': content_losses,
                'overlapping_nonpad_accuracy': content_accuracies,
                'codec_exact_nonpad': bool(((oracle == target) | ~nonpad).all()),
                'free_byte_accuracy_nonpad': float((output == target)[nonpad].float().mean()),
                'free_exact_chunk': text == row['target'], 'free_valid_chunk': text is not None,
                'generated_text': text, 'stop_probability': float(stop[0]),
                'expected_final_chunk': condition['is_final_chunk']})
            report = {'checkpoint_sha256': checkpoint_sha, 'selection_sha256': selection_sha,
                'implementation_sha256': digest(__file__), 'identity': identity,
                'records': records, 'complete': len(records) == len(selected),
                'quality_ready': False, 'scope': 'Paired validation diagnostic, not executable coding acceptance'}
            if report['complete']:
                report['summary'] = summarize(records)
            write_json(out / 'report.json', report)
            print(json.dumps({'evaluated': len(records), 'total': len(selected)}), flush=True)


def summarize(records):
    if not records:
        raise ValueError('No validation observations')
    means = {}
    for metric in ['teacher_forced_ce', 'teacher_forced_accuracy',
                   'overlapping_nonpad_ce', 'overlapping_nonpad_accuracy']:
        sizes = {len(row[metric]) for row in records}
        if len(sizes) != 1:
            raise ValueError('Concept level inventories differ')
        means[metric] = [sum(row[metric][i] for row in records) / len(records)
                         for i in range(next(iter(sizes)))]
    return {**means, 'windows': len(records),
        'mean_level_ce': sum(means['teacher_forced_ce']) / len(means['teacher_forced_ce']),
        'codec_exact_nonpad_windows': sum(r['codec_exact_nonpad'] for r in records),
        'free_exact_windows': sum(r['free_exact_chunk'] for r in records),
        'free_valid_windows': sum(r['free_valid_chunk'] for r in records),
        'mean_free_nonpad_byte_accuracy': sum(r['free_byte_accuracy_nonpad'] for r in records) / len(records),
        'stop_accuracy': sum((r['stop_probability'] >= .5) == r['expected_final_chunk'] for r in records) / len(records)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    for name in ['data', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--seed', type=int, default=24001)
    p.add_argument('--max-prompt', type=int, default=2048)
    p = subs.add_parser('run')
    for name in ['source', 'codec', 'checkpoint', 'selection', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)

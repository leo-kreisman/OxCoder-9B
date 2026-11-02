"""Measure whether frozen NCM predictions benefit from the correct prompt.

Both arms receive the same gold earlier concept levels. Only the prompt changes.
This diagnoses conditioning, not free generation or coding correctness.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def pairs(prepared, count=16):
    if not 2 <= count <= len(prepared):
        raise ValueError('At least two distinct validation conversations required')
    selected = sorted(prepared, key=lambda item: hashlib.sha256(
        ('prompt-control-v24:' + item[0]['id']).encode()).hexdigest())[:count]
    if (any(row['split'] != 'validation' for row, _ in selected)
            or len({row['group'] for row, _ in selected}) != count):
        raise ValueError('Prompt control requires disjoint validation conversations')
    # Deterministic derangement selected before model scoring. Prefer nearby
    # lengths to reduce positional differences, without pretending to isolate
    # semantic effects perfectly. All replacements are different conversations.
    selected.sort(key=lambda item: (len(item[1]['ids']), item[0]['id']))
    return [(item, selected[(index + 1) % count]) for index, item in enumerate(selected)]


def summarize(records):
    if not records:
        raise ValueError('No prompt-control observations')
    levels = len(records[0]['correct_ce'])
    keys = ('correct_ce', 'replaced_ce', 'correct_accuracy', 'replaced_accuracy',
            'argmax_agreement', 'kl_correct_to_replaced')
    if any(len(row[key]) != levels or not all(math.isfinite(x) for x in row[key])
           for row in records for key in keys):
        raise ValueError('Invalid prompt-control metrics')
    means = {key: [sum(row[key][i] for row in records) / len(records)
                   for i in range(levels)] for key in keys}
    means['replacement_ce_increase'] = [b-a for a, b in zip(means['correct_ce'], means['replaced_ce'])]
    return means


def run(a):
    import torch
    from torch.nn import functional as F
    from .backbone import digest
    from .codec import encode
    from .gguf_training_backbone import load_candidate
    from .train import admission, headroom, write_json
    torch.set_num_threads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'admission.json', admission([a.device], 8))
    folder = Path(a.selection)
    manifest = json.loads((folder / 'manifest.json').read_text())
    selection_sha = digest(folder / 'selection.pt')
    if selection_sha != manifest['selection_sha256']:
        raise ValueError('Validation selection checksum differs')
    prepared = torch.load(folder / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
    if len(prepared) != manifest['count']:
        raise ValueError('Validation selection count differs')
    selected = pairs(prepared, a.count)
    model, codec, identity, checkpoint_sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
    if identity['prepared_sha256'] != manifest['prepared_sha256']:
        raise ValueError('Checkpoint data identity differs')
    if max(len(c['ids']) for pair in selected for _, c in pair) > identity['prompt_tokens']:
        raise ValueError('Prompt exceeds checkpoint context')
    model.eval().requires_grad_(False)
    records = []
    with torch.inference_mode():
        for (row, condition), (other, replacement) in selected:
            headroom([a.device])
            codes, inputs = codec.targets(torch.tensor(
                [encode(row['target'], codec.config.length)], device=a.device))
            correct_prompt = torch.tensor([condition['ids']], dtype=torch.long)
            wrong_prompt = torch.tensor([replacement['ids']], dtype=torch.long)
            record = {'id': row['id'], 'replacement_id': other['id'],
                'prompt_tokens': len(condition['ids']), 'replacement_tokens': len(replacement['ids']),
                **{key: [] for key in ('correct_ce', 'replaced_ce', 'correct_accuracy',
                                      'replaced_accuracy', 'argmax_agreement', 'kl_correct_to_replaced')}}
            for level, inp in enumerate(inputs):
                correct = model(inp, correct_prompt, level).float()
                replaced = model(inp, wrong_prompt, level).float()
                target = codes[level].to(correct.device)
                for label, logits in [('correct', correct), ('replaced', replaced)]:
                    record[label + '_ce'].append(float(F.cross_entropy(
                        logits.flatten(0, 1), target.flatten())))
                    record[label + '_accuracy'].append(float((logits.argmax(-1) == target).float().mean()))
                record['argmax_agreement'].append(float((correct.argmax(-1) == replaced.argmax(-1)).float().mean()))
                p, q = correct.log_softmax(-1), replaced.log_softmax(-1)
                record['kl_correct_to_replaced'].append(float((p.exp() * (p-q)).sum(-1).mean()))
            records.append(record)
            write_json(out / 'report.json', {'records': records, 'summary': summarize(records),
                'checkpoint_sha256': checkpoint_sha, 'selection_sha256': selection_sha,
                'implementation_sha256': digest(__file__), 'identity': identity,
                'complete': len(records) == len(selected), 'quality_ready': False,
                'scope': 'Gold earlier concepts; prompt replacement only; different prompt lengths remain a confound'})
            print(json.dumps({'completed': len(records), 'total': len(selected)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=16)
    run(parser.parse_args())

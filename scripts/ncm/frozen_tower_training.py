"""Shared training-only controls for frozen-context original-backbone NCM."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import torch

from .backbone import digest
from .codec import encode
from . import frozen_tower as tower
from . import frozen_tower_batch as batch_tower
from .frozen_tower_objective import trajectory
from .train_accumulated import example_loss


def qualify(path, source_sha, codec_sha):
    report = json.loads(Path(path).read_text())
    if (not report.get('complete') or not report.get('passed')
            or report['source_sha256'] != source_sha or report['codec_sha256'] != codec_sha
            or report.get('compute_dtype') != 'float16' or report.get('prefix_alignment') != 64):
        raise ValueError('Passing FP16 frozen-tower execution audit required')
    records = report['records']
    if [r['prompt_tokens'] for r in records] != [64, 69, 512, 2048]:
        raise ValueError('Incomplete execution shape coverage')
    for record in records:
        comparisons = [record['gradient_comparison'], *record['gradient_groups'].values()]
        if (not record['passed'] or any(not c['finite'] or c['relative_l2'] >= .05 or c['cosine'] <= .99 for c in comparisons)
                or not record['logit_comparison']['finite'] or record['logit_comparison']['relative_l2'] >= .02):
            raise ValueError('Execution numerical gate failed')
    for name, sha in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Execution implementation changed: ' + name)
    return {'audit_sha256': digest(path), 'batch_levels': bool(report.get('batch_levels')),
            'compute_dtype': report['compute_dtype'], 'selection_sha256': report['selection_sha256']}


def load_corpus(selection, manifest, source_sha, codec_sha):
    selected = Path(selection) / 'selection.pt'
    report = json.loads(Path(manifest).read_text())
    identity = report['identity']
    if (not report.get('complete') or not identity.get('complete_training_responses_only')
            or identity['source_sha256'] != source_sha or identity['codec_sha256'] != codec_sha
            or digest(selected) != identity['selection_sha256']):
        raise ValueError('Wrong or incomplete corpus admission')
    prepared = torch.load(selected, weights_only=True, map_location='cpu')['prepared']
    admitted, seen, groups = [], set(), {}
    for record in report['records']:
        row, condition = prepared[record['index']]
        if (row['id'] in seen or any(row[k] != record[k] for k in ('id', 'group', 'split', 'source'))
                or bool(row.get('distilled')) != bool(record['distilled'])
                or len(condition['ids']) != record['prompt_tokens']
                or not 0 < len(condition['ids']) <= 2048
                or groups.setdefault(row['group'], row['split']) != row['split']
                or (row['split'] == 'train' and not record['codec_exact'])):
            raise ValueError('Corpus identity, prompt bound, or split changed')
        seen.add(row['id'])
        admitted.append((row, condition))
    training = [item for item in admitted if item[0]['split'] == 'train']
    validation = [item for item in admitted if item[0]['split'] == 'validation']
    responses = defaultdict(list)
    for row, condition in training:
        responses[condition['response_id']].append(condition)
    for chunks in responses.values():
        chunks.sort(key=lambda c: c['chunk_index'])
        if ([c['chunk_index'] for c in chunks] != list(range(len(chunks)))
                or not chunks[-1]['is_final_chunk'] or any(c['is_final_chunk'] for c in chunks[:-1])):
            raise ValueError('Incomplete training response')
    sources = dict(Counter(row['source'] for row, _ in training))
    if (len(validation) != 64 or set(sources) != {'TeichAI/Ox-Alpha-10k',
            'TeichAI/DeepSeek-v4-Pro-Agent', 'nvidia/OpenCodeInstruct'}
            or not any(row.get('distilled') for row, _ in training)):
        raise ValueError('Missing required creator data, distillation, or held-out windows')
    return training, validation, {'admission_sha256': digest(manifest),
        'selection_sha256': identity['selection_sha256'], 'training_chunks': len(training),
        'validation_chunks': len(validation), 'complete_training_responses': len(responses),
        'sources': sources, 'distilled_chunks': sum(bool(r.get('distilled')) for r, _ in training)}


def backward_example(model, codec, item, accumulation, loss_scale, self_conditioned, batch_levels):
    row, condition = item
    if row['split'] != 'train':
        raise ValueError('Only training examples may receive updates')
    target = torch.tensor([encode(row['target'], codec.config.length)], device=codec.position.device)
    prompt = torch.tensor([condition['ids']])
    model.eval()
    rollout = trajectory(model, codec, prompt, target_bytes=target,
                         self_conditioned=self_conditioned)
    model.train()
    inputs, labels, state = rollout['inputs'], rollout['labels'], rollout['state']
    values = batch_tower.forward_levels(model, inputs, prompt, state) if batch_levels else None
    total, accuracy, joint = 0., [], 0
    for level, inp in enumerate(inputs):
        logits, stop, _ = (values[level] if values is not None else
                          tower.level_output(model, inp, prompt, level, True, state))
        loss = example_loss(logits, labels[level], stop, condition['is_final_chunk'],
                            level, len(inputs), accumulation)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite concept loss')
        total += float(loss.detach())
        accuracy.append(float((logits.argmax(-1) == labels[level]).float().mean()))
        if values is None:
            (loss * loss_scale).backward()
        else:
            joint = joint + loss
    if values is not None:
        (joint * loss_scale).backward()
    return {'loss': total, 'code_accuracy': accuracy, 'self_conditioned': self_conditioned}

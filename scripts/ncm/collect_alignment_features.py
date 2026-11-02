"""Cache training-only teacher representations for an NCM alignment experiment.

The teacher is the unadapted original OxCoder body, evaluated causally. It never
runs during student inference. Validation features remain labelled validation;
no task from coding_eval is loaded or sent to the teacher by this command.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoTokenizer

from .audit_backbone import audit
from .backbone import Source, ConceptGenerator, digest
from .codec import Codec, CodecConfig, encode
from .gguf_training_backbone import PackedOxBackbone
from .representation_alignment import tokenize_alignment, pool_teacher
from .train import admission, headroom, save, write_json


def select(prepared, counts, max_prompt, seed):
    rng = random.Random(seed)
    inventory, observed = {}, {}
    for row, condition in prepared:
        if len(condition['ids']) > max_prompt or len(row['target'].encode()) < 16:
            continue
        split = row['split']
        if split not in counts:
            continue
        if observed.setdefault(row['group'], split) != split:
            raise ValueError('Conversation appears in multiple splits')
        inventory.setdefault(split, {}).setdefault(row['source'], []).append((row, condition))
    selected = []
    for split, count in counts.items():
        buckets = inventory.get(split, {})
        for rows in buckets.values():
            rng.shuffle(rows)
        used = set()
        chosen = []
        while len(chosen) < count:
            before = len(chosen)
            for source in sorted(buckets):
                while buckets[source]:
                    row, condition = buckets[source].pop()
                    if row['group'] not in used:
                        used.add(row['group'])
                        chosen.append((row, condition))
                        break
                if len(chosen) == count:
                    break
            if before == len(chosen):
                break
        if not chosen:
            raise ValueError('No eligible examples for ' + split)
        selected.extend(chosen)
    return selected


def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'admission.json', admission([a.device], 8))
    folder = Path(a.data)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(folder / 'prepared.pt') != manifest['prepared_sha256']:
        raise ValueError('Prepared data checksum differs')
    prepared = torch.load(folder / 'prepared.pt', map_location='cpu', weights_only=True)['prepared']
    selected = select(prepared, {'train': a.training, 'validation': a.validation}, a.max_prompt, a.seed)
    del prepared
    checkpoint = torch.load(a.codec, map_location='cpu', weights_only=True)
    codec = Codec(CodecConfig(**checkpoint['config'])).to(a.device)
    codec.load_state_dict(checkpoint['model'])
    codec.eval().requires_grad_(False)
    del checkpoint
    source = Source(a.source)
    teacher = PackedOxBackbone(source, [a.device], rank=32)
    teacher.eval().requires_grad_(False)
    parity = audit(teacher, source, a.reference)
    write_json(out / 'teacher-parity.json', parity)
    if not parity['passed']:
        raise RuntimeError('Original-teacher AR fixture parity failed')
    student = None
    adapted = []
    if a.student_checkpoint:
        student_sha = digest(a.student_checkpoint)
        checkpoint = torch.load(a.student_checkpoint, map_location='cpu', weights_only=True)
        parent = checkpoint['identity']
        if (parent['source_sha256'] != source.sha or parent['codec_sha256'] != digest(a.codec)
                or parent['head_kind'] != 'untied' or parent.get('cached_prefix')):
            raise ValueError('Student source, codec, or concept architecture differs')
        for adapter in teacher.adapters:
            adapter.requires_grad_(True)
        student = ConceptGenerator(teacher, codec.config, parallel_attention=True, termination=True,
            concept_input_norm=parent['concept_input_norm'], head_kind='untied',
            codebooks=codec.books, head_input_norm=parent['head_input_norm'])
        if set(student.trainable_state()) != set(checkpoint['model']):
            raise ValueError('Student parameter inventory differs')
        with torch.no_grad():
            for name, parameter in student.named_parameters():
                if name in checkpoint['model']:
                    parameter.copy_(checkpoint['model'][name].to(parameter))
        adapted = [adapter.up.weight.detach().clone() for adapter in teacher.adapters]
        student.eval().requires_grad_(False)
        del checkpoint
        if digest(a.student_checkpoint) != student_sha:
            raise ValueError('Student checkpoint changed during load')
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    identity = {'source_sha256': source.sha, 'codec_sha256': digest(a.codec),
        'prepared_sha256': manifest['prepared_sha256'], 'reference_sha256': digest(a.reference),
        'tokenizer_sha256': digest(Path(a.tokenizer) / 'tokenizer.json'),
        'levels': list(codec.config.levels), 'seed': a.seed, 'maximum_prompt_tokens': a.max_prompt,
        'teacher': 'original frozen causal body; zero residual adapters; no vocabulary head in feature extraction',
        'mapping': 'byte-overlap weighted causal predictor states; per-scale mean; EOS included; PAD excluded',
        'scope': 'experimental final-layer representation targets, not a published NCM conversion recipe',
        'implementation_sha256': {name: digest(Path(__file__).parent / name) for name in
            ('collect_alignment_features.py', 'representation_alignment.py', 'gguf_training_backbone.py',
             'gguf_training_weights.py', 'backbone.py', 'codec.py')},
        'quality_ready': False}
    if student is not None:
        identity['student_checkpoint_sha256'] = student_sha
        identity['student_features'] = 'gold earlier levels; same frozen body, restored learned adapters; reference forward'
    write_json(out / 'identity.json', identity)
    records, excluded = [], []
    for index, (row, condition) in enumerate(selected):
        began = time.monotonic()
        headroom([a.device])
        try:
            ids, weights, valid = tokenize_alignment(tokenizer, condition['ids'], row['target'], codec.config.length)
        except ValueError as error:
            excluded.append({'id': row['id'], 'split': row['split'], 'reason': str(error)})
            write_json(out / 'excluded.json', excluded)
            continue
        with torch.no_grad():
            for adapter in teacher.adapters:
                adapter.up.weight.zero_()
            hidden = teacher(teacher.embed_prompt(torch.tensor([ids])))[0]
            targets, masks = pool_teacher(hidden, weights, valid, codec.config.levels)
            byte_ids = torch.tensor([encode(row['target'], codec.config.length)], device=a.device)
            codes, inputs = codec.targets(byte_ids)
            oracle = codec.reconstruct_codes(codes).argmax(-1)
            exact = bool(torch.equal(oracle, byte_ids))
            features = []
            if student is not None:
                for adapter, saved in zip(teacher.adapters, adapted):
                    adapter.up.weight.copy_(saved)
                prompt = torch.tensor([condition['ids']])
                for level, inp in enumerate(inputs):
                    features.append(student.hidden_features(inp, prompt, level)[0].half().cpu())
        # Do not silently filter validation reconstruction failures.
        path = out / f'{index:06d}.pt'
        save(path, {'row': row, 'condition': condition, 'teacher_targets': [x.half().cpu() for x in targets],
                    'valid': [x.cpu() for x in masks], 'codes': [x.cpu() for x in codes],
                    'inputs': [x.cpu() for x in inputs], 'codec_exact': exact,
                    'student_features': features})
        record = {'id': row['id'], 'group': row['group'], 'split': row['split'], 'source': row['source'],
                  'path': path.name, 'sha256': digest(path), 'codec_exact': exact,
                  'prompt_tokens': len(condition['ids']), 'teacher_tokens': len(ids),
                  'seconds': time.monotonic() - began}
        records.append(record)
        write_json(out / 'progress.json', {'complete': False, 'records': records, 'excluded': excluded})
        print(json.dumps({**record, 'phase': 'alignment_features'}), flush=True)
        del hidden, targets, masks, codes, inputs, byte_ids, oracle
    if not any(r['split'] == 'train' for r in records) or not any(r['split'] == 'validation' for r in records):
        raise RuntimeError('Both training and validation records are required')
    write_json(out / 'manifest.json', {'identity_sha256': digest(out / 'identity.json'),
        'counts': dict(Counter(r['split'] for r in records)), 'records': records, 'excluded': excluded,
        'complete': True, 'quality_ready': False, 'teacher_used_for_student_inference': False})
    print(json.dumps({'phase': 'alignment_features_complete', 'records': len(records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reference', default='scripts/kindi_q4/kindi/reference-oxcoder-q4.json')
    parser.add_argument('--tokenizer', default='results/tokenizer')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--student-checkpoint', help='Optional immutable checkpoint for paired concept-readout control')
    parser.add_argument('--training', type=int, default=512)
    parser.add_argument('--validation', type=int, default=64)
    parser.add_argument('--max-prompt', type=int, default=512)
    parser.add_argument('--seed', type=int, default=21004)
    args = parser.parse_args()
    if min(args.training, args.validation, args.max_prompt) < 1:
        parser.error('Positive data bounds required')
    run(args)

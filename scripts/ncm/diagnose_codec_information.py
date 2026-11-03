"""Measure training-only code usage and the effect of removing level information.

Modal replacement is an in-sample diagnostic, not a learned predictor or a
model-quality evaluation. No parameter, dataset or inference policy is changed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import torch
from .backbone import digest
from .codec import Codec, CodecConfig, encode, PAD, resize
from .frozen_tower_training import load_corpus
from .train import admission, headroom, write_json


@torch.no_grad()
def measure(codec, examples, device):
    targets, encoded = [], [[] for _ in codec.books]
    for row, _ in examples:
        headroom([device])
        target = torch.tensor([encode(row['target'], codec.config.length)], device=device)
        codes, _ = codec.targets(target)
        targets.append(target)
        for saved, code in zip(encoded, codes):
            saved.append(code)
    targets = torch.cat(targets)
    encoded = [torch.cat(values) for values in encoded]
    modes, usage = [], []
    for size, codes in zip(codec.config.levels, encoded):
        counts = torch.bincount(codes.flatten(), minlength=codec.config.codes).float()
        probabilities = counts[counts > 0] / counts.sum()
        modal = codes.mode(0).values
        modes.append(modal)
        entropy = float(-(probabilities * probabilities.log2()).sum())
        usage.append({'positions': size, 'used_codes': int((counts > 0).sum()),
            'entropy_bits': entropy, 'perplexity': 2 ** entropy,
            'global_mode_fraction': float(counts.max() / counts.sum()),
            'position_mode_agreement': float((codes == modal).float().mean())})
    variants = ['gold', *['modal_' + str(i) for i in range(len(encoded))],
                *['prefix_' + str(i + 1) for i in range(len(encoded) - 1)]]
    paths = {}
    for variant in variants:
        correct = total = exact = 0
        for start in range(0, len(examples), 8):
            headroom([device])
            target = targets[start:start + 8]
            z = torch.zeros(len(target), codec.config.length, codec.config.width, device=device)
            for level, (book, ids) in enumerate(zip(codec.books, encoded)):
                if variant.startswith('prefix_') and level >= int(variant.split('_')[1]):
                    break
                code = ids[start:start + 8]
                if variant == 'modal_' + str(level):
                    code = modes[level].expand(len(target), -1)
                z += resize(book(code), codec.config.length)
            output = codec.decode_latents(z).argmax(-1)
            mask = target != PAD
            correct += int(((output == target) & mask).sum())
            total += int(mask.sum())
            exact += int(((output == target) | ~mask).all(-1).sum())
        paths[variant] = {'nonpad_byte_accuracy': correct / total, 'exact_windows': exact}
    return {'windows': len(examples), 'usage': usage, 'paths': paths,
            'ids': [r['id'] for r, _ in examples]}


def run(a):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {'quality_ready': False, 'training_started': False,
        'scope': 'Training-only modal-code and prefix ablations; not autonomous generation',
        'resources': admission([a.device], 2), 'codec_sha256': digest(a.codec)}
    manifest = json.loads(Path(a.manifest).read_text())
    training, _, corpus = load_corpus(a.selection, a.manifest,
        manifest['identity']['source_sha256'], report['codec_sha256'])
    report['corpus'] = corpus
    checkpoint = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**checkpoint['config'])).to(a.device)
    codec.load_state_dict(checkpoint['model']); codec.eval().requires_grad_(False)
    report['codec_config'] = checkpoint['config']
    groups = defaultdict(list)
    for row, condition in training:
        groups[(row['source'], condition['chunk_index'] == 0)].append((row, condition))
    rng = random.Random(33103)
    report['partitions'] = {}
    for first in (True, False):
        examples = []
        for source in sorted({r['source'] for r, _ in training}):
            candidates = sorted(groups[source, first], key=lambda x: x[0]['id'])
            examples.extend(rng.sample(candidates, min(a.per_source, len(candidates))))
        if examples:
            name = 'first_chunks' if first else 'later_chunks'
            report['partitions'][name] = measure(codec, examples, a.device)
            write_json(Path(a.output), report)
    report['complete'] = True
    report['implementation_sha256'] = digest(__file__)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(a.device)
    write_json(Path(a.output), report)
    print(json.dumps({k: {n: v for n, v in r.items() if n != 'ids'}
                      for k, r in report['partitions'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'manifest', 'codec', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--per-source', type=int, default=64)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error('per-source must be positive')
    run(args)

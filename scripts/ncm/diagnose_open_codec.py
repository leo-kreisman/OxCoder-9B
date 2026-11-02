"""Separate reconstruction loss from quantization and precision effects."""
import argparse
from collections import Counter
from contextlib import nullcontext
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .open_baseline import Corpus, PAD, upstream
from .open_corpus import file_sha
from .train import write_json


def load_codec(checkpoint, upstream_path, device):
    VQVAE, _ = upstream(upstream_path)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    codec = VQVAE(**OmegaConf.create(ck['identity']['codec_config'])).to(device)
    codec.load_state_dict(ck['model'])
    codec.eval().requires_grad_(False)
    return codec, ck


@torch.no_grad()
def diagnose(codec, data, device, batches=16, batch=8):
    results = {}
    for split in ('train', 'validation'):
        counts, errors = Counter(), Counter()
        rng = random.Random(29071)
        examples = []
        for _ in range(batches):
            _, target = data.batch(rng, batch, .2, split)
            y = target.to(device)
            z = codec._encode_pre_quant(y)
            ids = codec.encode(y)
            q = codec.vq.idx_to_z_q(ids)
            predictions = {'unquantized': codec.decoder(z).argmax(-1),
                           'fp32': codec.decoder(q).argmax(-1)}
            ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
            with ctx:
                low_ids = codec.encode(y)
                predictions['bf16'] = codec.decode(low_ids).argmax(-1)
            counts['codes_same'] += int((ids == low_ids).sum())
            counts['codes_total'] += ids.numel()
            valid = y != PAD
            counts['tokens'] += int(valid.sum())
            counts['windows'] += len(y)
            for key, pred in predictions.items():
                counts[key+'_correct'] += int(((pred == y) & valid).sum())
                counts[key+'_exact'] += int((pred == y).all(-1).sum())
            pred = predictions['fp32']
            for expected, got in zip(y[(pred != y) & valid].tolist(), pred[(pred != y) & valid].tolist()):
                errors[expected, got] += 1
            if len(examples) < 3:
                examples.append({'reference': data.tokenizer.decode(y[0].tolist()),
                                 'reconstruction': data.tokenizer.decode(pred[0].tolist())})
        results[split] = {'windows': counts['windows'], 'nonpad_tokens': counts['tokens'],
            'code_precision_agreement': counts['codes_same']/counts['codes_total'],
            'paths': {key: {'nonpad_accuracy': counts[key+'_correct']/counts['tokens'],
                           'exact_windows': counts[key+'_exact']} for key in predictions},
            'top_errors': [{'expected': data.tokenizer.id_to_token(e),
                            'got': data.tokenizer.id_to_token(g), 'count': n}
                           for (e,g),n in errors.most_common(30)], 'examples': examples}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--upstream', default='results/ncm-upstream')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batches', type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(2)
    codec, ck = load_codec(a.checkpoint, a.upstream, a.device)
    data = Corpus(a.data)
    if ck['identity']['data_sha256'] != file_sha(Path(a.data)/'manifest.json'):
        raise ValueError('Codec and data identities differ')
    report = {'checkpoint_sha256': file_sha(a.checkpoint),
              'data_sha256': ck['identity']['data_sha256'],
              'diagnostic': diagnose(codec, data, a.device, a.batches), 'quality_ready': False}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.output), report)
    for split, result in report['diagnostic'].items():
        print(split, result['paths'], 'code precision agreement:', result['code_precision_agreement'], flush=True)


if __name__ == '__main__':
    main()

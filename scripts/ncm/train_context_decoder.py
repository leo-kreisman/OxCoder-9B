"""Train a native multiscale decoder over cached, frozen OxCoder prompt states.

This is a separate encoder-decoder experiment. It does not resume the v24
adapter generator and does not convert the frozen encoder's layers into NCM
decoder layers. Every generated answer byte comes from predicted concept codes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .anvil import OptimizerBundle
from .backbone import digest
from .codec import Codec, CodecConfig, PAD, decode, resize
from .context_cache import FeatureCache
from .context_decoder import ContextConceptDecoder, DecoderConfig
from .train import admission, headroom, save, write_json, cpu_tree


def make_optimizer(model, lr):
    matrices, other = [], []
    for name, parameter in model.named_parameters():
        hidden = parameter.ndim == 2 and not name.startswith(('heads.', 'stop.', 'level.', 'position'))
        (matrices if hidden else other).append(parameter)
    return OptimizerBundle([
        torch.optim.AdamW(other, lr=lr, weight_decay=.01),
        torch.optim.Muon(matrices, lr=lr, weight_decay=.01, momentum=.95,
                         nesterov=True, ns_steps=5, adjust_lr_fn='match_rms_adamw')])


def batch(cache, indices, device):
    items = [cache.get(i) for i in indices]
    maximum = max(x['hidden'].shape[0] for x in items)
    hidden = torch.zeros(len(items), maximum, cache.config.source_width, dtype=torch.bfloat16)
    valid = torch.zeros(len(items), maximum, dtype=torch.bool)
    for i, item in enumerate(items):
        size = item['hidden'].shape[0]
        hidden[i, :size] = item['hidden']; valid[i, :size] = True
    return {'hidden': hidden.to(device), 'valid': valid.to(device),
            'codes': [torch.stack([item['codes'][j] for item in items]).to(device)
                      for j in range(len(cache.config.levels))],
            'bytes': torch.stack([item['bytes'] for item in items]).to(device),
            'stop': torch.tensor([float(item['condition']['is_final_chunk']) for item in items], device=device),
            'items': items}


@torch.no_grad()
def previous_concepts(codec, codes, corruption=0.):
    if not 0 <= corruption <= 1:
        raise ValueError('Invalid concept corruption probability')
    z = torch.zeros(codes[0].shape[0], codec.config.length, codec.config.width, device=codes[0].device)
    inputs = []
    for level, (book, code) in enumerate(zip(codec.books, codes)):
        inputs.append(codec.resample(z, codec.config.levels[level]))
        used = code
        if corruption and level + 1 < len(codes):
            used = torch.where(torch.rand(code.shape, device=code.device) < corruption,
                               torch.randint(codec.config.codes, code.shape, device=code.device), code)
        z = z + resize(book(used), codec.config.length)
    return inputs


@torch.no_grad()
def validation(model, codec, cache, indices, device):
    was_training = model.training
    model.eval()
    observations = []
    for index in indices:
        b = batch(cache, [index], device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            logits, ending = model(previous_concepts(codec, b['codes']), b['hidden'], b['valid'])
            output, _, stop = model.generate_chunk(codec, b['hidden'], b['valid'])
        try:
            text = decode(output[0].tolist())
        except (ValueError, UnicodeError):
            text = None
        target = b['bytes']
        observations.append({'id': b['items'][0]['row']['id'],
            'teacher_forced_ce': [float(F.cross_entropy(p.flatten(0, 1), q.flatten())) for p, q in zip(logits, b['codes'])],
            'teacher_forced_accuracy': [float((p.argmax(-1) == q).float().mean()) for p, q in zip(logits, b['codes'])],
            'free_exact_chunk': text == b['items'][0]['row']['target'], 'free_valid_chunk': text is not None,
            'free_byte_accuracy_nonpad': float((output == target)[target != PAD].float().mean()),
            'generated_text': text, 'stop_probability': float(stop[0]),
            'expected_final_chunk': bool(b['stop'][0]), 'codec_exact_nonpad': b['items'][0]['codec_exact']})
    model.train(was_training)
    return {'records': observations, 'windows': len(observations),
        'mean_level_ce': sum(sum(x['teacher_forced_ce']) / len(x['teacher_forced_ce']) for x in observations) / len(observations),
        'free_exact_windows': sum(x['free_exact_chunk'] for x in observations),
        'free_valid_windows': sum(x['free_valid_chunk'] for x in observations),
        'mean_free_nonpad_byte_accuracy': sum(x['free_byte_accuracy_nonpad'] for x in observations) / len(observations),
        'quality_ready': False, 'scope': 'Fixed validation chunks, including gold earlier response chunks in some prompts; not complete coding answers'}


def run(a):
    if (min(a.steps, a.batch, a.microbatch, a.save_every, a.eval_every, a.warmup) < 1
            or a.batch % a.microbatch or not 0 <= a.corruption <= 1 or not math.isfinite(a.lr) or a.lr <= 0):
        raise ValueError('Positive training sizes and divisible microbatches required')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume)
    write_json(out / 'admission.json', admission([a.device], 8))
    codec_sha = digest(a.codec)
    ck = torch.load(a.codec, weights_only=True, map_location='cpu')
    if digest(a.codec) != codec_sha:
        raise ValueError('Codec changed during loading')
    codec = Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model']); codec.eval().requires_grad_(False)
    config = DecoderConfig(latent_width=codec.config.width, codes=codec.config.codes,
                           levels=codec.config.levels, width=a.width, heads=a.heads, layers=a.layers)
    cache = FeatureCache(a.cache, config)
    if cache.metadata['identity']['codec_sha256'] != codec_sha:
        raise ValueError('Feature labels use a different codec')
    training = [i for i, r in enumerate(cache.records) if r['split'] == 'train']
    heldout = [i for i, r in enumerate(cache.records) if r['split'] == 'validation']
    if not training or len(heldout) != 64:
        raise ValueError('Expected training rows and all 64 frozen validation rows')
    model = ContextConceptDecoder(config).to(a.device)
    model.checkpoint_blocks = True
    optimizer = make_optimizer(model, a.lr)
    code_names = ('context_decoder.py', 'context_features.py', 'context_cache.py',
                  'train_context_decoder.py', 'context_runtime.py', 'codec.py', 'anvil.py')
    identity = {'architecture': 'frozen-oxcoder-prompt-encoder-native-ncm-decoder-v25',
        'config': asdict(config), 'source_sha256': cache.metadata['identity']['source_sha256'],
        'codec_sha256': codec_sha, 'feature_manifest_sha256': cache.sha,
        'selection_sha256': cache.metadata['identity']['selection_sha256'],
        'prepared_sha256': cache.metadata['identity']['prepared_sha256'],
        'prompt_tokens': config.maximum_prompt, 'autoregressive_answer_fallback': False,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_scope': 'native concept decoder only; all original 9B prompt encoder weights frozen',
        'optimizer': 'FP32-parameter Muon matrices + AdamW embeddings, norms and output heads',
        'lr': a.lr, 'batch': a.batch, 'microbatch': a.microbatch, 'seed': a.seed,
        'planned_steps': a.steps, 'warmup': a.warmup, 'max_corruption': a.corruption,
        'precision': 'FP32 parameters/optimizer, BF16 autocast, full precision reductions',
        'torch_version': str(torch.__version__), 'device_type': torch.device(a.device).type,
        'implementation_sha256': {n: digest(Path(__file__).parent / n) for n in code_names}}
    # Tuples become arrays in JSON; checkpoint identity comparison stays native.
    sampler = random.Random(a.seed)
    order, position, epoch, start, best = [], 0, 0, 0, float('inf')
    if a.resume:
        state = torch.load(out / 'latest.pt', weights_only=True, map_location='cpu')
        if state['identity'] != identity:
            raise ValueError('Training resume identity changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        order, position, epoch = state['sampler']['order'], state['sampler']['position'], state['sampler']['epoch']
        sampler.setstate(state['sampler']['random'])
        torch.set_rng_state(state['torch_rng'])
        if a.device.startswith('cuda'): torch.cuda.set_rng_state(state['cuda_rng'], a.device)
        start, best = state['step'], state['best_validation_ce']
    else:
        save(out / 'initial.pt', {'identity': identity, 'model': cpu_tree(model.state_dict()), 'step': 0})
    write_json(out / 'identity.json', identity)
    model.train()
    sequence = hashlib.sha256()
    for step in range(start + 1, a.steps + 1):
        headroom([a.device])
        begin = time.monotonic()
        selected = []
        while len(selected) < a.batch:
            if position == len(order):
                order = training.copy(); sampler.shuffle(order); position = 0; epoch += 1
            take = min(a.batch - len(selected), len(order) - position)
            selected.extend(order[position:position + take]); position += take
        # Length grouping changes padding cost, not the selected batch or loss.
        selected.sort(key=lambda i: cache.records[i]['prompt_tokens'])
        for i in selected: sequence.update((cache.records[i]['id'] + '\n').encode())
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * max(0., (step - a.warmup) / max(1, a.steps - a.warmup))))
        learning_rate = a.lr * min(1., step / a.warmup) * decay
        for opt in optimizer.optimizers:
            for group in opt.param_groups: group['lr'] = learning_rate
        corruption = a.corruption * min(1., max(0., (step - a.warmup) / (2 * a.warmup)))
        optimizer.zero_grad()
        loss_total, acc = 0., [0.] * len(config.levels)
        for offset in range(0, a.batch, a.microbatch):
            b = batch(cache, selected[offset:offset + a.microbatch], a.device)
            inputs = previous_concepts(codec, b['codes'], corruption)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.device.startswith('cuda')):
                logits, ending = model(inputs, b['hidden'], b['valid'])
                loss = sum(F.cross_entropy(p.flatten(0, 1), q.flatten()) for p, q in zip(logits, b['codes'])) / len(config.levels)
                loss = loss + .1 * F.binary_cross_entropy_with_logits(ending, b['stop'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite NCM objective')
            (loss * a.microbatch / a.batch).backward()
            loss_total += float(loss.detach()) * a.microbatch / a.batch
            for j, (p, q) in enumerate(zip(logits, b['codes'])):
                acc[j] += float((p.argmax(-1) == q).float().mean()) * a.microbatch / a.batch
            del loss, logits, ending, inputs, b
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        if a.device.startswith('cuda'): torch.cuda.synchronize(a.device)
        record = {'step': step, 'loss': loss_total, 'code_accuracy': acc,
                  'seconds': time.monotonic() - begin, 'examples': a.batch,
                  'epoch': epoch, 'lr': learning_rate, 'corruption': corruption, 'gradient_norm': gradient,
                  'cuda_peak_bytes': torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None,
                  'sequence_since_resume_sha256': sequence.hexdigest(), 'quality_ready': False}
        with (out / 'training.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        write_json(out / 'progress.json', record)
        do_eval = step % a.eval_every == 0 or step == a.steps
        if do_eval:
            report = validation(model, codec, cache, heldout, a.device)
            report['step'] = step
            write_json(out / f'validation-{step}.json', report)
            improved = report['mean_level_ce'] < best
            best = min(best, report['mean_level_ce'])
        if step % a.save_every == 0 or do_eval:
            state = {'identity': identity, 'model': cpu_tree(model.state_dict()),
                'optimizer': cpu_tree(optimizer.state_dict()), 'step': step, 'best_validation_ce': best,
                'sampler': {'order': order, 'position': position, 'epoch': epoch, 'random': sampler.getstate()},
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(a.device) if a.device.startswith('cuda') else None}
            save(out / 'latest.pt', state)
            if do_eval:
                save(out / f'evaluated-{step}.pt', {'identity': identity, 'model': state['model'], 'step': step})
                if improved: save(out / 'best.pt', {'identity': identity, 'model': state['model'], 'step': step})
            del state
        if (out / 'STOP_AFTER_CHECKPOINT').exists() and (step % a.save_every == 0 or do_eval):
            write_json(out / 'stopped.json', {'step': step, 'checkpoint_sha256': digest(out / 'latest.pt'), 'quality_ready': False})
            return
    write_json(out / 'completed.json', {'step': a.steps, 'quality_ready': False,
                                      'scope': 'Training complete; executable generation acceptance remains separate'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'codec', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--corruption', type=float, default=.15)
    p.add_argument('--seed', type=int, default=25001)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())

"""Two-stage coding baseline using pinned upstream NCM modules.

This is an independently implemented trainer, not an OxCoder conversion or an
exact reproduction of the paper's dataset, tokenizer, scale or training budget.
"""
import argparse
import ast
import copy
import fcntl
from collections import Counter
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from .open_corpus import file_sha
from .train import admission, headroom, save, write_json, cpu_tree, rng_state, set_rng

PIN = 'c70274273f489abc73832b2b98584a8483b6b203'
SPECIAL = ['<|ncm_pad|>', '<|ncm_bos|>', '<|ncm_eos|>']
PAD, BOS, EOS, LENGTH = 0, 1, 2, 128
LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]


def upstream(path):
    path = Path(path).resolve()
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()
    if rev != PIN or dirty:
        raise ValueError('Require clean, reviewed upstream revision')
    sys.path.insert(0, str(path))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    return VQVAE, NextConceptTransformer


def prepare(a):
    source = Path(a.corpus)
    manifest = json.loads((source/'manifest.json').read_text())
    if file_sha(source/'documents.jsonl') != manifest['documents_sha256']:
        raise ValueError('Source checksum differs')
    prompts = {}
    # Earlier acquisition snapshots stored assistant targets only. Recover their
    # complete original prompts from checksummed provenance, never from targets.
    for record in manifest.get('sources', []):
        if 'local_corpus' not in record:
            continue
        path = Path(record['local_corpus'])
        if file_sha(path) != record['sha256']:
            raise ValueError('Assistant provenance differs')
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                prompts[row['id']] = row.get('prompt', '')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=2,
        special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    def documents():
        with (source/'documents.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                if row['split'] == 'train':
                    yield row['text']
    tokenizer.train_from_iterator(documents(), trainer=trainer)
    tokenizer.save(str(out/'tokenizer.json'))
    streams = {s: (out/(s+'.bin')).open('wb') for s in ('train', 'validation', 'test')}
    offsets, counts, source_counts = Counter(), Counter(), Counter()
    metadata = []
    groups = {}
    try:
        with (source/'documents.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                s = row['split']
                if groups.setdefault(row['group'], s) != s:
                    raise ValueError('Group leakage')
                if any(t in row['text'] for t in SPECIAL):
                    raise ValueError('Literal reserved control token in corpus')
                ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
                if tokenizer.decode(ids, skip_special_tokens=False) != row['text']:
                    raise ValueError('Pre-tokenizer is not text-exact')
                prompt = row.get('prompt', prompts.get(row['id'], ''))
                if any(t in prompt for t in SPECIAL):
                    raise ValueError('Literal reserved control token in prompt')
                context = tokenizer.encode(prompt, add_special_tokens=False).ids
                stored = np.asarray([BOS]+context+ids+[EOS], dtype='<u2')
                streams[s].write(stored.tobytes())
                metadata.append({k: v for k, v in row.items() if k not in ('text','prompt')} |
                    {'offset': offsets[s], 'length': len(stored), 'target_offset': 1+len(context),
                     'windows': math.ceil((len(ids)+1)/LENGTH)})
                offsets[s] += len(stored)
                counts[s+'_'+row['kind']+'_tokens'] += len(ids)
                source_counts[row.get('source','fixture')] += 1
                if row.get('distilled'):
                    counts[s+'_distilled_documents'] += 1
    finally:
        for stream in streams.values():
            stream.close()
    with (out/'documents.jsonl').open('w') as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    write_json(out/'manifest.json', {'source_manifest_sha256': file_sha(source/'manifest.json'),
        'vocab_size': tokenizer.get_vocab_size(), 'length': LENGTH, 'counts': dict(counts),
        'source_documents': dict(source_counts),
        'documents': len(metadata), 'roundtrip_exact': True,
        'files': {name: file_sha(out/name) for name in
                  ['tokenizer.json', 'documents.jsonl', 'train.bin', 'validation.bin', 'test.bin']},
        'tokenizer_training_split': 'train only', 'quality_ready': False})


class Corpus:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        for name, digest in self.manifest['files'].items():
            if file_sha(self.path/name) != digest:
                raise ValueError('Prepared corpus checksum mismatch: '+name)
        self.rows = [json.loads(x) for x in (self.path/'documents.jsonl').read_text().splitlines()]
        self.arrays = {s: np.memmap(self.path/(s+'.bin'), dtype='<u2', mode='r')
                       for s in ('train', 'validation', 'test')}
        self.tokenizer = Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.buckets = {}
        for s in self.arrays:
            for kind in ('code', 'assistant'):
                rows = [r for r in self.rows if r['split'] == s and r['kind'] == kind]
                self.buckets[s, kind] = rows, np.cumsum([r['windows'] for r in rows])

    def ids(self, row):
        return self.arrays[row['split']][row['offset']:row['offset']+row['length']]

    def window(self, row, index):
        ids = self.ids(row)
        start = row.get('target_offset', 1) + index * LENGTH
        prefix = np.full(LENGTH, PAD, dtype=np.int64)
        prior = ids[max(0, start-LENGTH):start]
        prefix[-len(prior):] = prior
        target = np.full(LENGTH, PAD, dtype=np.int64)
        part = ids[start:start+LENGTH]
        target[:len(part)] = part
        return prefix, target

    def batch(self, rng, n, assistant_fraction, split='train'):
        items = []
        self.last_batch_tokens = Counter()
        for _ in range(n):
            kind = 'assistant' if rng.random() < assistant_fraction else 'code'
            rows, ends = self.buckets[split, kind]
            if not rows:
                raise ValueError('Empty sampling bucket')
            ticket = rng.randrange(int(ends[-1]))
            i = int(np.searchsorted(ends, ticket, side='right'))
            win = ticket - (int(ends[i-1]) if i else 0)
            item = self.window(rows[i], win)
            items.append(item)
            self.last_batch_tokens[kind] += int(np.count_nonzero(item[1] != PAD))
        return tuple(torch.from_numpy(np.stack([x[j] for x in items])) for j in (0, 1))


def codec_config(vocab):
    return dict(in_vocab_size=vocab, context_length=LENGTH, embed_dim=256,
        vqvae_vocab_size=[4096]*len(LEVELS), n_layers=4, compression_factor=1,
        dropout=.1, pre_quant_groupnorm=8, use_qk_norm=True,
        enc_layers=0, dec_layers=1,
        vector_quantizer_config=dict(clss='multiscale_residual_vector_quantizer',
            decay=.95, epsilon=1e-5, commitment_cost=.25,
            learned_l1_sampling=True, learned_l1_cascade=True, learned_l1_cascade_base=4,
            restart_threshold=0., quant_resi=dict(enabled=True, ratio=.5, share_mode=0),
            aux=dict(fine_drop=dict(prob=.5, min_keep=1)),
            levels=dict(use_manual_levels=True, manual_levels=LEVELS)))


def generator_config():
    return dict(n_embd=384, n_head=6, n_layer=6, dropout=.1, bias=True,
        prefix_len=LENGTH, context_length=LENGTH*2, vocab_size=[4096]*len(LEVELS),
        attn_pattern='block_diagonal', use_positional_encoding=True, use_level_encoding=True,
        use_rope=True, rope_base=100000, use_qk_norm=True, raw_token_prefix=True,
        prefix_pad_token_id=PAD, query_level0=True, predict_level0=False,
        tie_heads_to_codebook=True, tie_heads_learn_scale=True, use_flex_attention=False)


def validate_codec_config(cfg, vocab):
    """Accept the reviewed baseline or its explicit no-dropout refinement."""
    reviewed=codec_config(vocab)
    actual=copy.deepcopy(cfg)
    if actual.get('dropout') not in (0.,.1):
        raise ValueError('Unreviewed codec dropout')
    actual['dropout']=reviewed['dropout']
    fine=actual['vector_quantizer_config']['aux']['fine_drop']
    if fine.get('prob') not in (0.,.5):
        raise ValueError('Unreviewed codec fine dropout')
    fine['prob']=reviewed['vector_quantizer_config']['aux']['fine_drop']['prob']
    if actual!=reviewed:
        raise ValueError('Unreviewed codec architecture')


@torch.no_grad()
def initialize_books(codec, data, device, seed, windows=1024):
    """Use upstream k-means and consistent EMA sums/counts, training data only.

    Both arms initialize from the identical code-only sample. Dead-code restarts
    remain disabled. This explicit initialization is part of our trainer recipe.
    """
    codec.eval()
    rng = random.Random(seed+190)
    latents = []
    for _ in range(windows//8):
        _, y = data.batch(rng, 8, 0.)
        latents.append(codec._encode_pre_quant(y.to(device)).float())
    residual = torch.cat(latents)
    book = codec.vq
    for i, size in enumerate(LEVELS):
        flat = book._down_for_quant(residual, i, size).reshape(-1, codec.embed_dim)
        indexes = torch.randperm(len(flat), device=device)[:8192]
        book.kmeans_init_level(i, flat[indexes], n_iters=5)
        selected = []
        for start in range(0, len(flat), 512):
            selected.append(book._distances(flat[start:start+512], book.embeddings[i].weight).argmin(-1))
        codes = torch.cat(selected).reshape(len(residual), size)
        q = book.embeddings[i](codes)
        if size != LENGTH:
            q = book._up_to_full(q, i, LENGTH)
        residual = residual-book._apply_quant_resi(q, i)


def corrupt_codes(clean, levels, cardinality, maximum=.1):
    """Only conditioning is perturbed; callers retain clean CE targets."""
    pieces = clean.split(levels, dim=1)
    noisy = []
    for i, part in enumerate(pieces):
        rate = maximum * (1-i/max(1, len(levels)-1))
        mask = torch.rand(part.shape, device=part.device) < rate
        replacement = torch.randint(cardinality, part.shape, device=part.device)
        noisy.append(torch.where(mask, replacement, part))
    return torch.cat(noisy, dim=1)


def generator_inputs(model, codec, prefix, codes):
    with torch.no_grad():
        p = codec.encoder.wte(prefix)
        x = codec.vq.idx_to_ncm_input(codes)
    q = model.build_l0_query_input(prefix.shape[0], p.dtype)
    return torch.cat([p, q, x], dim=1)


@torch.no_grad()
def codec_evaluate(codec, data, device, documents=1000):
    codec.eval()
    rows = sorted((r for r in data.rows if r['split'] == 'validation'), key=lambda r: r['text_sha256'])
    # Common evaluation set for both arms. Include assistants without replacing code coverage.
    chosen = [r for r in rows if r['kind'] == 'code'][:documents]
    chosen += [r for r in rows if r['kind'] == 'assistant'][:100]
    report, correct, total = [], 0, 0
    code_counts = [torch.zeros(4096, dtype=torch.long) for _ in LEVELS]
    for row in chosen:
        ids = data.ids(row)
        predictions = []
        all_padding_exact = True
        for start in range(0, row['windows'], 16):
            batch = [data.window(row, i)[1] for i in range(start, min(start+16, row['windows']))]
            target = torch.from_numpy(np.stack(batch)).to(device)
            codes = codec.encode(target)
            for i, ids_level in enumerate(codes.cpu().split(LEVELS,1)):
                code_counts[i] += torch.bincount(ids_level.flatten(), minlength=4096)
            predicted = codec.decode(codes).argmax(-1).cpu()
            valid = torch.from_numpy(np.stack(batch)) != PAD
            correct += int(((predicted == torch.from_numpy(np.stack(batch))) & valid).sum())
            total += int(valid.sum())
            all_padding_exact &= bool((predicted[~valid] == PAD).all())
            predictions.extend(predicted.flatten().tolist())
        expected = ids[row.get('target_offset', 1):].tolist()
        got = predictions[:len(expected)]
        text = data.tokenizer.decode(got[:-1], skip_special_tokens=False)
        original = data.tokenizer.decode(expected[:-1], skip_special_tokens=False)
        syntax = None
        if row['kind'] == 'code':
            try:
                ast.parse(text)
                syntax = True
            except (SyntaxError, ValueError, RecursionError):
                syntax = False
        report.append({'id': row['id'], 'kind': row['kind'], 'tokens': len(expected),
            'exact_tokens': got == expected, 'exact_text': text == original,
            'termination_exact': bool(got and got[-1] == EOS and EOS not in got[:-1]),
            'padding_exact': all_padding_exact, 'syntax_valid': syntax})
    exact = sum(r['exact_tokens'] and r['padding_exact'] for r in report)
    by_length = {}
    for low, high in ((0,128),(128,512),(512,2048),(2048,10**9)):
        subset = [r for r in report if low < r['tokens'] <= high]
        by_length[f'{low+1}-{high}'] = {'documents':len(subset),
            'exact_documents':sum(r['exact_tokens'] and r['padding_exact'] for r in subset)}
    return {'documents': len(report), 'exact_documents': exact, 'token_accuracy': correct/max(1,total),
        'unique_codes_per_level': [int((x>0).sum()) for x in code_counts],
        'by_length': by_length, 'syntax_valid_documents':sum(r['syntax_valid'] is True for r in report),
        'rows': report, 'passed': bool(report) and exact == len(report),
        'scope': 'development reconstruction; not generated-code correctness'}


@torch.no_grad()
def generator_evaluate(model, codec, data, device, count=64):
    model.eval()
    prefix, target = data.batch(random.Random(918), count, .2, 'validation')
    records = []
    for i in range(count):
        p, y = prefix[i:i+1].to(device), target[i:i+1].to(device)
        clean = codec.encode(y)
        logits = model(generator_inputs(model, codec, p, clean), prefix_pad_mask=p == PAD)
        accuracy = float((logits.argmax(-1) == clean).float().mean())
        # No gold coarse code and no gold earlier level at generation time.
        generated = model.generate(None, p, temperature=0., top_k=0, top_p=1., use_kv_cache=False)
        got = codec.decode(generated).argmax(-1)
        records.append({'index': i, 'teacher_forced_concept_accuracy': accuracy,
            'free_concept_accuracy': float((generated == clean).float().mean()),
            'free_nonpad_token_accuracy': float(((got == y) & (y != PAD)).sum()/(y != PAD).sum().clamp(min=1)),
            'exact_window': bool((got == y).all()),
            'prediction': data.tokenizer.decode(got[0].tolist(), skip_special_tokens=False),
            'reference': data.tokenizer.decode(y[0].tolist(), skip_special_tokens=False)})
    return {'windows': count, 'exact_windows': sum(r['exact_window'] for r in records),
        'mean_teacher_forced_concept_accuracy': sum(r['teacher_forced_concept_accuracy'] for r in records)/count,
        'mean_free_concept_accuracy': sum(r['free_concept_accuracy'] for r in records)/count,
        'mean_free_nonpad_token_accuracy': sum(r['free_nonpad_token_accuracy'] for r in records)/count,
        'rows': records, 'quality_ready': False,
        'scope': 'held-out code continuation diagnostic; exact matching is not a functional coding benchmark'}


@torch.no_grad()
def coding_diagnostic(model, codec, data, device):
    from .coding_eval import TASKS, check_case, extract
    model.eval()
    results = []
    for name, question, cases in TASKS:
        prompt = json.dumps([{'role':'system','content':'Return only Python code defining the requested function.'},
                             {'role':'user','content':question}], ensure_ascii=False, sort_keys=True)
        history = [BOS]+data.tokenizer.encode(prompt, add_special_tokens=False).ids
        initial_tokens = len(history)
        output, stopped, invalid = [], False, False
        start = time.monotonic()
        for _ in range(8):
            prefix = ([PAD]*max(0,LENGTH-len(history))+history)[-LENGTH:]
            p = torch.tensor([prefix], device=device)
            codes = model.generate(None,p,temperature=0.,top_k=0,top_p=1.,use_kv_cache=False)
            ids = codec.decode(codes).argmax(-1)[0].tolist()
            for token in ids:
                if token == EOS:
                    stopped = True
                    break
                if token in (PAD,BOS):
                    invalid = True
                    break
                output.append(token)
                history.append(token)
            if stopped or invalid:
                break
        text = data.tokenizer.decode(output, skip_special_tokens=False)
        checks = [check_case(extract(text),name,args,want,preserve_arguments=name=='merge_intervals')
                  for args,want in cases] if stopped and not invalid else []
        results.append({'task':name,'text':text,'predicted_end':stopped,'invalid_control_token':invalid,
            'initial_prompt_tokens':initial_tokens,'prefix_window':LENGTH,'seconds':time.monotonic()-start,
            'checks':checks,'passed':len(checks)==len(cases) and all(c['passed'] for c in checks)})
    return {'tasks':results,'passed':sum(r['passed'] for r in results),'evaluated':len(results),
        'scope':'reused 12-task development diagnostic; 128-token rolling context; not OxCoder or professional qualification',
        'quality_ready':False}


def train(a):
    from omegaconf import OmegaConf
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    VQVAE, Generator = upstream(a.upstream)
    limits = admission([a.device], 8)
    data = Corpus(a.data)
    cfg = codec_config(data.manifest['vocab_size'])
    ck = None
    if a.mode == 'generator':
        ck = torch.load(a.codec, map_location='cpu', weights_only=True)
        if ck['identity']['data_sha256'] != file_sha(Path(a.data)/'manifest.json'):
            raise ValueError('Codec dataset differs')
        cfg = ck['identity']['codec_config']
        validate_codec_config(cfg,data.manifest['vocab_size'])
    codec = VQVAE(**OmegaConf.create(cfg)).to(a.device)
    codec_identity = None
    gcfg = generator_config()
    if a.untied_heads:
        gcfg['tie_heads_to_codebook'] = False
    if a.mode == 'generator':
        codec.load_state_dict(ck['model'])
        codec.eval().requires_grad_(False)
        codec_identity = file_sha(a.codec)
        model = Generator(codec, OmegaConf.create(gcfg)).to(a.device)
    else:
        model = codec
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume)
    lock = (out/'.training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'upstream': PIN, 'data_sha256': file_sha(Path(a.data)/'manifest.json'),
        'codec_config': cfg, 'generator_config': gcfg if a.mode == 'generator' else None,
        'codec_sha256': codec_identity, 'mode': a.mode, 'assistant_fraction': a.assistant_fraction,
        'microbatch': a.batch, 'accumulation': a.accumulation, 'seed': a.seed, 'lr': a.lr,
        'steps': a.steps, 'corruption': a.corruption if a.mode == 'generator' else None,
        'optimizer': 'AdamW', 'precision': 'FP32' if a.fp32 else 'BF16 autocast',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initialization': 'upstream kmeans; 1024 train-code windows; max8192 samples/level; 5 iterations; no restarts',
        'trainer_sha256': file_sha(__file__), 'quality_ready': False}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=.01)
    rng = random.Random(a.seed)
    begin, best, history, exposure = 0, -1., [], Counter()
    if a.resume:
        ck = torch.load(out/'latest.pt', map_location='cpu', weights_only=True)
        if ck['identity'] != identity:
            raise ValueError('Resume recipe differs')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        begin, best, history = ck['step'], ck['best'], ck['history']
        exposure.update(ck['exposure'])
        rng.setstate(ck['data_rng'])
        set_rng(ck['rng'])
    elif a.mode == 'codec':
        initialize_books(codec, data, a.device, a.seed)
    write_json(out/'identity.json', identity)
    write_json(out/'admission.json', limits)
    start_time = time.monotonic()
    for step in range(begin, a.steps):
        headroom([a.device])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = a.lr * min(1., (step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses, valid_tokens = [], 0
        for _ in range(a.accumulation):
            prefix, target = data.batch(rng, a.batch, a.assistant_fraction)
            exposure.update(data.last_batch_tokens)
            prefix, target = prefix.to(a.device), target.to(a.device)
            valid_tokens += int((target != PAD).sum())
            # Casting z to float32 inside the upstream quantizer does not disable
            # surrounding autocast. Full FP32 is an explicit recipe choice.
            if a.mode == 'codec':
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.device.startswith('cuda') and not a.fp32):
                    logits, vq_loss, *_ = model(target)
                    loss = F.cross_entropy(logits.flatten(0,1).float(), target.flatten()) + vq_loss
            else:
                with torch.no_grad():
                    clean = codec.encode(target)
                    noisy = corrupt_codes(clean, LEVELS, 4096, a.corruption)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.device.startswith('cuda') and not a.fp32):
                    logits = model(generator_inputs(model, codec, prefix, noisy), prefix_pad_mask=prefix == PAD)
                    # Equal weighting per hierarchy level, preserving coarse-level signal.
                    loss = torch.stack([F.cross_entropy(l.flatten(0,1).float(), t.flatten())
                        for l, t in zip(logits.split(LEVELS,1), clean.split(LEVELS,1))]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss/a.accumulation).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % a.save_every == 0 or step+1 == a.steps or step+1 == a.stop_after:
            item = {'step': step+1, 'loss': sum(losses)/len(losses), 'lr': lr,
                'elapsed_seconds': time.monotonic()-start_time, 'nonpad_tokens_this_update': valid_tokens,
                'cumulative_nonpad_tokens_including_eos': dict(exposure),
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else 0}
            if (step+1) % a.eval_every == 0 or step+1 == a.steps:
                if a.mode == 'codec':
                    metric = codec_evaluate(codec, data, a.device, 1000 if step+1 == a.steps else 64)
                    score = metric['exact_documents']/metric['documents'] + .001*metric['token_accuracy']
                else:
                    metric = generator_evaluate(model, codec, data, a.device)
                    score = metric['mean_free_concept_accuracy']
                write_json(out/f'eval-{step+1}.json', metric)
                item['evaluation'] = {k:v for k,v in metric.items() if k != 'rows'}
                # Final codec is evaluated with the same 64+100 selection set too.
                if a.mode == 'codec' and step+1 == a.steps:
                    quick = codec_evaluate(codec, data, a.device, 64)
                    score = quick['exact_documents']/quick['documents']+.001*quick['token_accuracy']
                if score > best:
                    best = score
                    save(out/'best.pt', {'identity': identity, 'step': step+1, 'model': cpu_tree(model.state_dict())})
            history.append(item)
            save(out/'latest.pt', {'identity': identity, 'step': step+1, 'model': cpu_tree(model.state_dict()),
                'optimizer': cpu_tree(optimizer.state_dict()), 'rng': rng_state(), 'data_rng': rng.getstate(),
                'history': history, 'best': best, 'exposure':dict(exposure)})
            write_json(out/'progress.json', {'step': step+1, 'training_complete': False, 'history': history, 'quality_ready': False})
            print(json.dumps(item), flush=True)
            if step+1 == a.stop_after and step+1 < a.steps:
                return
    chosen = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(chosen['model'])
    final = codec_evaluate(codec, data, a.device, 1000) if a.mode == 'codec' else generator_evaluate(model, codec, data, a.device)
    write_json(out/'selected-evaluation.json', final | {'checkpoint_sha256': file_sha(out/'best.pt'),
        'selected_step': chosen['step'], 'quality_ready': False})
    if a.mode == 'generator':
        write_json(out/'coding-development.json',coding_diagnostic(model,codec,data,a.device))
    write_json(out/'progress.json', {'step': a.steps, 'training_complete': True, 'history': history, 'quality_ready': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'codec', 'generator'])
    p.add_argument('--corpus')
    p.add_argument('--data')
    p.add_argument('--output', required=True)
    p.add_argument('--upstream', default='results/ncm-upstream')
    p.add_argument('--codec')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--assistant-fraction', type=float, default=.2)
    p.add_argument('--corruption', type=float, default=.1)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--untied-heads', action='store_true', help='Generator: independently learned classifiers instead of codebook distance heads')
    p.add_argument('--fp32', action='store_true', help='Disable training autocast; record as a distinct recipe')
    p.add_argument('--stop-after', type=int, help='Checkpoint and exit at this update; for interruption/resume controls')
    a = p.parse_args()
    if not 0 <= a.assistant_fraction <= 1 or not 0 <= a.corruption <= 1:
        p.error('Sampling and corruption must be probabilities')
    if min(a.batch, a.accumulation, a.steps, a.save_every, a.eval_every) < 1:
        p.error('Positive training counts required')
    if a.mode == 'prepare':
        prepare(a)
    else:
        train(a)


if __name__ == '__main__':
    main()

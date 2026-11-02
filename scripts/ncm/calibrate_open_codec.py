"""Controlled reconstruction refinement of a pinned upstream NCM codec.

The encoder, residual transforms, codebooks and EMA buffers remain immutable.
Only full-resolution reconstruction is optimized, on training documents only.
This tests whether a fixed concept representation retains recoverable details;
it is not evidence that the concept generator can predict those details.
The optional joint scope starts a distinct full-resolution experiment that
also updates the representation. Its old generator targets are not reusable.
"""
import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from .diagnose_open_codec import load_codec, diagnose
from .open_baseline import Corpus, PAD, codec_evaluate, upstream
from .open_corpus import file_sha
from .train import admission, headroom, cpu_tree, save, write_json, rng_state, set_rng


def representation_digest(codec):
    """Include buffers as well as parameters: accidental EMA updates must fail."""
    h = hashlib.sha256()
    for name, tensor in sorted(codec.state_dict().items()):
        if name.startswith('decoder.'):
            continue
        h.update(name.encode())
        h.update(str((tensor.shape, tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def calibration_loss(codec, target):
    # eval mode is intentional; it disables dropout/EMA, not decoder gradients.
    codec.eval()
    with torch.no_grad(), torch.autocast(target.device.type, enabled=False):
        codes = codec.encode(target)
        latent = codec.vq.idx_to_z_q(codes)
    logits = codec.decoder(latent)
    return F.cross_entropy(logits.flatten(0,1).float(), target.flatten())


def selection(metric):
    return (metric['exact_documents'], metric['token_accuracy'])


def token_coverage(batch, vocab, device):
    """Uniform token identities exercise rare symbols; no task answers."""
    ids=torch.randint(3,vocab,(batch,128),device=device)
    if batch>=4:
        ids[:batch//4]=ids[:batch//4,:1].clone()
    # Half are interior windows; half explicitly exercise termination/padding.
    n=batch//2
    if n:
        lengths=torch.randint(32,128,(n,),device=device)
        positions=torch.arange(128,device=device)[None]
        ids[:n]=torch.where(positions>lengths[:,None],PAD,ids[:n])
        ids[torch.arange(n,device=device),lengths]=2
    return ids


def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume)
    with (out/'.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        limits = admission([a.device], 8)
        codec, parent = load_codec(a.checkpoint, a.upstream, a.device)
        cfg = copy.deepcopy(parent['identity']['codec_config'])
        if a.scope == 'joint':
            # Use the normal upstream EMA/training path, without stochastic
            # masking of detail. No replacement or extra residual side channel.
            cfg['dropout'] = 0.
            cfg['vector_quantizer_config']['aux']['fine_drop']['prob'] = 0.
            V, _ = upstream(a.upstream)
            codec = V(**OmegaConf.create(cfg)).to(a.device)
            codec.load_state_dict(parent['model'])
        data = Corpus(a.data)
        data_sha = file_sha(Path(a.data)/'manifest.json')
        if parent['identity']['data_sha256'] != data_sha:
            raise ValueError('Codec and data differ')
        if a.scope == 'decoder':
            codec.decoder.requires_grad_(True)
        frozen_sha = representation_digest(codec)
        identity = {'stage':'full-resolution-codec-refinement', 'scope':a.scope,
            'codec_config':cfg, 'data_sha256':data_sha,
            'initial_checkpoint_sha256':file_sha(a.checkpoint), 'representation_sha256':frozen_sha,
            'trainer_sha256':file_sha(__file__),
            'diagnostic_sha256':file_sha(Path(__file__).with_name('diagnose_open_codec.py')),
            'baseline_sha256':file_sha(Path(__file__).with_name('open_baseline.py')),
            'trainable_parameters':sum(p.numel() for p in codec.parameters() if p.requires_grad),
            'steps':a.steps,'batch':a.batch,'accumulation':a.accumulation,'seed':a.seed,
            'lr':a.lr,'assistant_fraction':.2,'optimizer':'AdamW','precision':'FP32',
            'dropout':False,'ema_updates':a.scope=='joint','fine_dropout':False,
            'synthetic_coverage_fraction':getattr(a,'coverage_fraction',0.),'quality_ready':False}
        trainable=[p for p in codec.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=.01)
        rng = random.Random(a.seed)
        begin, best, history, exposure = 0, (-1,-1.), [], Counter()
        if a.resume:
            ck = torch.load(out/'latest.pt', map_location='cpu', weights_only=True)
            if ck['identity'] != identity:
                raise ValueError('Resume recipe differs')
            codec.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            begin, best, history = ck['step'], ck['best'], ck['history']
            rng.setstate(ck['data_rng'])
            set_rng(ck['rng'])
            exposure.update(ck['exposure'])
            if a.scope == 'decoder' and representation_digest(codec) != frozen_sha:
                raise ValueError('Resume representation changed')
        else:
            metric = codec_evaluate(codec, data, a.device, 64)
            best = selection(metric)
            write_json(out/'eval-0.json',metric)
            save(out/'best.pt',{'identity':identity,'step':0,'model':cpu_tree(codec.state_dict())})
        write_json(out/'identity.json',identity)
        write_json(out/'admission.json',limits)
        started = time.monotonic()
        for step in range(begin+1,a.steps+1):
            headroom([a.device])
            if shutil.disk_usage(out).free < 1 << 30:
                raise RuntimeError('Disk reserve below 1 GiB; previous checkpoint retained')
            lr = a.lr*min(1.,step/50)*(.1+.9*(1+math.cos(math.pi*(step-1)/a.steps))/2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for _ in range(a.accumulation):
                synthetic=int(a.batch*getattr(a,'coverage_fraction',0.))
                _, target = data.batch(rng,a.batch-synthetic,.2)
                exposure.update(data.last_batch_tokens)
                target = target.to(a.device)
                if synthetic:
                    coverage=token_coverage(synthetic,data.manifest['vocab_size'],a.device)
                    exposure['synthetic_token_coverage']+=int((coverage!=PAD).sum())
                    target=torch.cat([target,coverage])
                if a.scope == 'decoder':
                    loss = calibration_loss(codec,target)
                else:
                    codec.train()
                    logits, vq_loss, *_ = codec(target)
                    loss = F.cross_entropy(logits.flatten(0,1), target.flatten())+vq_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite calibration loss')
                (loss/a.accumulation).backward()
                total_loss += float(loss.detach())/a.accumulation
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step()
            if step%a.save_every == 0 or step == a.steps or step == a.stop_after:
                if a.scope == 'decoder' and representation_digest(codec) != frozen_sha:
                    raise RuntimeError('Frozen representation changed')
                item={'step':step,'loss':total_loss,'lr':lr,'elapsed_seconds':time.monotonic()-started,
                      'cumulative_nonpad_tokens_including_eos':dict(exposure)}
                if step%a.eval_every == 0 or step == a.steps:
                    metric=codec_evaluate(codec,data,a.device,64)
                    write_json(out/f'eval-{step}.json',metric)
                    item['evaluation']={k:v for k,v in metric.items() if k!='rows'}
                    if selection(metric)>best:
                        best=selection(metric)
                        save(out/'best.pt',{'identity':identity,'step':step,'model':cpu_tree(codec.state_dict())})
                history.append(item)
                save(out/'latest.pt',{'identity':identity,'step':step,'model':cpu_tree(codec.state_dict()),
                    'optimizer':cpu_tree(optimizer.state_dict()),'data_rng':rng.getstate(),'rng':rng_state(),
                    'best':best,'history':history,'exposure':dict(exposure)})
                write_json(out/'progress.json',{'step':step,'training_complete':False,'history':history,'quality_ready':False})
                print(item,flush=True)
                if step==a.stop_after and step<a.steps:
                    return
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        codec.load_state_dict(ck['model'])
        metric=codec_evaluate(codec,data,a.device,1000)
        write_json(out/'selected-evaluation.json',metric | {'checkpoint_sha256':file_sha(out/'best.pt'),
            'selected_step':ck['step'],'quality_ready':False,'representation_unchanged':representation_digest(codec)==frozen_sha})
        write_json(out/'diagnostic.json',diagnose(codec,data,a.device))
        write_json(out/'progress.json',{'step':a.steps,'training_complete':True,'history':history,'quality_ready':False})
        print('Selected:',ck['step'],'exact documents:',metric['exact_documents'],'/',metric['documents'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--scope',choices=['decoder','joint'],default='decoder')
    p.add_argument('--coverage-fraction',type=float,default=0.,help='Fraction of batch replaced by synthetic token-identity coverage (0 to 0.5)')
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--upstream',default='results/ncm-upstream')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=47)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if min(a.steps,a.batch,a.accumulation,a.eval_every,a.save_every)<1 or a.lr<=0:
        p.error('Positive counts and learning rate required')
    if not 0<=a.coverage_fraction<=.5:
        p.error('Coverage fraction must lie between zero and one half')
    run(a)


if __name__=='__main__':
    main()

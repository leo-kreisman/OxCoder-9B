"""Training-only memorization control, never a coding qualification result."""
import argparse
import random
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from .diagnose_open_codec import load_codec
from .open_baseline import Corpus, PAD, LEVELS, upstream, generator_inputs
from .open_corpus import file_sha
from .train import admission, headroom, write_json


@torch.no_grad()
def measure(model,codec,prefix,codes,target):
    model.eval()
    teacher=model(generator_inputs(model,codec,prefix,codes),prefix_pad_mask=prefix==PAD).argmax(-1)
    free=model.generate(None,prefix,temperature=0.,top_k=0,top_p=1.,use_kv_cache=False)
    decoded=codec.decode(free).argmax(-1)
    oracle=codec.decode(codes).argmax(-1)
    def levels(got):
        return [float((p==t).float().mean()) for p,t in zip(got.split(LEVELS,1),codes.split(LEVELS,1))]
    return {'teacher_accuracy_per_level':levels(teacher),'free_accuracy_per_level':levels(free),
        'free_exact_concept_windows':int((free==codes).all(-1).sum()),'windows':len(prefix),
        'free_nonpad_token_accuracy':float(((decoded==target)&(target!=PAD)).sum()/(target!=PAD).sum()),
        'oracle_nonpad_token_accuracy':float(((oracle==target)&(target!=PAD)).sum()/(target!=PAD).sum()),
        'free_matches_oracle_windows':int((decoded==oracle).all(-1).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generator',required=True)
    p.add_argument('--codec',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--upstream',default='results/ncm-upstream')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--untied-heads',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7129)
    admission([a.device],5)
    codec,ck=load_codec(a.codec,a.upstream,a.device)
    data=Corpus(a.data)
    gck=torch.load(a.generator,map_location='cpu',weights_only=True)
    if gck['identity']['codec_sha256']!=file_sha(a.codec) or ck['identity']['data_sha256']!=file_sha(Path(a.data)/'manifest.json'):
        raise ValueError('Parent identities differ')
    _,G=upstream(a.upstream)
    cfg=dict(gck['identity']['generator_config'])
    cfg['dropout']=0.
    if a.untied_heads:
        cfg['tie_heads_to_codebook']=False
    model=G(codec,OmegaConf.create(cfg)).to(a.device)
    if a.untied_heads:
        weights={k:v for k,v in gck['model'].items() if not k.startswith('lm_heads.')}
        missing,unexpected=model.load_state_dict(weights,strict=False)
        if unexpected or any(not n.startswith('lm_heads.') for n in missing):
            raise ValueError('Only output heads may be reinitialized')
    else:
        model.load_state_dict(gck['model'])
    prefix,target=data.batch(random.Random(7129),8,.5,'train')
    prefix,target=prefix.to(a.device),target.to(a.device)
    with torch.no_grad():
        clean=codec.encode(target)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    history=[{'step':0,**measure(model,codec,prefix,clean,target)}]
    started=time.monotonic()
    for step in range(1,a.steps+1):
        headroom([a.device])
        model.train()
        opt.zero_grad(set_to_none=True)
        total=0.
        for i in range(0,8,2):
            pp,cc=prefix[i:i+2],clean[i:i+2]
            logits=model(generator_inputs(model,codec,pp,cc),prefix_pad_mask=pp==PAD)
            loss=torch.stack([F.cross_entropy(l.flatten(0,1),t.flatten())
                for l,t in zip(logits.split(LEVELS,1),cc.split(LEVELS,1))]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss/4).backward()
            total+=float(loss.detach())/4
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if step%100==0 or step==a.steps:
            item={'step':step,'loss':total,'seconds':time.monotonic()-started,**measure(model,codec,prefix,clean,target)}
            history.append(item)
            print(item,flush=True)
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,{'scope':'memorization of eight training windows; NOT held-out quality',
        'generator_sha256':file_sha(a.generator),'codec_sha256':file_sha(a.codec),
        'head_mode':'fresh untied classifiers' if a.untied_heads else 'original tied distance heads',
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'audit_sha256':file_sha(__file__),
        'data_sha256':ck['identity']['data_sha256'],'seed':7129,'steps':a.steps,
        'precision':'FP32','dropout':0,'conditioning_corruption':0,'history':history,'quality_ready':False})


if __name__=='__main__':
    main()

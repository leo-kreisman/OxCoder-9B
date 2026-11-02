"""Head-only learning control on the actual frozen OxCoder NCM adapter.

Only eight training chunks are memorized. Hidden features are computed with
gold earlier levels; free generation is evaluated separately without gold codes.
Neither result establishes held-out coding ability or an 8 GB deployment.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import time
import torch
from torch.nn import functional as F

from .backbone import digest
from .codec import encode,PAD
from .generate_candidate import load_candidate
from .geometry import IndependentConceptHeads
from .train import admission,headroom,write_json


def select_examples(prepared,count,max_prompt,seed):
    buckets={}
    for row,c in prepared:
        if row['split']!='train' or len(c['ids'])>max_prompt:continue
        if len(row['target'].encode())<32:continue
        key=(row['source'],bool(row.get('distilled',False)))
        buckets.setdefault(key,[]).append((row,c))
    rng=random.Random(seed)
    for rows in buckets.values():rng.shuffle(rows)
    selected=[];seen=set()
    while len(selected)<count:
        before=len(selected)
        for key in sorted(buckets):
            rows=buckets[key]
            while rows:
                row,c=rows.pop()
                if c['response_id'] not in seen:
                    seen.add(c['response_id']);selected.append((row,c));break
            if len(selected)==count:break
        if len(selected)==before:raise ValueError('Insufficient distinct training responses')
    return selected


@torch.no_grad()
def free_evaluation(model,codec,examples,codes,targets):
    records=[]
    for index,((row,c),wanted) in enumerate(zip(examples,targets)):
        got,predicted,_=model.generate_chunk(codec,torch.tensor([c['ids']]))
        mask=wanted!=PAD
        records.append({'id':row['id'],
            'exact_concepts':all(torch.equal(p,codes[i][index:index+1]) for i,p in enumerate(predicted)),
            'code_accuracy':[float((p==codes[i][index:index+1]).float().mean()) for i,p in enumerate(predicted)],
            'exact_nonpad_reconstruction':bool(((got[0]==wanted)|~mask).all()),
            'nonpad_accuracy':float((got[0][mask]==wanted[mask]).float().mean())})
    return {'windows':len(records),'exact_concept_windows':sum(r['exact_concepts'] for r in records),
        'exact_nonpad_windows':sum(r['exact_nonpad_reconstruction'] for r in records),
        'mean_nonpad_accuracy':sum(r['nonpad_accuracy'] for r in records)/len(records),'records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','codec','checkpoint','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--examples',type=int,default=8)
    p.add_argument('--max-prompt',type=int,default=2048)
    p.add_argument('--seed',type=int,default=8229)
    a=p.parse_args()
    if min(a.steps,a.examples,a.max_prompt)<1:raise ValueError('Positive bounds required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    implementation=out/'implementation';implementation.mkdir()
    for name in ('audit_oxcoder_heads.py','backbone.py','geometry.py','generate_candidate.py'):
        shutil.copyfile(Path(__file__).parent/name,implementation/name)
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    devices=['cuda:0','cuda:1'];limits=admission(devices)
    path=Path(a.data)/'prepared.pt'
    manifest=json.loads((Path(a.data)/'manifest.json').read_text())
    if digest(path)!=manifest['prepared_sha256']:raise ValueError('Prepared checksum mismatch')
    prepared=torch.load(path,map_location='cpu',weights_only=True)
    examples=select_examples(prepared['prepared'],a.examples,a.max_prompt,a.seed)
    model,codec,identity,checkpoint_sha=load_candidate(a.source,a.codec,a.checkpoint,devices)
    if identity['prepared_sha256']!=manifest['prepared_sha256']:raise ValueError('Initialization dataset differs')
    if model.cached_prefix:raise ValueError('Use the uncached numerically checked path')
    model.eval();codec.eval()
    features=[[] for _ in codec.books];codes=[[] for _ in codec.books];targets=[]
    started=time.monotonic()
    with torch.no_grad():
        for row,c in examples:
            headroom(devices)
            target=torch.tensor([encode(row['target'],codec.config.length)],device=devices[-1])
            wanted,inputs=codec.targets(target);targets.append(target[0])
            prompt=torch.tensor([c['ids']])
            for level,inp in enumerate(inputs):
                h=model.hidden_features(inp,prompt,level).detach()
                if not torch.isfinite(h).all():raise RuntimeError('Nonfinite hidden features')
                features[level].append(h);codes[level].append(wanted[level])
            print(json.dumps({'phase':'features','completed':len(targets),'total':len(examples)}),flush=True)
    features=[torch.cat(x) for x in features];codes=[torch.cat(x) for x in codes]
    targets=torch.stack(targets)
    oracle=codec.reconstruct_codes(codes).argmax(-1)
    mask=targets!=PAD
    original=copy.deepcopy(model.head.state_dict())
    report={'checkpoint_sha256':checkpoint_sha,'codec_sha256':digest(a.codec),
        'prepared_sha256':manifest['prepared_sha256'],'source_sha256':identity['source_sha256'],
        'implementation_sha256':{k:digest(Path(__file__).parent/k) for k in
             ('audit_oxcoder_heads.py','backbone.py','geometry.py','generate_candidate.py')},
        'seed':a.seed,'steps':a.steps,'admission':limits,
        'selection':[{'id':r['id'],'source':r['source'],'distilled':bool(r.get('distilled',False)),
                      'split':r['split'],'prompt_tokens':len(c['ids'])} for r,c in examples],
        'oracle_nonpad_accuracy':float((oracle[mask]==targets[mask]).float().mean()),
        'feature_collection_seconds':time.monotonic()-started,
        'head_initialization':'geometry retains trained weights; independent classifiers start random',
        'training':'full fixed feature batch, mean level CE, AdamW 0.001, weight decay 0.01',
        'scope':'training-set head memorization; backbone and adapters fixed; no held-out quality claim',
        'complete':False,'quality_ready':False,'arms':{}}
    report['before']=free_evaluation(model,codec,examples,codes,targets)
    write_json(out/'report.json',report)
    for kind in ('geometry','untied'):
        torch.manual_seed(a.seed)
        if kind=='geometry':model.head.load_state_dict(original)
        else:
            model.head=IndependentConceptHeads(model.backbone.width,codec.books,input_norm=True).to(devices[-1])
        model.head_kind=kind
        params=[v for v in model.head.parameters() if v.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.01,foreach=False)
        arm={'trainable_head_parameters':sum(v.numel() for v in params),'history':[]}
        report['arms'][kind]=arm
        begin=time.monotonic()
        for step in range(a.steps):
            headroom(devices);optimizer.zero_grad(set_to_none=True)
            loss_value=0.;accuracy=[]
            for level,h in enumerate(features):
                logits=model.head(h,level)
                loss=F.cross_entropy(logits.flatten(0,1),codes[level].flatten())/len(codes)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite classifier loss')
                loss.backward();loss_value+=float(loss.detach())
                accuracy.append(float((logits.argmax(-1)==codes[level]).float().mean()))
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%50==0 or step+1==a.steps:
                record={'step':step+1,'loss':loss_value,'teacher_forced_code_accuracy':accuracy,
                        'seconds':time.monotonic()-begin}
                arm['history'].append(record);write_json(out/'report.json',report)
                print(json.dumps({'head':kind,**record}),flush=True)
        del optimizer
        torch.cuda.empty_cache()
        arm['free_generation']=free_evaluation(model,codec,examples,codes,targets)
        write_json(out/'report.json',report)
    report['complete']=True
    report['peak_allocated_bytes']={d:torch.cuda.max_memory_allocated(d) for d in devices}
    write_json(out/'report.json',report)
    print(json.dumps({'complete':True,'quality_ready':False,
        'free_exact_windows':{k:v['free_generation']['exact_concept_windows'] for k,v in report['arms'].items()}}),flush=True)


if __name__=='__main__':main()

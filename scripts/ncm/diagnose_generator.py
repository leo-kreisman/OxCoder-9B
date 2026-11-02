"""Separate concept prediction errors from codec errors; test one-example fitting.

Oracle runs and memorization are diagnostic controls, never coding scores.
The baseline checkpoint is read-only. No autoregressive fallback is used.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from .codec import encode, decode, resize, PAD
from .conditioning import index_responses, condition_chunk
from .generate_candidate import load_candidate
from .train import admission, load_data, write_json, save


@torch.no_grad()
def oracle_prefix(model, codec, prompt, target_codes, count):
    if not 0 <= count <= len(codec.books):
        raise ValueError('Invalid oracle prefix length')
    z=torch.zeros(1,codec.config.length,codec.config.width,device=codec.position.device)
    chosen=[]
    for i,(size,book) in enumerate(zip(codec.config.levels,codec.books)):
        if i<count:
            ids=target_codes[i]
        else:
            ids=model(codec.resample(z,size),prompt,i).argmax(-1).to(z.device)
        chosen.append(ids)
        z+=resize(book(ids),codec.config.length)
    return codec.decode_latents(z).argmax(-1),chosen


def compare(output,target,text):
    mask=target!=PAD
    try:
        generated=decode(output[0].tolist());error=None
    except (ValueError,UnicodeError) as e:
        generated=None;error=str(e)
    return {'nonpad_byte_accuracy':float(((output==target)&mask).sum()/mask.sum()),
            'generated_text':generated,'decode_error':error,'exact_chunk':generated==text,
            'byte_ids':output[0].tolist()}


def fit_head(head, features, targets, steps=1000, lr=.001):
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=head(features)
        loss=F.cross_entropy(logits,targets)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite probe loss')
        loss.backward();optimizer.step()
        if step%25==0 or step==steps-1:
            with torch.no_grad():accuracy=float((head(features).argmax(-1)==targets).float().mean())
            history.append({'step':step+1,'loss':float(loss.detach()),'code_accuracy':accuracy})
            if accuracy==1.:break
    return history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','codec','checkpoint','data','corpus','teacher','baseline-run','probe-id','output']:
        p.add_argument('--'+k,required=True)
    p.add_argument('--devices',default='cuda:0,cuda:1')
    p.add_argument('--probe-steps',type=int,default=1000)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(73)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    devices=a.devices.split(',');admission(devices)
    rows,manifest=load_data(a.data);corpus,cm=load_data(a.corpus);teacher,tm=load_data(a.teacher)
    if cm['examples_sha256']!=manifest['complete_corpus_sha256'] or tm['examples_sha256']!=manifest['teacher_sha256']:
        raise ValueError('Diagnostic corpus differs from source mixture')
    by_id={r['id']:r for r in rows};responses=index_responses(corpus,teacher)
    probe=by_id[a.probe_id]
    if probe['split']!='train':raise ValueError('Memorization control requires a training-only example')
    previous=json.loads((Path(a.baseline_run)/'generation-evaluations.json').read_text())[-1]
    selected=[probe];sources=set()
    for entry in previous['examples']:
        r=by_id[entry['id']]
        if r['source'] not in sources:
            selected.append(r);sources.add(r['source'])
    model,codec,identity,sha=load_candidate(a.source,a.codec,a.checkpoint,devices)
    if identity['data_sha256']!=manifest['examples_sha256']:raise ValueError('Checkpoint uses different data')
    tok=AutoTokenizer.from_pretrained('results/tokenizer',local_files_only=True,trust_remote_code=False)
    report={'checkpoint_sha256':sha,'quality_ready':False,'scope':'oracle-prefix and single-training-example controls, not held-out coding qualification','examples':[]}
    probe_cache=None
    for row in selected:
        c=condition_chunk(row,responses,tok,codec.config.length,identity['prompt_tokens'])
        if c is None:raise ValueError('Selected diagnostic exceeds original context budget')
        prompt=torch.tensor([c['ids']],dtype=torch.long)
        target=torch.tensor([encode(row['target'],codec.config.length)],device=devices[-1])
        with torch.no_grad():codes,inputs=codec.targets(target)
        r={'id':row['id'],'source':row['source'],'split':row['split'],'target':row['target'],'oracle_ladder':[]}
        for count in range(len(codes)+1):
            begin=time.monotonic();got,_=oracle_prefix(model,codec,prompt,codes,count)
            result={'oracle_levels':count,'seconds':time.monotonic()-begin,**compare(got,target,row['target'])}
            r['oracle_ladder'].append(result)
            print(json.dumps({'id':row['id'],'oracle_levels':count,'accuracy':result['nonpad_byte_accuracy'],'exact':result['exact_chunk']}),flush=True)
        report['examples'].append(r);write_json(out/'diagnosis.json',report)
        if row['id']==a.probe_id:
            if not r['oracle_ladder'][-1]['exact_chunk']:raise ValueError('Probe does not reconstruct exactly with known codes')
            probe_cache=(prompt,target,codes,inputs)
    prompt,target,codes,inputs=probe_cache
    features=[]
    hook=model.head.register_forward_pre_hook(lambda module,args:features.append(args[0].detach().flatten(0,1)))
    try:
        with torch.no_grad():
            for i in range(len(codes)):model(inputs[i],prompt,i)
    finally:hook.remove()
    features=torch.cat(features).detach();labels=torch.cat([x.flatten() for x in codes]).to(features.device)
    original={k:v.detach().clone() for k,v in model.head.state_dict().items()}
    model.requires_grad_(False);model.head.requires_grad_(True)
    history=fit_head(model.head,features,labels,steps=a.probe_steps)
    output,_=oracle_prefix(model,codec,prompt,codes,0)
    with torch.no_grad():teacher_accuracy=float((model.head(features).argmax(-1)==labels).float().mean())
    report['memorization_control']={'training_id':a.probe_id,'history':history,
        'teacher_forced_code_accuracy':teacher_accuracy,'free_generation':compare(output,target,probe['target']),
        'quality_ready':False,'scope':'one training chunk; does not establish generalization or complete-response termination'}
    save(out/'probe-head.pt',{'head':model.head.state_dict(),'quality_ready':False,
                            'scope':'memorization-only diagnostic; not a coding model','training_id':a.probe_id})
    model.head.load_state_dict(original)
    write_json(out/'diagnosis.json',report)
    print(json.dumps(report['memorization_control']),flush=True)


if __name__=='__main__':main()

"""Adapt only an official codec's decoder on frozen codes from both corpora + teacher.

Encoder, codebooks and residual transforms never change, making cached discrete
targets valid across optimizer arms. Validation/test examples never train.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .official_codec import OfficialCodec, PROBES, evaluate
from .anvil import make_optimizers
from .backbone import digest
from .train import admission, headroom, load_data, save, write_json, cpu_tree, rng_state, set_rng


def select_windows(codec,rows,limit):
    windows=[];provenance=[]
    for r in rows:
        ids=codec.encode_text(r['target']);pieces=codec.windows(ids)
        indexes=list(range(len(pieces)))
        if len(indexes)>limit:
            indexes=sorted({round(i*(len(pieces)-1)/(limit-1)) for i in range(limit)}) if limit>1 else [0]
        for i in indexes:
            windows.append(pieces[i]);provenance.append({'id':r['id'],'source':r['source'],'window':i,
                'distilled':r.get('distilled',False),'valid_tokens':min(codec.config.length,len(ids)-i*codec.config.length)})
    return torch.stack(windows),provenance


@torch.no_grad()
def build_cache(a):
    out=Path(a.cache)
    if out.exists():raise ValueError('Cache already exists')
    torch.set_num_threads(2);admission([a.device],5)
    codec=OfficialCodec(a.upstream,a.artifacts,a.device)
    rows,manifest=load_data(a.data);teacher,tm=load_data(a.teacher)
    if any(r['split']!='train' for r in teacher):raise ValueError('Teacher cache must be training-only')
    groups={r['group']:r['split'] for r in rows}
    if any(groups.get(r['group'])!='train' for r in teacher):raise ValueError('Teacher provenance mismatch')
    training=[r for r in rows if r['split']=='train']+teacher
    ids,provenance=select_windows(codec,training,a.windows_per_response)
    codes=[]
    for start in range(0,len(ids),8):
        codes.append(codec.model.encode(ids[start:start+8].to(a.device)).cpu())
    identity={'codec':codec.identity,'corpus_sha256':manifest['examples_sha256'],
              'teacher_sha256':tm['examples_sha256'],'windows_per_response':a.windows_per_response}
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,{'identity':identity,'ids':ids,'codes':torch.cat(codes),'provenance':provenance})
    print(json.dumps({'windows':len(ids),'tokens':sum(r['valid_tokens'] for r in provenance),
        'source_counts':{s:sum(r['source']==s for r in provenance) for s in {r['source'] for r in provenance}},
        'teacher_windows':sum(r['distilled'] for r in provenance),'sha256':digest(out)},indent=2),flush=True)


def documents(rows,per_source):
    chosen=[]
    for source in sorted({r['source'] for r in rows}):
        selected=sorted((r for r in rows if r['source']==source and r['split']=='validation'),key=lambda r:r['id'])[:per_source]
        chosen.extend((r['id'],r['target']) for r in selected)
    return chosen


def train(a):
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    limits=admission([a.device],7)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    codec=OfficialCodec(a.upstream,a.artifacts,a.device)
    cache=torch.load(a.cache,map_location='cpu',weights_only=True)
    if cache['identity']['codec']!=codec.identity:raise ValueError('Cache/codec identity mismatch')
    rows,manifest=load_data(a.data)
    if manifest['examples_sha256']!=cache['identity']['corpus_sha256']:raise ValueError('Corpus differs')
    decoder=codec.model.decoder;decoder.requires_grad_(True)
    trained=codec.model if a.joint else decoder
    key='model' if a.joint else 'decoder'
    best_name='best-codec.pt' if a.joint else 'best-decoder.pt'
    if a.joint:
        codec.model.requires_grad_(True)
        # Upstream codebooks are updated by EMA, not by the optimizer.
        for book in codec.model.vq.embeddings:book.requires_grad_(False)
        codec.model.vq.fine_drop_prob=0.
        codec.model.vq.fine_drop_prob_start=None
    optimizers=make_optimizers(trained,a.optimizer,a.lr,a.anvil_lr,muon_lr=a.muon_lr)
    ident={'cache_sha256':digest(a.cache),'codec':codec.identity,'seed':a.seed,'batch':a.batch,
           'optimizer':a.optimizer,'lr':a.lr,'anvil_lr':a.anvil_lr,'dtype':'FP32 parameters, BF16 autocast forward',
           'train_scope':'decoder only; dropout disabled; frozen encoder/VQ; uniform sampled training windows'}
    if a.joint:ident['train_scope']='joint codec; fresh codes every batch; upstream EMA books; dropout/fine-drop disabled; CE + VQ loss'
    if a.optimizer=='muon':
        ident['muon']={'torch':str(torch.__version__),'lr':a.lr if a.muon_lr is None else a.muon_lr,
                       'ns_steps':5,'momentum':.95,'adjust_lr_fn':'match_rms_adamw'}
    begin=0;history=[];best=-1.;elapsed=0.
    if a.resume:
        ck=torch.load(out/'latest.pt',weights_only=True,map_location='cpu')
        if ck['identity']!=ident:raise ValueError('Resume configuration mismatch')
        trained.load_state_dict(ck[key])
        for o,state in zip(optimizers,ck['optimizers']):o.load_state_dict(state)
        begin=ck['step'];history=ck['history'];best=ck['best'];elapsed=ck['training_seconds'];set_rng(ck['rng'])
    if a.steps<=begin:raise ValueError('No new steps requested')
    # Separate data RNG makes batch order identical across optimizer choices.
    data_rng=torch.Generator().manual_seed(a.seed)
    if a.resume:data_rng.set_state(ck['data_rng'])
    quick=documents(rows,4);write_json(out/'identity.json',ident);write_json(out/'admission.json',limits)
    for step in range(begin,a.steps):
        headroom([a.device]);codec.model.eval()
        if a.joint:codec.model.vq.train()
        idx=torch.randint(len(cache['ids']),(a.batch,),generator=data_rng)
        ids=cache['ids'][idx].to(a.device);codes=cache['codes'][idx].to(a.device)
        for o in optimizers:o.zero_grad(set_to_none=True)
        torch.cuda.synchronize(a.device);started=time.monotonic()
        if not a.joint:
            with torch.no_grad():z=codec.model.vq.idx_to_z_q(codes)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if a.joint:logits,vq_loss,*_=codec.model(ids)
            else:logits=decoder(z);vq_loss=0.
            loss=F.cross_entropy(logits.flatten(0,1).float(),ids.flatten(),ignore_index=codec.tokenizer.pad_token_id)
            loss=loss+vq_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reconstruction loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(trained.parameters(),1.,error_if_nonfinite=True)
        for o in optimizers:o.step()
        torch.cuda.synchronize(a.device);elapsed+=time.monotonic()-started
        if (step+1)%a.eval_every==0 or step+1==a.steps:
            codec.model.eval()
            metric=evaluate(codec,quick);score=metric['token_accuracy']
            item={'step':step+1,'training_seconds':elapsed,'loss':float(loss.detach()),
                  **{k:v for k,v in metric.items() if k!='rows'}}
            history.append(item);print(json.dumps(item),flush=True)
            checkpoint={'identity':ident,'step':step+1,key:cpu_tree(trained.state_dict()),
                'optimizers':[cpu_tree(o.state_dict()) for o in optimizers],'history':history,
                'rng':rng_state(),'data_rng':data_rng.get_state(),'best':max(best,score),'training_seconds':elapsed}
            save(out/'latest.pt',checkpoint)
            if score>best:
                best=score;save(out/best_name,{'identity':ident,'step':step+1,key:checkpoint[key]})
            write_json(out/'history.json',history)
    chosen=torch.load(out/best_name,map_location='cpu',weights_only=True);trained.load_state_dict(chosen[key]);codec.model.eval()
    result={'identity':ident,'selected_step':chosen['step'],'training_seconds':elapsed,
        'checkpoint_sha256':digest(out/best_name),'checkpoint_kind':key,
        'probes':evaluate(codec,[(f'probe-{i}',s) for i,s in enumerate(PROBES)]),
        'validation':evaluate(codec,documents(rows,16)),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(a.device),'quality_ready':False}
    result['codec_gate_passed']=result['probes']['passed'] and result['validation']['passed']
    write_json(out/'screen.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('identity','validation','probes')},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['cache','train'])
    p.add_argument('--upstream',default='results/ncm-upstream');p.add_argument('--artifacts',required=True)
    p.add_argument('--data',required=True);p.add_argument('--teacher');p.add_argument('--cache',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--windows-per-response',type=int,default=8)
    p.add_argument('--output');p.add_argument('--optimizer',choices=['adamw','anvil','muon'],default='adamw')
    p.add_argument('--muon-lr',type=float,default=None)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--lr',type=float,default=.0001);p.add_argument('--anvil-lr',type=float,default=.002)
    p.add_argument('--resume',action='store_true');p.add_argument('--joint',action='store_true');a=p.parse_args()
    if a.mode=='cache':build_cache(a)
    else:train(a)


if __name__=='__main__':main()

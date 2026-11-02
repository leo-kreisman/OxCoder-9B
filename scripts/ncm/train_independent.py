"""Experimental OxCoder adaptation with independent per-level classifiers.

The frozen codec has not passed broad reconstruction. This is a diagnostic
training stage, never an automatic release or a claim of usable conversion.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import shutil
import time
import torch
from torch.nn import functional as F

from .anvil import OptimizerBundle
from .backbone import digest
from .codec import encode,resize
from .coding_eval import evaluate_model
from .generate_candidate import load_candidate
from .geometry import IndependentConceptHeads
from .train import (admission,headroom,write_json,save,cpu_tree,rng_state,set_rng,
                    exact_training_curriculum,evaluate_candidate)
from .train_geometry import ResponseSampler
from .prepare_creator_expansion import frozen_evaluation_hash


@torch.no_grad()
def corrupted_inputs(codec,codes,maximum):
    """Uniform replacement of earlier code IDs; preserve all clean labels."""
    if not 0<=maximum<=1:raise ValueError('Corruption must be in [0,1]')
    z=torch.zeros(codes[0].shape[0],codec.config.length,codec.config.width,device=codes[0].device)
    inputs=[]
    for level,(ids,size,book) in enumerate(zip(codes,codec.config.levels,codec.books)):
        inputs.append(codec.resample(z,size))
        probability=maximum*(1-level/max(1,len(codes)-1))
        if probability:
            chosen=torch.where(torch.rand(ids.shape,device=ids.device)<probability,
                               torch.randint(book.num_embeddings,ids.shape,device=ids.device),ids)
        else:chosen=ids
        z=z+resize(book(chosen),codec.config.length)
    return inputs


def train(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    with (out/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a,out)


def validate_continuation(previous,current):
    """Resource-only continuation cannot silently change learning settings."""
    allowed={'implementation_sha256','allocator_cap_gib','continuation'}
    left={k:v for k,v in previous.items() if k not in allowed}
    right={k:v for k,v in current.items() if k not in allowed}
    if left!=right:raise ValueError('Continuation changes the learning recipe or data')


def validate_expansion(manifest,data,parent):
    if manifest.get('base_prepared_sha256')!=parent['prepared_sha256']:
        raise ValueError('Expansion does not descend from the initialization dataset')
    if manifest.get('evaluation_changed') is not False:
        raise ValueError('Expansion must preserve evaluation')
    if frozen_evaluation_hash(data['prepared'])!=manifest['frozen_evaluation_sha256']:
        raise ValueError('Expanded evaluation fingerprint differs')
    count=manifest['base_rows']
    if not 0<count<len(data['prepared']):raise ValueError('Expansion has no added rows')
    groups={}
    for row,c in data['prepared']:
        if groups.setdefault(row['group'],row['split'])!=row['split']:
            raise ValueError('Expanded conversation split conflict')
    if any(r['split']!='train' for r,c in data['prepared'][count:]):
        raise ValueError('Expansion adds evaluation examples')


def run(a,out):
    torch.set_num_threads(2);torch.manual_seed(a.seed);random.seed(a.seed)
    devices=['cuda:0','cuda:1'];limits=admission(devices,cap_gib=a.allocator_cap_gib)
    write_json(out/'admission.json',limits)
    folder=Path(a.data);manifest=json.loads((folder/'manifest.json').read_text())
    if digest(folder/'prepared.pt')!=manifest['prepared_sha256']:raise ValueError('Prepared data checksum differs')
    data=torch.load(folder/'prepared.pt',map_location='cpu',weights_only=True)
    model,codec,parent,parent_sha=load_candidate(a.source,a.codec,a.init,devices)
    if a.data_expansion:validate_expansion(manifest,data,parent)
    elif parent['prepared_sha256']!=manifest['prepared_sha256']:raise ValueError('Prepared lineage differs')
    if model.cached_prefix:raise ValueError('Prefix cache has not passed the numerical gate')
    if a.retain_heads:
        if model.head_kind!='untied' or not model.head.input_norm:raise ValueError('Retained heads must be independent and normalized')
    else:model.head=IndependentConceptHeads(model.backbone.width,codec.books,input_norm=True).to(devices[-1])
    model.head_kind='untied'
    prepared,curriculum=exact_training_curriculum(codec,data['prepared'],devices[-1])
    write_json(out/'curriculum.json',curriculum)
    sampler=ResponseSampler(prepared);write_json(out/'sampling.json',sampler.report)
    trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    head=[v for n,v in trainable if n.startswith(('head.','stop_head.'))]
    matrices=[v for n,v in trainable if (n.startswith('backbone.adapters.') and v.ndim==2) or n=='project.weight']
    used={id(v) for v in head+matrices}
    other=[v for n,v in trainable if id(v) not in used]
    optimizer=OptimizerBundle([
        torch.optim.AdamW(head,lr=a.head_lr,weight_decay=.01,foreach=False),
        torch.optim.AdamW(other,lr=a.body_lr,weight_decay=.01,foreach=False),
        torch.optim.Muon(matrices,lr=a.body_lr,weight_decay=.01,adjust_lr_fn='match_rms_adamw')])
    implementation={k:digest(Path(__file__).parent/k) for k in
        ('train_independent.py','backbone.py','geometry.py','train.py','train_geometry.py','generate_candidate.py','codec.py','prepare_creator_expansion.py')}
    identity={**parent,'head_kind':'untied','head_input_norm':True,'nearest_code':False,
        'calibrated_head':False,'cached_prefix':False,'training_recipe':'independent-classifiers-v16',
        'implementation_sha256':implementation,'seed':a.seed,'planned_steps':a.steps,
        'initialization':{'checkpoint_sha256':parent_sha,'scope':'retain body/adapters and independent classifiers; reset optimizer' if a.retain_heads else 'retain body/adapters; fresh independent classifiers'},
        'head_lr':a.head_lr,'body_lr':a.body_lr,'head_warmup':a.head_warmup,
        'loss':'mean level code CE + 0.1 end BCE; no continuous residual loss',
        'conditioning_corruption':{'maximum':a.corruption,'schedule':'linear from maximum at first level to zero at last; clean warmup; clean labels'},
        'optimizer':{'name':'Muon hidden matrices + AdamW classifiers/interfaces','head_lr':a.head_lr,'body_lr':a.body_lr},
        'trainable_parameters':sum(v.numel() for n,v in trainable),'quality_ready':False,
        'allocator_cap_gib':a.allocator_cap_gib,
        'admission':'exactly reconstructed train chunks only; all selected validation errors retained; broad codec gate failed'}
    identity.pop('residual_byte_loss_weight',None)
    identity.pop('continuation',None)
    if a.data_expansion:
        identity['prepared_sha256']=manifest['prepared_sha256']
        identity['dataset_expansion']={k:manifest[k] for k in
            ('base_prepared_sha256','addition_manifest_sha256','frozen_evaluation_sha256','counts')}
        identity['data_sha256_scope']='inherited base examples; full expanded inventory identified by prepared_sha256 and dataset_expansion'
    history=[];begin=0
    if a.continue_checkpoint:
        continuation_sha=digest(a.continue_checkpoint)
        previous=torch.load(a.continue_checkpoint,map_location='cpu',weights_only=True)
        if digest(a.continue_checkpoint)!=continuation_sha:raise ValueError('Continuation checkpoint changed during load')
        validate_continuation(previous['identity'],identity)
        if set(previous['model'])!=set(model.trainable_state()):raise ValueError('Continuation parameter inventory differs')
        with torch.no_grad():
            for n,v in trainable:v.copy_(previous['model'][n].to(v))
        optimizer.load_state_dict(previous['optimizer']);set_rng(previous['rng'])
        begin=previous['step'];history=previous['history']
        identity['continuation']={'checkpoint_sha256':continuation_sha,'source_step':begin,
            'scope':'model, optimizer and RNG retained; separately recorded resource/implementation continuation'}
        del previous
    if a.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        if identity!=ck['identity']:raise ValueError('Resume recipe or implementation differs')
        if set(ck['model'])!=set(model.trainable_state()):raise ValueError('Resume parameter inventory differs')
        with torch.no_grad():
            for n,v in trainable:v.copy_(ck['model'][n].to(v))
        optimizer.load_state_dict(ck['optimizer']);set_rng(ck['rng']);begin=ck['step'];history=ck['history']
        del ck
    else:
        directory=out/'implementation';directory.mkdir()
        for name in implementation:shutil.copyfile(Path(__file__).parent/name,directory/name)
    if begin>=a.steps:raise ValueError('No remaining updates')
    write_json(out/'identity.json',identity)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('results/tokenizer',local_files_only=True,trust_remote_code=False)
    stop=min(a.steps,a.stop_after) if a.stop_after else a.steps
    for step in range(begin,stop):
        if shutil.disk_usage(out).free<2<<30:raise RuntimeError('Keep 2 GiB free for checkpoints')
        headroom(devices);model.train();optimizer.zero_grad()
        started=time.monotonic();row,c=sampler.sample(random)
        target=torch.tensor([encode(row['target'],codec.config.length)],device=devices[-1])
        with torch.no_grad():codes,clean=codec.targets(target)
        warmup=step<a.head_warmup
        inputs=clean if warmup else corrupted_inputs(codec,codes,a.corruption)
        prompt=torch.tensor([c['ids']]);values=[];accuracies=[]
        for level,inp in enumerate(inputs):
            if warmup:
                with torch.no_grad():h=model.hidden_features(inp,prompt,level)
                logits=model.head(h,level);end=model.stop_head(h.mean(1)).squeeze(-1)
            else:logits,end,_=model(inp,prompt,level,return_aux=True)
            loss=F.cross_entropy(logits.flatten(0,1),codes[level].flatten())/len(codes)
            if level==len(codes)-1:
                loss=loss+.1*F.binary_cross_entropy_with_logits(end,torch.full_like(end,float(c['is_final_chunk'])))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();values.append(float(loss.detach()))
            accuracies.append(float((logits.argmax(-1)==codes[level]).float().mean()))
        if not warmup:
            for adapter in model.backbone.adapters:
                g=adapter.up.weight.grad
                if g is None or not torch.isfinite(g).all() or g.norm()==0:raise RuntimeError('Missing/nonfinite/zero adapter gradient')
        for d in devices:
            torch.nn.utils.clip_grad_norm_([v for n,v in trainable if str(v.device)==d],1.,error_if_nonfinite=True,foreach=False)
        optimizer.step()
        record={'step':step+1,'loss':sum(values),'code_accuracy':accuracies,'warmup':warmup,
            'source':row['source'],'distilled':bool(row.get('distilled',False)),'training_example':row['id'],
            'expansion':row.get('expansion'),
            'seconds':time.monotonic()-started,'prompt_tokens':len(c['ids']),
            'peak_allocated_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices}}
        history.append(record);print(json.dumps(record),flush=True)
        evaluate=(step+1)%a.eval_every==0 or step+1==a.steps
        if (step+1)%a.save_every==0 or step+1==stop or evaluate:
            save(out/'latest.pt',{'identity':identity,'model':model.trainable_state(),'optimizer':cpu_tree(optimizer.state_dict()),
                 'rng':rng_state(),'step':step+1,'history':history})
        write_json(out/'progress.json',{'step':step+1,'training_complete':step+1==a.steps,'history':history,'quality_ready':False})
        if evaluate:
            model.eval();write_json(out/f'chunks-{step+1}.json',evaluate_candidate(model,codec,prepared,devices[-1]))
            evaluate_model(model,codec,tokenizer,identity['prompt_tokens'],digest(out/'latest.pt'),out/f'coding-{step+1}.json')
    print(json.dumps({'stopped_at':stop,'training_complete':stop==a.steps,'quality_ready':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','codec','init','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--stop-after',type=int)
    p.add_argument('--head-warmup',type=int,default=25);p.add_argument('--head-lr',type=float,default=.0003)
    p.add_argument('--body-lr',type=float,default=.00003);p.add_argument('--corruption',type=float,default=.5)
    p.add_argument('--save-every',type=int,default=10);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=9161);p.add_argument('--resume',action='store_true')
    p.add_argument('--allocator-cap-gib',type=float,default=10.)
    p.add_argument('--continue-checkpoint',help='Immutable state for a new resource-only continuation directory')
    p.add_argument('--retain-heads',action='store_true',help='New weights-only stage retaining existing independent heads')
    p.add_argument('--data-expansion',action='store_true',help='Explicit new dataset stage with validated base lineage and frozen evaluation')
    a=p.parse_args()
    if min(a.steps,a.eval_every,a.save_every)<1 or a.head_warmup<0 or not 0<=a.corruption<=1:
        p.error('Invalid schedule')
    if not 0<a.allocator_cap_gib<=12:p.error('Allocator cap must be positive and at most 12 GiB')
    if a.data_expansion and a.continue_checkpoint:p.error('Data expansion starts a new optimizer stage, not resource-only continuation')
    train(a)

"""Experimental residual-geometry NCM adaptation; no autoregressive fallback.

Training uses only train-split targets, including the existing response-level
teacher distillation. The optional audited cache detaches prefix states and
invalidates them each update; --uncached computes full-context gradients.
The broad codec gate is still failed: this stage uses exact training targets and
retains every selected validation example, including reconstruction failures.
"""
import argparse
import collections
import json
import random
import shutil
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from .backbone import ConceptGenerator,digest
from .generate_candidate import load_candidate
from .codec import encode,resize
from .train import (admission,headroom,save,write_json,cpu_tree,rng_state,set_rng,
                    exact_training_curriculum,evaluate_candidate,decoded_byte_loss)
from .coding_eval import evaluate_model
from .anvil import OptimizerBundle


class ResponseSampler:
    """Favor coding responses without giving tiny ending strata equal mass."""
    def __init__(self,prepared):
        self.groups={}
        for row,c in prepared:
            if row['split']!='train':continue
            key=(bool(c['coding_response']),row['source'],bool(row.get('distilled',False)))
            self.groups.setdefault(key,{}).setdefault(c['response_id'],[]).append((row,c))
        if not self.groups:raise ValueError('No training responses')
        self.report=[{'coding':k[0],'source':k[1],'distilled':k[2],
                      'responses':len(v),'chunks':sum(map(len,v.values()))} for k,v in self.groups.items()]

    def sample(self,rng):
        coding=rng.random()<.8
        keys=[k for k in self.groups if k[0]==coding] or list(self.groups)
        key=rng.choices(keys,weights=[len(self.groups[k])**.5 for k in keys])[0]
        response=rng.choice(list(self.groups[key].values()))
        return rng.choice(response)


@torch.no_grad()
def count_codes(codec,training,device):
    counts=[torch.zeros(b.num_embeddings,device=device) for b in codec.books]
    for start in range(0,len(training),16):
        targets=torch.tensor([encode(r['target'],codec.config.length) for r,c in training[start:start+16]],device=device)
        codes,_=codec.targets(targets)
        for count,ids in zip(counts,codes):count.add_(torch.bincount(ids.flatten(),minlength=count.numel()))
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','codec','init','data','prefix-audit','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--head-warmup',type=int,default=100)
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--save-every',type=int,default=25)
    p.add_argument('--head-lr',type=float,default=.001);p.add_argument('--body-lr',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=7301);p.add_argument('--resume',action='store_true')
    p.add_argument('--head-input-norm',action='store_true',help='RMS-normalize final hidden features before residual projection')
    p.add_argument('--calibrate-head',action='store_true',help='Initialize heads to empirical code priors with training-weighted distance scales')
    p.add_argument('--batch-levels',action='store_true',help='Evaluate teacher-forced levels as independent padded batch members')
    p.add_argument('--batch-audit',help='Passing numerical comparison for --continue-checkpoint or --init')
    p.add_argument('--continue-checkpoint',help='Continue a compatible geometry stage in a new directory, retaining optimizer/RNG')
    p.add_argument('--nearest-code',action='store_true',help='Freeze class biases at zero; initialize residual projection offsets to training code means')
    p.add_argument('--uncached',action='store_true',help='Full-context training/inference; record but do not use a failed prefix-cache optimization')
    p.add_argument('--residual-byte-loss',type=float,default=0.,help='Final decoder CE on continuous predicted residual, before nearest-code selection')
    p.add_argument('--devices',default='cuda:0,cuda:1');a=p.parse_args()
    if min(a.steps,a.eval_every,a.save_every)<1 or a.head_warmup<0:raise ValueError('Invalid training schedule')
    if a.nearest_code and not a.calibrate_head:raise ValueError('Nearest-code stage requires explicit mean/scale calibration')
    torch.set_num_threads(2);torch.manual_seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    audit=json.loads(Path(a.prefix_audit).read_text())
    if not audit.get('complete') or not audit.get('full_precision_reductions') or (not audit.get('passed') and not a.uncached):
        raise ValueError('Passing full-precision prefix/target-gradient audit required')
    if a.uncached and a.batch_levels:raise ValueError('Level batching requires the audited cache path')
    devices=a.devices.split(',');limits=admission(devices)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume);write_json(out/'admission.json',limits)
    data_dir=Path(a.data);manifest=json.loads((data_dir/'manifest.json').read_text())
    if digest(data_dir/'prepared.pt')!=manifest['prepared_sha256']:raise ValueError('Prepared data checksum differs')
    data=torch.load(data_dir/'prepared.pt',map_location='cpu',weights_only=True)
    old,codec,identity,initial_sha=load_candidate(a.source,a.codec,a.init,devices)
    import fla
    if identity.get('kernel_backend',{}).get('version')!=fla.__version__:
        raise ValueError('Installed FLA differs from audited initialization')
    if audit['checkpoint_sha256']!=initial_sha:raise ValueError('Prefix audit applies to a different initialization')
    if data['data_manifest']['examples_sha256']!=identity['data_sha256']:raise ValueError('Dataset lineage differs')
    model=ConceptGenerator(old.backbone,codec.config,parallel_attention=True,termination=True,
                           concept_input_norm=.8,cached_prefix=not a.uncached,head_kind='geometry',codebooks=codec.books,
                           head_input_norm=a.head_input_norm,nearest_code=a.nearest_code)
    initial=old.trainable_state()
    with torch.no_grad():
        for name,v in model.named_parameters():
            if v.requires_grad and not name.startswith('head.'):
                if name not in initial or initial[name].shape!=v.shape:raise ValueError('Shared parameter differs: '+name)
                v.copy_(initial[name].to(v))
        model.head.log_temperature.fill_(2.3)
    del old,initial
    prepared,curriculum=exact_training_curriculum(codec,data['prepared'],devices[-1])
    write_json(out/'curriculum.json',curriculum)
    training=[x for x in prepared if x[0]['split']=='train']
    sampler=ResponseSampler(training);write_json(out/'sampling.json',sampler.report)
    counts=count_codes(codec,training,devices[-1])
    if a.calibrate_head:write_json(out/'head-calibration.json',model.head.calibrate(counts))
    else:model.head.initialize_priors(counts)
    trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    head=[v for n,v in trainable if n.startswith(('head.','stop_head.'))]
    matrices=[v for n,v in trainable if n.startswith('backbone.adapters.') and v.ndim==2 or n=='project.weight']
    matrix_ids={id(v) for v in matrices};head_ids={id(v) for v in head}
    other=[v for n,v in trainable if id(v) not in matrix_ids|head_ids]
    optimizer=OptimizerBundle([
        torch.optim.AdamW(head,lr=a.head_lr,weight_decay=.01,foreach=False),
        torch.optim.AdamW(other,lr=a.body_lr,weight_decay=.01,foreach=False),
        torch.optim.Muon(matrices,lr=a.body_lr,weight_decay=.01,adjust_lr_fn='match_rms_adamw')])
    identity={**identity,'head_kind':'geometry','cached_prefix':not a.uncached,'prompt_tokens':data['prompt_tokens'],
        'head_input_norm':a.head_input_norm,
        'calibrated_head':a.calibrate_head,
        'initialization':{'checkpoint_sha256':initial_sha,'scope':'shared trainable weights; new codebook-tied heads'},
        'prefix_audit_sha256':digest(a.prefix_audit),'prepared_sha256':manifest['prepared_sha256'],
        'training_recipe':'geometry-v7','prefix_gradients':'detached; target-side adapters train',
        'full_precision_reductions':True,'head_warmup':a.head_warmup,'head_lr':a.head_lr,'body_lr':a.body_lr,
        'sampling':'80% coding; sqrt response-count source/teacher buckets; uniform response then chunk',
        'loss':'mean level code CE + 0.2 normalized residual MSE + 0.2 decoded byte CE + 0.1 end BCE',
        'seed':a.seed,'quality_ready':False}
    if a.uncached:
        identity['prefix_gradients']='full-context backward; prefix cache disabled'
        identity['prefix_audit_outcome']='passed but cache disabled' if audit['passed'] else 'failed; cache disabled, no tolerance relaxation'
    # Remove inherited descriptions of the previous stage's optimizer/sampler.
    identity['optimizer']={'name':'Muon hidden matrices + AdamW interfaces','head_lr':a.head_lr,'body_lr':a.body_lr}
    if a.nearest_code:identity['nearest_code']=True
    if a.residual_byte_loss:
        identity['residual_byte_loss_weight']=a.residual_byte_loss
        identity['loss']+=f' + {a.residual_byte_loss} final continuous-residual decoder CE'
    if a.continue_checkpoint:
        prior_sha=digest(a.continue_checkpoint)
        prior=torch.load(a.continue_checkpoint,map_location='cpu',weights_only=True)
        if digest(a.continue_checkpoint)!=prior_sha:raise ValueError('Continuation checkpoint changed during load')
        for key in ['source_sha256','codec_sha256','prepared_sha256','rank','head_kind','head_input_norm',
                    'calibrated_head','full_precision_reductions','head_lr','body_lr','prompt_tokens','nearest_code','residual_byte_loss_weight']:
            if prior['identity'].get(key)!=identity.get(key):raise ValueError('Continuation configuration differs: '+key)
        if set(prior['model'])!=set(model.trainable_state()):raise ValueError('Continuation parameter inventory differs')
        with torch.no_grad():
            for name,v in trainable:v.copy_(prior['model'][name].to(v))
        optimizer.load_state_dict(prior['optimizer']);set_rng(prior['rng'])
        identity['continuation']={'checkpoint_sha256':prior_sha,'source_step':prior['step'],
                                  'scope':'weights, optimizer and RNG retained; new stage update count'}
        del prior
    if a.batch_levels:
        if not a.batch_audit:raise ValueError('Batched levels require an audit')
        batch_audit=json.loads(Path(a.batch_audit).read_text())
        expected=digest(a.continue_checkpoint or a.init)
        if not batch_audit.get('passed') or not batch_audit.get('complete') or batch_audit.get('checkpoint_sha256')!=expected:
            raise ValueError('Batched-level audit failed or belongs to another checkpoint')
        identity['level_batching']={'audit_sha256':digest(a.batch_audit),'mode':'first level separate; remaining independent padded levels share detached prefix'}
    history=[];begin=0
    if a.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        if identity!=ck['identity']:raise ValueError('Resume identity differs')
        if set(ck['model'])!=set(model.trainable_state()):raise ValueError('Resume parameter inventory differs')
        with torch.no_grad():
            for name,v in trainable:v.copy_(ck['model'][name].to(v))
        optimizer.load_state_dict(ck['optimizer']);set_rng(ck['rng']);history=ck['history'];begin=ck['step']
    if a.steps<=begin:raise ValueError('No remaining updates')
    tokenizer=AutoTokenizer.from_pretrained('results/tokenizer',local_files_only=True,trust_remote_code=False)
    write_json(out/'identity.json',identity)
    best_development=-1
    if a.resume and (out/'best-development.json').exists():
        best_development=json.loads((out/'best-development.json').read_text())['passed_tasks']
    for step in range(begin,a.steps):
        started=time.monotonic();headroom(devices);model.train()
        warmup=step<a.head_warmup
        # Keep all parameters marked trainable for complete checkpoint inventory.
        # Warmup detaches backbone features by disabling grads only in its forward.
        row,condition=sampler.sample(random)
        target=torch.tensor([encode(row['target'],codec.config.length)],device=devices[-1])
        with torch.no_grad():codes,inputs=codec.targets(target);z=codec.encode_latents(target)
        prompt=torch.tensor([condition['ids']],dtype=torch.long)
        prefix=model.prepare_prefix(prompt);optimizer.zero_grad()
        values=[];accuracies=[];components=[]
        batched=model.forward_levels(inputs,prompt,prefix_state=prefix,detach_backbone=warmup) if a.batch_levels else None
        batch_losses=[]
        for level in range(len(codes)):
            if batched is not None:
                logits,stop,residual=batched[level]
            elif warmup:
                # Capture detached final features without a second backbone pass.
                with torch.no_grad():
                    h=model.hidden_features(inputs[level],prompt,level,prefix_state=prefix)
                logits,residual=model.head(h,level,return_latent=True)
                stop=model.stop_head(h.mean(1)).squeeze(-1)
            else:
                logits,stop,residual=model(inputs[level],prompt,level,prefix_state=prefix,return_aux=True)
            wanted=codec.resample(z,codec.config.levels[level])-inputs[level]
            mse=F.mse_loss(residual,wanted)/wanted.square().mean().clamp_min(.01)
            ce=F.cross_entropy(logits.flatten(0,1),codes[level].flatten())
            loss=(ce+.2*mse)/len(codes)
            entry={'code_ce':float(ce.detach()),'normalized_residual_mse':float(mse.detach())}
            if level==len(codes)-1:
                byte_loss=decoded_byte_loss(codec,logits,inputs[level],target)
                end_loss=F.binary_cross_entropy_with_logits(stop,torch.full_like(stop,float(condition['is_final_chunk'])))
                loss=loss+.2*byte_loss+.1*end_loss
                entry.update(byte_ce=float(byte_loss.detach()),end_bce=float(end_loss.detach()))
                if a.residual_byte_loss:
                    continuous=codec.decode_latents(inputs[level]+residual)
                    direct_ce=F.cross_entropy(continuous.flatten(0,1),target.flatten())
                    loss=loss+a.residual_byte_loss*direct_ce
                    entry['continuous_residual_byte_ce']=float(direct_ce.detach())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            if batched is None:loss.backward()
            else:batch_losses.append(loss)
            values.append(float(loss.detach()))
            components.append(entry)
            accuracies.append(float((logits.argmax(-1)==codes[level]).float().mean()))
        if batched is not None:
            torch.stack(batch_losses).sum().backward()
            del batched,batch_losses
        adapter_gradients=[]
        if not warmup:
            for adapter in model.backbone.adapters:
                grad=adapter.up.weight.grad
                if grad is None or not torch.isfinite(grad).all() or grad.norm()==0:raise RuntimeError('Adapter gradient missing/nonfinite/zero')
                adapter_gradients.append(float(grad.norm()))
        for d in devices:
            torch.nn.utils.clip_grad_norm_([v for n,v in trainable if v.device==torch.device(d)],1.,foreach=False,error_if_nonfinite=True)
        optimizer.step()
        del prefix
        record={'step':step+1,'loss':sum(values),'code_accuracy':accuracies,'head_warmup':warmup,
                'loss_components':components,
                'training_example':row['id'],'coding_response':condition['coding_response'],
                'distilled':bool(row.get('distilled',False)),'prompt_tokens':prompt.numel(),
                'seconds':time.monotonic()-started,'adapter_gradient_norms':adapter_gradients,
                'peak_allocated_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices}}
        history.append(record);print(json.dumps(record),flush=True)
        last=step+1==a.steps;evaluate=(step+1)%a.eval_every==0 or last
        if (step+1)%a.save_every==0 or evaluate:
            save(out/'latest.pt',{'identity':identity,'model':model.trainable_state(),
                 'optimizer':cpu_tree(optimizer.state_dict()),'rng':rng_state(),'step':step+1,'history':history})
        write_json(out/'training.json',{'identity':identity,'history':history,'complete':last,'quality_ready':False})
        if evaluate:
            model.eval()
            write_json(out/f'chunks-{step+1}.json',evaluate_candidate(model,codec,prepared,devices[-1]))
            report=evaluate_model(model,codec,tokenizer,identity['prompt_tokens'],digest(out/'latest.pt'),out/f'coding-{step+1}.json')
            if report['passed']>best_development:
                snapshot=out/f'development-step-{step+1}.pt'
                if snapshot.exists():raise RuntimeError('Refusing to overwrite a development snapshot')
                shutil.copyfile(out/'latest.pt',snapshot)
                if digest(snapshot)!=report['checkpoint_sha256']:raise RuntimeError('Development snapshot checksum differs')
                best_development=report['passed']
                write_json(out/'best-development.json',{'checkpoint':snapshot.name,
                    'checkpoint_sha256':report['checkpoint_sha256'],'step':step+1,
                    'passed_tasks':report['passed'],'evaluated_tasks':report['evaluated'],
                    'quality_ready':False,'scope':'selection on reused development tasks, not qualification'})
    print(json.dumps({'complete':True,'quality_ready':False,'steps':a.steps}),flush=True)


if __name__=='__main__':main()

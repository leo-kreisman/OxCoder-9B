"""Train an existing learned residual-VQ codec with explicit byte coverage.

Synthetic symbols exercise representation, not reasoning or coding knowledge.
No answer lookup, residual side channel, or autoregressive reconstruction is used.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time
import torch
from .codec import Codec,CodecConfig,encode,EOS,PAD,resize
from .backbone import digest
from .train import admission,headroom,load_data,save,write_json,cpu_tree,reconstruction,rng_state,set_rng
from .evaluate_documents import evaluate
from .adapt_codec import documents
from .official_codec import PROBES


def coverage_batch(batch,length,rng):
    # Random and repeated-symbol runs exercise high/low entropy residual levels.
    ids=torch.randint(256,(batch,length),generator=rng)
    ids[:batch//4]=ids[:batch//4,:1]
    lengths=torch.randint(length,(batch,),generator=rng)
    positions=torch.arange(length)[None]
    ids[positions>lengths[:,None]]=PAD
    ids[torch.arange(batch),lengths]=EOS
    return ids


def complete_selection_score(validation,whole,probes):
    """Prioritize intact development documents over averaged byte accuracy."""
    exact=sum(r['exact_text'] for r in whole['rows'])+sum(r['exact_text'] for r in probes['rows'])
    return (int(whole['passed'] and probes['passed']),exact,validation['nonpad_symbol_accuracy'])


def frozen_parameter_digest(model):
    h=hashlib.sha256()
    for name,value in model.named_parameters():
        if not value.requires_grad:
            h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def coverage_audit(model,device,batches=32):
    rng=torch.Generator().manual_seed(71023);correct=total=exact=0
    for _ in range(batches):
        ids=coverage_batch(32,model.config.length,rng).to(device)
        codes,_=model.targets(ids);got=model.reconstruct_codes(codes).argmax(-1)
        correct+=int((got==ids).sum());total+=ids.numel();exact+=int((got==ids).all(-1).sum())
    return {'symbol_accuracy':correct/total,'exact_window_rate':exact/(batches*32),
            'windows':batches*32,'seed':71023,'scope':'synthetic byte representation, not text generation'}


@torch.no_grad()
def revive_fine_codes(model,optimizer,ids,usage,rng):
    """Reinitialize unused fine codes from observed residuals; reset their moments."""
    dead=(usage==0).nonzero().flatten()
    if len(dead)==0:return 0
    z=model.encode_latents(ids);_,codes,_,_=model.quantize(z);earlier=torch.zeros_like(z)
    for book,index in zip(model.books[:-1],codes[:-1]):earlier+=resize(book(index),model.config.length)
    residuals=(z-earlier)[ids!=PAD]
    if not len(residuals):return 0
    sample=torch.randint(len(residuals),(len(dead),),generator=rng).to(z.device)
    weight=model.books[-1].weight;weight[dead]=residuals[sample]
    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if key in optimizer.state.get(weight,{}):optimizer.state[weight][key][dead]=0
    return len(dead)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--corpus',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--zero-position',action='store_true',help='Tokenwise control: remove unnecessary positional variation')
    p.add_argument('--revive-every',type=int,default=0,help='Reinitialize unused fine codes at this interval; zero disables')
    p.add_argument('--decoder-only',action='store_true',help='Freeze learned representations and fit only the reconstruction decoder')
    p.add_argument('--select-complete',action='store_true',help='Select checkpoints on complete development responses and probes before per-symbol accuracy')
    p.add_argument('--disable-coarse-loss',action='store_true',help='Decoder-only calibration: train full reconstruction without the earlier auxiliary prefix loss')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(391)
    limits=admission([a.device],4);out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    ck=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    model=Codec(CodecConfig(**ck['config'])).to(a.device);model.load_state_dict(ck['model'])
    if a.zero_position:
        if model.config.layers!=0:raise ValueError('Zero-position control requires a tokenwise codec')
        with torch.no_grad():model.position.zero_()
        model.position.requires_grad_(False)
    if a.decoder_only:
        if a.revive_every:raise ValueError('Frozen-code decoder training cannot revive codebooks')
        model.requires_grad_(False);model.decoder.requires_grad_(True);model.head.requires_grad_(True)
    if a.disable_coarse_loss:
        if not a.decoder_only:raise ValueError('Disabling coarse loss is restricted to frozen-encoder decoder calibration')
        model.config.coarse_loss_weight=0.
    frozen_sha=frozen_parameter_digest(model) if a.decoder_only else None
    rows,manifest=load_data(a.data);corpus,cm=load_data(a.corpus)
    if manifest['complete_corpus_sha256']!=cm['examples_sha256']:raise ValueError('Corpus identity mismatch')
    training=torch.tensor([encode(r['target'],model.config.length) for r in rows if r['split']=='train'])
    validation=[r for r in rows if r['split']=='validation']
    identity={'stage':'codec','recipe':'byte-coverage-v1','data_sha256':manifest['examples_sha256'],
        'config':dataclasses.asdict(model.config),'initialization_sha256':digest(a.checkpoint),
        'optimizer':'AdamW','lr':a.lr,'batch':a.batch,'seed':391,
        'synthetic_fraction':.5,'synthetic_seed':291,'quality_ready':False}
    if a.zero_position:identity['position']='zero, frozen; tokenwise encoder and decoder'
    if a.revive_every:identity['revive_every']=a.revive_every
    if a.decoder_only:identity['train_scope']='decoder only; frozen encoder, positions, codebooks'
    if a.disable_coarse_loss:identity['coarse_loss_override']='disabled for decoder-only full-reconstruction calibration'
    if a.select_complete:identity['selection']='complete-development-documents-and-probes-before-symbols-v1'
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr)
    rng=torch.Generator().manual_seed(291);best=-1.;history=[];start=time.monotonic()
    usage=torch.zeros(model.config.codes,device=a.device,dtype=torch.long)
    begin=0
    if a.select_complete:best=(-1,-1,-1.)
    if a.resume:
        previous=torch.load(out/'latest.pt',weights_only=True,map_location='cpu')
        stored_identity=dict(previous['identity'])
        stored_identity['config']=dataclasses.asdict(CodecConfig(**stored_identity['config']))
        if stored_identity!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        rng.set_state(previous['rng']);begin=previous['step']
        if 'global_rng' in previous:set_rng(previous['global_rng'])
        elif model.config.coarse_loss_weight:raise ValueError('Stochastic codec resume requires saved global RNG state')
        if a.revive_every:
            if 'fine_usage' not in previous:raise ValueError('Missing fine-code usage state')
            usage.copy_(previous['fine_usage'].to(a.device))
        history=json.loads((out/'training.json').read_text())['history']
        best=max(complete_selection_score(item['validation'],item['whole_validation'],item['probes']) for item in history) if a.select_complete else max(item['validation']['nonpad_symbol_accuracy'] for item in history)
    if a.steps<=begin:raise ValueError('No additional updates requested')
    write_json(out/'identity.json',identity);write_json(out/'admission.json',limits)
    for step in range(begin+1,a.steps+1):
        headroom([a.device]);model.train();n=a.batch//2
        natural=training[torch.randint(len(training),(n,),generator=rng)]
        ids=torch.cat((natural,coverage_batch(a.batch-n,model.config.length,rng))).to(a.device)
        optimizer.zero_grad(set_to_none=True);loss,_,codes,_=model(ids)
        usage+=torch.bincount(codes[-1].flatten(),minlength=model.config.codes)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if a.revive_every and step%a.revive_every==0:
            count=revive_fine_codes(model,optimizer,ids,usage,rng);usage.zero_()
            print(json.dumps({'step':step,'fine_codes_reinitialized':count}),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            model.eval();metric=reconstruction(model,validation,a.device)
            cov=coverage_audit(model,a.device)
            report={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start,
                    'validation':metric,'coverage':cov}
            if a.select_complete:
                report['whole_validation']=evaluate(model,documents(corpus,16),a.device)
                report['probes']=evaluate(model,[(f'probe-{i}',s) for i,s in enumerate(PROBES)],a.device)
            history.append(report);print(json.dumps(report),flush=True)
            # Choose by natural validation accuracy; synthetic data are never the sole gate.
            score=complete_selection_score(metric,report['whole_validation'],report['probes']) if a.select_complete else metric['nonpad_symbol_accuracy']
            if score>best:
                best=score;save(out/'best.pt',{'config':model.metadata(),'identity':identity,
                     'model':cpu_tree(model.state_dict()),'step':step,'validation':metric})
            save(out/'latest.pt',{'identity':identity,'config':model.metadata(),'model':cpu_tree(model.state_dict()),
                'optimizer':cpu_tree(optimizer.state_dict()),'rng':rng.get_state(),'step':step,'validation':metric,
                'fine_usage':usage.cpu(),'global_rng':rng_state()})
            write_json(out/'training.json',{'history':history,'complete':step==a.steps,'quality_ready':False})
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu');model.load_state_dict(chosen['model']);model.eval()
    if frozen_sha is not None and frozen_parameter_digest(model)!=frozen_sha:
        raise RuntimeError('Frozen representation changed during decoder calibration')
    report={'checkpoint_sha256':digest(out/'best.pt'),'corpus_sha256':cm['examples_sha256'],
            'checkpoint_step':chosen['step'],'validation':evaluate(model,documents(corpus,16),a.device),
            'probes':evaluate(model,[(f'probe-{i}',s) for i,s in enumerate(PROBES)],a.device),
            'coverage':coverage_audit(model,a.device),'quality_ready':False}
    if frozen_sha is not None:report['unchanged_frozen_parameters_sha256']=frozen_sha
    report['passed']=report['validation']['passed'] and report['probes']['passed']
    write_json(out/'complete-gate.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('validation','probes')}),flush=True)


if __name__=='__main__':main()

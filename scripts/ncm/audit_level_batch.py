"""Compare independent sequential and padded-batch concept-level training."""
import argparse
import time
import torch
from .codec import encode
from .generate_candidate import load_candidate
from .train import admission,write_json


def synchronize(devices):
    for d in devices:torch.cuda.synchronize(d)


def main():
    p=argparse.ArgumentParser()
    for k in ['source','codec','checkpoint','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(14)
    devices=['cuda:0','cuda:1'];admission(devices)
    model,codec,identity,sha=load_candidate(a.source,a.codec,a.checkpoint,devices)
    if not identity.get('cached_prefix'):raise ValueError('A cached-prefix candidate is required')
    target=torch.tensor([encode('def square(x):\n    return x * x\n',codec.config.length)],device=devices[-1])
    with torch.no_grad():codes,inputs=codec.targets(target)
    records=[];model.train()
    for length in [64,1024]:
        prompt=torch.tensor([[100,200,300,400]*(length//4)])
        state=model.prepare_prefix(prompt);outputs=[];gradients=[];times=[]
        for batched in [False,True]:
            xs=[x.detach().clone().requires_grad_(True) for x in inputs]
            synchronize(devices);started=time.monotonic()
            if batched:result=model.forward_levels(xs,prompt,prefix_state=state)
            else:result=[model(x,prompt,i,prefix_state=state,return_aux=True) for i,x in enumerate(xs)]
            loss=sum(torch.nn.functional.cross_entropy(r[0].flatten(0,1),c.flatten()) for r,c in zip(result,codes))/len(codes)
            probes=xs+[model.project.weight,model.backbone.adapters[0].up.weight,model.backbone.adapters[-1].up.weight]
            gs=torch.autograd.grad(loss,probes)
            synchronize(devices);times.append(time.monotonic()-started)
            outputs.append([r[0].detach() for r in result]);gradients.append([g.detach().float() for g in gs])
            del result,loss,gs,xs
        errors=[float((x-y).norm()/x.norm().clamp_min(1e-12)) for x,y in zip(*outputs)]
        ge=[float((x-y).norm()/x.norm().clamp_min(1e-12)) for x,y in zip(*gradients)]
        cos=[float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0)) for x,y in zip(*gradients)]
        finite=all(torch.isfinite(x).all() for group in gradients for x in group)
        records.append({'prompt_tokens':length,'relative_logit_errors':errors,'relative_gradient_errors':ge,
                        'gradient_cosines':cos,'sequential_forward_backward_seconds':times[0],
                        'batched_forward_backward_seconds':times[1],
                        'passed':bool(finite and max(errors)<.02 and max(ge)<.05)})
        write_json(a.output,{'checkpoint_sha256':sha,'records':records,'complete':False,'quality_ready':False})
    report={'checkpoint_sha256':sha,'records':records,'complete':True,'passed':all(r['passed'] for r in records),
            'torch':str(torch.__version__),'quality_ready':False,
            'full_precision_reductions':identity.get('full_precision_reductions',False),
            'peak_allocated_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices},
            'scope':'synthetic token-ID prompts, forward and input/project/first-last-adapter gradients; not coding or inference throughput'}
    write_json(a.output,report)
    if not report['passed']:raise SystemExit('Batched-level numerical audit failed')


if __name__=='__main__':main()

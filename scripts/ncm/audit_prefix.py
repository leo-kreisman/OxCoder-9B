"""Whole-model prefix-cache fidelity, target-gradient and timing comparison."""
import argparse
import time
import torch
from .generate_candidate import load_candidate
from .codec import encode
from .train import admission,write_json


def sync(devices):
    for d in devices:torch.cuda.synchronize(d)


def main():
    p=argparse.ArgumentParser()
    for k in ['source','codec','checkpoint','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--full-precision-reductions',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(9)
    if a.full_precision_reductions:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    devices=['cuda:0','cuda:1'];admission(devices)
    model,codec,identity,sha=load_candidate(a.source,a.codec,a.checkpoint,devices)
    target=torch.tensor([encode('def square(x):\n    return x * x\n',codec.config.length)],device=devices[-1])
    with torch.no_grad():codes,inputs=codec.targets(target)
    records=[]
    for length in [64,512,2048]:
        prompt=torch.tensor([[100,200,300,400]*(length//4)])
        outputs=[];timings=[]
        for cached in [False,True]:
            model.cached_prefix=cached;model.eval()
            sync(devices);start=time.monotonic()
            state=model.prepare_prefix(prompt)
            with torch.no_grad():out=[model(inp,prompt,i,prefix_state=state).detach() for i,inp in enumerate(inputs)]
            sync(devices);timings.append(time.monotonic()-start);outputs.append(out)
        errors=[float((x-y).norm()/x.norm().clamp_min(1e-12)) for x,y in zip(*outputs)]
        agreements=[float((x.argmax(-1)==y.argmax(-1)).float().mean()) for x,y in zip(*outputs)]
        records.append({'prompt_tokens':length,'relative_logit_errors':errors,'code_argmax_agreement':agreements,
            'full_seconds':timings[0],'cached_seconds_including_prefill':timings[1],
            'passed':max(errors)<.02})
        write_json(a.output,{'checkpoint_sha256':sha,'records':records,'complete':False,'quality_ready':False})
    gradients=[];prompt=torch.tensor([[100,200,300,400]*16])
    model.train()
    for level in sorted({0,len(inputs)-1}):
        pair=[]
        for cached in [False,True]:
            model.cached_prefix=cached;state=model.prepare_prefix(prompt)
            x=inputs[level].detach().clone().requires_grad_(True)
            logits=model(x,prompt,level,prefix_state=state)
            g=torch.autograd.grad(logits.square().mean(),x)[0]
            pair.append(g.detach())
        error=float((pair[0]-pair[1]).norm()/pair[0].norm().clamp_min(1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(pair[0].flatten(),pair[1].flatten(),dim=0))
        gradients.append({'level':level,'relative_error':error,'cosine':cosine,
                          'max_absolute_error':float((pair[0]-pair[1]).abs().max()),
                          'finite':all(bool(torch.isfinite(g).all()) for g in pair),
                          'passed':error<.05})
    report={'checkpoint_sha256':sha,'records':records,'target_gradients':gradients,
            'full_precision_reductions':a.full_precision_reductions,
            'passed':all(r['passed'] for r in records+gradients),'complete':True,'quality_ready':False,
            'scope':'synthetic token-ID prompts; numerical/cache check, not coding quality or deployment throughput'}
    write_json(a.output,report)
    if not report['passed']:raise SystemExit('Prefix audit failed')


if __name__=='__main__':main()

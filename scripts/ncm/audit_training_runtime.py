"""Compare full NCM four-level training passes at identical weights and data.

No optimizer updates. Tests gradients for every trainable parameter, including
the prompt path, and records memory and synchronized wall time. This is a
training execution audit, not a claim about generated-code quality.
"""
import argparse
import gc
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .backbone import digest
from .codec import encode
from .generate_candidate import load_candidate
from .train import admission, write_json
from .train_independent import corrupted_inputs
from .training_runtime import execution


def compare(a, b):
    square_error = square_reference = dot = square_actual = 0.
    finite = True
    maximum = 0.
    if set(a) != set(b):
        raise ValueError('Gradient inventories differ')
    for name in a:
        x, y = a[name].double(), b[name].double()
        finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        square_error += float((x-y).square().sum())
        square_reference += float(y.square().sum())
        square_actual += float(x.square().sum())
        dot += float((x*y).sum())
        maximum = max(maximum, float((x-y).abs().max()))
    return {'relative_l2': (square_error/max(square_reference, 1e-30))**.5,
            'cosine': dot/max((square_actual*square_reference)**.5, 1e-30),
            'max_absolute': maximum, 'finite': finite, 'parameters_compared': len(a)}


def run(a):
    torch.set_num_threads(2)
    devices = ['cuda:0', 'cuda:1']
    limits = admission(devices, a.allocator_cap_gib)
    model, codec, identity, sha = load_candidate(a.source, a.codec, a.checkpoint, devices)
    model.train()
    prepared = torch.load(Path(a.data)/'prepared.pt', weights_only=True, map_location='cpu')['prepared']
    rows = [(r,c) for r,c in prepared if r['split']=='train']
    report = {'checkpoint_sha256': sha, 'prepared_sha256': digest(Path(a.data)/'prepared.pt'),
              'model_identity': identity,
              'implementation_sha256': {name:digest(Path(__file__).parent/name) for name in
                  ('audit_training_runtime.py','training_runtime.py','shared_training.py','vendor/veomni_shared_prefix.py','backbone.py')},
              'runtime_implementation_sha256': digest(Path(__file__).with_name('training_runtime.py')),
              'admission': limits, 'records': [], 'complete': False, 'quality_ready': False,
              'scope': 'four-level forward/backward at fixed weights; no optimizer step or quality claim',
              'optimizer_state_in_memory': False,
              'thresholds': {'logit_relative_l2': .02, 'gradient_relative_l2': .05,
                             'gradient_cosine_minimum': .99}}
    variants = [(0,'reference',False)] + [(int(k),'reference',False) for k in a.keep.split(',') if int(k)]
    if a.convolution:
        variants += [(0,'causal-conv1d',False)]
    if a.shared:
        variants += [(0,'reference',True)]
    for requested in map(int,a.lengths.split(',')):
        row,c = min(rows,key=lambda item:abs(len(item[1]['ids'])-requested))
        target = torch.tensor([encode(row['target'],codec.config.length)],device=devices[-1])
        with torch.no_grad(): codes,_ = codec.targets(target)
        torch.manual_seed(18001)
        inputs = corrupted_inputs(codec,codes,.5)
        prompt = torch.tensor([c['ids']])
        reference = None
        for keep,conv,shared in variants:
            timings=[]; peaks=[]
            try:
                for repeat in range(a.repeats+1):
                    model.zero_grad(set_to_none=True)
                    for d in devices:
                        torch.cuda.synchronize(d); torch.cuda.reset_peak_memory_stats(d)
                    start=time.perf_counter(); outputs={}; losses=[]
                    with execution(model,keep,conv) as runtime:
                        runtime['shared_prefix_training']=shared
                        shared_loss=0
                        if shared:
                            from .shared_training import forward_levels
                            level_outputs=forward_levels(model,inputs,prompt)
                        for level,inp in enumerate(inputs):
                            logits,end,_=level_outputs[level] if shared else model(inp,prompt,level,return_aux=True)
                            loss=F.cross_entropy(logits.flatten(0,1),codes[level].flatten())/len(codes)
                            if level==len(codes)-1:
                                loss=loss+.1*F.binary_cross_entropy_with_logits(end,torch.full_like(end,float(c['is_final_chunk'])))
                            if shared:shared_loss=shared_loss+loss
                            else:loss.backward()
                            losses.append(float(loss.detach()))
                            outputs[str(level)]=logits.detach().cpu()
                        if shared:
                            shared_loss.backward()
                            del level_outputs,shared_loss
                    for d in devices:torch.cuda.synchronize(d)
                    elapsed=time.perf_counter()-start
                    if repeat:timings.append(elapsed)
                    peaks.append({d:torch.cuda.max_memory_allocated(d) for d in devices})
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()
                           if p.requires_grad and p.grad is not None}
                expected={n for n,p in model.named_parameters() if p.requires_grad}
                if set(gradients)!=expected:raise ValueError('Missing trainable gradients')
                if reference is None:reference=(outputs,gradients,statistics.median(timings))
                ge=compare(gradients,reference[1]); le=compare(outputs,reference[0])
                groups={}
                for group,predicate in (
                    ('adapters',lambda n:n.startswith('backbone.adapters.')),
                    ('classifiers',lambda n:n.startswith(('head.','stop_head.'))),
                    ('interfaces',lambda n:not n.startswith(('backbone.adapters.','head.','stop_head.')))):
                    groups[group]=compare({n:g for n,g in gradients.items() if predicate(n)},
                                          {n:g for n,g in reference[1].items() if predicate(n)})
                record={'requested_prompt_tokens':requested,'prompt_tokens':len(c['ids']),
                        'example_id':row['id'],'runtime':runtime,'loss':sum(losses),
                        'median_seconds':statistics.median(timings),'seconds':timings,
                        'speed_ratio':reference[2]/statistics.median(timings),
                        'peak_allocated_bytes':{d:max(p[d] for p in peaks) for d in devices},
                        'gradient_comparison':ge,'logit_comparison':le,
                        'gradient_groups':groups,
                        'passed':ge['finite'] and le['finite'] and ge['relative_l2']<.05
                                 and ge['cosine']>.99 and le['relative_l2']<.02
                                 and all(g['finite'] and g['relative_l2']<.05 and g['cosine']>.99 for g in groups.values())}
                del gradients,outputs,logits,end,loss
            except torch.cuda.OutOfMemoryError as error:
                record={'requested_prompt_tokens':requested,'runtime':{'keep_per_device':keep,'convolution':conv,'shared_prefix_training':shared},
                        'passed':False,'error':'CUDA allocator cap exceeded'}
                model.zero_grad(set_to_none=True); gc.collect()
                for d in devices:
                    with torch.cuda.device(d):torch.cuda.empty_cache()
            report['records'].append(record); write_json(a.output,report)
            print(record,flush=True)
        del reference
    report['complete']=True;write_json(a.output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','codec','checkpoint','data','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--keep',default='2,4')
    p.add_argument('--lengths',default='64,512,2048')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--convolution',action='store_true')
    p.add_argument('--shared',action='store_true')
    p.add_argument('--allocator-cap-gib',type=float,default=11.25)
    a=p.parse_args()
    if a.repeats<1 or any(int(n)<1 or int(n)>2048 for n in a.lengths.split(',')):
        p.error('Require repeats >=1 and prompt lengths within 1..2048')
    run(a)

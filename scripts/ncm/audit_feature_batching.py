"""Numerical and timing gate for batched frozen-prompt feature extraction.

Right padding is after every real prompt token; all source layers stay causal.
No current target bytes enter these calls. The gate is fixed before measuring:
relative feature L2 <= .001, identical concept IDs and decoded bytes, and stop
probability error <= 1e-5 on every audited prompt. Failed batching is not enabled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from .backbone import digest
from .context_cache import FeatureCache
from .context_decoder import DecoderConfig
from .context_runtime import load
from .train import admission,write_json


@torch.no_grad()
def encode_batch(backbone, sequences):
    if not sequences or min(map(len,sequences))<1:raise ValueError('Nonempty prompts required')
    maximum=max(map(len,sequences))
    ids=torch.zeros(len(sequences),maximum,dtype=torch.long)
    for i,values in enumerate(sequences):ids[i,:len(values)]=torch.tensor(values)
    hidden=backbone(backbone.embed_prompt(ids))
    return [hidden[i,:len(values)].clone() for i,values in enumerate(sequences)]


def run(a):
    torch.set_num_threads(2); admission([a.device],8)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    runtime,codec,identity,checksum=load(a.source,a.codec,a.checkpoint,a.device)
    cache=FeatureCache(a.cache,DecoderConfig(**identity['config']))
    if cache.sha!=identity['feature_manifest_sha256']:raise ValueError('Checkpoint feature identity differs')
    indices=sorted((i for i,r in enumerate(cache.records) if r['split']=='validation'),
        key=lambda i:hashlib.sha256(('batch-features-v27:'+cache.records[i]['id']).encode()).hexdigest())[:a.count]
    items=[cache.get(i) for i in indices]
    # Length sorting reduces padding; the audit retains short and long prompts.
    items.sort(key=lambda x:len(x['condition']['ids']))
    records=[]
    for batch_size in a.batches:
        elapsed=0.; rows=[]
        try:
            for start in range(0,len(items),batch_size):
                group=items[start:start+batch_size]
                torch.cuda.synchronize(a.device);beg=time.monotonic()
                hidden=encode_batch(runtime.backbone,[x['condition']['ids'] for x in group])
                torch.cuda.synchronize(a.device);elapsed+=time.monotonic()-beg
                for item,fresh in zip(group,hidden):
                    original=item['hidden'][None].to(a.device);fresh=fresh[None]
                    valid=torch.ones(original.shape[:2],dtype=torch.bool,device=a.device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        reference=runtime.decoder.generate_chunk(codec,original,valid)
                        candidate=runtime.decoder.generate_chunk(codec,fresh,valid)
                    relative=float((fresh.float()-original.float()).norm()/original.float().norm())
                    row={'id':item['row']['id'],'prompt_tokens':len(item['condition']['ids']),
                         'feature_relative_l2':relative,'features_exact':bool(torch.equal(fresh,original)),
                         'codes_exact':all(torch.equal(x,y) for x,y in zip(reference[1],candidate[1])),
                         'bytes_exact':bool(torch.equal(reference[0],candidate[0])),
                         'stop_max_abs':float((reference[2]-candidate[2]).abs().max())}
                    row['passed']=relative<=.001 and row['codes_exact'] and row['bytes_exact'] and row['stop_max_abs']<=1e-5
                    rows.append(row)
                del hidden
            result={'batch':batch_size,'records':rows,'feature_encoding_seconds':elapsed,
                    'passed':len(rows)==len(items) and all(r['passed'] for r in rows)}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache();result={'batch':batch_size,'passed':False,'error':'allocator limit reached'}
        records.append(result)
        report={'checkpoint_sha256':checksum,'source_sha256':identity['source_sha256'],'cache_sha256':cache.sha,
                'implementation_sha256':digest(__file__),'results':records,'complete':len(records)==len(a.batches),
                'scope':'Prompt feature extraction audit; not coding quality or generation speed','quality_ready':False}
        write_json(out/'report.json',report)
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','codec','checkpoint','cache','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--count',type=int,default=16)
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4])
    run(p.parse_args())

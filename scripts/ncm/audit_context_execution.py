"""Measure trained-decoder differences between packed training and level inference.

Fixed validation conditioning and canonical earlier levels isolate execution
shape and precision. This is a numerical diagnostic, not generated-code quality.
No parameters are updated or saved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .backbone import digest
from .codec import Codec, CodecConfig
from .context_cache import FeatureCache
from .context_decoder import ContextConceptDecoder, DecoderConfig
from .train_context_decoder import batch, previous_concepts
from .train import admission, write_json


def comparison(actual, expected):
    if len(actual) != len(expected):
        raise ValueError('Mismatched level inventories')
    values = []
    for a, b in zip(actual, expected):
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Invalid compared logits')
        values.append({
            'relative_logit_l2': float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)),
            'argmax_agreement': float((a.argmax(-1)==b.argmax(-1)).float().mean()),
            'maximum_absolute_difference': float((a.float()-b.float()).abs().max())})
    return values


@torch.no_grad()
def run(a):
    torch.set_num_threads(2)
    admission([a.device], 8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    checksum = digest(a.checkpoint)
    state = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    if digest(a.checkpoint) != checksum:
        raise ValueError('Checkpoint changed')
    identity = state['identity']
    config = DecoderConfig(**identity['config'])
    if digest(a.codec) != identity['codec_sha256']:
        raise ValueError('Wrong codec')
    model = ContextConceptDecoder(config).to(a.device).eval()
    model.load_state_dict(state['model'])
    codec_state = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**codec_state['config'])).to(a.device).eval()
    codec.load_state_dict(codec_state['model'])
    cache = FeatureCache(a.cache, config)
    if cache.sha != identity['feature_manifest_sha256']:
        raise ValueError('Wrong conditioning cache')
    indices = sorted((i for i,r in enumerate(cache.records) if r['split']=='validation'),
        key=lambda i: hashlib.sha256(('execution-v28:'+cache.records[i]['id']).encode()).hexdigest())[:a.count]
    if len(indices)!=a.count or a.count<1:
        raise ValueError('Insufficient validation rows')
    records=[]
    for start in range(0,len(indices),a.batch):
        group=indices[start:start+a.batch]
        packed=batch(cache,group,a.device)
        packed_inputs=previous_concepts(codec,packed['codes'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            batched,batched_stop=model(packed_inputs,packed['hidden'],packed['valid'])
        for position,index in enumerate(group):
            item=batch(cache,[index],a.device)
            inputs=previous_concepts(codec,item['codes'])
            fp32,_=model(inputs,item['hidden'],item['valid'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                combined,combined_stop=model(inputs,item['hidden'],item['valid'])
                memory,mask=model.prepare_memory(item['hidden'],item['valid'])
                separate=[]
                for level,inp in enumerate(inputs):
                    logits,ending=model.forward_level(inp,level,memory,mask)
                    separate.append(logits)
            row={'id':cache.records[index]['id'],
                'combined_vs_independent_levels':comparison(combined,separate),
                'batched_vs_single':comparison([x[position:position+1] for x in batched],combined),
                'bf16_vs_fp32':comparison(combined,fp32),
                'combined_vs_separate_stop_abs':float((combined_stop-ending).abs().max()),
                'batch_vs_single_stop_abs':float((batched_stop[position:position+1]-combined_stop).abs().max())}
            records.append(row)
    report={'complete':True,'checkpoint_sha256':checksum,'cache_sha256':cache.sha,
        'implementation_sha256':digest(__file__),'records':records,'batch':a.batch,
        'scope':'Canonical earlier concepts; execution-shape and precision diagnostic only',
        'quality_ready':False}
    write_json(out,report)
    print(json.dumps({'complete':True,'records':len(records),
        'minimum_level_agreement':min(x['argmax_agreement'] for r in records for x in r['combined_vs_independent_levels']),
        'maximum_level_relative_l2':max(x['relative_logit_l2'] for r in records for x in r['combined_vs_independent_levels'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','codec','cache','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--count',type=int,default=16)
    p.add_argument('--batch',type=int,default=8)
    a=p.parse_args()
    if a.batch<1:p.error('Positive batch required')
    run(a)

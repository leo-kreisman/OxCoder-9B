"""Audit CUDA graph replay of the unchanged frozen encoder at exact prompt lengths.

No padding to a new length, quantization change, or target information is used.
This is a feature-extraction optimization audit, not an NCM quality evaluation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import torch

from .backbone import Source,digest
from .context_cache import FeatureCache
from .context_decoder import DecoderConfig
from .gguf_training_backbone import PackedOxBackbone
from .train import admission,write_json


class EncoderGraph:
    def __init__(self, encoder, example):
        self.input=example.clone();self.device=example.device;self.encoder=encoder
        self.graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(2):encoder(self.input)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.no_grad(),torch.cuda.graph(self.graph,stream=stream):self.output=encoder(self.input)
        torch.cuda.current_stream(self.device).wait_stream(stream)

    @torch.no_grad()
    def __call__(self,example):
        if example.shape!=self.input.shape or example.dtype!=self.input.dtype or example.device!=self.device:
            raise ValueError('Graph input signature changed')
        self.input.copy_(example);self.graph.replay();return self.output


def run(a):
    torch.set_num_threads(2);admission([a.device],8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    suspended=False;report={}
    try:
        if a.pause_unit:
            if subprocess.run(['systemctl','--user','is-active','--quiet',a.pause_unit]).returncode:
                raise RuntimeError('Requested extraction worker is not active')
            subprocess.run(['systemctl','--user','kill','--kill-who=main','--signal=STOP',a.pause_unit],check=True)
            suspended=True
        source=Source(a.source);encoder=PackedOxBackbone(source,[a.device],rank=1).eval().requires_grad_(False)
        cache=FeatureCache(a.cache,DecoderConfig())
        indices=sorted((i for i,r in enumerate(cache.records) if r['split']=='train'),
            key=lambda i:hashlib.sha256(('encoder-graph:'+cache.records[i]['id']).encode()).hexdigest())[:a.count]
        records=[]
        with torch.cuda.device(a.device),torch.no_grad():
            for index in indices:
                item=cache.get(index);ids=torch.tensor([item['condition']['ids']]);example=encoder.embed_prompt(ids)
                eager=encoder(example);expected=item['hidden'][None].to(a.device)
                if not torch.equal(eager,expected):raise ValueError('Baseline encoder differs from qualified features')
                torch.cuda.synchronize(a.device);start=time.monotonic();graph=EncoderGraph(encoder,example)
                torch.cuda.synchronize(a.device);capture=time.monotonic()-start
                replay=graph(example).clone();torch.cuda.synchronize(a.device)
                # A different real prompt of the same length exercises dynamic
                # input replacement. Reversing token order is not training data.
                changed=encoder.embed_prompt(ids.flip(1));reference=encoder(changed)
                altered=graph(changed).clone();torch.cuda.synchronize(a.device)
                times={'eager':[],'graph':[]}
                for kind in times:
                    for _ in range(5):
                        torch.cuda.synchronize(a.device);start=time.monotonic()
                        value=encoder(example) if kind=='eager' else graph(example)
                        torch.cuda.synchronize(a.device);times[kind].append(time.monotonic()-start)
                row={'id':item['row']['id'],'prompt_tokens':len(item['condition']['ids']),
                    'original_features_exact':bool(torch.equal(replay,expected)),
                    'changed_prompt_features_exact':bool(torch.equal(altered,reference)),
                    'capture_seconds':capture,'eager_median_seconds':statistics.median(times['eager']),
                    'replay_median_seconds':statistics.median(times['graph']),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(a.device)}
                row['passed']=row['original_features_exact'] and row['changed_prompt_features_exact']
                records.append(row);print(json.dumps(row),flush=True)
                del graph,example,eager,expected,replay,reference,altered,value
                gc.collect();torch.cuda.empty_cache()
        report={'complete':True,'passed':all(r['passed'] for r in records),'records':records,
            'source_sha256':source.sha,'feature_manifest_sha256':cache.sha,'implementation_sha256':digest(__file__),
            'scope':'Exact-length frozen prompt encoder graph replay; excludes embedding and cache I/O', 'quality_ready':False}
    except Exception as error:
        report={'complete':False,'passed':False,'error':repr(error),'quality_ready':False}
        raise
    finally:
        write_json(out/'report.json',report)
        if suspended:subprocess.run(['systemctl','--user','kill','--kill-who=main','--signal=CONT',a.pause_unit],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','cache','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--count',type=int,default=4)
    p.add_argument('--pause-unit');run(p.parse_args())

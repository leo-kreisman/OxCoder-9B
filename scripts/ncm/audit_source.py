"""Compare GGUF inverse transformations with bounded reads of original HF weights.

No whole-checkpoint download. Range responses are mandatory and size checked.
This validates selected norms, decay/head order, and quantized projection rows;
it does not establish full-forward numerical parity.
"""
import argparse
import json
import struct
import urllib.request
from pathlib import Path
import torch
from .backbone import Source, SOURCE_REVISION, mapping, to_hf
from .train import write_json


def remote_range(url, start, end):
    if end-start > 2 << 20:
        raise ValueError('Remote audit range exceeds 2 MiB')
    request=urllib.request.Request(url+f'?ncm_range={start}_{end}',headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('Server did not honor bounded range')
        raw=response.read(end-start+2)
        if len(raw)!=end-start+1:raise ValueError('Wrong range length')
        return raw


def audit(source_path, output):
    source=Source(source_path)
    url=f'https://huggingface.co/OrionLLM/OxCoder-9B/resolve/{SOURCE_REVISION}/model.safetensors'
    size=struct.unpack('<Q',remote_range(url,0,7))[0]
    header=json.loads(remote_range(url,8,size+7))
    checks=[]
    for layer,name in [(0,'input_layernorm.weight'),(0,'linear_attn.A_log'),(0,'linear_attn.dt_bias'),
                       (3,'self_attn.q_norm.weight'),(3,'post_attention_layernorm.weight'),
                       (0,'linear_attn.in_proj_a.weight')]:
        key=f'model.language_model.layers.{layer}.{name}'
        record=header[key];start,end=record['data_offsets']
        raw=remote_range(url,size+8+start,size+8+end-1)
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16}[record['dtype']]
        original=torch.frombuffer(bytearray(raw),dtype=dtype).float().reshape(record['shape'])
        gguf=to_hf(name,source.read(f'blk.{layer}.'+mapping(name)))
        error=(gguf-original).abs()
        relative=float(error.norm()/original.norm().clamp_min(1e-8))
        # Projection quantization is lossy; scalar transforms should be close.
        limit=.15 if name.endswith('in_proj_a.weight') else .015
        checks.append({'tensor':key,'relative_l2_error':relative,'max_abs_error':float(error.max()),
                       'relative_l2_limit':limit,'passed':relative<limit,'bytes_downloaded':len(raw)})
    report={'source_sha256':source.sha,'original_revision':SOURCE_REVISION,'checks':checks,
            'passed':all(c['passed'] for c in checks),'scope':'selected inverse transforms, not full-forward parity'}
    write_json(output,report)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Source inverse transform audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.output)

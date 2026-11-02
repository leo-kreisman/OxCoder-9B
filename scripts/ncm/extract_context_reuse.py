"""Resumable prompt-only extraction with checksum-checked reuse and audited batches.

Source and codec numerical policies match context_features.py. Existing feature
payloads are hardlinked only after exact row/condition and identity comparison.
New current-target bytes never enter the source encoder. The frozen validation
features must all come from the qualified original cache.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import torch

from .audit_feature_batching import encode_batch
from .audit_backbone import audit
from .backbone import Source,digest
from .codec import Codec,CodecConfig,encode,PAD
from .context_cache import FeatureCache
from .context_decoder import DecoderConfig
from .gguf_training_backbone import PackedOxBackbone
from .train import admission,headroom,save,write_json


def groups(pending, batch_size, maximum_tokens):
    current=[]
    for item in sorted(pending,key=lambda x:(len(x[2]['ids']),x[0])):
        size=len(item[2]['ids'])
        if current and (len(current)==batch_size or (len(current)+1)*size>maximum_tokens):
            yield current;current=[]
        current.append(item)
    if current:yield current


def run(a):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    admitted=admission([a.device],8)
    selected=Path(a.selection);meta=json.loads((selected/'manifest.json').read_text())
    if digest(selected/'selection.pt')!=meta['selection_sha256']:raise ValueError('Selection changed')
    prepared=torch.load(selected/'selection.pt',weights_only=True,map_location='cpu')['prepared']
    old=FeatureCache(a.reuse,DecoderConfig()); oldmap={r['id']:i for i,r in enumerate(old.records)}
    source=Source(a.source); codec_sha=digest(a.codec)
    if source.sha!=old.metadata['identity']['source_sha256'] or codec_sha!=old.metadata['identity']['codec_sha256']:
        raise ValueError('Reuse source/codec differs')
    batch_report=json.loads(Path(a.audit).read_text())
    gate=[r for r in batch_report.get('results',[]) if r['batch']==a.batch]
    if (not batch_report.get('complete') or len(gate)!=1 or not gate[0]['passed']
            or batch_report['source_sha256']!=source.sha
            or batch_report['implementation_sha256']!=digest(Path(__file__).with_name('audit_feature_batching.py'))):
        raise ValueError('Batch extraction lacks current numerical qualification')
    identity={'architecture':'frozen-oxcoder-prompt-encoder-native-ncm-decoder-v25',
        'source_sha256':source.sha,'codec_sha256':codec_sha,'selection_sha256':meta['selection_sha256'],
        'prepared_sha256':meta['prepared_sha256'],'encoder':'all 32 frozen original layers; zero adapters',
        'encoder_inputs':'condition.ids only; never current target bytes or codes',
        'feature_dtype':'bfloat16','maximum_prompt_tokens':2048,'reuse_manifest_sha256':old.sha,
        'feature_batch':a.batch,'maximum_batch_tokens':a.maximum_batch_tokens,'batch_audit_sha256':digest(a.audit),
        'implementation_sha256':{n:digest(Path(__file__).with_name(n)) for n in
            ('extract_context_reuse.py','audit_feature_batching.py','context_cache.py','codec.py','gguf_training_backbone.py')},
        'shard':a.shard,'shards':a.shards,'quality_ready':False}
    if a.resume:
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Extraction resume identity changed')
        state=json.loads((out/'progress.json').read_text());records=state['records'];rejected=state['rejected']
        for r in records:
            if digest(out/r['path'])!=r['sha256']:raise ValueError('Resumed payload changed')
    else:
        records=[];rejected=[];write_json(out/'identity.json',identity);write_json(out/'admission.json',admitted)
    done={r['index'] for r in records+rejected}
    def persist():
        write_json(out/'progress.json',{'records':records,'rejected':rejected,'quality_ready':False})
        print(json.dumps({'saved':len(records),'rejected':len(rejected),
            'reused':sum(bool(x.get('reused')) for x in records)}),flush=True)
    pending=[]
    remaining=sum(len(c['ids'])*8192 for i,(r,c) in enumerate(prepared)
                  if i%a.shards==a.shard and i not in done and r['id'] not in oldmap)
    if shutil.disk_usage(out).free<remaining+(20<<30):raise RuntimeError('Insufficient disk plus 20 GiB reserve')
    for i,(row,condition) in enumerate(prepared):
        if i%a.shards!=a.shard or i in done:continue
        if row['id'] in oldmap:
            j=oldmap[row['id']];payload=old.get(j);oldrec=old.records[j]
            if payload['row']!=row or payload['condition']!=condition:raise ValueError('Reuse content differs')
            path=out/f'{i:06d}.pt';original=(old.path.parent/oldrec['path']).resolve()
            if not path.exists():os.link(original,path)
            if digest(path)!=oldrec['sha256']:raise ValueError('Existing reused payload differs')
            records.append({**oldrec,'index':i,'path':path.name,'reused':True,
                            'reuse_original':str(original)})
        else:
            if row['split']=='validation':raise ValueError('Frozen validation feature missing from original cache')
            pending.append((i,row,condition))
    persist()
    ck=torch.load(a.codec,weights_only=True,map_location='cpu');codec=Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model']);codec.eval().requires_grad_(False)
    encoder=PackedOxBackbone(source,[a.device],rank=1).eval().requires_grad_(False)
    parity=audit(encoder,source,a.reference);write_json(out/'source-parity.json',parity)
    if not parity['passed']:raise RuntimeError('Original source parity failed')
    with torch.no_grad():
        for number,group in enumerate(groups(pending,a.batch,a.maximum_batch_tokens)):
            headroom([a.device]);accepted=[]
            for i,row,condition in group:
                target=torch.tensor([encode(row['target'],codec.config.length)],device=a.device)
                codes,_=codec.targets(target);reconstructed=codec.reconstruct_codes(codes).argmax(-1)
                exact=bool(((reconstructed==target)|(target==PAD)).all())
                if not exact:rejected.append({'index':i,'id':row['id'],'split':'train','reason':'codec reconstruction not exact'})
                else:accepted.append((i,row,condition,[c[0].cpu() for c in codes],target[0].cpu()))
            if accepted:
                features=encode_batch(encoder,[x[2]['ids'] for x in accepted])
                for (i,row,condition,codes,target),hidden in zip(accepted,features):
                    hidden=hidden.bfloat16().cpu()
                    if not torch.isfinite(hidden).all():raise FloatingPointError('Nonfinite prompt features')
                    path=out/f'{i:06d}.pt'
                    save(path,{'row':row,'condition':condition,'hidden':hidden,'codes':codes,'bytes':target,'codec_exact':True})
                    records.append({'index':i,'id':row['id'],'group':row['group'],'split':row['split'],
                        'source':row['source'],'distilled':bool(row.get('distilled')),'path':path.name,
                        'sha256':digest(path),'prompt_tokens':len(condition['ids']),'codec_exact':True,'reused':False})
                del features
            if number%100==0:persist()
    persist()
    write_json(out/'manifest.json',{'identity_sha256':digest(out/'identity.json'),'records':records,'rejected':rejected,
        'counts':dict(Counter(r['split'] for r in records)),'complete':True,'quality_ready':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','codec','selection','output','reuse','audit'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=1)
    p.add_argument('--maximum-batch-tokens',type=int,default=4096)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--reference',default='scripts/kindi_q4/kindi/reference-oxcoder-q4.json')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.batch<1 or args.maximum_batch_tokens<2048 or not 0<=args.shard<args.shards:p.error('Invalid batch or shard')
    run(args)

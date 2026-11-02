"""Add admitted creator responses to the frozen OxCoder conditioning inventory.

No new validation/test examples, tokenizer training, codec update or synthetic
teacher generation occurs here. Oversized prompts are skipped without truncation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

from .acquire_creator_corpora import archived_digest
from .admit_creator_corpora import read_rows
from .backbone import digest
from .conditioning import condition_chunk
from .data import chunks
from .prepare_geometry import CODE
from .train import save,write_json


def frozen_evaluation_hash(prepared):
    import hashlib
    rows=[(r,c) for r,c in prepared if r['split']!='train']
    return hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def add_response(original,groups,tokenizer,length,token_limit,maximum):
    if original['split']!='train':raise ValueError('Expansion contains nontraining examples')
    if groups.get(original['group'],'train')!='train':raise ValueError('Expansion crosses a frozen holdout group')
    parts=list(chunks(original['target'],length-1));eligible=[];memo={};rejected=0
    for index,part in enumerate(parts):
        row={k:original[k] for k in ('group','split','source')}
        row.update(id=f"{original['id']}:byte-{index}",target=part,distilled=False,
                   expansion='creator-v15b')
        c=condition_chunk(row,{original['id']:original},tokenizer,length,token_limit,memo)
        if c is None:rejected+=1;continue
        c['coding_response']=bool(CODE.search(original['target']))
        eligible.append((row,c))
    if len(eligible)>maximum:
        indices=sorted({round(i*(len(eligible)-1)/(maximum-1)) for i in range(maximum)}) if maximum>1 else [0]
        eligible=[eligible[i] for i in indices]
    return eligible,{'total_chunks':len(parts),'over_context':rejected,'selected_chunks':len(eligible)}


def prepare(a):
    base=Path(a.base);addition=Path(a.addition)
    bm=json.loads((base/'manifest.json').read_text());am=json.loads((addition/'manifest.json').read_text())
    if not am['admission_complete'] or am['targets_are_verified']:raise ValueError('Unexpected admission identity')
    if digest(base/'prepared.pt')!=bm['prepared_sha256']:raise ValueError('Base checksum differs')
    path=addition/'examples.jsonl.gz'
    if digest(path)!=am['examples_archive_sha256'] or archived_digest(path)[0]!=am['examples_uncompressed_sha256']:
        raise ValueError('Addition checksum differs')
    data=torch.load(base/'prepared.pt',weights_only=True,map_location='cpu')
    if data['data_manifest']['length']!=a.length:raise ValueError('Byte length differs from the base dataset')
    prepared=data['prepared'];before=frozen_evaluation_hash(prepared)
    groups={r['group']:r['split'] for r,c in prepared};known={r['id'] for r,c in prepared}
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    counts=Counter();base_count=len(prepared)
    for index,original in enumerate(read_rows(path)):
        rows,stats=add_response(original,groups,tokenizer,a.length,data['prompt_tokens'],a.chunks_per_response)
        counts['responses_considered']+=1
        for k,v in stats.items():counts[k]+=v
        if rows:counts['responses_selected']+=1
        for r,c in rows:
            if r['id'] in known:raise ValueError('Duplicate chunk ID in expansion')
            known.add(r['id']);prepared.append((r,c));groups[r['group']]='train'
            counts['source:'+r['source']]+=1
        if (index+1)%500==0:
            write_json(out/'progress.json',{'counts':dict(counts),'quality_ready':False})
            print(json.dumps({'responses':index+1,'counts':dict(counts)}),flush=True)
    if before!=frozen_evaluation_hash(prepared):raise ValueError('Frozen evaluation inventory changed')
    if not counts['selected_chunks']:raise ValueError('No training chunks admitted')
    save(out/'prepared.pt',data)
    report={'prepared_sha256':digest(out/'prepared.pt'),'base_prepared_sha256':bm['prepared_sha256'],
        'addition_manifest_sha256':digest(addition/'manifest.json'),'base_rows':base_count,
        'counts':dict(counts),'frozen_evaluation_sha256':before,'evaluation_changed':False,
        'context_tokens':data['prompt_tokens'],'byte_length':a.length,'chunks_per_response':a.chunks_per_response,
        'source_tokenizer_sha256':digest(Path(a.tokenizer)/'tokenizer.json'),
        'implementation_sha256':{n:digest(Path(__file__).parent/n) for n in
             ('prepare_creator_expansion.py','conditioning.py','data.py','distill.py','prepare_geometry.py')},
        'training_started':False,'quality_ready':False,
        'scope':'conditioning-ready addition only; frozen codec reconstruction admission is still required by the trainer'}
    write_json(out/'manifest.json',report);write_json(out/'progress.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','addition','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--tokenizer',default='results/tokenizer');p.add_argument('--length',type=int,default=128)
    p.add_argument('--chunks-per-response',type=int,default=8)
    a=p.parse_args()
    if a.length<2 or a.chunks_per_response<1:p.error('Invalid chunk bounds')
    prepare(a)

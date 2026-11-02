"""Prepare explicit UTF-8 chunks of the complete corpus plus cached teacher text."""
import argparse
import json
from pathlib import Path
from .data import chunks
from .backbone import digest
from .train import load_data,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--teacher',required=True);p.add_argument('--output',required=True)
    p.add_argument('--length',type=int,default=128);p.add_argument('--train-chunks',type=int,default=16)
    a=p.parse_args();rows,manifest=load_data(a.data);teacher,tm=load_data(a.teacher)
    groups={r['group']:r['split'] for r in rows}
    if any(r['split']!='train' or groups.get(r['group'])!='train' for r in teacher):raise ValueError('Teacher split mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=[];seen=set()
    # Evaluation chunks take precedence in deduplication; no shared target enters training.
    ordered=sorted(rows+teacher,key=lambda r:({'test':0,'validation':1,'train':2}[r['split']],r['id']))
    for r in ordered:
        parts=list(chunks(r['target'],a.length-1));indexes=set(range(len(parts)))
        if r['split']=='train' and len(parts)>a.train_chunks:
            indexes={round(i*(len(parts)-1)/(a.train_chunks-1)) for i in range(a.train_chunks)}
        previous=''
        for i,text in enumerate(parts):
            if i in indexes and text not in seen:
                item={k:v for k,v in r.items() if k!='messages'}
                item.update(id=r['id']+f':byte-{i}',target=text,chunked=True,
                    prompt=r['prompt']+'\nContinue this assistant response exactly:\n'+previous[-2048:])
                result.append(item);seen.add(text)
            previous+=text
    path=out/'examples.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    report={'format':'ncm-byte-followup-v3','length':a.length,'sources':manifest['sources'],
        'complete_corpus_sha256':manifest['examples_sha256'],'teacher_sha256':tm['examples_sha256'],
        'examples_sha256':digest(path),'counts':{s:sum(r['split']==s for r in result) for s in ['train','validation','test']},
        'teacher_chunks':sum(r.get('distilled',False) for r in result),
        'scope':'complete validation/test chunk coverage before dedup; spread bounded training chunks; fresh codec initialization',
        'targets_are_verified':False}
    write_json(out/'manifest.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

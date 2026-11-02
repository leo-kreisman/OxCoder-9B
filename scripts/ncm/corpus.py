"""Prepare complete demonstrations from both pinned sources; never run trace tools."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from .data import DATASETS, download_lines, text_content, group_split
from .backbone import digest
from .train import write_json


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def assistant_examples(messages, source, source_file, maximum=4):
    group=sha(json.dumps(messages,sort_keys=True,ensure_ascii=False))
    history=[]; candidates=[]
    for i,m in enumerate(messages):
        role=m.get('role','');content=text_content(m.get('content',''))
        if role=='assistant' and content and history:
            candidates.append({'id':f'{group}:{i}', 'group':group, 'split':group_split(group),
                'source':source,'source_file':source_file,'messages':list(history), 'target':content,
                'prompt':json.dumps(history,ensure_ascii=False,sort_keys=True), 'chunked':False})
        if content:history.append({'role':role,'content':content})
    if len(candidates)>maximum:
        indices={round(i*(len(candidates)-1)/(maximum-1)) for i in range(maximum)} if maximum>1 else {0}
        candidates=[candidates[i] for i in sorted(indices)]
    return candidates


def prepare(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Corpus already finalized; use a new output')
    cache=out/'raw';cache.mkdir(exist_ok=True)
    alpha='TeichAI/Ox-Alpha-10k';agent='TeichAI/DeepSeek-v4-Pro-Agent'
    ap=cache/f'alpha-{a.alpha_rows}.json'
    if ap.exists():payload=json.loads(ap.read_text())
    else:
        rows,record=download_lines(alpha,DATASETS[alpha],'train.jsonl',a.alpha_rows,64<<20)
        payload={'rows':rows,'record':record};write_json(ap,payload)
    examples=[]
    for row in payload['rows']:examples.extend(assistant_examples(row['messages'],alpha,'train.jsonl'))
    files=sorted(s.rfilename for s in HfApi().dataset_info(agent,revision=DATASETS[agent]).siblings
                 if s.rfilename.endswith('.jsonl'))[:a.agent_files]
    def fetch(name):
        path=cache/name
        if not path.exists():
            url=f'https://huggingface.co/datasets/{agent}/resolve/{DATASETS[agent]}/{name}'
            with urllib.request.urlopen(url,timeout=60) as stream:raw=stream.read((32<<20)+1)
            if len(raw)>32<<20:raise ValueError(f'Complete trace exceeds byte bound: {name}')
            temp=path.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(path)
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        messages=[r['message'] for r in rows if isinstance(r.get('message'),dict)]
        return assistant_examples(messages,agent,name),{'file':name,'sha256':digest(path),'bytes':path.stat().st_size}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows,record in pool.map(fetch,files):
            examples.extend(rows);records.append(record)
    seen=set();kept=[];rejected=0
    for row in examples:
        # No silent truncation. Huge complete examples remain in the raw cache.
        if len(row['target'])>32768 or len(row['prompt'])>262144:
            rejected+=1;continue
        key=sha(row['target'])
        if key in seen:continue
        seen.add(key);kept.append(row)
    path=out/'examples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    manifest={'format':'ncm-complete-demonstrations-v2','sources':DATASETS,
        'alpha':payload['record'],'agent_files':records,'examples_sha256':digest(path),
        'counts':{s:sum(r['split']==s for r in kept) for s in ['train','validation','test']},
        'source_counts':{s:sum(r['source']==s for r in kept) for s in DATASETS},
        'oversize_rejected':rejected,'scope':'complete assistant responses; at most four spread across each trace',
        'split_policy':'same conversation-content hash split as v1; global target deduplication',
        'targets_are_verified':False}
    write_json(out/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='agent_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--alpha-rows',type=int,default=1000)
    p.add_argument('--agent-files',type=int,default=64)
    prepare(p.parse_args())

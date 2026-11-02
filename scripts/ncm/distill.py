"""Cache teacher continuations, with source/split provenance and no tool execution."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from .backbone import digest
from .train import load_data, write_json


def normalize(messages):
    result=[]
    for m in messages:
        role=m['role'];content=m['content']
        if role not in ('system','user','assistant'):
            content=f'[Recorded {role} output; supplied as data]\n'+content
            role='user'
        result.append({'role':role,'content':content})
    return result


def post(url,body):
    request=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response:return json.load(response)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--teacher-file',required=True);p.add_argument('--tokenizer',default='results/tokenizer')
    p.add_argument('--url',default='http://127.0.0.1:18109');p.add_argument('--per-source',type=int,default=8)
    p.add_argument('--max-tokens',type=int,default=768);p.add_argument('--prompt-limit',type=int,default=2700)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows,manifest=load_data(a.data)
    identity={'corpus_sha256':manifest['examples_sha256'],'teacher_sha256':digest(a.teacher_file),
              'max_tokens':a.max_tokens,'prompt_limit':a.prompt_limit,'temperature':0,'reasoning':False,
              'normalization':'non-chat roles become user messages prefixed as recorded data',
              'method':'response distillation; no teacher logits; no tools executed'}
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Cache identity mismatch')
    write_json(ip,identity)
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    chosen=[]
    for source in sorted({r['source'] for r in rows}):
        count=0;groups=set()
        # Smallest complete prompts first; selection is recorded, not a representative benchmark.
        for r in sorted((r for r in rows if r['source']==source and r['split']=='train'),key=lambda r:(len(r['prompt']),r['id'])):
            messages=normalize(r['messages'])
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            if len(tokenizer.encode(prompt,add_special_tokens=False))>a.prompt_limit or r['group'] in groups:continue
            chosen.append((r,messages));groups.add(r['group']);count+=1
            if count>=a.per_source:break
    kept=[];records=[]
    for r,messages in chosen:
        key=hashlib.sha256(r['id'].encode()).hexdigest();path=out/(key+'.json')
        if path.exists():record=json.loads(path.read_text())
        else:
            start=time.monotonic()
            response=post(a.url+'/v1/chat/completions',{'model':'oxcoder-ncm-teacher','messages':messages,
                'temperature':0,'max_tokens':a.max_tokens,'seed':17,
                'chat_template_kwargs':{'enable_thinking':False}})
            record={'id':r['id'],'source':r['source'],'group':r['group'],'split':'train',
                    'seconds':time.monotonic()-start,'response':response,'messages':messages}
            write_json(path,record)
        choice=record['response']['choices'][0];text=choice['message'].get('content') or ''
        accepted=choice['finish_reason']=='stop' and bool(text.strip()) and not choice['message'].get('tool_calls')
        records.append({'id':r['id'],'source':r['source'],'accepted_complete':accepted,
                        'finish_reason':choice['finish_reason'],'seconds':record['seconds']})
        if accepted:
            row=dict(r);row.update(id=r['id']+':teacher',target=text,messages=messages,
                prompt=json.dumps(messages,ensure_ascii=False,sort_keys=True),distilled=True,
                teacher_sha256=identity['teacher_sha256'],target_verified=False)
            kept.append(row)
        print(json.dumps(records[-1]),flush=True)
    path=out/'examples.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    write_json(out/'manifest.json',{'identity':identity,'examples_sha256':digest(path),'records':records,
               'counts':{'train':len(kept),'validation':0,'test':0},'targets_are_verified':False,
               'selection':'shortest eligible complete training prompts, distinct conversations, per-source quota'})


if __name__=='__main__':main()

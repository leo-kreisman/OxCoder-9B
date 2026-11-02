"""Prepare untruncated longer-context conditioning and a coding-response tag."""
import argparse
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from .train import load_data,save,write_json
from .backbone import digest
from .conditioning import index_responses,select_conditioned

CODE=re.compile(r'```(?:python|py|javascript|js|typescript|ts|java|c|cpp|rust|go|ruby|php|bash|sh|sql|html|css|json)\b|(?:^|\n)\s*(?:def|class) \w+|"type": "toolCall"',re.I)


def main():
    p=argparse.ArgumentParser()
    for k in ['data','corpus','teacher','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--prompt-tokens',type=int,default=4096);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows,manifest=load_data(a.data);corpus,cm=load_data(a.corpus);teacher,tm=load_data(a.teacher)
    if cm['examples_sha256']!=manifest['complete_corpus_sha256'] or tm['examples_sha256']!=manifest['teacher_sha256']:
        raise ValueError('Conditioning source checksum mismatch')
    responses=index_responses(corpus,teacher)
    tokenizer=AutoTokenizer.from_pretrained('results/tokenizer',local_files_only=True,trust_remote_code=False)
    prepared,selection=select_conditioned(rows,responses,tokenizer,manifest['length'],a.prompt_tokens)
    compact=[];counts={}
    for row,c in prepared:
        row={k:row[k] for k in ['id','group','split','target','source','distilled'] if k in row}
        c['coding_response']=bool(CODE.search(responses[c['response_id']]['target']))
        key=f"{row['split']}|{row['source']}|code={c['coding_response']}"
        counts[key]=counts.get(key,0)+1
        compact.append((row,c))
    save(out/'prepared.pt',{'prepared':compact,'data_manifest':manifest,'prompt_tokens':a.prompt_tokens})
    write_json(out/'manifest.json',{'prepared_sha256':digest(out/'prepared.pt'),
        'source_dataset_sha256':manifest['examples_sha256'],'selection':selection,
        'classification_counts':counts,'quality_ready':False,
        'scope':'response-level code/tool syntax tag, not a correctness verifier; no prompt truncation'})
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()

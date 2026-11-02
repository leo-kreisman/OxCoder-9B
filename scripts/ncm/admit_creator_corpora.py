"""Admit a train-only creator-data expansion against frozen evaluation material.

This produces an additive corpus, not a replacement split. Source targets are
demonstrations, not correctness-verified programs. Downloaded code is not run.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
import urllib.request

from .acquire_creator_corpora import archived_digest
from .coding_eval import TASKS
from .open_corpus import NearDuplicates,file_sha,sha,shingles,tokens
from .train import write_json


def read_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def instruction_keys(prompt):
    try:messages=json.loads(prompt)
    except (ValueError,TypeError):return []
    if not isinstance(messages,list):return []
    return [sha(json.dumps([m['role'],m.get('content','')],sort_keys=True,ensure_ascii=False))
            for m in messages if isinstance(m,dict) and m.get('role') in ('system','developer')]


def prompt_text(prompt,shared_instructions):
    try:messages=json.loads(prompt)
    except (ValueError,TypeError):return prompt
    if not isinstance(messages,list):return prompt
    result=[]
    for m in messages:
        if not isinstance(m,dict):return prompt
        key=sha(json.dumps([m.get('role'),m.get('content','')],sort_keys=True,ensure_ascii=False))
        if m.get('role') in ('system','developer') and key in shared_instructions:continue
        content=m.get('content','')
        result.append(content if isinstance(content,str) else json.dumps(content,ensure_ascii=False))
    return '\n'.join(result)


def shared_instruction_inventory(paths):
    groups={}
    for path in paths:
        for row in read_rows(path):
            split='train' if row['split']=='train' else 'heldout'
            for key in instruction_keys(row.get('prompt','')):
                groups.setdefault(key,{'train':set(),'heldout':set()})[split].add(row['group'])
    return {key:{k:len(v) for k,v in value.items()} for key,value in groups.items()
            if len(value['train'])>=20 and len(value['heldout'])>=5}


class AdmissionFilter:
    def __init__(self,benchmark_hashes,shared_instructions=()):
        self.benchmarks=set(benchmark_hashes)
        self.protected=set()
        self.protected_prompts=set()
        self.shared_instructions=set(shared_instructions)
        self.groups={}
        self.exact=set()
        self.near=NearDuplicates()
        self.local_prompts=[' '.join(tokens(prompt)) for _,prompt,_ in TASKS]

    def register(self,row,protect=False):
        text=row.get('text',row.get('target',''))
        group=row['group'];split=row['split']
        if self.groups.setdefault(group,split)!=split:raise ValueError('Frozen group split conflict')
        self.exact.add(sha(text))
        self.near.duplicate(text)
        if protect:
            self.protected.update(shingles(text))
            # Protect the known prompt as well as the target. No model reads
            # these values; they are used only by the exclusion filter.
            self.protected_prompts.update(shingles(prompt_text(row.get('prompt',''),self.shared_instructions)))

    def consider(self,row):
        if row['split']!='train':return 'reserved_nontraining'
        if self.groups.get(row['group'],'train')!='train':return 'protected_group'
        text=row['target'];prompt=row.get('prompt','')
        if sha(text) in self.exact:return 'existing_exact_target'
        combined=' '.join(tokens(prompt+'\n'+text))
        if any(p in combined for p in self.local_prompts):return 'local_coding_task_prompt'
        if self.benchmarks.intersection(shingles(text)) or self.benchmarks.intersection(shingles(prompt)):
            return 'benchmark_or_holdout_shingle'
        if (self.protected.intersection(shingles(text)) or self.protected.intersection(shingles(prompt))
                or self.protected_prompts.intersection(shingles(text))
                or self.protected_prompts.intersection(shingles(prompt_text(prompt,self.shared_instructions)))):
            return 'benchmark_or_holdout_shingle'
        if self.near.duplicate(text):return 'near_duplicate'
        self.exact.add(sha(text));self.groups[row['group']]='train'
        return None


def benchmark_hashes(records,folder):
    import pyarrow.parquet as pq
    result=set()
    for index,r in enumerate(records):
        path=folder/f'benchmark-{index}.parquet'
        if not path.exists():
            url=f"https://huggingface.co/datasets/{r['dataset']}/resolve/{r['revision']}/{r['file']}"
            with urllib.request.urlopen(url,timeout=90) as stream:
                raw=stream.read(r['bytes']+1)
            if len(raw)!=r['bytes']:raise ValueError('Benchmark file size differs')
            tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(path)
        if file_sha(path)!=r['sha256']:raise ValueError('Benchmark checksum differs from frozen corpus')
        for row in pq.read_table(path).to_pylist():
            for key in ('prompt','canonical_solution','code','text','test'):
                if isinstance(row.get(key),str):result.update(shingles(row[key]))
            if 'canonical_solution' in row:result.update(shingles(row.get('prompt','')+row['canonical_solution']))
    for _,prompt,cases in TASKS:
        result.update(shingles(prompt,n=min(32,len(tokens(prompt)))))
    return result


def admit(a):
    reservoir=Path(a.reservoir);frozen=Path(a.frozen)
    rm=json.loads((reservoir/'manifest.json').read_text())
    fm=json.loads((frozen/'manifest.json').read_text())
    if not rm['sources_complete']:raise ValueError('Source acquisition is incomplete')
    rp=reservoir/'examples.jsonl.gz';fp=frozen/'documents.jsonl'
    if file_sha(rp)!=rm['examples_archive_sha256'] or archived_digest(rp)[0]!=rm['examples_uncompressed_sha256']:
        raise ValueError('Creator archive checksum differs')
    if file_sha(fp)!=fm['documents_sha256']:raise ValueError('Frozen corpus checksum differs')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(out).free<1<<30:raise RuntimeError('Keep 1 GiB free')
    inventory=shared_instruction_inventory([fp,rp])
    write_json(out/'shared-instructions.json',{'hashes':inventory,
        'scope':'identical system/developer messages in >=20 training and >=5 held-out groups; excluded only from held-out prompt comparison, never benchmark or target comparison'})
    filters=AdmissionFilter(benchmark_hashes(fm['benchmark_exclusions'],out),inventory)
    counts=Counter()
    for row in read_rows(fp):
        filters.register(row,protect=row['split']!='train')
        counts['frozen_'+row['split']]+=1
    # Reserve the new corpus's original validation/test groups before looking
    # at any train candidates. Iteration order cannot move them into training.
    for row in read_rows(rp):
        if row['split']!='train':filters.register(row,protect=True)
    write_json(out/'progress.json',{'phase':'filtering','counts':dict(counts),'quality_ready':False})
    destination=out/'examples.jsonl.gz';temp=out/'examples.jsonl.gz.tmp'
    with gzip.open(temp,'wt',encoding='utf-8',compresslevel=3) as f:
        for index,row in enumerate(read_rows(rp)):
            reason=filters.consider(row)
            if reason:counts['rejected_'+reason]+=1
            else:
                row={**row,'targets_are_verified':False,'admission':'creator-expansion-v15'}
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                counts['admitted']+=1;counts['source:'+row['source']]+=1
                counts['target_utf8_bytes']+=len(row['target'].encode())
            if (index+1)%1000==0:
                report={'phase':'filtering','processed':index+1,'counts':dict(counts),'quality_ready':False}
                write_json(out/'progress.json',report);print(json.dumps(report),flush=True)
    temp.replace(destination)
    digest,size=archived_digest(destination)
    report={'format':'creator-training-expansion-v15','counts':dict(counts),
        'reservoir_manifest_sha256':file_sha(reservoir/'manifest.json'),
        'frozen_manifest_sha256':file_sha(frozen/'manifest.json'),
        'examples_archive_sha256':file_sha(destination),'examples_uncompressed_sha256':digest,
        'examples_uncompressed_bytes':size,'source_revisions':rm['sources'],
        'benchmark_exclusions':fm['benchmark_exclusions'],
        'implementation_sha256':{name:file_sha(Path(__file__).parent/name) for name in
             ('admit_creator_corpora.py','open_corpus.py','coding_eval.py')},
        'filter':'frozen groups, exact targets, SimHash distance <=3, 32-token benchmark/holdout overlap in target and prompt; shared instruction wrappers excluded only from holdout prompt comparison',
        'shared_instructions_sha256':file_sha(out/'shared-instructions.json'),
        'split':'train only; original and newly reserved evaluation groups remain excluded',
        'limitations':'Conservative lexical filters can reject unrelated examples and cannot prove semantic decontamination.',
        'admission_complete':True,'used_in_training':False,'targets_are_verified':False,'quality_ready':False}
    write_json(out/'manifest.json',report);write_json(out/'progress.json',report)
    print(json.dumps({'admission_complete':True,'counts':dict(counts)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('reservoir','frozen','output'):parser.add_argument('--'+key,required=True)
    admit(parser.parse_args())

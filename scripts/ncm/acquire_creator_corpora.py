"""Archive both complete pinned creator sources and extract bounded responses.

Source text is never executed. Full downloaded sources are gzip archived;
response-size filtering applies only to the extracted training reservoir.
This reservoir is not admitted to existing training/evaluation splits without
the later contamination and provenance checks described in the documentation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi

from .corpus import assistant_examples
from .data import DATASETS
from .open_corpus import file_sha
from .train import write_json


def archived_digest(path):
    h=hashlib.sha256();size=0
    with gzip.open(path,'rb') as f:
        for piece in iter(lambda:f.read(1<<20),b''):
            h.update(piece);size+=len(piece)
    return h.hexdigest(),size


def fetch_once(item,root):
    source,revision,name,size=item
    if Path(name).name!=name:
        raise ValueError('Only flat, reviewed source paths are accepted')
    folder=root/('alpha' if source.endswith('Ox-Alpha-10k') else 'agent')
    folder.mkdir(exist_ok=True)
    path=folder/(name+'.gz')
    meta=folder/(name+'.metadata.json')
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        digest,actual=archived_digest(path)
        if (record['source']!=source or record['revision']!=revision or record['file']!=name
                or actual!=size or record['sha256']!=digest or file_sha(path)!=record['archive_sha256']):
            raise ValueError('Cached source identity mismatch')
        return path,record
    if shutil.disk_usage(root).free<(1536<<20):
        raise RuntimeError('Keep 1.5 GiB free while acquiring sources')
    limit=160<<20 if source.endswith('Ox-Alpha-10k') else 32<<20
    if size is None or size>limit:
        raise ValueError('Source exceeds reviewed per-file size bound')
    h=hashlib.sha256();actual=0
    temp=path.with_suffix('.tmp')
    url=f'https://huggingface.co/datasets/{source}/resolve/{revision}/{name}'
    with urllib.request.urlopen(url,timeout=90) as response, gzip.open(temp,'wb',compresslevel=3) as f:
        for piece in iter(lambda:response.read(1<<20),b''):
            actual+=len(piece)
            if actual>limit or actual>size:
                raise ValueError('Downloaded source exceeds metadata bound')
            h.update(piece);f.write(piece)
    if actual!=size:
        raise ValueError('Incomplete source download')
    temp.replace(path)
    record={'source':source,'revision':revision,'file':name,'bytes':actual,
            'sha256':h.hexdigest(),'archive_sha256':file_sha(path)}
    write_json(meta,record)
    return path,record


def fetch(item,root):
    for attempt in range(6):
        try:
            return fetch_once(item,root)
        except (urllib.error.URLError,TimeoutError) as exc:
            if attempt==5 or (isinstance(exc,urllib.error.HTTPError) and exc.code not in (408,429,499,500,502,503,504)):
                raise
            time.sleep(min(30,2**(attempt+1)))


def extract(path,record):
    source=record['source']
    if source.endswith('Ox-Alpha-10k'):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row=json.loads(line)
                    yield from assistant_examples(row['messages'],source,record['file'])
    else:
        with gzip.open(path,'rt',encoding='utf-8') as f:
            rows=[json.loads(line) for line in f if line.strip()]
        messages=[r['message'] for r in rows if isinstance(r.get('message'),dict)]
        yield from assistant_examples(messages,source,record['file'])


def acquire(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'.acquisition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'manifest.json').exists():
            raise ValueError('Finalized corpus exists; use its manifest')
        raw=out/'raw';raw.mkdir(exist_ok=True)
        items=[]
        for source,revision in DATASETS.items():
            info=HfApi().dataset_info(source,revision=revision,files_metadata=True)
            files=sorted((f for f in info.siblings if f.rfilename.endswith('.jsonl')),key=lambda f:f.rfilename)
            items.extend((source,revision,f.rfilename,f.size) for f in files)
        write_json(out/'source-plan.json',{'sources':DATASETS,'files':items,
            'expected_uncompressed_bytes':sum(i[3] for i in items),'extractor_sha256':file_sha(__file__),
            'download_workers':getattr(a,'workers',4)})
        stats=Counter();seen=set();records=[]
        destination=out/'examples.jsonl.gz'
        temp=out/'examples.jsonl.gz.tmp'
        with gzip.open(temp,'wt',encoding='utf-8',compresslevel=3) as stream, ThreadPoolExecutor(max_workers=getattr(a,'workers',4)) as pool:
            # Bounded submission prevents thousands of completed futures from
            # accumulating while a preceding download is slow.
            for start in range(0,len(items),32):
                for path,record in pool.map(lambda item:fetch(item,raw),items[start:start+32]):
                    records.append(record)
                    for row in extract(path,record):
                        stats['candidate_responses']+=1
                        if len(row['target'])>32768 or len(row['prompt'])>262144:
                            stats['oversize_rejected']+=1;continue
                        h=hashlib.sha256(row['target'].encode()).hexdigest()
                        if h in seen:
                            stats['duplicate_targets_rejected']+=1;continue
                        seen.add(h)
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        stats[row['source']]+=1
                        stats[row['split']+'_documents']+=1
                    if len(records)%100==0:
                        stream.flush()
                        report={'files_completed':len(records),'files_planned':len(items),'counts':dict(stats),'quality_ready':False}
                        write_json(out/'progress.json',report);print(report,flush=True)
                if shutil.disk_usage(out).free<(1536<<20):
                    raise RuntimeError('Disk reserve reached; archived sources retained for resumption')
        temp.replace(destination)
        digest,size=archived_digest(destination)
        manifest={'format':'complete-creator-source-reservoir-v1','sources':DATASETS,
            'source_records':records,'sources_complete':len(records)==len(items),
            'counts':dict(stats),'examples_uncompressed_sha256':digest,'examples_uncompressed_bytes':size,
            'examples_archive_sha256':file_sha(destination),'extractor_sha256':file_sha(__file__),
            'selection':'complete targets; at most four spread across each conversation; explicit size bounds',
            'split_policy':'same content-hash conversation splits as earlier creator corpus',
            'targets_are_verified':False,'benchmark_decontamination_pending':True,
            'admitted_to_training':False,'quality_ready':False}
        write_json(out/'manifest.json',manifest)
        write_json(out/'progress.json',{'files_completed':len(records),'files_planned':len(items),
            'counts':dict(stats),'acquisition_complete':True,'admitted_to_training':False,'quality_ready':False})
        print('Source acquisition complete:',len(records),'files',dict(stats),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=16:
        parser.error('Workers must be between one and sixteen')
    acquire(args)

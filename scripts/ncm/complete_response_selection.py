"""Prepare full, short training responses without using evaluation predictions.

All chunks of an eligible response remain together. Eligibility depends only
on existing split/provenance, contiguous chunks and a fixed length bound. Where
a checked teacher response is available, prefer it over the original answer.
The frozen 64 validation windows are copied unchanged.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .context_features import unit_hash


def select(prepared, maximum_chunks=8):
    if maximum_chunks < 1: raise ValueError('Positive response length required')
    responses, splits, ids = defaultdict(list), {}, set()
    for row, condition in prepared:
        if row['id'] in ids: raise ValueError('Duplicate chunk identity')
        ids.add(row['id'])
        if splits.setdefault(row['group'],row['split']) != row['split']:
            raise ValueError('Conversation crosses splits')
        if row['split']=='train': responses[condition['response_id']].append((row,condition))
    eligible = []
    for entries in responses.values():
        entries.sort(key=lambda item:item[1]['chunk_index'])
        if (len(entries)<=maximum_chunks and entries[-1][1]['is_final_chunk']
                and [c['chunk_index'] for _,c in entries]==list(range(len(entries)))):
            if len({r['group'] for r,_ in entries})!=1:raise ValueError('Response crosses conversations')
            eligible.append(entries)
    teacher_groups={rows[0][0]['group'] for rows in eligible if rows[0][0].get('distilled')}
    return [item for rows in eligible if rows[0][0].get('distilled') or rows[0][0]['group'] not in teacher_groups
            for item in rows]


def prepare(a):
    import torch
    from .backbone import digest
    from .prepared_reader import load
    from .distillation_curriculum import filter_teacher_actions
    from .train import save,write_json
    metadata=json.loads((Path(a.data)/'manifest.json').read_text())
    data=load(a.json_cache,metadata['prepared_sha256'])
    filtered,removed=filter_teacher_actions(data['prepared'],metadata)
    training=select(filtered,a.maximum_chunks)
    old=Path(a.validation); oldmeta=json.loads((old/'manifest.json').read_text())
    if (oldmeta['prepared_sha256']!=metadata['prepared_sha256'] or
            digest(old/'selection.pt')!=oldmeta['selection_sha256']):raise ValueError('Validation changed')
    validation=torch.load(old/'selection.pt',weights_only=True,map_location='cpu')['prepared']
    if (len(validation)!=64 or any(r['split']!='validation' for r,_ in validation) or
            {r['group'] for r,_ in training}&{r['group'] for r,_ in validation}):raise ValueError('Invalid split')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    selected=sorted(training+validation,key=lambda x:unit_hash(f"{a.seed}:order:{x[0]['id']}"))
    save(out/'selection.pt',{'prepared':selected})
    report={'prepared_sha256':metadata['prepared_sha256'],'selection_sha256':digest(out/'selection.pt'),
        'validation_selection_sha256':oldmeta['selection_sha256'],'seed':a.seed,
        'maximum_response_chunks':a.maximum_chunks,'counts':dict(Counter(r['split'] for r,_ in selected)),
        'training_sources':dict(Counter(r['source'] for r,_ in training)),
        'training_responses':len({c['response_id'] for _,c in training}),
        'training_distilled':sum(bool(r.get('distilled')) for r,_ in training),
        'training_first_chunks':sum(c['chunk_index']==0 for _,c in training),
        'training_final_chunks':sum(c['is_final_chunk'] for _,c in training),
        'feature_storage_bytes_before_reuse':sum(len(c['ids'])*8192 for _,c in selected),
        'teacher_filter':removed,'implementation_sha256':digest(__file__),
        'scope':'Complete responses at most eight byte chunks; evaluation unchanged; not a quality result',
        'quality_ready':False}
    if len(report['training_sources'])!=3 or not report['training_distilled']:raise ValueError('Lost source or distillation')
    write_json(out/'manifest.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','json-cache','validation','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--maximum-chunks',type=int,default=8);p.add_argument('--seed',type=int,default=27001)
    prepare(p.parse_args())

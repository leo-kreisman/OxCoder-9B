"""Drop an entire training response if any selected chunk failed codec screening.

Validation is retained verbatim. This runs after ordinary shard completeness
checks and preserves the original merged cache as provenance.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def retain_complete(records, prepared):
    response_ids=defaultdict(set)
    for row,condition in prepared:
        if row['split']=='train':response_ids[condition['response_id']].add(row['id'])
    available={r['id'] for r in records if r['split']=='train'}
    removed={name:sorted(ids-available) for name,ids in response_ids.items() if not ids<=available}
    admitted=set().union(*(ids for name,ids in response_ids.items() if name not in removed))
    return [r for r in records if r['split']=='validation' or r['id'] in admitted],removed


def run(a):
    import torch
    from .backbone import digest
    from .train import write_json
    original=Path(a.input);report=json.loads(original.read_text());selection=Path(a.selection)
    if (not report.get('complete') or digest(selection/'selection.pt')!=report['identity']['selection_sha256']):
        raise ValueError('Incomplete cache or changed selection')
    prepared=torch.load(selection/'selection.pt',weights_only=True,map_location='cpu')['prepared']
    records,removed=retain_complete(report['records'],prepared)
    # Record paths are relative to the original manifest. Keep that resolution
    # when the derived manifest is written elsewhere.
    import os
    out=Path(a.output).resolve()
    records=[{**r,'path':os.path.relpath((original.parent/r['path']).resolve(),out.parent)} for r in records]
    old_validation=[r['id'] for r in report['records'] if r['split']=='validation']
    if [r['id'] for r in records if r['split']=='validation']!=old_validation or len(old_validation)!=64:
        raise ValueError('Validation changed')
    if {r['source'] for r in records if r['split']=='train'}!={r['source'] for r in report['records'] if r['split']=='train'}:
        raise ValueError('Source lost')
    if not any(r.get('distilled') for r in records if r['split']=='train'):raise ValueError('Distillation lost')
    report={**report,'identity':{**report['identity'],'complete_training_responses_only':True},
        'parent_feature_manifest_sha256':digest(original),'parent_feature_manifest':str(original.resolve()),
        'records':records,'counts':dict(Counter(r['split'] for r in records)),
        'incomplete_training_responses_excluded':removed,
        'scope':'All surviving training responses complete after codec screening; validation unchanged','quality_ready':False}
    if out.exists():raise FileExistsError(out)
    write_json(out,report);print(json.dumps({'counts':report['counts'],'excluded_responses':len(removed)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','selection','output'):p.add_argument('--'+key,required=True)
    run(p.parse_args())

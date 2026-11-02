"""Select shorter complete prepared prompts without truncating any example."""
import argparse
import collections
import json
from pathlib import Path
import torch
from .backbone import digest
from .train import save,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prompt-tokens',type=int,required=True);a=p.parse_args()
    root=Path(a.data);manifest=json.loads((root/'manifest.json').read_text())
    parent_sha=digest(root/'prepared.pt')
    if parent_sha!=manifest['prepared_sha256']:raise ValueError('Parent prepared checksum differs')
    data=torch.load(root/'prepared.pt',map_location='cpu',weights_only=True)
    if not 0<a.prompt_tokens<=data['prompt_tokens']:raise ValueError('Restriction must not expand the parent limit')
    kept=[(r,c) for r,c in data['prepared'] if len(c['ids'])<=a.prompt_tokens]
    expected={r['source'] for r,c in data['prepared'] if r['split']=='train'}
    if {r['source'] for r,c in kept if r['split']=='train'}!=expected:raise ValueError('Restriction removed an entire training source')
    counts=collections.Counter(r['split'] for r,c in kept)
    if any(counts[s]==0 for s in ['train','validation','test']):raise ValueError('Restriction removed an entire split')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data={**data,'prepared':kept,'prompt_tokens':a.prompt_tokens}
    save(out/'prepared.pt',data)
    report={'prepared_sha256':digest(out/'prepared.pt'),'parent_prepared_sha256':parent_sha,
            'source_dataset_sha256':data['data_manifest']['examples_sha256'],
            'selection':{'prompt_tokens':a.prompt_tokens,'truncation':False,'counts':dict(counts),
                'source_training_counts':dict(collections.Counter(r['source'] for r,c in kept if r['split']=='train'))},
            'quality_ready':False,'scope':'complete prompts selected by length; no token or target truncation'}
    write_json(out/'manifest.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

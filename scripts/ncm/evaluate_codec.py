"""Evaluate a frozen codec; never use the test split to select checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from .codec import Codec, CodecConfig
from .backbone import digest
from .train import load_data, reconstruction, write_json, admission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    if Path(a.output).exists():
        p.error('Evaluation output already exists; retain the prior result')
    torch.set_num_threads(2)
    admission([a.device], 4)
    rows, manifest = load_data(a.data)
    ck = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    if ck['identity']['data_sha256'] != manifest['examples_sha256']:
        raise ValueError('Dataset identity differs from training')
    model = Codec(CodecConfig(**ck['config'])).to(a.device)
    model.load_state_dict(ck['model'])
    report = dict(reconstruction(model, [r for r in rows if r['split'] == a.split], a.device),
        split=a.split, checkpoint_step=ck['step'], checkpoint_sha256=digest(a.checkpoint),
        data_sha256=manifest['examples_sha256'], quality_ready=False)
    write_json(a.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

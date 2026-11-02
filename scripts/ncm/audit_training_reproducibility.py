"""Compare two independently loaded, unchanged BF16 training references."""
import argparse
import json
from pathlib import Path

import torch

from .audit_packed_training import compare_groups
from .audit_training_runtime import compare
from .backbone import digest
from .train import write_json


def run(a):
    first, second = Path(a.first), Path(a.second)
    left, right = [json.loads((p / 'report.json').read_text()) for p in (first, second)]
    for key in ('source_checkpoint_sha256', 'prepared_sha256', 'model_identity'):
        if left[key] != right[key]:
            raise ValueError('Reference runs differ: ' + key)
    if not left['complete'] or not right['complete']:
        raise ValueError('Reference runs incomplete')
    for name in ('backbone.py', 'shared_training.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py'):
        if left['implementation_sha256'][name] != right['implementation_sha256'][name]:
            raise ValueError('Reference implementation changed')
    report = {key: left[key] for key in ('source_checkpoint_sha256', 'prepared_sha256', 'model_identity')}
    report.update(complete=False, quality_ready=False, records=[],
                  scope='Unchanged resident BF16 backend; two independent loads; no compact weights in compared tensors',
                  implementation_sha256=left['implementation_sha256'],
                  input_report_sha256=[digest(p / 'report.json') for p in (first, second)])
    for length in (64, 69, 512, 2048):
        paths = [p / f'reference-{length}.pt' for p in (first, second)]
        x, y = [torch.load(p, weights_only=True, map_location='cpu') for p in paths]
        for key in ('outputs', 'gradients'):
            if set(x[key]) != set(y[key]):
                raise ValueError('Reference inventory differs')
        g, l = compare(x['gradients'], y['gradients']), compare(x['outputs'], y['outputs'])
        groups = compare_groups(x['gradients'], y['gradients'])
        exact = all(torch.equal(v, y['outputs'][n]) for n, v in x['outputs'].items())
        report['records'].append({'prompt_tokens': length, 'file_sha256': [digest(p) for p in paths],
            'gradient_comparison': g, 'logit_comparison': l, 'gradient_groups': groups,
            'bit_exact_outputs': exact,
            'bit_exact_gradients': all(torch.equal(v, y['gradients'][n]) for n, v in x['gradients'].items()),
            'passed': exact and g['finite'] and g['relative_l2'] < .05 and g['cosine'] > .99 and
                all(v['finite'] and v['relative_l2'] < .05 and v['cosine'] > .99 for v in groups.values())})
    report['complete'] = True
    report['passed'] = all(r['passed'] for r in report['records'])
    write_json(a.output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('first', 'second', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())

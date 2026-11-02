import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.ncm.backbone import digest
from scripts.ncm.codec import Codec, CodecConfig, encode
from scripts.ncm import train_context_decoder as trainer
from scripts.ncm import train_context_residual as residual


class InterruptedForTest(Exception): pass


class ResidualResumeTests(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted_model_optimizer_and_sampler(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.manual_seed(1)
            codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=0, codes=17, levels=(1, 3, 8)))
            codec_path = root / 'codec.pt'
            torch.save({'config': codec.metadata(), 'model': codec.state_dict()}, codec_path)
            records = []
            for index in range(68):
                split = 'train' if index < 4 else 'validation'
                row = {'id': str(index), 'group': str(index), 'split': split, 'target': 'x',
                       'source': 'synthetic-test-only', 'distilled': False}
                condition = {'ids': [1, 2], 'is_final_chunk': index % 2 == 0}
                # Synthetic feature labels test the trainer, not codec quality
                # or any real model's generation ability.
                path = root / f'{index}.pt'
                torch.save({'row': row, 'condition': condition,
                            'hidden': torch.randn(2, 4096).bfloat16(),
                            'codes': [torch.randint(17, (n,)) for n in (1, 3, 8)],
                            'bytes': torch.tensor(encode('x', 8)), 'codec_exact': True}, path)
                records.append({**{k: row[k] for k in ('id', 'group', 'split')}, 'path': path.name,
                                'sha256': digest(path), 'prompt_tokens': 2, 'codec_exact': True})
            manifest = root / 'features.json'
            manifest.write_text(json.dumps({'complete': True, 'records': records,
                'identity': {'codec_sha256': digest(codec_path), 'source_sha256': 'synthetic-unloadable-source',
                             'selection_sha256': 'synthetic', 'prepared_sha256': 'synthetic'}}))
            def args(name, resume=False):
                return argparse.Namespace(cache=str(manifest), codec=str(codec_path), output=str(root / name),
                    device='cpu', steps=6, batch=2, microbatch=1, save_every=2, eval_every=6,
                    warmup=1, seed=25001, lr=.001, corruption=0., width=32, heads=4, layers=2, resume=resume)
            original_save = trainer.save
            def parent_boundary(path, value):
                original_save(path, value)
                if Path(path).name == 'latest.pt' and value['step'] == 2:
                    raise InterruptedForTest()
            def branch_args(name, resume=False):
                result = args(name, resume)
                result.init = str(root / 'parent/latest.pt')
                result.on_policy = .5
                return result
            def branch_boundary(path, value):
                original_save(path, value)
                if Path(path).name == 'latest.pt' and value['step'] == 4:
                    raise InterruptedForTest()
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(trainer, 'save', parent_boundary):
                    with self.assertRaises(InterruptedForTest): trainer.run(args('parent'))
                residual.run(branch_args('full'))
                with patch.object(residual, 'save', branch_boundary):
                    with self.assertRaises(InterruptedForTest): residual.run(branch_args('resumed'))
                residual.run(branch_args('resumed', True))
            one = torch.load(root / 'full/latest.pt', weights_only=True)
            two = torch.load(root / 'resumed/latest.pt', weights_only=True)
            def equal(a, b):
                if isinstance(a, torch.Tensor): self.assertTrue(torch.equal(a, b))
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a: equal(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b): equal(x, y)
                else: self.assertEqual(a, b)
            equal(one, two)
            self.assertEqual(one['identity']['parent_step'], 2)
            self.assertEqual(one['identity']['parent_sha256'], digest(root / 'parent/latest.pt'))
            self.assertEqual(one['step'], 6)
            report = json.loads((root / 'resumed/validation-6.json').read_text())
            self.assertEqual(report['windows'], 64)
            self.assertFalse(report['quality_ready'])


if __name__ == '__main__': unittest.main()

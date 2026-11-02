import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.ncm.compact_training_policy import logical_gradient_groups, packs, qualified, prefix_policy, placement_policy, storage_policy


class CompactPolicyTests(unittest.TestCase):
    def test_relocation_preserves_reference_gradient_clipping(self):
        names = ['position', 'project.weight', 'level.weight',
                 'backbone.adapters.0.down.weight', 'backbone.adapters.11.up.weight',
                 'backbone.adapters.12.down.weight', 'backbone.adapters.31.up.weight',
                 'head.weight', 'stop_head.bias']
        parameters = [torch.nn.Parameter(torch.ones(2)) for _ in names]
        for i, p in enumerate(parameters):
            p.grad = torch.tensor([i + 1., -i - .5])
        reference = copy.deepcopy(parameters)
        for p, q in zip(parameters, reference):
            q.grad = p.grad.clone()
        expected = [reference[:5], reference[5:]]
        actual = logical_gradient_groups(list(zip(names, parameters)), [12, 20])
        for group in expected + actual:
            torch.nn.utils.clip_grad_norm_(group, 1., foreach=False)
        for p, q in zip(parameters, reference):
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_unknown_parameter_and_bad_split_are_rejected(self):
        with self.assertRaises(ValueError):
            logical_gradient_groups([('surprise.weight', torch.ones(1))], [12, 20])
        with self.assertRaises(ValueError):
            logical_gradient_groups([], [12, 19])

    def test_packing_preserves_order_and_short_example_boundaries(self):
        lengths = [64, 69, 40, 128, 256, 2048, 64]
        examples = [{'id': i, 'prompt': torch.zeros(1, n)} for i, n in enumerate(lengths)]
        result = list(packs(examples, 800, [4, 16, 64, 128]))
        self.assertEqual([[x['id'] for x in group] for group in result], [[0, 1], [2], [3], [4], [5], [6]])
        self.assertEqual([x['id'] for group in result for x in group], list(range(7)))
        for group in result:
            if len(group) > 1:
                self.assertLessEqual(sum(x['prompt'].shape[1] + 212 for x in group), 800)

    def test_compact_weight_admission_requires_bit_exact_complete_model(self):
        identity = {name: name for name in ('source_sha256', 'codec_sha256', 'rank', 'head_kind',
                                            'concept_input_norm', 'layer_split')}
        implementation = {name: 'hash' for name in ('gguf_training_weights.py', 'gguf_training_backbone.py',
            'backbone.py', 'shared_training.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py')}
        comparison = {'finite': True, 'relative_l2': 0., 'cosine': 1.}
        report = {'complete': True, 'prepared_sha256': 'data', 'model_identity': identity,
            'implementation_sha256': implementation, 'records': [
                {'compact_weights': True, 'requested_prompt_tokens': length, 'passed': True,
                 'gradient_comparison': comparison, 'logit_comparison': comparison,
                 'gradient_groups': {name: comparison for name in ('adapters', 'classifiers', 'interfaces')},
                 'bit_exact_outputs': True, 'bit_exact_gradients': True}
                for length in (64, 69, 512, 2048)]}
        with tempfile.TemporaryDirectory() as d, patch('scripts.ncm.compact_training_policy.digest', return_value='hash'):
            file = Path(d) / 'audit.json'
            file.write_text(json.dumps(report))
            self.assertFalse(qualified(file, identity, 'data')['packing'])
            report['records'][0]['bit_exact_gradients'] = False
            file.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'bit-exact'):
                qualified(file, identity, 'data')
            report['records'][0]['bit_exact_gradients'] = True
            report['records'][-1]['requested_prompt_tokens'] = 512
            file.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                qualified(file, identity, 'data')
            report['records'][-1]['requested_prompt_tokens'] = 2048
            report['source_checkpoint_sha256'] = 'checkpoint'
            for record in report['records']:
                record['devices'] = ['cuda:0', 'cuda:1']
            file.write_text(json.dumps(report))
            relocated = copy.deepcopy(report)
            for record in relocated['records']:
                record['devices'] = ['cuda:1']
                record['bit_exact_gradients'] = False
            moved = Path(d) / 'placement.json'
            moved.write_text(json.dumps(relocated))
            self.assertIn('relocation', placement_policy(file, moved, identity, 'data'))
            report['records'][0]['bit_exact_gradients'] = False
            file.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'bit-exact'):
                placement_policy(file, moved, identity, 'data')

    def test_failed_long_prefix_does_not_disable_qualified_short_path_or_weaken_gate(self):
        comparison = {'finite': True, 'relative_l2': 0., 'cosine': 1.}
        records = [{'requested_prompt_tokens': n, 'passed': True,
                    'runtime': {'shared_prefix_training': True, 'keep_per_device': 0, 'convolution': 'reference'},
                    'gradient_comparison': comparison, 'logit_comparison': comparison,
                    'gradient_groups': {k: dict(comparison) for k in ('adapters', 'classifiers', 'interfaces')}}
                   for n in (64, 69, 512, 2048)]
        with tempfile.TemporaryDirectory() as d, patch('scripts.ncm.compact_training_policy.validate_audit', return_value={}):
            file = Path(d) / 'audit.json'
            file.write_text(json.dumps({'records': records}))
            self.assertEqual(prefix_policy(file, {}, 'data')['maximum_shared_prefix'], 2048)
            records[-1]['gradient_groups']['interfaces']['relative_l2'] = .051
            file.write_text(json.dumps({'records': records}))
            self.assertEqual(prefix_policy(file, {}, 'data')['maximum_shared_prefix'], 512)
            records[0]['passed'] = False
            file.write_text(json.dumps({'records': records}))
            self.assertEqual(prefix_policy(file, {}, 'data')['maximum_shared_prefix'], 0)

    def test_packing_respects_qualified_prefix_limit(self):
        examples = [{'prompt': torch.zeros(1, n)} for n in (64, 1024, 69)]
        self.assertEqual([len(g) for g in packs(examples, 4000, [4, 16, 64, 128], 512)], [1, 1, 1])

    def test_reference_rounding_never_waives_matrix_exactness_or_gradient_limits(self):
        identity = {name: name for name in ('source_sha256', 'codec_sha256', 'rank', 'head_kind',
                                            'concept_input_norm', 'layer_split')}
        comparison = {'finite': True, 'relative_l2': .003, 'cosine': .99999}
        impl = {name: 'hash' for name in ('gguf_training_weights.py', 'gguf_training_backbone.py',
                 'backbone.py', 'shared_training.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py')}
        baseline = {'complete': True, 'passed': True, 'prepared_sha256': 'data', 'model_identity': identity,
            'source_checkpoint_sha256': 'checkpoint', 'implementation_sha256': impl, 'records': [
                {'prompt_tokens': n, 'passed': True, 'bit_exact_outputs': True, 'bit_exact_gradients': False,
                 'gradient_comparison': dict(comparison),
                 'gradient_groups': {k: dict(comparison) for k in ('adapters', 'classifiers', 'interfaces')}}
                for n in (64, 69, 512, 2048)]}
        runtime = copy.deepcopy(baseline)
        for row in runtime['records']:
            row.update(compact_weights=True, devices=['cuda:0', 'cuda:1'],
                       requested_prompt_tokens=row['prompt_tokens'],
                       logit_comparison={'finite': True, 'relative_l2': 0., 'cosine': 1.})
        matrices = {'complete': True, 'passed': True, 'all_linear_layers': True,
            'source_sha256': identity['source_sha256'], 'implementation_sha256': 'hash',
            'expected_matrices': 248, 'records': [
                {'name': f'blk.{layer}.matrix{j}', 'passed': True, 'unpacked_weight_bit_exact': True,
                 'forward_bit_exact': True, 'input_gradient_bit_exact': True}
                for layer in range(32) for j in range(7 if layer % 4 == 3 else 8)]}
        with tempfile.TemporaryDirectory() as d, patch('scripts.ncm.compact_training_policy.digest', return_value='hash'):
            m, r, b = [Path(d) / name for name in ('matrix.json', 'runtime.json', 'baseline.json')]
            m.write_text(json.dumps(matrices));r.write_text(json.dumps(runtime));b.write_text(json.dumps(baseline))
            self.assertIn('matrix_audit_sha256', storage_policy(m, r, b, identity, 'data'))
            matrices['records'][0]['input_gradient_bit_exact'] = False
            m.write_text(json.dumps(matrices))
            with self.assertRaisesRegex(ValueError, 'linear matrices'):
                storage_policy(m, r, b, identity, 'data')
            matrices['records'][0]['input_gradient_bit_exact'] = True
            m.write_text(json.dumps(matrices))
            runtime['records'][0]['gradient_comparison']['relative_l2'] = .06
            r.write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, 'gradient gate'):
                storage_policy(m, r, b, identity, 'data')


if __name__ == '__main__':
    unittest.main()

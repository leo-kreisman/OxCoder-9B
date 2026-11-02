import copy
import unittest

import torch

from scripts.ncm.change_body_lr import branch


class LearningRateBranchTests(unittest.TestCase):
    def fixture(self):
        state = lambda lr: {'param_groups': [{'lr': lr, 'params': [0], 'weight_decay': .01}],
                            'state': {0: {'step': torch.tensor(350.), 'momentum': torch.arange(3.)}}}
        return {'identity': {'body_lr': 3e-5, 'head_lr': 3e-4, 'optimizer': {'body_lr': 3e-5}},
                'optimizer': {'format': 'mixed-optimizers-v1',
                    'classes': ['AdamW', 'AdamW', 'FP32MasterMuon'],
                    'states': [state(3e-4), state(3e-5), {'format': 'fp32-master-muon-v1',
                        'masters': [torch.arange(3.)], 'inner': state(3e-5)}]},
                'step': 350, 'model': {'weight': torch.ones(3)},
                'rng': {'python': (3, (1, 2), None), 'torch': torch.arange(3, dtype=torch.uint8)},
                'history': [{'step': 350, 'loss': 4.}]}

    def test_only_declared_learning_rate_fields_change(self):
        original = self.fixture()
        changed = branch(original, 1e-4, 'immutable-parent')
        changed['identity'].pop('learning_rate_transition')
        changed['identity']['body_lr'] = 3e-5
        changed['identity']['optimizer']['body_lr'] = 3e-5
        changed['optimizer']['states'][1]['param_groups'][0]['lr'] = 3e-5
        changed['optimizer']['states'][2]['inner']['param_groups'][0]['lr'] = 3e-5

        def equal(left, right):
            self.assertIs(type(left), type(right))
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
                self.assertEqual(left.dtype, right.dtype)
                self.assertNotEqual(left.data_ptr(), right.data_ptr())
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left: equal(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right): equal(a, b)
            else:
                self.assertEqual(left, right)
        equal(original, changed)

    def test_rejects_ambiguous_or_invalid_changes(self):
        for rate in [0., -1., float('nan'), float('inf'), 3e-5]:
            with self.assertRaises(ValueError): branch(self.fixture(), rate, 'parent')
        for broken in ['class', 'master', 'head_lr', 'interface_lr', 'muon_lr', 'identity']:
            value = copy.deepcopy(self.fixture())
            if broken == 'class': value['optimizer']['classes'][2] = 'Muon'
            elif broken == 'master': value['optimizer']['states'][2]['format'] = 'legacy'
            elif broken == 'head_lr': value['optimizer']['states'][0]['param_groups'][0]['lr'] = .1
            elif broken == 'interface_lr': value['optimizer']['states'][1]['param_groups'][0]['lr'] = .1
            elif broken == 'muon_lr': value['optimizer']['states'][2]['inner']['param_groups'][0]['lr'] = .1
            else: value['identity']['optimizer']['body_lr'] = .1
            with self.assertRaises(ValueError): branch(value, 1e-4, 'parent')


if __name__ == '__main__':
    unittest.main()

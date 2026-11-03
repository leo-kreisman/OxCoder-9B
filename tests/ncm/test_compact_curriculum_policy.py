import copy
import ast
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import torch

from test_frozen_tower_optimizer import Toy
from scripts.ncm.frozen_tower_optimizer import TowerOptimizer
from scripts.ncm.compact_curriculum_policy import CompactSampler, qualify_expansion, import_parent_state
from scripts.ncm.prepare_creator_expansion import frozen_evaluation_hash


def example(name, compact=False, split='train', source='creator'):
    row = {'id': name, 'split': split, 'target': 'x', 'source': source}
    if compact:
        row['curriculum'] = 'compact-code-v33'
    return row, {'coding_response': True, 'response_id': name, 'ids': [1, 2]}


class CompactPolicyTests(unittest.TestCase):
    def test_frozen_evaluators_are_unchanged(self):
        root = Path(__file__).resolve().parents[2] / 'scripts/ncm'
        def definitions(name):
            tree = ast.parse((root / name).read_text())
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in ('evaluate', 'validate')}
        self.assertEqual(definitions('train_frozen_clean.py'), definitions('train_compact_tower.py'))

    def test_partition_sampling_and_rng_replay(self):
        rows = [example('a'), example('b', source='second'), example('short', True)]
        sampler = CompactSampler(rows)
        rng = random.Random(73)
        saved = rng.getstate()
        ids = [sampler.sample(rng)[0]['id'] for _ in range(1000)]
        rng.setstate(saved)
        self.assertEqual(ids, [sampler.sample(rng)[0]['id'] for _ in range(1000)])
        self.assertEqual(set(ids), {'a', 'b', 'short'})
        self.assertTrue(400 < ids.count('short') < 600)
        with self.assertRaises(ValueError):
            CompactSampler(rows + [example('held', split='validation')])

    def test_expansion_preserves_base_and_validation(self):
        base = [example('a'), example('held', split='validation')]
        added = base + [example('short', True)]
        corpus = {'training_chunks': 1}
        report = {'format': 'frozen-tower-text-admission-v33', 'complete': True,
            'base_corpus': corpus, 'validation_unchanged': True,
            'frozen_evaluation_sha256': frozen_evaluation_hash(base)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'admission.json'
            path.write_text(json.dumps(report))
            self.assertEqual(qualify_expansion(base, added, corpus, path)['additional_chunks'], 1)
            changed = copy.deepcopy(added)
            changed[0][0]['target'] = 'different'
            with self.assertRaises(ValueError):
                qualify_expansion(base, changed, corpus, path)
            changed = copy.deepcopy(added)
            changed[1][1]['ids'].append(3)
            with self.assertRaises(ValueError):
                qualify_expansion(base, changed, corpus, path)
            with self.assertRaises(ValueError):
                qualify_expansion(base, base, corpus, path)

    def test_preserve_optimizer_and_reject_objective_change(self):
        torch.manual_seed(311)
        model, rng = Toy(), random.Random(11)
        optimizer = TowerOptimizer(model)
        x = torch.randn(2, 4)
        (model(x) * optimizer.loss_scale).backward(); optimizer.step()
        args = SimpleNamespace(steps=10000, warmup=100, batch=4, seed=29004)
        parent = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v32',
            'planned_steps': args.steps, 'warmup': args.warmup, 'effective_batch_examples': args.batch,
            'seed': args.seed, 'compute_dtype': 'float16', 'loss_scale': optimizer.loss_scale,
            'body_lr': 1e-4, 'head_lr': 1e-3, 'on_policy_probability': 0.,
            'conditioning_corruption_maximum': .15, 'autoregressive_answer_fallback': False,
            'implementation_sha256': {}}
        state = {'identity': parent, 'step': 1000,
            'model': {n: p.detach().clone() for n, p in model.named_parameters()},
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'rng': {'python': rng.getstate(), 'torch': torch.get_rng_state()}}
        draw, tensor = rng.random(), torch.rand(4)
        other, other_rng = Toy(), random.Random(91)
        other_optimizer = TowerOptimizer(other)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parent.pt'; torch.save(state, path)
            self.assertEqual(import_parent_state(path, parent, other, other_optimizer,
                              other_rng, 'cpu', args), 1000)
            self.assertEqual(draw, other_rng.random())
            torch.testing.assert_close(tensor, torch.rand(4), rtol=0, atol=0)
            for m, o in ((model, optimizer), (other, other_optimizer)):
                o.zero_grad(); (m(x) * o.loss_scale).backward(); o.step()
            for p, q in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            changed = {**parent, 'conditioning_corruption_maximum': .5}
            with self.assertRaises(ValueError):
                import_parent_state(path, changed, other, other_optimizer, other_rng, 'cpu', args)


if __name__ == '__main__':
    unittest.main()

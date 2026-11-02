import unittest

from scripts.ncm.probe_context_checkpoint import select


class ProbeSelectionTests(unittest.TestCase):
    def test_training_only_disjoint_balanced_and_order_independent(self):
        rows = [{'id': str(i), 'group': str(i // 2), 'source': f's{i // 8}',
                 'split': 'train'} for i in range(24)]
        rows += [{'id': 'v', 'group': 'v', 'source': 'heldout-only', 'split': 'validation'}]
        indices = select(rows, 9)
        chosen = [rows[i] for i in indices]
        self.assertEqual({x['split'] for x in chosen}, {'train'})
        self.assertEqual(len({x['group'] for x in chosen}), 9)
        self.assertEqual([sum(x['source'] == s for x in chosen) for s in ('s0', 's1', 's2')], [3, 3, 3])
        reverse = list(reversed(rows))
        self.assertEqual([x['id'] for x in chosen], [reverse[i]['id'] for i in select(reverse, 9)])
        with self.assertRaises(ValueError): select(rows, 0)
        with self.assertRaises(ValueError): select(rows, 13)


if __name__ == '__main__': unittest.main()

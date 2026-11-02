import random
import unittest

from scripts.ncm.context_features import select_training


def fixture():
    rows = []
    for i in range(48):
        for chunk in range(2):
            rows.append(({'id': f'{i}-{chunk}', 'group': f'g{i}', 'split': 'train',
                          'source': f's{i % 3}', 'distilled': i % 5 == 0, 'target': 'unread'},
                         {'ids': [1, 2], 'response_id': f'r{i}', 'chunk_index': chunk,
                          'coding_response': i % 2 == 0}))
    return rows


class SelectionTests(unittest.TestCase):
    def test_order_independent_and_conversation_disjoint(self):
        rows = fixture()
        chosen = select_training(rows, 32, 25001)
        random.Random(33).shuffle(rows)
        self.assertEqual(chosen, select_training(rows, 32, 25001))
        self.assertEqual(len({row['group'] for row, _ in chosen}), 32)
        self.assertEqual({row['source'] for row, _ in chosen}, {'s0', 's1', 's2'})
        self.assertTrue(any(row['distilled'] for row, _ in chosen))

    def test_validation_targets_do_not_choose_training_rows(self):
        rows = fixture()
        chosen = [r['id'] for r, _ in select_training(rows, 32, 25001)]
        for r, _ in rows: r['target'] = 'a completely different answer'
        rows.append(({'id': 'val', 'group': 'validation', 'split': 'validation', 'target': 'unseen'}, {}))
        self.assertEqual(chosen, [r['id'] for r, _ in select_training(rows, 32, 25001)])

    def test_rejects_split_leakage_duplicate_ids_and_invalid_count(self):
        rows = fixture()
        with self.assertRaises(ValueError): select_training(rows, 0, 1)
        with self.assertRaises(ValueError): select_training(rows + rows[:1], 32, 1)
        with self.assertRaises(ValueError):
            select_training(rows + [({'id': 'v', 'group': 'g0', 'split': 'validation'}, {})], 32, 1)


if __name__ == '__main__': unittest.main()

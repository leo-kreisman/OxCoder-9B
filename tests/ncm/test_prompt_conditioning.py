import unittest

from scripts.ncm.audit_prompt_conditioning import pairs, summarize


def example(i, split='validation'):
    return ({'id': str(i), 'group': str(i), 'split': split}, {'ids': [1] * (i+2)})


class PromptConditioningTests(unittest.TestCase):
    def test_frozen_derangement_without_self_pairs(self):
        rows = [example(i) for i in range(24)]
        selected = pairs(rows, 16)
        self.assertEqual(selected, pairs(list(reversed(rows)), 16))
        self.assertEqual(len({a[0]['group'] for a, b in selected}), 16)
        self.assertTrue(all(a[0]['group'] != b[0]['group'] for a, b in selected))
        self.assertEqual({a[0]['id'] for a, b in selected}, {b[0]['id'] for a, b in selected})

    def test_rejects_training_rows_and_duplicate_groups(self):
        with self.assertRaises(ValueError): pairs([example(0), example(1, 'train')], 2)
        with self.assertRaises(ValueError): pairs([example(0), example(0)], 2)
        with self.assertRaises(ValueError): pairs([example(0)], 1)

    def test_loss_change_sign_and_nonfinite_detection(self):
        record = {'correct_ce': [2., 3.], 'replaced_ce': [4., 2.],
                  'correct_accuracy': [.2, .3], 'replaced_accuracy': [.1, .4],
                  'argmax_agreement': [.5, .6], 'kl_correct_to_replaced': [.7, .8]}
        self.assertEqual(summarize([record])['replacement_ce_increase'], [2., -1.])
        with self.assertRaises(ValueError): summarize([{**record, 'correct_ce': [float('nan'), 3.]}])


if __name__ == '__main__':
    unittest.main()

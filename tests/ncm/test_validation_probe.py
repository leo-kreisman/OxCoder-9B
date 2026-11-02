import unittest

from scripts.ncm.validation_probe import select, summarize


def example(group, split='validation', source='creator-a', index=0, length=30):
    return ({'id': f'{group}:{index}', 'group': group, 'split': split,
             'source': source, 'target': 'unobserved target'},
            {'ids': list(range(length)), 'is_final_chunk': bool(index)})


class ValidationProbeTests(unittest.TestCase):
    def test_selection_is_order_independent_and_conversation_disjoint(self):
        rows = [example(f'g{i}', source=f'creator-{i % 2}', index=j)
                for i in range(20) for j in range(3)]
        rows.append(example('training', split='train'))
        rows.append(example('test', split='test'))
        selected = select(rows, 12)
        self.assertEqual(selected, select(list(reversed(rows)), 12))
        self.assertEqual(len({r['group'] for r, _ in selected}), 12)
        self.assertTrue(all(r['split'] == 'validation' for r, _ in selected))
        self.assertEqual(sum(r['source'] == 'creator-0' for r, _ in selected), 6)
        # It never looks at reconstruction, target contents, or model scores.
        changed = [({**r, 'target': 'different target'}, c) for r, c in rows]
        self.assertEqual([r['id'] for r, _ in selected], [r['id'] for r, _ in select(changed, 12)])

    def test_rejects_leakage_duplicates_and_insufficient_coverage(self):
        for rows in [[example('g'), example('g', split='train', index=1)],
                     [example('g'), example('g')]]:
            with self.assertRaises(ValueError): select(rows, 1)
        with self.assertRaises(ValueError): select([example('g', length=40)], 1, max_prompt=20)
        with self.assertRaises(ValueError): select([example('g'), example('g', index=1)], 2)

    def test_summary_includes_codec_and_invalid_generation_failures(self):
        row = {'teacher_forced_ce': [2., 4.], 'teacher_forced_accuracy': [.1, .2],
               'overlapping_nonpad_ce': [3., 5.], 'overlapping_nonpad_accuracy': [.05, .1],
               'codec_exact_nonpad': False, 'free_exact_chunk': False, 'free_valid_chunk': False,
               'free_byte_accuracy_nonpad': .1, 'stop_probability': .8, 'expected_final_chunk': False}
        good = {**row, 'codec_exact_nonpad': True, 'free_exact_chunk': True,
                'free_valid_chunk': True, 'free_byte_accuracy_nonpad': 1., 'expected_final_chunk': True}
        result = summarize([row, good])
        self.assertEqual(result['windows'], 2)
        self.assertEqual(result['mean_level_ce'], 3.)
        self.assertEqual(result['codec_exact_nonpad_windows'], 1)
        self.assertEqual(result['free_exact_windows'], 1)
        self.assertEqual(result['stop_accuracy'], .5)
        self.assertEqual(result['mean_free_nonpad_byte_accuracy'], .55)


if __name__ == '__main__':
    unittest.main()

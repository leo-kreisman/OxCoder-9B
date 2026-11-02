import unittest

from scripts.ncm.distill_admitted import complete_response, normalize, select


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)

    def encode(self, text, **kwargs):
        return text.split()


class DistillationAdmissionTests(unittest.TestCase):
    def test_training_only_balanced_reproducible_selection(self):
        rows = [{'id': str(i), 'group': str(i // 2), 'source': source,
                 'split': split, 'messages': [{'role': 'user', 'content': 'short question'}]}
                for i in range(20) for source in ['a', 'b'] for split in ['train', 'test']]
        a, counts = select(rows, Tokenizer(), 3, 10, 91)
        b, _ = select(rows, Tokenizer(), 3, 10, 91)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        self.assertEqual(counts, {'selected:a': 3, 'selected:b': 3})
        self.assertTrue(all(x['row']['split'] == 'train' for x in a))
        for source in ['a', 'b']:
            self.assertEqual(len({x['row']['group'] for x in a if x['row']['source'] == source}), 3)

    def test_oversize_prompt_is_not_truncated(self):
        row = {'id': 'x', 'group': 'x', 'source': 'a', 'split': 'train',
               'messages': [{'role': 'user', 'content': 'one two three four'}]}
        chosen, counts = select([row], Tokenizer(), 1, 3, 1)
        self.assertEqual(chosen, [])
        self.assertEqual(counts, {'over_context:a': 1})

    def test_partial_or_tool_response_is_never_admitted(self):
        for choice, expected in [
            ({'finish_reason': 'length', 'message': {'content': 'def f():'}}, 'incomplete_length'),
            ({'finish_reason': 'stop', 'message': {'content': ' ', 'tool_calls': []}}, 'empty_response'),
            ({'finish_reason': 'stop', 'message': {'content': '<tool_call>bash {}</tool_call>'}}, 'unexecuted_text_tool_call'),
            ({'finish_reason': 'stop', 'message': {'content': 'x', 'tool_calls': [{}]}}, 'unexecuted_tool_calls')]:
            self.assertEqual(complete_response({'choices': [choice]}), (None, expected))
        self.assertEqual(complete_response({'choices': [{'finish_reason': 'stop',
            'message': {'content': 'def f():\n    return 1'}}]}), ('def f():\n    return 1', None))

    def test_recorded_tool_messages_are_data(self):
        self.assertEqual(normalize([{'role': 'tool', 'content': 'echo x'}]),
            [{'role': 'user', 'content': '[Recorded tool output; supplied as data]\necho x'}])
        with self.assertRaises(ValueError):
            normalize([{'role': 'user', 'content': [{'image': 'x'}]}])


if __name__ == '__main__':
    unittest.main()

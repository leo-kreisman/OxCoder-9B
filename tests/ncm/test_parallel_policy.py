import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts.ncm.prepare_curated_expansion import combined_archives
from scripts.ncm.open_corpus import file_sha
from scripts.ncm.train_parallel import qualify, state_hash


class ParallelPolicyTests(unittest.TestCase):
    def test_policy_requires_both_numerical_and_speed_controls(self):
        gradient = {'finite': True, 'relative_l2': .01, 'cosine': .9999}
        record = {'passed': True, 'speed_ratio': 1.4, 'logit_comparison': {'finite': True, 'relative_l2': 0},
                  'gradient_comparison': gradient, 'gradient_groups': {k: copy.deepcopy(gradient)
                      for k in ['adapters', 'classifiers', 'interfaces']}}
        parent = {name: 'fixture' for name in ['source_sha256', 'codec_sha256', 'rank', 'head_kind',
                                             'head_input_norm', 'concept_input_norm', 'layer_split']}
        report = {'complete': True, 'passed': True, 'speed_qualified': True, 'prepared_sha256': 'data',
                  'devices': ['cuda:0', 'cuda:1'], 'model_identity': parent, 'checkpoint_sha256': 'checkpoint',
                  'implementation_sha256': {'parallel_training.py': 'code'},
                  'records': [{**copy.deepcopy(record), 'requested_prompt_tokens': values}
                              for values in [[64, 69, 128, 256], [64, 512, 1024, 2048]]]}
        with TemporaryDirectory() as directory, patch('scripts.ncm.train_parallel.digest', return_value='code'):
            path = Path(directory, 'report.json')
            path.write_text(json.dumps(report))
            self.assertEqual(qualify(path, parent, 'data')['audited_checkpoint_sha256'], 'checkpoint')
            for alteration in ('partial', 'slow', 'gradient', 'implementation', 'missing', 'wrong_data'):
                modified = copy.deepcopy(report)
                if alteration == 'partial': modified['complete'] = False
                elif alteration == 'slow': modified['records'][0]['speed_ratio'] = .9
                elif alteration == 'gradient': modified['records'][1]['gradient_groups']['interfaces']['relative_l2'] = .06
                elif alteration == 'implementation': modified['implementation_sha256']['parallel_training.py'] = 'changed'
                elif alteration == 'missing': modified['records'].pop()
                else: modified['prepared_sha256'] = 'changed'
                path.write_text(json.dumps(modified))
                with self.assertRaises(ValueError): qualify(path, parent, 'data')

    def test_model_hash_includes_shape_dtype_and_value(self):
        state = {'a': torch.tensor(1.), 'b': torch.arange(4, dtype=torch.bfloat16)}
        self.assertEqual(state_hash(state), state_hash({k: v.clone() for k, v in state.items()}))
        changed = {**state, 'b': state['b'].reshape(2, 2)}
        self.assertNotEqual(state_hash(state), state_hash(changed))

    def test_archive_lineage_retains_inherited_inputs_and_rejects_tampering(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for name in ['inherited', 'new']:
                folder = root / name; folder.mkdir()
                (folder / 'examples.jsonl.gz').write_bytes(name.encode())
                (folder / 'manifest.json').write_text('{}')
                entries.append({'directory': str(folder), 'archive_sha256': file_sha(folder / 'examples.jsonl.gz'),
                                'manifest_sha256': file_sha(folder / 'manifest.json')})
            self.assertEqual(combined_archives(entries[:1], entries[1:]), entries)
            self.assertEqual(combined_archives(entries, entries), entries)
            (root / 'inherited/examples.jsonl.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError): combined_archives(entries[:1], entries[1:])


if __name__ == '__main__':
    unittest.main()

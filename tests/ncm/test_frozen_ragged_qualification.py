import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.ncm.frozen_ragged_qualification import qualify


class RaggedQualificationTests(unittest.TestCase):
    def test_requires_shape_numerical_speed_and_implementation_evidence(self):
        metric = {'finite': True, 'relative_l2': .001, 'cosine': .999}
        report = {'complete': True, 'passed': True, 'speed_qualified': True,
                  'source_sha256': 'source', 'codec_sha256': 'codec',
                  'compute_dtype': 'float16', 'prefix_alignment': 1,
                  'selection_sha256': 'selection', 'implementation_sha256': {},
                  'records': []}
        for sizes in ((64, 69, 128, 256), (64, 512, 1024, 2048)):
            report['records'].append({'examples': [{'prompt_tokens': s} for s in sizes],
                'passed': True, 'gradient_comparison': metric,
                'gradient_groups': {k: metric for k in ('adapters', 'classifiers', 'interfaces')},
                'logit_comparison': metric, 'speed_ratio_over_v30': 1.5})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.json'
            path.write_text(json.dumps(report))
            self.assertTrue(qualify(path, 'source', 'codec')['unpadded_levels'])
            for mutation in ('incomplete', 'shape', 'group', 'gradient', 'logit', 'speed', 'hash'):
                bad = copy.deepcopy(report)
                if mutation == 'incomplete': bad['complete'] = False
                elif mutation == 'shape': bad['records'].pop()
                elif mutation == 'group': bad['records'][0]['gradient_groups'].pop('interfaces')
                elif mutation == 'gradient': bad['records'][0]['gradient_comparison']['relative_l2'] = .1
                elif mutation == 'logit': bad['records'][0]['logit_comparison']['relative_l2'] = .03
                elif mutation == 'speed': bad['records'][0]['speed_ratio_over_v30'] = .9
                else: bad['implementation_sha256']['frozen_tower_ragged.py'] = 'changed'
                path.write_text(json.dumps(bad))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    qualify(path, 'source', 'codec')


if __name__ == '__main__':
    unittest.main()

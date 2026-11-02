import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch

from scripts.ncm.acquire_creator_corpora import acquire, fetch, archived_digest
from scripts.ncm.data import DATASETS


class AcquisitionTests(unittest.TestCase):
    def test_transient_proxy_error_retries_but_missing_source_does_not(self):
        module='scripts.ncm.acquire_creator_corpora'
        transient=urllib.error.HTTPError('url',499,'proxy closed request',{},None)
        missing=urllib.error.HTTPError('url',404,'missing',{},None)
        with patch(module+'.fetch_once',side_effect=[transient,('path',{})]) as f, patch(module+'.time.sleep'):
            self.assertEqual(fetch(None,None),('path',{}))
            self.assertEqual(f.call_count,2)
        with patch(module+'.fetch_once',side_effect=missing) as f, patch(module+'.time.sleep'):
            with self.assertRaises(urllib.error.HTTPError):fetch(None,None)
            self.assertEqual(f.call_count,1)

    def test_complete_archive_and_filtered_reservoir_are_distinct(self):
        alpha,agent=list(DATASETS)
        messages=lambda text:[{'role':'user','content':'Write Python.'},{'role':'assistant','content':text}]
        payloads={
            'train.jsonl':(json.dumps({'messages':messages('print(1)')})+'\n'+
                           json.dumps({'messages':messages('x'*32769)})+'\n').encode(),
            'trace.jsonl':''.join(json.dumps({'message':m})+'\n' for m in messages('print(2)')).encode()}
        def info(source,**kw):
            name='train.jsonl' if source==alpha else 'trace.jsonl'
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=name,size=len(payloads[name]))])
        def response(url,**kw):
            return io.BytesIO(payloads[url.rsplit('/',1)[1]])
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.ncm.acquire_creator_corpora.HfApi') as api, \
             patch('scripts.ncm.acquire_creator_corpora.urllib.request.urlopen',side_effect=response):
            api.return_value.dataset_info.side_effect=info
            acquire(SimpleNamespace(output=tmp))
            root=Path(tmp);manifest=json.loads((root/'manifest.json').read_text())
            self.assertTrue(manifest['sources_complete'])
            self.assertFalse(manifest['admitted_to_training'])
            self.assertEqual(manifest['counts']['oversize_rejected'],1)
            with gzip.open(root/'examples.jsonl.gz','rt') as f:
                rows=[json.loads(line) for line in f]
            self.assertEqual(len(rows),2)
            self.assertEqual({r['target'] for r in rows},{'print(1)','print(2)'})
            with gzip.open(root/'raw/alpha/train.jsonl.gz','rb') as f:
                self.assertEqual(f.read(),payloads['train.jsonl'])
            self.assertEqual(archived_digest(root/'examples.jsonl.gz')[0],manifest['examples_uncompressed_sha256'])

    def test_incomplete_download_cannot_become_a_completed_archive(self):
        source,revision=next(iter(DATASETS.items()))
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.ncm.acquire_creator_corpora.urllib.request.urlopen',return_value=io.BytesIO(b'short')):
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                fetch((source,revision,'train.jsonl',100),Path(tmp))
            self.assertFalse((Path(tmp)/'alpha/train.jsonl.gz').exists())
            self.assertFalse((Path(tmp)/'alpha/train.jsonl.metadata.json').exists())


if __name__=='__main__':
    unittest.main()

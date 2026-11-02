import unittest
from scripts.ncm.complete_context_cache import retain_complete


class CompleteCacheTests(unittest.TestCase):
    def test_missing_codec_chunk_removes_whole_training_response_only(self):
        prepared=[({'id':str(i),'split':'train'},{'response_id': 'a' if i<3 else 'b'}) for i in range(5)]
        records=[{'id':str(i),'split':'train'} for i in (0,2,3,4)]+[{'id':'v','split':'validation'}]
        kept,removed=retain_complete(records,prepared)
        self.assertEqual([r['id'] for r in kept],['3','4','v'])
        self.assertEqual(removed,{'a':['1']})


if __name__=='__main__':unittest.main()

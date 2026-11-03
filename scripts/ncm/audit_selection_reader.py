"""Compare complete admitted corpora and deterministic sampling after caching."""
import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import time

from .backbone import digest
from .prepared_reader import identical
from .selection_reader import prepare
from .cached_tower_corpus import load_corpus as cached
from .frozen_tower_training import load_corpus as reference
from .compact_curriculum_policy import CompactSampler


def run(a):
    identity = json.loads(Path(a.manifest).read_text())['identity']
    cache = Path(a.cache)
    if not cache.exists():
        prepare(Path(a.selection) / 'selection.pt', identity['selection_sha256'], cache)
    arguments = (a.selection, a.manifest, identity['source_sha256'], identity['codec_sha256'])
    begin = time.monotonic()
    original = reference(*arguments)
    reference_seconds = time.monotonic() - begin
    times, exact = [], []
    for _ in range(3):
        begin = time.monotonic()
        candidate = cached(*arguments, a.cache)
        times.append(time.monotonic() - begin)
        exact.append(identical(original, candidate))
        del candidate
        gc.collect()
    candidate = cached(*arguments, a.cache)
    old_sampler, new_sampler = CompactSampler(original[0]), CompactSampler(candidate[0])
    old_rng, new_rng = random.Random(35004), random.Random(35004)
    sampling_equal = all(identical(old_sampler.sample(old_rng), new_sampler.sample(new_rng)) for _ in range(1000))
    report = {'complete': True, 'passed': all(exact) and sampling_equal, 'quality_ready': False,
              'source_selection_sha256': identity['selection_sha256'],
              'admission_sha256': digest(a.manifest), 'cache_manifest_sha256': digest(cache / 'manifest.json'),
              'reference_seconds': reference_seconds, 'warm_cached_seconds': times,
              'median_cached_seconds': statistics.median(times),
              'all_types_values_ordering_identical': all(exact), 'identical_sampler_draws': 1000 if sampling_equal else 0,
              'corpus': original[2], 'scope': 'Serialization/startup only; no training or quality claim',
              'implementation_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                  ('selection_reader.py', 'cached_tower_corpus.py', 'frozen_tower_training.py', 'prepared_reader.py',
                   'compact_curriculum_policy.py', 'audit_selection_reader.py')}}
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'manifest', 'cache', 'output'):
        p.add_argument('--' + name, required=True)
    run(p.parse_args())

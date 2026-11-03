"""Fresh-process coding gate with exact saved trainable-weight restoration."""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .backbone import digest
from .gguf_training_backbone import load_candidate
from .frozen_ragged_qualification import qualify
from . import frozen_tower as tower
from .exact_checkpoint import restore_trainable
from .train_compact_tower import evaluate
from .train import admission, write_json


def run(a):
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Never replace an existing coding report')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    admission([a.device], 8)
    model, codec, identity, checksum = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
    if identity.get('architecture') not in {'frozen-context-original-oxcoder-native-ncm-v' + str(v) for v in (32, 33, 34, 35)}:
        raise ValueError('Unsupported source-tower checkpoint architecture')
    for name, sha in identity['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Checkpoint implementation changed: ' + name)
    qualify(a.execution_audit, identity['source_sha256'], identity['codec_sha256'])
    tower.set_compute_dtype(model.backbone, torch.float16)
    restore_trainable(model, a.checkpoint, identity, checksum)
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer', local_files_only=True, trust_remote_code=False)
    report = evaluate(model, codec, tokenizer, checksum, output)
    report.update(trainable_checkpoint_values_restored_exactly=True,
                  exact_restore_sha256=digest(Path(__file__).with_name('exact_checkpoint.py')))
    write_json(output, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'execution-audit', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    run(p.parse_args())

#!/usr/bin/env bash
set -euo pipefail
cd /home/scribe/Projects/OxCoder-9B
while [[ "$(systemctl --user show ncm-generator-v6b-evaluate -p ActiveState --value)" =~ ^(active|activating|deactivating)$ ]]; do sleep 10; done
python3 - <<'PY'
import json, hashlib, shutil, tarfile
from pathlib import Path
root=Path('results/ncm-generator-v6b')
status=json.loads((root/'evaluation-status.json').read_text())
assert status['phase']=='coding_evaluation_finished',status
assert json.loads((root/'training.json').read_text())['complete']
dest=Path('results/ncm-v6c-initials');dest.mkdir(exist_ok=False)
shutil.copy2(root/'latest.pt',dest/'generator.pt')
with tarfile.open(dest/'implementation.tar.gz','w:gz') as tar:
 for folder in ['scripts/ncm','tests/ncm']:
  for p in sorted(Path(folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p)
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
(dest/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
PY
PYTHON=/home/scribe/Projects/JEV_experiment/.venv-derivation/bin/python
SOURCE=/home/scribe/Projects/JEV_experiment/results/oxcoder-github-mirror/OxCoder-9B.Q4_K_M.gguf
"$PYTHON" -u -m scripts.ncm.train generator \
 --data results/ncm-muon-distillation-v5/data --output results/ncm-generator-v6c \
 --source "$SOURCE" --codec results/ncm-coverage-decoder-v6/best.pt \
 --complete-gate results/ncm-coverage-decoder-v6/complete-gate.json \
 --exact-target-curriculum --complete-data results/ncm-corpus-v2 \
 --teacher-data results/ncm-muon-distillation-v5/teacher \
 --parallel-attention --layer-split 12,20 --devices cuda:0,cuda:1 \
 --rank 32 --prompt-tokens 1024 --optimizer muon --lr 0.0001 \
 --steps 1000 --eval-every 250 --save-every 100 \
 --audit-reference scripts/kindi_q4/kindi/reference-oxcoder-q4.json \
 --generator-init results/ncm-v6c-initials/generator.pt \
 --balanced-curriculum --byte-loss 0.2 \
 --kernel-backend fla --kernel-audit results/ncm-kernel-audit.json
"$PYTHON" -u -m scripts.ncm.coding_eval --source "$SOURCE" \
 --codec results/ncm-coverage-decoder-v6/best.pt \
 --checkpoint results/ncm-generator-v6c/latest.pt \
 --output results/ncm-generator-v6c/coding-evaluation.json

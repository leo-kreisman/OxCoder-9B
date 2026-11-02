#!/usr/bin/env bash
set -euo pipefail
# Finite jobs only. No interference with other inference servers or GPU settings.
if [[ $# -lt 1 ]]; then echo 'Usage: bash scripts/ncm/run_limited.sh PYTHON [ARGS...]' >&2; exit 2; fi
exec systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=0 \
  -p CPUQuota=400% -p IOWeight=10 \
  nice -n 10 ionice -c 2 -n 7 "$@"

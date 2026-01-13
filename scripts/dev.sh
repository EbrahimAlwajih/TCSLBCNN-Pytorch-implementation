#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-test}"

if [[ "$TASK" == "lint" ]]; then
  ruff check .
elif [[ "$TASK" == "test" ]]; then
  pytest -q
elif [[ "$TASK" == "smoke" ]]; then
  python -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
else
  echo "Unknown task: $TASK"
  exit 1
fi

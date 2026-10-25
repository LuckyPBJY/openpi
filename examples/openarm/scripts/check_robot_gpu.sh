#!/usr/bin/env bash
# Check GPU / JAX / PyTorch compatibility on the target machine.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
OPENPI_ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$OPENPI_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="python3"
fi

echo "== nvidia-smi =="
nvidia-smi 2>&1 | head -15
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 || true

echo "== python =="
"$PY" --version 2>&1

echo "== torch =="
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('cap', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)" 2>&1 || true

echo "== jax =="
"$PY" -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" 2>&1 || true

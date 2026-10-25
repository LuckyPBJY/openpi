#!/usr/bin/env bash
# Start the policy server locally (example).
#   scripts/serve_local.sh [joint|eef] [port] [checkpoint_step]
set -euo pipefail

MODE="${1:-joint}"
PORT="${2:-8000}"
STEP="${3:-}"
OPENPI_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

case "$MODE" in
    joint)
        CONFIG=pi05_openarm_20261025_lora
        EXP=openarm_20261025_joint_run1
        ;;
    eef)
        CONFIG=pi05_openarm_20261025_eef_raw_lora
        EXP=openarm_20261025_eef_raw_run1
        ;;
    *)
        echo "usage: $0 [joint|eef] [port] [checkpoint_step]" >&2
        exit 2
        ;;
esac

if [ -z "$STEP" ]; then
    echo "checkpoint step is required, e.g. 25000" >&2
    exit 2
fi
CKPT="$OPENPI_ROOT/checkpoints/$CONFIG/$EXP/$STEP"
if [ ! -d "$CKPT/params" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 2
fi

cd "$OPENPI_ROOT"
exec env \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    .venv/bin/python scripts/serve_policy.py policy:checkpoint \
        --policy.config="$CONFIG" \
        --policy.dir="$CKPT" \
        --port="$PORT"

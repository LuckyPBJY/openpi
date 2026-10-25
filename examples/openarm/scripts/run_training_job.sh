#!/usr/bin/env bash
# Run one OpenArm training job. Intended to run inside a GPU allocation
# (e.g. srun), but works locally too.
#
#   run_training_job.sh <config> <exp_name> [proxy_port] [fresh|resume]
#
# Optional environment:
#   OPENARM_ENV              path to an env file (default: scripts/openarm_env.sh)
#   OPENARM_LOGIN_HOST       if set, start an SSH proxy tunnel
#   OPENARM_REMOTE_PROXY_PORT  remote proxy port (default 6897)
#   OPENARM_PROXY_CHECK_URL  proxy connectivity check (default https://pypi.org)
#   NUM_TRAIN_STEPS/BATCH_SIZE/SAVE_INTERVAL/KEEP_PERIOD/LOG_INTERVAL
set -uo pipefail

CONFIG="${1:?config is required}"
EXP="${2:?exp_name is required}"
PORT="${3:-6897}"
MODE="${4:-fresh}"
OPENPI_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$OPENPI_ROOT/examples/openarm/logs"

case "$MODE" in
    resume) CKPT_FLAG=--resume ;;
    fresh) CKPT_FLAG=--overwrite ;;
    *) echo "mode must be 'resume' or 'fresh', got '$MODE'" >&2; exit 2 ;;
esac

ENV_SCRIPT="${OPENARM_ENV:-$OPENPI_ROOT/examples/openarm/scripts/openarm_env.sh}"
if [ -f "$ENV_SCRIPT" ]; then
    # shellcheck disable=SC1090
    source "$ENV_SCRIPT"
fi

TPID=""
if [ -n "${OPENARM_LOGIN_HOST:-}" ]; then
    bash "$(dirname "$0")/proxy_tunnel.sh" "$PORT" >"$OPENPI_ROOT/examples/openarm/logs/tunnel_${EXP}.log" 2>&1 &
    TPID=$!
    trap 'kill "$TPID" 2>/dev/null' EXIT
    for _ in $(seq 1 30); do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    export OPENARM_PROXY_PORT="$PORT"
fi

if [ -n "${HTTPS_PROXY:-}" ]; then
    for _ in $(seq 1 30); do
        if curl -s -o /dev/null --max-time 3 -x "$HTTPS_PROXY" \
            "${OPENARM_PROXY_CHECK_URL:-https://pypi.org}"; then
            break
        fi
        sleep 1
    done
fi

echo "[$(date)] config=$CONFIG exp=$EXP mode=$MODE wandb_mode=${WANDB_MODE:-online} gpu=${CUDA_VISIBLE_DEVICES:-?}"
cd "$OPENPI_ROOT"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    .venv/bin/python scripts/train.py "$CONFIG" \
        --exp-name="$EXP" \
        "$CKPT_FLAG" \
        --num-train-steps="${NUM_TRAIN_STEPS:-100000000}" \
        --batch-size="${BATCH_SIZE:-8}" \
        --save-interval="${SAVE_INTERVAL:-2500}" \
        --keep-period="${KEEP_PERIOD:-2500}" \
        --log-interval="${LOG_INTERVAL:-100}"
RC=$?
if [ -n "$TPID" ]; then
    kill "$TPID" 2>/dev/null
fi
exit $RC

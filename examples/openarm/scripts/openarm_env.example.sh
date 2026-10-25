#!/usr/bin/env bash
# Copy this file to openarm_env.sh and fill in your own environment.
# Do NOT commit your real proxy address, credentials, or private paths.
#
#   cp examples/openarm/scripts/openarm_env.example.sh \
#      examples/openarm/scripts/openarm_env.sh

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export OPENPI_ROOT

# LeRobot dataset cache. Set to the directory containing <repo_id>/.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$OPENPI_ROOT/lerobot_data}"

# FFmpeg shared libraries are needed by torchcodec for video decoding.
# export LD_LIBRARY_PATH="/path/to/ffmpeg/lib:${LD_LIBRARY_PATH:-}"

# Keep thread counts bounded on shared nodes.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# Optional outbound proxy (needed for wandb / downloads on some clusters).
# unset ALL_PROXY all_proxy
# export HTTP_PROXY="http://127.0.0.1:<your-proxy-port>"
# export HTTPS_PROXY="$HTTP_PROXY"

# Optional: SSH tunnel to a login node that hosts the proxy.
# export OPENARM_LOGIN_HOST="<ssh-login-host>"
# export OPENARM_REMOTE_PROXY_PORT="<remote-proxy-port>"

# Optional Weights & Biases configuration.
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_ENTITY="<your-entity>"
# export WANDB_PROJECT="openpi"

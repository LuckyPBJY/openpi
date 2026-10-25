#!/usr/bin/env bash
# Expose an SSH-reachable proxy on a local port. Optional; only needed when the
# compute node has no direct outbound network.
#
#   OPENARM_LOGIN_HOST=<ssh-host> OPENARM_REMOTE_PROXY_PORT=<port> \
#       scripts/proxy_tunnel.sh <local_port>
set -u

PORT="${1:-6897}"
LOGIN_HOST="${OPENARM_LOGIN_HOST:?Set OPENARM_LOGIN_HOST to your SSH login host}"
REMOTE_PORT="${OPENARM_REMOTE_PROXY_PORT:-6897}"

while true; do
    ssh -N \
        -o BatchMode=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=accept-new \
        -L "127.0.0.1:${PORT}:127.0.0.1:${REMOTE_PORT}" \
        "${LOGIN_HOST}"
    sleep 5
done

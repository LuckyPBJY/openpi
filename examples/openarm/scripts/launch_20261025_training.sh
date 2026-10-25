#!/usr/bin/env bash
# Example launcher: two trainings in tmux, one GPU each, no step limit.
#
# Required:
#   PARTITION=<your-slurm-partition>
# Optional:
#   TIME (default 7-00:00:00), CPUS (default 8)
#   PORT_JOINT (default 6897), PORT_EEF (default 6898)
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PARTITION="${PARTITION:?Set PARTITION to your SLURM partition}"
TIME="${TIME:-7-00:00:00}"
CPUS="${CPUS:-8}"
PORT_JOINT="${PORT_JOINT:-6897}"
PORT_EEF="${PORT_EEF:-6898}"
mkdir -p "$OPENPI_ROOT/examples/openarm/logs"

tmux new-session -d -s openarm_joint \
  "cd $OPENPI_ROOT && srun -c $CPUS -p $PARTITION --gres=gpu:1 --job-name=openarm_joint -t $TIME \
     bash $OPENPI_ROOT/examples/openarm/scripts/run_training_job.sh \
       pi05_openarm_20261025_lora openarm_20261025_joint_run1 $PORT_JOINT fresh \
     2>&1 | tee $OPENPI_ROOT/examples/openarm/logs/openarm_joint.log"

tmux new-session -d -s openarm_eef \
  "cd $OPENPI_ROOT && srun -c $CPUS -p $PARTITION --gres=gpu:1 --job-name=openarm_eef -t $TIME \
     bash $OPENPI_ROOT/examples/openarm/scripts/run_training_job.sh \
       pi05_openarm_20261025_eef_raw_lora openarm_20261025_eef_raw_run1 $PORT_EEF fresh \
     2>&1 | tee $OPENPI_ROOT/examples/openarm/logs/openarm_eef.log"

tmux ls
squeue -u "$USER" -o '%.10i %.32j %.8T %.10M %R'

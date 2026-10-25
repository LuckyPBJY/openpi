# OpenArm dual-arm VLA (π0.5) — data conversion & training

This directory contains the data conversion, fine-tuning, and evaluation
pipeline used for an OpenArm dual-arm manipulation task:

> Using the right arm, pick up the medicine box and the IV bottle from the table
> one at a time, place them into the basket, then use the left arm to lift the basket.

The model is π0.5 (`pi05_base`) with LoRA fine-tuning. Two action
representations are provided:

- **joint**: state = 16D qpos; action = 16D joint delta (grippers absolute)
- **eef raw**: state = 16D EE pose + gripper; action = 14D EE delta
  (position + rotation vector + gripper). **No `eef_obs_to_cmd` / mean-bias
  compensation is used.**

## What is included

```
examples/openarm/
  README.md
  TRAINING_HANDOFF.md
  scripts/
    convert_openarm_data_to_lerobot.py
    verify_lerobot_dataset.py
    compute_norm_stats_direct.py
    eval_checkpoints.py
    summarize_eval.py
    checkpoint_sanity.py
    download_openpi_gcs.py
    openarm_env.example.sh
    run_training_job.sh
    proxy_tunnel.sh
    launch_20261025_training.sh
    serve_local.sh
    check_robot_gpu.sh
  split/          # 120 train / 25 val episode indices
  norm_stats/     # two norm_stats.json (state/action statistics)
```

Code changes relative to upstream openpi (`215abfb`) live in:

```
pyproject.toml
src/openpi/training/config.py
src/openpi/training/data_loader.py
src/openpi/policies/openarm_policy.py   # new
uv.lock
```

## What is NOT included

- Raw HDF5 demonstrations
- Converted LeRobot datasets (videos / parquet)
- `pi05_base` and fine-tuned checkpoints
- W&B logs / caches
- Cluster-specific paths, proxy addresses, credentials

The datasets and weights are intentionally kept out of the repository.

## Quick start

```bash
# 0) Environment
uv sync --managed-python --python 3.11 --no-install-package rerun-sdk
uv pip install -e .
cp examples/openarm/scripts/openarm_env.example.sh \
   examples/openarm/scripts/openarm_env.sh
# edit openarm_env.sh for your paths/proxy/wandb

# 1) Convert HDF5 -> LeRobot (tactile fields are ignored)
python examples/openarm/scripts/convert_openarm_data_to_lerobot.py \
  --data-dir <PATH_TO_CROPPED_DATA> \
  --root <PATH_TO_LEROBOT_HOME> \
  --repo-id openarm/cropped_20261025_joint_eef \
  --include-eef \
  --task "Using the right arm, pick up the medicine box and the IV bottle from the table one at a time, place them into the basket, then use the left arm to lift the basket."

# 2) Norm stats (train split only)
HF_LEROBOT_HOME=<PATH_TO_LEROBOT_HOME> python examples/openarm/scripts/compute_norm_stats_direct.py \
  --config pi05_openarm_20261025_lora
HF_LEROBOT_HOME=<PATH_TO_LEROBOT_HOME> python examples/openarm/scripts/compute_norm_stats_direct.py \
  --config pi05_openarm_20261025_eef_raw_lora
# copy examples/openarm/norm_stats/* into ./assets/ if desired

# 3) Train (inside a GPU allocation; no step limit)
srun -c 8 -p <YOUR_PARTITION> --gres=gpu:1 -t 7-00:00:00 \
  bash examples/openarm/scripts/run_training_job.sh \
    pi05_openarm_20261025_lora openarm_20261025_joint_run1 <PROXY_PORT> fresh
# EE raw: pi05_openarm_20261025_eef_raw_lora
```

See `TRAINING_HANDOFF.md` for the full data schema, split, validation protocol,
known issues, and license notes.

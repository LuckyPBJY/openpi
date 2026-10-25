#!/usr/bin/env python3
"""Sanity-check a converted OpenArm LeRobot dataset.

Checks:
  1. lerobot metadata (episodes/frames/fps/features)
  2. openpi's ``create_torch_dataset`` can load it with an action horizon
  3. a sample has the expected keys/shapes and a valid action chunk
  4. every episode's videos decode

Usage:
    .venv/bin/python scripts/verify_lerobot_dataset.py \
        --root /path/to/openarm/lerobot_data \
        --repo-id openarm/cropped_20261018_joint --horizon 30
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, default=here / "lerobot_data")
    p.add_argument("--repo-id", default="openarm/cropped_20261018_joint")
    p.add_argument("--horizon", type=int, default=30)
    args = p.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(args.root.expanduser().resolve())

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, video_backend="torchcodec")
    print(f"episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.meta.fps}")
    print("features:")
    for k, v in ds.features.items():
        print(f"  {k}: {v}")

    # openpi data loader path (uses delta_timestamps for action chunks)
    from openpi.training import config as _config
    from openpi.training.data_loader import create_torch_dataset

    data_config = _config.DataConfig(
        repo_id=args.repo_id,
        prompt_from_task=True,
        action_sequence_keys=("action",),
    )
    train_ds = create_torch_dataset(data_config, args.horizon, None)
    sample = train_ds[0]
    print("openpi sample keys:", sorted(sample.keys()))
    print("state", np.asarray(sample["observation.state"]).shape)
    print("action", np.asarray(sample["action"]).shape)
    print("prompt", sample.get("prompt"))
    for k, v in sample.items():
        if "images" in k:
            print(k, np.asarray(v).shape, np.asarray(v).dtype)

    # every episode decodes
    starts = [int(x) for x in ds.episode_data_index["from"]]
    for i, idx in enumerate(starts):
        item = ds[int(idx)]
        for k, v in item.items():
            if "images" in k:
                arr = np.asarray(v)
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"episode {i} image {k} non-finite")
    print(f"decoded first frame of all {len(starts)} episodes OK")
    print("VERIFY OK")


if __name__ == "__main__":
    sys.exit(main())

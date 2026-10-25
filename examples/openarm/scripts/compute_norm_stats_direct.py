#!/usr/bin/env python3
"""Compute norm stats directly from LeRobot parquet state/action columns.

Avoids video decoding entirely (the normal data loader decodes 3 videos per
frame, which is slow and thread-hungry on the login node). The transforms are
replicated exactly:

  joint: state=qpos(16); action delta = command - state on arm dims, grippers absolute
  eef:   state=[L p3,q4,grip, R p3,q4,grip]; action delta = dp3+rotvec3+grip per arm

Action chunks use the same clamp-to-episode-end padding as LeRobot.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from openpi.shared import normalize
from openpi.training import config as cfg

ARM_L = list(range(0, 7))
GRIP_L = 7
ARM_R = list(range(8, 15))
GRIP_R = 15
ARM = ARM_L + ARM_R
GRIP = [GRIP_L, GRIP_R]


def rot_delta_world(q_cmd: np.ndarray, q_state: np.ndarray) -> np.ndarray:
    q_cmd = q_cmd / np.linalg.norm(q_cmd, axis=-1, keepdims=True)
    q_state = q_state / np.linalg.norm(q_state, axis=-1, keepdims=True)
    rc = Rotation.from_quat(q_cmd.reshape(-1, 4))
    rs = Rotation.from_quat(np.broadcast_to(q_state, q_cmd.shape).reshape(-1, 4))
    return (rc * rs.inv()).as_rotvec().reshape(q_cmd.shape[:-1] + (3,))


def ee_state(eef: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    # [L p3,q4, L grip, R p3,q4, R grip]
    return np.concatenate(
        [eef[:, :7], qpos[:, [GRIP_L]], eef[:, 7:14], qpos[:, [GRIP_R]]], axis=-1
    )


def ee_absolute_action(ae: np.ndarray, aj: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [ae[..., :7], aj[..., [GRIP_L]], ae[..., 7:14], aj[..., [GRIP_R]]], axis=-1
    )


def eef_delta(state: np.ndarray, actions_abs: np.ndarray) -> np.ndarray:
    # actions_abs: (T,H,16), state: (T,16) anchored at each t
    out = np.empty(actions_abs.shape[:-1] + (14,), dtype=np.float64)
    for s, o in ((0, 0), (8, 7)):
        p_s, q_s = state[:, s:s + 3], state[:, s + 3:s + 7]
        p_a, q_a = actions_abs[..., s:s + 3], actions_abs[..., s + 3:s + 7]
        out[..., o:o + 3] = p_a - p_s[:, None, :]
        out[..., o + 3:o + 6] = rot_delta_world(q_a, q_s[:, None, :])
        out[..., o + 6] = actions_abs[..., s + 7]
    return out


def joint_delta(state: np.ndarray, actions_abs: np.ndarray) -> np.ndarray:
    out = actions_abs.copy()
    out[..., ARM] -= state[:, None, ARM]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-root", default=None)
    args = p.parse_args()

    config = cfg.get_config(args.config)
    H = config.model.action_horizon
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    episodes = data_config.episodes
    if episodes is None:
        raise SystemExit("config has no episodes list")
    is_eef = "_eef" in args.config

    default_root = pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME", str(pathlib.Path.cwd() / "lerobot_data"))
    ) / repo_id
    root = pathlib.Path(args.dataset_root) if args.dataset_root else default_root
    cols = ["observation.state", "action"]
    if is_eef:
        cols += ["observation.eef", "action.eef"]

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    t0 = time.time()
    n_frames = 0
    for i, ep in enumerate(episodes):
        path = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        t = pq.read_table(path, columns=cols)
        qpos = np.stack(t["observation.state"].to_pylist()).astype(np.float64)
        act = np.stack(t["action"].to_pylist()).astype(np.float64)
        T = len(qpos)
        idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None, :], T - 1)
        if is_eef:
            eef = np.stack(t["observation.eef"].to_pylist()).astype(np.float64)
            ae = np.stack(t["action.eef"].to_pylist()).astype(np.float64)
            state = ee_state(eef, qpos)
            actions_abs = ee_absolute_action(ae[idx], act[idx])
            actions = eef_delta(state, actions_abs)
        else:
            state = qpos
            actions_abs = act[idx]
            actions = joint_delta(state, actions_abs)
        state_stats.update(state)
        action_stats.update(actions)
        n_frames += T
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(episodes)} episodes, {n_frames} frames", flush=True)

    out = config.assets_dirs / data_config.repo_id
    normalize.save(out, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
    print(f"config={args.config} eef={is_eef} episodes={len(episodes)} frames={n_frames} "
          f"elapsed={time.time() - t0:.1f}s -> {out}/norm_stats.json")
    print("state mean", np.round(state_stats.get_statistics().mean, 4))
    print("actions mean", np.round(action_stats.get_statistics().mean, 4))


if __name__ == "__main__":
    main()

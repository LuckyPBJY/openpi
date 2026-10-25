#!/usr/bin/env python3
"""P0: load a trained checkpoint and sanity-check inference on real dataset frames.

Checks:
  - output shape / NaN / range
  - quaternion norms (EE route)
  - action MAE against the recorded commands (training frames; NOT a
    generalization metric, only a pipeline sanity check)
  - predicted delta magnitude vs recorded delta magnitude
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import openarm_policy
from openpi.policies import policy_config
from openpi.training import config as cfg


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def parse_episodes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--episodes", default="0,50,117")
    p.add_argument("--offsets", default="100,200")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    config = cfg.get_config(args.config)
    repo_id = args.repo_id or config.data.repo_id
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"config={args.config} repo_id={repo_id} norm_stats={data_config.norm_stats is not None}")

    ds = LeRobotDataset(
        repo_id,
        delta_timestamps={k: [t / 30 for t in range(config.model.action_horizon)]
                          for k in data_config.action_sequence_keys},
        video_backend="torchcodec",
    )

    print("loading policy...")
    policy = policy_config.create_trained_policy(
        config, args.checkpoint, norm_stats=data_config.norm_stats
    )
    print("policy loaded")

    starts = [int(x) for x in ds.episode_data_index["from"]]
    ends = [int(x) for x in ds.episode_data_index["to"]]
    samples = []
    for ep in parse_episodes(args.episodes):
        for off in parse_episodes(args.offsets):
            idx = starts[ep] + off
            if idx < ends[ep]:
                samples.append((ep, off, idx))

    is_eef = "_eef" in args.config
    records = []
    for ep, off, idx in samples:
        s = ds[idx]
        images = {
            "head": np.asarray(s["observation.images.head"]),
            "left_wrist": np.asarray(s["observation.images.left_wrist"]),
            "right_wrist": np.asarray(s["observation.images.right_wrist"]),
        }
        qpos = np.asarray(s["observation.state"], dtype=np.float32)
        task = s["task"] if isinstance(s["task"], str) else s["task"].decode()

        if is_eef:
            eef = np.asarray(s["observation.eef"], dtype=np.float32)
            obs = {"images": images, "state_eef": eef, "state_joint": qpos, "prompt": task}
        else:
            obs = {"images": images, "state": qpos, "prompt": task}

        out = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
        finite = bool(np.isfinite(out).all())

        rec = {"episode": ep, "offset": off, "index": idx, "shape": list(out.shape), "finite": finite}
        if not finite:
            records.append(rec)
            continue

        if not is_eef:
            gt = np.asarray(s["action"], dtype=np.float64)  # (H,16) absolute joint commands
            arm = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
            grip = [7, 15]
            rec["mae_arm"] = float(np.abs(out[:, arm] - gt[:, arm]).mean())
            rec["mae_grip"] = float(np.abs(out[:, grip] - gt[:, grip]).mean())
            rec["mae_all"] = float(np.abs(out - gt).mean())
            rec["pred_min"] = float(out.min())
            rec["pred_max"] = float(out.max())
            # delta relative to the current qpos
            pred_delta = out[:, arm] - qpos[arm]
            gt_delta = gt[:, arm] - qpos[arm]
            rec["pred_delta_abs_mean"] = float(np.abs(pred_delta).mean())
            rec["gt_delta_abs_mean"] = float(np.abs(gt_delta).mean())
            rec["pred_delta_max"] = float(np.abs(pred_delta).max())
        else:
            ae = np.asarray(s["action.eef"], dtype=np.float64)  # (H,14)
            aj = np.asarray(s["action"], dtype=np.float64)      # (H,16)
            gt = np.concatenate([ae[:, :7], aj[:, [7]], ae[:, 7:14], aj[:, [15]]], axis=-1)
            # positions / quaternions / grippers
            lp, lq, lg = [0, 1, 2], [3, 4, 5, 6], 7
            rp, rq, rg = [8, 9, 10], [11, 12, 13, 14], 15
            rec["mae_pos"] = float(np.abs(out[:, lp + rp] - gt[:, lp + rp]).mean())
            rec["mae_grip"] = float(np.abs(out[:, [lg, rg]] - gt[:, [lg, rg]]).mean())
            rec["quat_angle_deg_mean"] = float(np.mean(
                np.concatenate([quat_angle_deg(out[:, lq], gt[:, lq]),
                                quat_angle_deg(out[:, rq], gt[:, rq])])))
            rec["quat_angle_deg_max"] = float(np.max(
                np.concatenate([quat_angle_deg(out[:, lq], gt[:, lq]),
                                quat_angle_deg(out[:, rq], gt[:, rq])])))
            qn = np.concatenate([np.linalg.norm(out[:, lq], axis=-1),
                                 np.linalg.norm(out[:, rq], axis=-1)])
            rec["quat_norm_min"] = float(qn.min())
            rec["quat_norm_max"] = float(qn.max())
            rec["pred_pos_min"] = float(out[:, lp + rp].min())
            rec["pred_pos_max"] = float(out[:, lp + rp].max())

            # predicted delta vs current corrected observation-frame state
            # Raw route: no eef_obs_to_cmd / mean-bias compensation.
            eef0 = np.asarray(s["observation.eef"], np.float32)
            grip0 = qpos[[7, 15]]
            curr = np.concatenate([eef0[:7], grip0[:1], eef0[7:14], grip0[1:2]]).astype(np.float64)
            for name, src in (("pred", out), ("gt", gt)):
                dp = np.linalg.norm(src[:, lp] - curr[lp], axis=-1)
                dr = np.linalg.norm(openarm_policy._rot_delta_world(src[:, lq], curr[lq]), axis=-1)
                rec[f"{name}_delta_pos_mean"] = float(dp.mean())
                rec[f"{name}_delta_pos_max"] = float(dp.max())
                rec[f"{name}_delta_rot_deg_mean"] = float(np.degrees(dr).mean())
                rec[f"{name}_delta_rot_deg_max"] = float(np.degrees(dr).max())

        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    # aggregate over valid samples
    keys = [k for k in records[0] if k not in ("episode", "offset", "index", "shape", "finite")] if records else []
    agg = {}
    for k in keys:
        vals = [r[k] for r in records if k in r and isinstance(r[k], (int, float))]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    print("AGGREGATE")
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"config": args.config, "checkpoint": args.checkpoint, "records": records, "aggregate": agg},
            indent=2, ensure_ascii=False))
        print("wrote", args.out)


if __name__ == "__main__":
    main()

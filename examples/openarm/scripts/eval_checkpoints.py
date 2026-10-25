#!/usr/bin/env python3
"""Evaluate one checkpoint on the held-out cropped_test episodes.

Offline success proxy: at sampled frames, compare the first predicted action
(and the action chunk) against the demonstrated command.

  joint: arm MAE <= 0.05 rad and gripper <= 0.005 m
  eef:   position <= 0.01 m, rotation <= 5 deg, gripper <= 0.005 m

Strict/loose thresholds are also reported. This is NOT closed-loop task success.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import re

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from openpi.policies import policy_config
from openpi.training import config as cfg

PROMPT = "manipulate the object with both arms"
CAMS = ("head", "left_wrist", "right_wrist")


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def decode_img(g, lengths, t):
    return np.asarray(Image.open(io.BytesIO(bytes(g[t][: int(lengths[t])]))).convert("RGB"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="cropped_test")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    p.add_argument("--episodes", default=None, help="comma-separated source episode numbers")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    config = cfg.get_config(args.config)
    H = config.model.action_horizon
    is_eef = "_eef" in args.config
    print(f"config={args.config} checkpoint={args.checkpoint} H={H} eef={is_eef}", flush=True)
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    print("policy loaded", flush=True)

    files = sorted(glob.glob(args.data_dir + "/*.hdf5"),
                   key=lambda x: int(re.findall(r"\d+", pathlib.Path(x).name)[0]))
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        files = [f for f in files if int(re.findall(r"\d+", pathlib.Path(f).name)[0]) in wanted]
    if args.max_episodes:
        files = files[: args.max_episodes]
    records = []
    for path in files:
        ep = int(re.findall(r"\d+", pathlib.Path(path).name)[0])
        with h5py.File(path, "r") as f:
            d = f["data/demo_0"]
            T = int(d["actions"].shape[0])
            qpos = d["observations/qpos"][:]
            actions = d["actions"][:]
            eef = d["observations/eef"][:]
            actions_eef = d["actions_eef"][:]
            imgs = {c: d[f"observations/images/{c}"] for c in CAMS}
            lens = {c: d[f"observations/images/{c}_lengths"][:] for c in CAMS}

            for t in range(30, T - H + 1, args.stride):
                images = {c: decode_img(imgs[c], lens[c], t) for c in CAMS}
                if is_eef:
                    obs = {"images": images, "state_eef": eef[t].astype(np.float32),
                           "state_joint": qpos[t].astype(np.float32), "prompt": PROMPT}
                else:
                    obs = {"images": images, "state": qpos[t].astype(np.float32), "prompt": PROMPT}

                pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
                if not np.isfinite(pred).all():
                    continue

                if not is_eef:
                    gt = actions[t:t + H].astype(np.float64)
                    arm = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
                    grip = [7, 15]
                    arm_first = float(np.abs(pred[0, arm] - gt[0, arm]).mean())
                    grip_first = float(np.abs(pred[0, grip] - gt[0, grip]).mean())
                    arm_chunk = float(np.abs(pred[:, arm] - gt[:, arm]).mean())
                    grip_chunk = float(np.abs(pred[:, grip] - gt[:, grip]).mean())
                    rec = {"episode": ep, "t": t, "arm_first": arm_first, "grip_first": grip_first,
                           "arm_chunk": arm_chunk, "grip_chunk": grip_chunk,
                           "success_strict": arm_first <= 0.02 and grip_first <= 0.002,
                           "success_medium": arm_first <= 0.05 and grip_first <= 0.005,
                           "success_loose": arm_first <= 0.10 and grip_first <= 0.01}
                else:
                    ae = actions_eef[t:t + H].astype(np.float64)
                    aj = actions[t:t + H].astype(np.float64)
                    gt = np.concatenate([ae[:, :7], aj[:, [7]], ae[:, 7:14], aj[:, [15]]], axis=-1)
                    lp, lq, lg = [0, 1, 2], [3, 4, 5, 6], 7
                    rp, rq, rg = [8, 9, 10], [11, 12, 13, 14], 15
                    pos_first = float(np.mean([
                        np.linalg.norm(pred[0, lp] - gt[0, lp]),
                        np.linalg.norm(pred[0, rp] - gt[0, rp]),
                    ]))
                    rot_first = float(np.mean([
                        quat_angle_deg(pred[0, lq], gt[0, lq]),
                        quat_angle_deg(pred[0, rq], gt[0, rq]),
                    ]))
                    grip_first = float(np.abs(pred[0, [lg, rg]] - gt[0, [lg, rg]]).mean())
                    pos_chunk = float(np.mean([
                        np.linalg.norm(pred[:, lp] - gt[:, lp], axis=-1).mean(),
                        np.linalg.norm(pred[:, rp] - gt[:, rp], axis=-1).mean(),
                    ]))
                    rot_chunk = float(np.mean([
                        quat_angle_deg(pred[:, lq], gt[:, lq]).mean(),
                        quat_angle_deg(pred[:, rq], gt[:, rq]).mean(),
                    ]))
                    grip_chunk = float(np.abs(pred[:, [lg, rg]] - gt[:, [lg, rg]]).mean())
                    rec = {"episode": ep, "t": t, "pos_first": pos_first, "rot_first": rot_first,
                           "grip_first": grip_first, "pos_chunk": pos_chunk, "rot_chunk": rot_chunk,
                           "grip_chunk": grip_chunk,
                           "success_strict": pos_first <= 0.005 and rot_first <= 2.0 and grip_first <= 0.002,
                           "success_medium": pos_first <= 0.01 and rot_first <= 5.0 and grip_first <= 0.005,
                           "success_loose": pos_first <= 0.02 and rot_first <= 10.0 and grip_first <= 0.01}
                records.append(rec)

    def agg_success(key):
        vals = [bool(r[key]) for r in records]
        if not vals:
            return {"frame": 0.0, "episode_mean": 0.0, "episode_ge80": 0.0}
        per_ep = {}
        for r in records:
            per_ep.setdefault(r["episode"], []).append(bool(r[key]))
        ep_means = [float(np.mean(v)) for v in per_ep.values()]
        return {"frame": float(np.mean(vals)),
                "episode_mean": float(np.mean(ep_means)),
                "episode_ge80": float(np.mean([m >= 0.8 for m in ep_means]))}

    error_keys = (["arm_first", "grip_first", "arm_chunk", "grip_chunk"] if not is_eef
                  else ["pos_first", "rot_first", "grip_first", "pos_chunk", "rot_chunk", "grip_chunk"])
    errors = {}
    for k in error_keys:
        vals = np.array([r[k] for r in records], dtype=np.float64)
        errors[k] = {"median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)),
                     "mean": float(vals.mean())}

    out = {
        "config": args.config, "checkpoint": args.checkpoint,
        "n_samples": len(records), "n_episodes": len({r["episode"] for r in records}),
        "success": {k: agg_success(f"success_{k}") for k in ("strict", "medium", "loose")},
        "errors": errors,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    m = out["success"]["medium"]
    print(f"RESULT {args.config} {pathlib.Path(args.checkpoint).name} "
          f"n={out['n_samples']} ep={out['n_episodes']} "
          f"succ strict/med/loose={out['success']['strict']['frame']:.3f}/"
          f"{m['frame']:.3f}/{out['success']['loose']['frame']:.3f} "
          f"ep_mean={m['episode_mean']:.3f} ep>=80%={m['episode_ge80']:.3f} "
          f"{ {k: round(v['median'], 4) for k, v in errors.items()} }", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize checkpoint evaluation JSONs and recommend a checkpoint."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

DEFAULT_EVAL_ROOT = pathlib.Path(__file__).resolve().parents[1] / "logs"


def load(route: str, eval_root: pathlib.Path):
    rows = []
    for f in glob.glob(str(eval_root / f"eval_{route}" / "*.json")):
        d = json.loads(pathlib.Path(f).read_text())
        step = int(re.findall(r"(\d+)\.json$", f)[0])
        e = d["errors"]
        rows.append({
            "step": step,
            "n": d["n_samples"],
            "episodes": d["n_episodes"],
            "strict": d["success"]["strict"]["frame"],
            "medium": d["success"]["medium"]["frame"],
            "loose": d["success"]["loose"]["frame"],
            "ep_mean": d["success"]["medium"]["episode_mean"],
            "ep_ge80": d["success"]["medium"]["episode_ge80"],
            "err_med": (e["arm_first"]["median"] if route == "joint" else e["pos_first"]["median"]),
            "err_p90": (e["arm_first"]["p90"] if route == "joint" else e["pos_first"]["p90"]),
            "rot_med": e.get("rot_first", {}).get("median"),
            "grip_med": e["grip_first"]["median"],
        })
    return sorted(rows, key=lambda r: r["step"])


def recommendation(route: str, rows):
    if not rows:
        return None, "no data"
    # Primary: medium frame success; tie-break strict success, then lower median error,
    # then earlier checkpoint (less overfitting).
    best = max(rows, key=lambda r: (round(r["medium"], 6), round(r["strict"], 6),
                                    -r["err_med"], -r["step"]))
    reasons = [
        f"medium-threshold success {best['medium']:.3f} (strict {best['strict']:.3f})",
        f"episode-mean success {best['ep_mean']:.3f}, episodes >=80% success {best['ep_ge80']:.3f}",
        f"first-step median error {best['err_med']:.4g}, p90 {best['err_p90']:.4g}",
        f"gripper median {best['grip_med']:.4g}",
    ]
    if route == "eef":
        reasons.append(f"rotation median {best['rot_med']:.4g} deg")
    return best, "; ".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=pathlib.Path, default=DEFAULT_EVAL_ROOT,
                        help="directory containing eval_joint/ and eval_eef/")
    args = parser.parse_args()
    for route in ("joint", "eef"):
        rows = load(route, args.eval_root)
        print(f"\n### {route.upper()}")
        if not rows:
            print("no results yet")
            continue
        if route == "joint":
            print("| step | n | strict | medium | loose | ep_mean | ep>=80% | arm_med(rad) | arm_p90 | grip_med(m) |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                print(f"| {r['step']} | {r['n']} | {r['strict']:.3f} | {r['medium']:.3f} | "
                      f"{r['loose']:.3f} | {r['ep_mean']:.3f} | {r['ep_ge80']:.3f} | "
                      f"{r['err_med']:.4f} | {r['err_p90']:.4f} | {r['grip_med']:.4f} |")
        else:
            print("| step | n | strict | medium | loose | ep_mean | ep>=80% | pos_med(m) | pos_p90 | rot_med(deg) | grip_med(m) |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                print(f"| {r['step']} | {r['n']} | {r['strict']:.3f} | {r['medium']:.3f} | "
                      f"{r['loose']:.3f} | {r['ep_mean']:.3f} | {r['ep_ge80']:.3f} | "
                      f"{r['err_med']:.4f} | {r['err_p90']:.4f} | {r['rot_med']:.4f} | {r['grip_med']:.4f} |")
        best, why = recommendation(route, rows)
        print(f"\n**recommended {route}: step {best['step']}** — {why}")


if __name__ == "__main__":
    main()

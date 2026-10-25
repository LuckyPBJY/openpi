#!/usr/bin/env python3
"""Convert OpenArm HDF5 episodes (cropped_20261018) to a LeRobot dataset.

Joint-space route (route A):
    observation.state = observations/qpos  (16,)  # 7 joints + 1 gripper per arm, rad / m
    action            = actions           (16,)  # controller commands, rad / m
    images            = head (1280x720), left_wrist (640x480), right_wrist (640x480)

Delta conversion is intentionally NOT applied here: it belongs in the openpi
data config (``DeltaActions``) so the stored dataset remains absolute.

Usage (from the openpi repo, using its venv):
    .venv/bin/python /path/to/scripts/convert_openarm_data_to_lerobot.py \
        --data-dir /path/to/openarm/cropped_20261018 \
        --root /path/to/openarm/lerobot_data \
        --repo-id openarm/cropped_20261018_joint \
        --task "manipulate the object with both arms"
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import pathlib
import re
import shutil
import time

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

CAMS = ("head", "left_wrist", "right_wrist")
IMG_SHAPES = {
    "head": (720, 1280, 3),
    "left_wrist": (480, 640, 3),
    "right_wrist": (480, 640, 3),
}
STATE_DIM = 16
ACTION_DIM = 16
EEF_DIM = 14  # 3 pos + 4 quat (xyzw) per arm


@dataclasses.dataclass(frozen=True)
class Episode:
    source_file: str
    source_index: int


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=pathlib.Path, default=here / "cropped_20261018")
    p.add_argument("--root", type=pathlib.Path, default=here / "lerobot_data",
                   help="HF_LEROBOT_HOME; dataset is written to <root>/<repo-id>")
    p.add_argument("--repo-id", default="openarm/cropped_20261018_joint")
    p.add_argument("--task", default="manipulate the object with both arms",
                   help="language instruction stored for every frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--robot-type", default="openarm")
    p.add_argument("--episodes", type=str, default=None,
                   help="comma-separated source episode numbers, e.g. '0,1,5' (default: all)")
    p.add_argument("--limit", type=int, default=None, help="convert only the first N episodes (smoke test)")
    p.add_argument("--image-writer-processes", type=int, default=2)
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--vcodec", default="h264", choices=("h264", "libsvtav1", "hevc"),
                   help="video codec; h264 is much faster than lerobot's AV1 default")
    p.add_argument("--decoder", default="torchvision", choices=("torchvision", "cv2", "pil"))
    p.add_argument("--png-compress-level", type=int, default=0,
                   help="PIL PNG compression for temporary frames (0 is fastest; files are deleted)")
    p.add_argument("--include-eef", action="store_true",
                   help="also store observations/eef and actions_eef (14-dim world pose, quat xyzw)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def list_episodes(data_dir: pathlib.Path) -> list[Episode]:
    eps = []
    for name in os.listdir(data_dir):
        if not name.endswith(".hdf5"):
            continue
        m = re.search(r"(\d+)", name)
        if m is None:
            continue
        eps.append(Episode(source_file=name, source_index=int(m.group(1))))
    eps.sort(key=lambda e: e.source_index)
    return eps


def make_decoder(kind: str):
    if kind == "torchvision":
        import torch
        import torchvision.io as tio

        def decode(packed: np.ndarray, length: int) -> np.ndarray:
            buf = torch.frombuffer(bytearray(packed[:length]), dtype=torch.uint8)
            return tio.decode_jpeg(buf).permute(1, 2, 0).numpy()  # CHW -> HWC, RGB uint8

    elif kind == "cv2":
        import cv2

        def decode(packed: np.ndarray, length: int) -> np.ndarray:
            bgr = cv2.imdecode(np.frombuffer(bytes(packed[:length]), np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    else:

        def decode(packed: np.ndarray, length: int) -> np.ndarray:
            img = Image.open(io.BytesIO(bytes(packed[:length]))).convert("RGB")
            return np.asarray(img, dtype=np.uint8)

    return decode


def make_features(include_eef: bool = False) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["action"]},
    }
    if include_eef:
        # observations/eef = FK pose of qpos; actions_eef = FK pose of the controller command.
        # Layout per arm: xyz + quat(xyzw) in world frame.
        features["observation.eef"] = {"dtype": "float32", "shape": (EEF_DIM,), "names": ["eef"]}
        features["action.eef"] = {"dtype": "float32", "shape": (EEF_DIM,), "names": ["eef"]}
    for cam in CAMS:
        features[f"observation.images.{cam}"] = {
            # "video" makes lerobot encode frames into per-episode mp4 files.
            # "image" would embed PNG bytes directly in the parquet (far too large here).
            "dtype": "video",
            "shape": IMG_SHAPES[cam],
            "names": ["height", "width", "channel"],
        }
    return features


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    # HF_LEROBOT_HOME is read at import time by lerobot.
    os.environ["HF_LEROBOT_HOME"] = str(args.root.expanduser().resolve())

    from lerobot.common.datasets import image_writer as _image_writer
    from lerobot.common.datasets import lerobot_dataset as _lerobot_dataset
    from lerobot.common.datasets import video_utils as _video_utils
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # lerobot hard-codes AV1 (libsvtav1) for episode video encoding, which is very slow.
    # Patch the module-level function used by LeRobotDataset.encode_episode_videos.
    _orig_encode = _video_utils.encode_video_frames

    def _encode(imgs_dir, video_path, fps, **kwargs):
        kwargs.setdefault("vcodec", args.vcodec)
        return _orig_encode(imgs_dir, video_path, fps, **kwargs)

    _lerobot_dataset.encode_video_frames = _encode

    # Temporary frames are PNGs and PIL's default zlib level (6) dominates conversion time
    # (~300 ms per 720p image). They are deleted after video encoding, so use level 0.
    # The image-writer processes are forked, so patch before LeRobotDataset.create.
    _orig_write_image = _image_writer.write_image

    def _write_image(image, fpath):
        if str(fpath).endswith(".png") and args.png_compress_level >= 0:
            if isinstance(image, np.ndarray):
                img = _image_writer.image_array_to_pil_image(image)
            else:
                img = image
            img.save(fpath, compress_level=args.png_compress_level)
        else:
            _orig_write_image(image, fpath)

    _image_writer.write_image = _write_image

    decode_jpeg = make_decoder(args.decoder)

    episodes = list_episodes(args.data_dir)
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        episodes = [e for e in episodes if e.source_index in wanted]
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("no episodes selected")

    out_dir = pathlib.Path(os.environ["HF_LEROBOT_HOME"]) / args.repo_id
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {out_dir} (use --overwrite)")
        shutil.rmtree(out_dir)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=make_features(args.include_eef),
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    joint_names: list[str] = []
    manifest = []
    t_start = time.time()
    total_frames = 0

    for lerobot_ep_idx, ep in enumerate(tqdm(episodes, desc="episodes")):
        path = args.data_dir / ep.source_file
        with h5py.File(path, "r") as f:
            demo = f["data/demo_0"]
            n = int(demo["actions"].shape[0])
            qpos = demo["observations/qpos"][:].astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)
            eef = demo["observations/eef"][:].astype(np.float32) if args.include_eef else None
            eef_cmd = demo["actions_eef"][:].astype(np.float32) if args.include_eef else None
            packed = {cam: demo[f"observations/images/{cam}"][:] for cam in CAMS}
            lengths = {cam: demo[f"observations/images/{cam}_lengths"][:] for cam in CAMS}
            if not joint_names:
                joint_names = [x.decode() if isinstance(x, bytes) else str(x)
                               for x in demo.file["meta/joint_names"][:]]

        if qpos.shape != (n, STATE_DIM) or actions.shape != (n, ACTION_DIM):
            raise RuntimeError(f"{ep.source_file}: unexpected shapes {qpos.shape} {actions.shape}")
        if args.include_eef and (eef.shape != (n, EEF_DIM) or eef_cmd.shape != (n, EEF_DIM)):
            raise RuntimeError(f"{ep.source_file}: unexpected eef shapes {eef.shape} {eef_cmd.shape}")

        for t in range(n):
            frame = {
                "observation.state": qpos[t],
                "action": actions[t],
                "task": args.task,
            }
            if args.include_eef:
                frame["observation.eef"] = eef[t]
                frame["action.eef"] = eef_cmd[t]
            for cam in CAMS:
                frame[f"observation.images.{cam}"] = decode_jpeg(packed[cam][t], int(lengths[cam][t]))
            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += n
        manifest.append({"lerobot_episode_index": lerobot_ep_idx,
                         "source_file": ep.source_file,
                         "source_index": ep.source_index,
                         "num_frames": n,
                         "duration_s": n / args.fps})
        del packed, lengths, qpos, actions, eef, eef_cmd

    elapsed = time.time() - t_start
    meta = {
        "source_data_dir": str(args.data_dir.resolve()),
        "repo_id": args.repo_id,
        "root": str(out_dir),
        "fps": args.fps,
        "robot_type": args.robot_type,
        "task": args.task,
        "state_source": "observations/qpos",
        "action_source": "actions",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "include_eef": args.include_eef,
        "eef_features": ({"observation.eef": EEF_DIM, "action.eef": EEF_DIM,
                          "layout": "per arm: xyz + quat(xyzw), world frame"}
                         if args.include_eef else None),
        "joint_names": joint_names,
        "camera_features": {f"observation.images.{c}": list(IMG_SHAPES[c]) for c in CAMS},
        "num_episodes": len(manifest),
        "num_frames": total_frames,
        "duration_s": total_frames / args.fps,
        "elapsed_s": round(elapsed, 1),
        "episodes": manifest,
    }
    with open(out_dir / "conversion_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

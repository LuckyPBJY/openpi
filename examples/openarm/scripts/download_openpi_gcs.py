#!/usr/bin/env python3
"""Mirror a public GCS object/prefix into openpi's cache layout.

openpi's ``download.maybe_download`` routes ``gs://openpi-assets`` through
``gsutil`` and everything else through gcsfs. Neither works here: gsutil is not
installed and gcsfs/aiohttp ignores HTTP(S)_PROXY (DNS for
storage.googleapis.com fails without the proxy). This script uses plain HTTPS
through the proxy and the public JSON API instead.

Cache layout matches openpi: ``~/.cache/openpi/<bucket>/<object path>``.

Usage:
    HTTPS_PROXY=http://YOUR_PROXY_HOST:PORT \
    .venv/bin/python scripts/download_openpi_gcs.py \
        gs://openpi-assets/checkpoints/pi05_base
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
OBJ = "https://storage.googleapis.com/{bucket}/{name}"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def get_object_meta(bucket: str, name: str) -> dict | None:
    url = API.format(bucket=bucket) + "/" + urllib.parse.quote(name, safe="")
    try:
        return _get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def list_objects(bucket: str, prefix: str):
    token = None
    while True:
        q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        d = _get_json(API.format(bucket=bucket) + "?" + urllib.parse.urlencode(q))
        for item in d.get("items", []):
            yield item
        token = d.get("nextPageToken")
        if not token:
            break


def download_one(bucket: str, name: str, dest: pathlib.Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = OBJ.format(bucket=bucket, name=urllib.parse.quote(name))
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
    if done != expected_size:
        raise RuntimeError(f"size mismatch for {name}: got {done}, expected {expected_size}")
    os.replace(tmp, dest)
    dt = time.time() - t0
    print(f"  {name}  {done / 1e9:.2f} GB  {done / 1e6 / max(dt, 1e-9):.1f} MB/s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url", help="gs://bucket/path (object or directory prefix)")
    p.add_argument("--cache-dir", default=os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi"))
    args = p.parse_args()

    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "gs":
        raise SystemExit("expected a gs:// URL")
    bucket, path = parsed.netloc, parsed.path.strip("/")
    cache = pathlib.Path(args.cache_dir).expanduser().resolve()

    meta = get_object_meta(bucket, path)
    if meta is not None:
        dest = cache / bucket / path
        print(f"{args.url} -> {dest}")
        download_one(bucket, path, dest, int(meta["size"]))
        print("done")
        return

    prefix = path.rstrip("/") + "/"
    items = list(list_objects(bucket, prefix))
    if not items:
        raise SystemExit(f"no objects found under {args.url}")
    total = sum(int(x.get("size", 0)) for x in items)
    print(f"mirroring {len(items)} objects ({total / 1e9:.2f} GB) to {cache / bucket / path}")
    for i, item in enumerate(items, 1):
        name = item["name"]
        dest = cache / bucket / name
        print(f"[{i}/{len(items)}]", end=" ", flush=True)
        download_one(bucket, name, dest, int(item.get("size", 0)))
    print("done")


if __name__ == "__main__":
    main()

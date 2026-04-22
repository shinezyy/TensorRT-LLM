#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify a Motion JPEG AVI file contains a non-dark, non-constant frame.

AVI MJPEG stores each frame as a complete JPEG blob inside an ``00dc``
chunk. We extract the first JPEG (SOI=FFD8FF...EOI=FFD9) via raw byte
scan, decode with Pillow, and reject:
    (a) mean luminance below --dark-mean-max, OR
    (b) pixel std below --flat-std-max (i.e. constant frame).

Either condition alone is strong evidence of a broken generation path.

Exit code 0 when the frame passes both checks; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SOI = b"\xff\xd8\xff"
EOI = b"\xff\xd9"


def extract_first_jpeg(avi_bytes: bytes) -> bytes:
    start = avi_bytes.find(SOI)
    if start < 0:
        raise ValueError("no JPEG SOI marker found in file")
    end = avi_bytes.find(EOI, start)
    if end < 0:
        raise ValueError("no JPEG EOI marker found after SOI")
    return avi_bytes[start : end + len(EOI)]


def brightness_stats(img: Image.Image) -> tuple[float, float, float]:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float(arr.mean()), float(arr.std()), float(arr.max())


def verify(path: Path, dark_mean_max: float, flat_std_max: float) -> int:
    data = path.read_bytes()
    jpeg = extract_first_jpeg(data)
    img = Image.open(io.BytesIO(jpeg))
    w, h = img.size
    mean, std, maxv = brightness_stats(img)

    is_dark = mean <= dark_mean_max
    is_flat = std <= flat_std_max

    verdict = "PASS" if not (is_dark or is_flat) else "FAIL"
    print(
        f"{verdict} {path} size={w}x{h} mean={mean:.2f} std={std:.3f} max={maxv:.1f} "
        f"thresholds(dark_mean<={dark_mean_max}, flat_std<={flat_std_max})",
        flush=True,
    )
    if is_dark:
        print(f"  reason: mean luminance {mean:.2f} <= dark_mean_max {dark_mean_max}")
    if is_flat:
        print(f"  reason: pixel std {std:.3f} <= flat_std_max {flat_std_max}")

    return 0 if verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to AVI file (Motion JPEG)")
    parser.add_argument(
        "--dark-mean-max",
        type=float,
        default=5.0,
        help="Frame fails if mean luminance <= this value (0-255 scale).",
    )
    parser.add_argument(
        "--flat-std-max",
        type=float,
        default=1.0,
        help="Frame fails if pixel std <= this value (constant frame).",
    )
    args = parser.parse_args()
    if not args.video.exists():
        print(f"ERROR: no such file: {args.video}", file=sys.stderr)
        return 2
    return verify(args.video, args.dark_mean_max, args.flat_std_max)


if __name__ == "__main__":
    sys.exit(main())

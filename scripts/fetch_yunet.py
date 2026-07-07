#!/usr/bin/env python3
"""Fetch the YuNet face detection model used by the subject-anchor analysis.

Downloads face_detection_yunet_2023mar.onnx (~232 KB) from the official
opencv_zoo repository into models/ at the repo root. The analysis code in
src/video_analysis.py looks for it at:

    1. $BEATSYNC_YUNET_MODEL (explicit override / kill switch)
    2. models/face_detection_yunet_2023mar.onnx  <- this script's target
    3. otherwise the face path is disabled (motion/detail centroids still work)

Usage: python scripts/fetch_yunet.py [--force]
"""

import argparse
import os
import sys
import urllib.request

URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT_DIR, "models", "face_detection_yunet_2023mar.onnx")


def sha256_of(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if os.path.isfile(TARGET) and not args.force:
        if sha256_of(TARGET) == SHA256:
            print(f"Already present and verified: {TARGET}")
            return 0
        print("Existing file failed checksum; re-downloading.")

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    print(f"Downloading {URL}")
    tmp = TARGET + ".part"
    urllib.request.urlretrieve(URL, tmp)

    actual = sha256_of(tmp)
    if actual != SHA256:
        os.remove(tmp)
        print(f"Checksum mismatch (got {actual}); upstream may have changed. Aborting.")
        return 1
    os.replace(tmp, TARGET)
    print(f"Saved and verified: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

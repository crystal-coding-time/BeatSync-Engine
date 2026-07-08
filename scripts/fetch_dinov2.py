#!/usr/bin/env python3
"""Fetch the DINOv2 ViT-S/14 ONNX model used for visual-embedding diversity.

Downloads dinov2_vits14.onnx (~86.6 MB) from a public Hugging Face export into
models/ at the repo root. The embedding code in src/visual_embeddings.py looks
for it at:

    1. $BEATSYNC_EMBED_MODEL (explicit override / kill switch)
    2. models/dinov2_vits14.onnx  <- this script's target
    3. otherwise the feature is disabled (planner treats missing keys as zero
       penalty, so absence is silent-but-logged)

The model exports a single output ``output`` [1,384] (the pooled CLS embedding)
from input ``input`` [1,3,224,224]. See src/visual_embeddings.py for the exact
preprocessing (ImageNet-normalized 224x224 center crop).

Usage: python scripts/fetch_dinov2.py [--force]
"""

import argparse
import os
import sys
import urllib.request

URL = (
    "https://huggingface.co/sefaburak/dinov2-small-onnx/resolve/main/"
    "dinov2_vits14.onnx"
)
SHA256 = "4df36ef0716a8f17d984fc7546a3a5d670fda6911eb298592250cb9e26756063"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT_DIR, "models", "dinov2_vits14.onnx")


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
        actual = sha256_of(TARGET)
        if actual == SHA256:
            print(f"Already present and verified: {TARGET}")
            print(f"SHA256: {actual}")
            return 0
        print("Existing file failed checksum; re-downloading.")

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    print(f"Downloading {URL}")
    tmp = TARGET + ".part"
    urllib.request.urlretrieve(URL, tmp)

    actual = sha256_of(tmp)
    print(f"SHA256: {actual}")
    if actual != SHA256:
        os.remove(tmp)
        print(f"Checksum mismatch (got {actual}); upstream may have changed. Aborting.")
        return 1
    os.replace(tmp, TARGET)
    print(f"Saved and verified: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

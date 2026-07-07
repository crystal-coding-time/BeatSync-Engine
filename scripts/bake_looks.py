#!/usr/bin/env python3
"""Bake the built-in color looks into looks/ as HaldCLUT PNGs + .cube LUTs.

Each look is an ffmpeg core-filter chain applied to an identity Hald CLUT
(haldclutsrc=8 -> 512x512 PNG -> 64^3 LUT). The PNG is converted to a .cube
file so renders apply the look with the single-input lut3d filter (no extra
ffmpeg inputs, works in a plain -vf chain). Both artifacts are generated
from chains authored here, so they carry no third-party license.

Run from the repo root after changing a chain:  python3 scripts/bake_looks.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from logger import FFMPEG_EXE  # resolves bundled bin/ first, then PATH
from looks import HALD_LEVEL, LOOKS_DIR, png_to_cube

LOOKS = {
    'vintage': "curves=preset=vintage",
    'cross_process': "curves=preset=cross_process",
    'cool': "colorbalance=bs=0.18:bm=0.08:bh=0.04,eq=saturation=0.95",
    'warm': "colorbalance=rs=0.15:rm=0.06:rh=0.08:bs=-0.08,eq=saturation=1.05",
    'high_contrast': "curves=preset=strong_contrast,eq=saturation=1.12",
    'day_for_night': (
        "eq=brightness=-0.15:saturation=0.55,"
        "colorbalance=bs=0.25:bm=0.12:bh=0.05,"
        "curves=master='0/0 0.5/0.42 1/0.9'"
    ),
}


def bake_png(name: str, chain: str) -> str:
    png_path = os.path.join(LOOKS_DIR, f"{name}.png")
    cmd = [
        FFMPEG_EXE, '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'haldclutsrc={HALD_LEVEL}',
        '-vf', chain, '-frames:v', '1', '-update', '1', '-y', png_path,
    ]
    subprocess.run(cmd, check=True)
    return png_path


def main() -> None:
    os.makedirs(LOOKS_DIR, exist_ok=True)
    for name, chain in LOOKS.items():
        png_path = bake_png(name, chain)
        cube_path = os.path.join(LOOKS_DIR, f"{name}.cube")
        png_to_cube(png_path, cube_path)
        print(f"baked {name}: {os.path.basename(png_path)} + {os.path.basename(cube_path)}")


if __name__ == '__main__':
    main()

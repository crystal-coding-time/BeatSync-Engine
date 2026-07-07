#!/usr/bin/env python3
"""Color looks: baked HaldCLUT PNGs applied per segment via ffmpeg lut3d.

The committed artifacts are the small HaldCLUT PNGs in looks/ (baked by
scripts/bake_looks.py from filter chains authored in this repo, so no
third-party license applies). lut3d needs a .cube file; those are ~7 MB
each, so they are derived from the PNGs on demand and gitignored.
"""

import os
from typing import List, Tuple

from logger import ROOT_DIR

LOOKS_DIR = os.path.join(ROOT_DIR, 'looks')
HALD_LEVEL = 8  # 512x512 PNG, 64^3 LUT


def png_to_cube(png_path: str, cube_path: str) -> None:
    """Hald raster order (r fastest, then g, then b) matches .cube order."""
    from PIL import Image

    img = Image.open(png_path).convert('RGB')
    if img.size != (HALD_LEVEL ** 3, HALD_LEVEL ** 3):
        raise ValueError(f"Unexpected Hald image size {img.size} for {png_path}")
    tmp_path = cube_path + '.tmp'
    with open(tmp_path, 'w', encoding='ascii') as f:
        f.write(f"# Derived from {os.path.basename(png_path)}\n"
                f"LUT_3D_SIZE {HALD_LEVEL * HALD_LEVEL}\n")
        for r, g, b in img.getdata():
            f.write(f"{r / 255:.6f} {g / 255:.6f} {b / 255:.6f}\n")
    os.replace(tmp_path, cube_path)


def ensure_look_cubes() -> None:
    """Derive any missing/stale .cube next to its committed PNG."""
    if not os.path.isdir(LOOKS_DIR):
        return
    for entry in sorted(os.listdir(LOOKS_DIR)):
        if not entry.endswith('.png'):
            continue
        png_path = os.path.join(LOOKS_DIR, entry)
        cube_path = os.path.join(LOOKS_DIR, entry[:-4] + '.cube')
        if (not os.path.exists(cube_path)
                or os.path.getmtime(cube_path) < os.path.getmtime(png_path)):
            try:
                png_to_cube(png_path, cube_path)
            except Exception as e:
                print(f"   ⚠️  Could not derive LUT for look {entry}: {e}")


def list_looks() -> List[Tuple[str, str]]:
    """(display label, .cube path) pairs for the GUI; 'None' first."""
    choices: List[Tuple[str, str]] = [('None', '')]
    if os.path.isdir(LOOKS_DIR):
        for entry in sorted(os.listdir(LOOKS_DIR)):
            if entry.endswith('.cube'):
                label = entry[:-5].replace('_', ' ').title()
                choices.append((label, os.path.join(LOOKS_DIR, entry)))
    return choices

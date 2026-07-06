#!/usr/bin/env python3
"""Text overlay system.

Text entries (quotes, captions, watermarking lines — any text) are rendered
to transparent full-frame PNGs with Pillow, then composited per segment via
ffmpeg's core overlay filter with alpha fades. This avoids drawtext entirely:
Homebrew ffmpeg ships without libfreetype, and Pillow gives proper wrapping
and font control on every platform.

Planning distributes entries chronologically across the video, anchors them
on calmer segments (never a 'drop' cut), and lets one entry span consecutive
segments until it has been readable for ~3 seconds. Fades are cut-aligned:
each segment gets fade timings expressed in its own local clock.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

FADE_SECONDS = 0.35
MIN_READABLE_SECONDS = 3.0

_FONT_CANDIDATES = [
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'C:/Windows/Fonts/arialbd.ttf',
    'C:/Windows/Fonts/arial.ttf',
]


def find_font() -> Optional[str]:
    env_font = os.environ.get('BEATSYNC_FONT')
    if env_font and os.path.exists(env_font):
        return env_font
    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines() or ['']:
        words = raw_line.split()
        if not words:
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + ' ' + word
            if font.getlength(trial) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [text]


def render_text_png(text: str, target_size: Tuple[int, int], out_path: str,
                    position: str = 'bottom', scale: float = 1.0) -> Optional[str]:
    """Render one text entry to a transparent full-frame PNG. None if no font."""
    font_path = find_font()
    if not font_path:
        print("   ⚠️  Text overlay skipped: no usable font found (set BEATSYNC_FONT)")
        return None

    width, height = target_size
    font_size = max(14, int(height / 14 * max(0.2, float(scale))))
    font = ImageFont.truetype(font_path, font_size)
    stroke = max(2, font_size // 12)

    lines = _wrap_text(text, font, max_width=width * 0.86)
    wrapped = '\n'.join(lines)

    canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=stroke,
                                   align='center', spacing=font_size // 4)
    block_w = bbox[2] - bbox[0]
    block_h = bbox[3] - bbox[1]

    x = (width - block_w) / 2 - bbox[0]
    if position == 'top':
        y = height * 0.08 - bbox[1]
    elif position == 'center':
        y = (height - block_h) / 2 - bbox[1]
    else:  # bottom (lower third)
        y = height * 0.88 - block_h - bbox[1]

    # Soft shadow behind the stroke keeps text readable on bright footage.
    shadow_offset = max(2, font_size // 16)
    draw.multiline_text((x + shadow_offset, y + shadow_offset), wrapped, font=font,
                        fill=(0, 0, 0, 140), stroke_width=stroke, stroke_fill=(0, 0, 0, 140),
                        align='center', spacing=font_size // 4)
    draw.multiline_text((x, y), wrapped, font=font, fill=(255, 255, 255, 255),
                        stroke_width=stroke, stroke_fill=(0, 0, 0, 220),
                        align='center', spacing=font_size // 4)

    canvas.save(out_path, 'PNG')
    return out_path


def plan_text_overlays(texts: Sequence[str], segment_durations: Sequence[float],
                       planned_clip_sequence: Optional[Sequence[Dict]]) -> Dict[int, Tuple[str, float, float]]:
    """Map segment index -> (text, window_offset, window_duration).

    Entries are spread chronologically (one per equal timeline bin). Each is
    anchored on the longest non-drop segment in its bin and extended across
    following unclaimed segments until readable (~3s).
    """
    total = len(segment_durations)
    if not texts or total == 0:
        return {}

    def is_drop(i: int) -> bool:
        if not planned_clip_sequence or i >= len(planned_clip_sequence):
            return False
        clip = planned_clip_sequence[i] or {}
        return str(clip.get('target', '')) == 'drop'

    plan: Dict[int, Tuple[str, float, float]] = {}
    n = len(texts)
    for t_idx, text in enumerate(texts):
        lo = round(t_idx * total / n)
        hi = max(lo + 1, round((t_idx + 1) * total / n))
        candidates = [i for i in range(lo, min(hi, total)) if i not in plan]
        if not candidates:
            continue
        anchor = max(candidates, key=lambda i: (not is_drop(i), segment_durations[i]))

        window = [anchor]
        acc = segment_durations[anchor]
        j = anchor + 1
        while acc < MIN_READABLE_SECONDS and j < total and j not in plan:
            window.append(j)
            acc += segment_durations[j]
            j += 1

        offset = 0.0
        for i in window:
            plan[i] = (text, offset, acc)
            offset += segment_durations[i]
    return plan


def overlay_fade_times(window_offset: float, window_duration: float) -> Tuple[float, float]:
    """(fade_in_duration, fade_out_start) in the segment's local clock.

    ffmpeg's fade filter rejects negative start times, so the fade-in always
    starts at 0: continuation segments (offset past the fade) get duration 0,
    meaning "skip the fade-in filter"; partial overlaps get the remainder.
    """
    fade_in_duration = max(0.0, min(FADE_SECONDS, FADE_SECONDS - window_offset))
    fade_out_start = max(0.0, (window_duration - window_offset) - FADE_SECONDS)
    return fade_in_duration, fade_out_start

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

import bisect
import math
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


def find_font(preferred: Optional[str] = None) -> Optional[str]:
    """Resolve the render font file. `preferred` (the UI's font picker) wins
    when it points at a real file; otherwise the historic ladder applies:
    BEATSYNC_FONT env var, then the platform candidates."""
    if preferred:
        if os.path.exists(preferred):
            return preferred
        print(f"   ⚠️  Selected font not found ({preferred}) — using auto font")
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
                    position: str = 'bottom', scale: float = 1.0,
                    font_path: Optional[str] = None) -> Optional[str]:
    """Render one text entry to a transparent full-frame PNG. None if no font."""
    font_path = find_font(font_path)
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


def _parse_stamp(stamp: str) -> Optional[float]:
    """'15' / '1:23' / '1:23:45' → seconds. None when malformed or non-finite.

    The two-part branch keeps the historic `int(minutes) * 60 + float(seconds)`
    expression so valid '@m:ss' pins parse to bit-identical floats."""
    try:
        if ':' in stamp:
            parts = stamp.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                value = int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                value = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            else:
                return None
        else:
            value = float(stamp)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_text_entries(lines: Sequence[str]) -> List[Tuple[str, Optional[float]]]:
    """Parse entry lines into (text, pinned_time_seconds_or_None).

    A line starting with '@<time> ' pins the entry: '@15 Finish strong',
    '@1:23 Halfway there' or '@1:23:45 Deep cut'. Anything else is
    auto-placed. An '@'-led line whose stamp doesn't parse (or is nan/inf)
    is kept but unpinned, with the bad stamp token stripped so it never
    renders on screen — a typo'd timestamp shouldn't burn '@1:2:3:4' into
    the video or silently drop the caption.
    """
    entries: List[Tuple[str, Optional[float]]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        pin = None
        if line.startswith('@'):
            head, _, rest = line.partition(' ')
            pin = _parse_stamp(head[1:])
            if pin is None:
                print(f"   ⚠️  Text pin {head!r} not understood (use @15, @1:23 "
                      f"or @1:23:45) — placing the line automatically")
            line = rest.strip()
        if line:
            entries.append((line, pin))
    return entries


def _segment_index_at(cut_times: Sequence[float], time_s: float) -> int:
    """Index i with cut_times[i] <= time_s < cut_times[i+1] (cut_times is
    sorted ascending). Falls back to the last segment index when time_s lies
    outside the timeline, matching the previous linear scan."""
    i = bisect.bisect_right(cut_times, time_s) - 1
    if 0 <= i < len(cut_times) - 1:
        return i
    return len(cut_times) - 2


def _occupied_segments(cut_times: Sequence[float], start: float, end: float) -> Tuple[int, int]:
    """First/last segment a (start, end) window occupies, by the same >0.01s
    overlap rule the segment projection uses. (-1, -1) if it touches none.

    Overlap requires cut_times[i+1] > start and cut_times[i] < end, so bisect
    narrows the candidates to the few segments the window spans; the exact
    original overlap comparison then decides (no epsilon arithmetic on the
    bisect side, so float behavior is unchanged)."""
    lo = max(0, bisect.bisect_right(cut_times, start) - 1)
    hi = min(len(cut_times) - 2, bisect.bisect_left(cut_times, end) - 1)
    first = last = -1
    for i in range(lo, hi + 1):
        if min(end, cut_times[i + 1]) - max(start, cut_times[i]) > 0.01:
            if first < 0:
                first = i
            last = i
    return first, last


def _solve_placements(entries: Sequence[Tuple[str, Optional[float]]],
                      cut_times: List[float],
                      snap_points: List[float],
                      planned_clip_sequence: Optional[Sequence[Dict]],
                      timeline_start: float, timeline_end: float,
                      min_durations: Optional[Sequence[Optional[float]]] = None,
                      ) -> List[Tuple[str, float, float]]:
    """Give every entry its own disjoint window on the audio timeline.

    Pinned entries claim their time first; auto entries then flow around them
    (centered in equal timeline shares, retried after/before a clashing
    window, skipped when no room remains). Returns the placed windows sorted
    by start time.

    min_durations (aligned with entries, None per entry = default) lets an
    entry demand a LONGER window than the readability cap — Motion-style
    progress widgets need their full '[widget:progress duration:Ns]' run.
    When any entry has one, min-duration entries seat before plain auto
    entries and every entry may retry into remaining free gaps. None (the
    classic path) keeps the historic window math bit-for-bit."""
    def segment_at(time_s: float) -> int:
        return _segment_index_at(cut_times, time_s)

    def is_drop_segment(i: int) -> bool:
        if not planned_clip_sequence or i >= len(planned_clip_sequence):
            return False
        return str((planned_clip_sequence[i] or {}).get('target', '')) == 'drop'

    def snap(time_s: float, radius: float) -> float:
        nearby = [b for b in snap_points if abs(b - time_s) <= radius]
        if not nearby:
            return time_s
        # Prefer beats that don't start on a drop cut, then the closest.
        return min(nearby, key=lambda b: (is_drop_segment(segment_at(b)), abs(b - time_s)))

    n = len(entries)
    share = (timeline_end - timeline_start) / n
    duration_cap = max(1.0, min(MIN_READABLE_SECONDS + 2 * FADE_SECONDS, share * 0.95))

    def cap_for(k: int) -> float:
        want = min_durations[k] if min_durations else None
        if not want:
            return duration_cap
        return max(duration_cap, min(want + 2 * FADE_SECONDS,
                                     (timeline_end - timeline_start) * 0.95))

    placed: List[Tuple[str, float, float]] = []

    def occupied_segments(start: float, end: float) -> Tuple[int, int]:
        return _occupied_segments(cut_times, start, end)

    def overlapping(start: float, end: float) -> Optional[Tuple[str, float, float]]:
        # A segment can only carry one text overlay, so windows clash when
        # they intersect in time OR merely share a segment.
        first, last = occupied_segments(start, end)
        for w in placed:
            if min(end, w[2]) - max(start, w[1]) > 0.0:
                return w
            w_first, w_last = occupied_segments(w[1], w[2])
            if first >= 0 and w_first >= 0 and first <= w_last and w_first <= last:
                return w
        return None

    def push_after(clash: Tuple[str, float, float]) -> float:
        # Start after the clashing window AND outside the last segment it
        # occupies, so the retry cannot clash with the same window again.
        _, last = occupied_segments(clash[1], clash[2])
        seg_exit = cut_times[min(last + 1, len(cut_times) - 1)] if last >= 0 else clash[2]
        return max(clash[2] + 0.15, seg_exit)

    def pull_before(clash: Tuple[str, float, float]) -> float:
        # End before the clashing window AND before the first segment it occupies.
        first, _ = occupied_segments(clash[1], clash[2])
        seg_entry = cut_times[first] if first >= 0 else clash[1]
        return min(clash[1] - 0.15, seg_entry)

    # Pinned entries claim their time first; auto entries then flow around them.
    for k, (text, pin) in enumerate(entries):
        if pin is None:
            continue
        cap = cap_for(k)
        start = snap(max(timeline_start, min(pin, timeline_end - cap)), radius=1.5)
        end = min(start + cap, timeline_end)
        clash = overlapping(start, end)
        while clash is not None:
            start = push_after(clash)
            end = start + cap
            clash = overlapping(start, end) if end <= timeline_end else None
        if end > timeline_end or end - start < 0.8:
            print(f"   ⚠️  Text overlay skipped (no room at pinned time): {text[:40]!r}")
            continue
        placed.append((text, start, end))

    # Gate: the widened solver below runs ONLY when some entry actually
    # demands a min duration. The classic path (min_durations absent or
    # all-None) must stay bit-identical to the historic solver, so it keeps
    # the original loop verbatim and never evaluates the new logic.
    has_min = min_durations is not None and any(
        d is not None and d > 0 for d in min_durations)

    if not has_min:
        for k, (text, pin) in enumerate(entries):
            if pin is not None:
                continue
            cap = cap_for(k)
            share_start = timeline_start + k * share
            start = snap(share_start + (share - cap) / 2, radius=share / 3)
            end = start + cap
            clash = overlapping(start, end)
            if clash is not None:
                # Try after the clashing window, then before it.
                after_start = push_after(clash)
                after = (after_start, after_start + cap)
                before_end = pull_before(clash)
                before = (before_end - cap, before_end)
                if after[1] <= timeline_end and overlapping(*after) is None:
                    start, end = after
                elif before[0] >= timeline_start and overlapping(*before) is None:
                    start, end = before
                else:
                    print(f"   ⚠️  Text overlay skipped (no room left on timeline): {text[:40]!r}")
                    continue
            placed.append((text, start, end))

        return sorted(placed, key=lambda w: w[1])

    # Widened solver: min-duration entries seat right after pinned ones (a
    # widget window overflows its timeline share, so it must claim space
    # before the plain neighbours fill it), and every entry gets a final
    # retry into the remaining free gaps instead of only once-after +
    # once-before. All arithmetic is deterministic — no rng.
    def free_gaps() -> List[Tuple[float, float]]:
        # Placed windows block their full segment span (the one-overlay-per-
        # segment rule), so gaps are computed between segment-expanded blocks.
        blocks: List[Tuple[float, float]] = []
        for w in placed:
            first, last = occupied_segments(w[1], w[2])
            if first >= 0:
                blocks.append((min(w[1], cut_times[first]),
                               max(w[2], cut_times[last + 1])))
            else:
                blocks.append((w[1], w[2]))
        blocks.sort()
        gaps: List[Tuple[float, float]] = []
        cursor = timeline_start
        for bs, be in blocks:
            if bs > cursor:
                gaps.append((cursor, bs))
            cursor = max(cursor, be)
        if timeline_end > cursor:
            gaps.append((cursor, timeline_end))
        return gaps

    def place_in_gap(desired: float, cap: float,
                     min_cap: Optional[float] = None) -> Optional[Tuple[float, float]]:
        gaps = free_gaps()
        # Prefer a gap that fits the full cap, nearest the desired start.
        best: Optional[Tuple[float, float, float]] = None
        for gs, ge in gaps:
            if ge - gs + 1e-9 < cap:
                continue
            start = min(max(desired, gs), ge - cap)
            dist = abs(start - desired)
            if best is None or dist < best[0] - 1e-9:
                best = (dist, start, cap)
        if best is None and min_cap is not None and min_cap < cap:
            # Nothing fits whole: shrink into the largest usable fragment
            # (least shrink first, then nearest) rather than dropping the line.
            for gs, ge in gaps:
                size = ge - gs
                if size + 1e-9 < min_cap:
                    continue
                eff = min(cap, size)
                start = min(max(desired, gs), ge - eff)
                dist = abs(start - desired)
                key = (-eff, dist)
                if best is None or key < (-best[2], best[0]):
                    best = (dist, start, eff)
        if best is None:
            return None
        start, eff = best[1], best[2]
        end = start + eff
        if overlapping(start, end) is not None:
            return None
        return start, end

    def attempt(k: int, cap: float, min_cap: Optional[float] = None,
                ) -> Optional[Tuple[float, float]]:
        share_start = timeline_start + k * share
        desired = max(timeline_start, min(share_start + (share - cap) / 2,
                                          timeline_end - cap))
        start = snap(desired, radius=share / 3)
        end = start + cap
        clash = overlapping(start, end)
        if clash is None:
            return start, end
        after_start = push_after(clash)
        after = (after_start, after_start + cap)
        before_end = pull_before(clash)
        before = (before_end - cap, before_end)
        if after[1] <= timeline_end and overlapping(*after) is None:
            return after
        if before[0] >= timeline_start and overlapping(*before) is None:
            return before
        return place_in_gap(desired, cap, min_cap=min_cap)

    order = ([k for k, (_, pin) in enumerate(entries)
              if pin is None and (min_durations[k] or 0) > 0]
             + [k for k, (_, pin) in enumerate(entries)
                if pin is None and not ((min_durations[k] or 0) > 0)])
    # Plain captions may shrink into leftover fragments (a shorter caption
    # beats a dropped one; pinned windows already accept 0.8s). Widget
    # windows never shrink below the classic cap — the bar would end early.
    shrink_floor = 0.8
    for k in order:
        text = entries[k][0]
        cap = cap_for(k)
        is_widget = (min_durations[k] or 0) > 0
        window = attempt(k, cap, min_cap=None if is_widget else shrink_floor)
        if window is None and cap > duration_cap:
            # No room for the full widget run anywhere — a shorter classic
            # window beats dropping the line (the widget just ends early).
            window = attempt(k, duration_cap)
        if window is None:
            print(f"   ⚠️  Text overlay skipped (no room left on timeline): {text[:40]!r}")
            continue
        placed.append((text, window[0], window[1]))

    return sorted(placed, key=lambda w: w[1])


def _project_onto_segments(schedule: List[Tuple[str, float, float]],
                           cut_times: List[float],
                           ) -> Dict[int, Tuple[str, float, float, float]]:
    """Project each window onto every segment it intersects, translating fade
    timings into the segment's local clock."""
    seg_map: Dict[int, Tuple[str, float, float, float]] = {}
    for text, ws, we in schedule:
        for i in range(len(cut_times) - 1):
            seg_start, seg_end = cut_times[i], cut_times[i + 1]
            if min(we, seg_end) - max(ws, seg_start) <= 0.01:
                continue
            local_start = ws - seg_start
            local_end = we - seg_start
            if local_start >= 0:
                fade_in_start, fade_in_duration = local_start, FADE_SECONDS
            else:
                # The fade began in an earlier segment. Restarting the
                # remainder here would drop alpha back to 0 at the cut (a
                # visible pop), so continuation segments show the text at
                # full opacity instead; ffmpeg's fade rejects st<0 anyway.
                fade_in_start, fade_in_duration = 0.0, 0.0
            fade_out_start = max(0.0, local_end - FADE_SECONDS)
            seg_map[i] = (text, fade_in_start, fade_in_duration, fade_out_start)
    return seg_map


def plan_text_windows(entries: Sequence[Tuple[str, Optional[float]]],
                      cut_times: Sequence[float],
                      beat_times: Optional[Sequence[float]] = None,
                      planned_clip_sequence: Optional[Sequence[Dict]] = None,
                      min_durations: Optional[Sequence[Optional[float]]] = None,
                      ) -> Tuple[Dict[int, Tuple[str, float, float, float]], List[Tuple[str, float, float]]]:
    """Plan text on the global audio timeline, then project onto segments.

    Every entry gets its own disjoint time window: unpinned entries are
    centered in equal timeline shares, pinned entries go where asked; window
    starts snap to the nearest beat (preferring non-drop segments). Windows
    are then projected onto every segment they intersect, with fade timings
    translated into each segment's local clock.

    Returns (seg_map, schedule):
      seg_map:  segment index -> (text, fade_in_start, fade_in_duration, fade_out_start)
      schedule: [(text, window_start, window_end)] for logging.
    """
    cut_times = [float(t) for t in cut_times]
    if not entries or len(cut_times) < 2:
        return {}, []
    timeline_start, timeline_end = cut_times[0], cut_times[-1]
    if timeline_end - timeline_start <= 0.5:
        return {}, []

    snap_points = sorted(float(b) for b in (beat_times if beat_times is not None and len(beat_times) else cut_times))
    schedule = _solve_placements(entries, cut_times, snap_points,
                                 planned_clip_sequence, timeline_start, timeline_end,
                                 min_durations=min_durations)
    seg_map = _project_onto_segments(schedule, cut_times)
    return seg_map, schedule

#!/usr/bin/env python3
"""
UI Content for BeatSync Engine
Focused on Auto Mode.
"""

# ============================================================================
# MAIN UI CONTENT
# ============================================================================

UI_TITLE = "🎵 BeatSync Engine"

UI_MAIN_DESCRIPTION = """Create music videos that cut to the beat. Upload audio and video clips 
to automatically generate a video synchronized with your music's rhythm."""

# ============================================================================
# STATUS MESSAGES
# ============================================================================

def get_ready_status():
    """Ready status message."""
    return '✅ Ready to process!\n\nUpload audio and video files to begin.'

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

def _format_auto_section_summary(sections_info):
    """Return a safe section summary for all Auto Mode versions.

    Supports both the older flat dictionaries:
        {"section": "chorus", "selected_beats": 8, "total_beats": 32, "selection_ratio": 0.25}
    and the newer V3.2 wave dictionaries:
        {"section": {"type": "chorus", ...}, "selected_count": 8, "beat_count": 32, "density": 0.25}
    """
    if not sections_info:
        return ""

    lines = ["Sections analyzed and processed:"]
    for item in sections_info[:12]:
        if not isinstance(item, dict):
            continue

        raw_section = item.get('section', item.get('type', 'section'))
        if isinstance(raw_section, dict):
            section_name = raw_section.get('type') or raw_section.get('section') or raw_section.get('name') or 'section'
        else:
            section_name = raw_section

        section_name = str(section_name).replace('_', ' ').strip().title() or 'Section'

        selected = item.get('selected_beats', item.get('selected_count', item.get('cuts', 0)))
        total = item.get('total_beats', item.get('beat_count', item.get('beats', 0)))
        ratio = item.get('selection_ratio', item.get('density', None))

        try:
            selected_i = int(selected)
        except Exception:
            selected_i = 0
        try:
            total_i = int(total)
        except Exception:
            total_i = 0

        if ratio is None:
            ratio = selected_i / total_i if total_i > 0 else 0.0
        try:
            ratio_f = float(ratio)
        except Exception:
            ratio_f = 0.0

        lines.append(f"      - {section_name}: {selected_i}/{total_i} beats ({ratio_f * 100:.1f}%)")

    if len(sections_info) > 12:
        lines.append(f"      - ... plus {len(sections_info) - 12} more sections")

    return "\n".join(lines)


def get_success_message_auto(total_cuts, total_beats, tempo, sections_info,
                            encoder_info, fps_info, filename,
                            audio_duration=None, output_fps=None,
                            total_processing_seconds=None, processing_label=None):
    """Success message for Auto mode. Compatible with Auto Mode V1/V2/V3/V3.2."""

    section_summary = _format_auto_section_summary(sections_info)
    try:
        audio_duration_f = float(audio_duration)
    except Exception:
        audio_duration_f = 0.0
    try:
        output_fps_f = float(output_fps)
    except Exception:
        output_fps_f = 0.0
    try:
        total_seconds_i = int(round(float(total_processing_seconds)))
    except Exception:
        total_seconds_i = 0
    processing_text = processing_label or encoder_info
    fps_text = f"{output_fps_f:.1f}" if output_fps_f else str(fps_info).split()[0]

    return f"""✅ Video created successfully!

Statistics:
Video processing: {processing_text}
Total cuts: {total_cuts}
Audio duration: {audio_duration_f:.2f} seconds
Output FPS: {fps_text}
{total_beats} beats detected at {tempo:.1f} BPM

{section_summary}
Total time processing: {total_seconds_i} seconds

Output: {filename}"""


# ============================================================================
# CONSOLE MESSAGES
# ============================================================================

CONSOLE_SEPARATOR = "=" * 70

# ============================================================================
# INPUT LABELS & INFO
# ============================================================================

LABEL_AUDIO_FILE = "🎵 Audio File (MP3/WAV/FLAC)"
LABEL_VIDEO_FILES = "🎥 Video Files (MP4/MKV/MOV/WebM/M4V/AVI/GIF)"

LABEL_CUSTOM_FPS = "🎞️ Custom FPS (Frame Rate)"
INFO_CUSTOM_FPS = "Leave empty for auto-detect, or enter value (24/30/60)"

LABEL_PROCESSING_MODE = "🎬 Processing Mode"

# Output
LABEL_OUTPUT_FILENAME = "📝 Output Filename"
INFO_OUTPUT_FILENAME = "Timestamp added automatically (.mkv or .mov)"

def get_processing_mode_info_nvenc():
    """Processing mode info with NVENC."""
    return 'GPU (NVENC): High quality | CPU: High quality | ProRes: Max quality'

def get_processing_mode_info_videotoolbox():
    """Processing mode info with Apple VideoToolbox."""
    return 'GPU (VideoToolbox): Fast hardware encode | CPU: High quality | ProRes: Max quality'

def get_processing_mode_info_cpu():
    """Processing mode info without NVENC."""
    return 'CPU: H.264 encoding | ProRes: Max quality (NVENC not available)'

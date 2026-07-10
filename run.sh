#!/usr/bin/env bash
# macOS/Linux equivalent of run.bat — uses the .venv created for this port.
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: .venv not found. Create it with:"
    echo "  python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt  (remove the cupy line first)"
    exit 1
fi

export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8

# Default to the beat-this transformer for beat AND downbeat tracking (cuts
# land on real bar lines). Respects a user-set value; the ~8 MB small0
# checkpoint auto-downloads on first use, and any failure falls back to
# librosa automatically. Export BEATSYNC_BEAT_BACKEND=librosa to opt out.
export BEATSYNC_BEAT_BACKEND="${BEATSYNC_BEAT_BACKEND:-beat_this}"

exec .venv/bin/python src/gui.py "$@"

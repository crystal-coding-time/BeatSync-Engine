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

exec .venv/bin/python src/gui.py "$@"

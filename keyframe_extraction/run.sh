#!/usr/bin/env bash
# Chạy cắt keyframe. Tham số thêm được chuyển thẳng cho extract_keyframes.py, vd:
#   ./run.sh --limit 1          # thử 1 video
#   ./run.sh --gpu 1            # dùng GPU 1
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$DIR/.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Chưa cài môi trường. Chạy ./setup.sh trước." >&2
    exit 1
fi
exec "$VENV_DIR/bin/python" "$DIR/extract_keyframes.py" "$@"

#!/usr/bin/env bash
# Cài môi trường (chạy 1 lần): tạo venv .venv/ và cài torch + OmniShotCut.
# Có thể đổi chỗ đặt venv:  VENV_DIR=/duong/dan/venv ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[venv] tạo $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip

# Trên Linux, torch trên PyPI đã kèm CUDA. Muốn bản CUDA cụ thể thì đặt
# TORCH_INDEX_URL, vd: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup.sh
if [ -n "${TORCH_INDEX_URL:-}" ]; then
    "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
    "$PY" -m pip install torch torchvision
fi
"$PY" -m pip install -r requirements.txt

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
PY
echo
echo "Cài xong. Sửa video_dir / output_dir trong config.yaml rồi chạy:  ./run.sh"

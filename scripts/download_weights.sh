#!/usr/bin/env bash
# Download the pretrained, frozen weights into ml-service/weights/.
#
#   bash scripts/download_weights.sh
#
# Needs: python with huggingface_hub and gdown (both in ml-service/requirements-ml.txt), curl.
# If the Hugging Face download fails behind a strict proxy, retry with HF_HUB_DISABLE_XET=1.
# Set PYTHON to choose the interpreter (default: ml-service/.venv/bin/python, else python3).
#
# What it fetches:
#   1. GenD (CLIP ViT-L/14), MIT          huggingface.co/yermandy/GenD_CLIP_L_14
#   2. CLIP ViT-L/14 config files only    huggingface.co/openai/clip-vit-large-patch14
#   3. MediaPipe Face Landmarker, Apache-2.0
#   4. SBI EfficientNet-B4, FF++ c23      Google Drive link in the official SBI README
#      Research use only: https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="$ROOT/ml-service/weights"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/ml-service/.venv/bin/python" ]; then PYTHON="$ROOT/ml-service/.venv/bin/python"; else PYTHON=python3; fi
fi
mkdir -p "$W/gend" "$W/clip-vit-large-patch14" "$W/sbi" "$W/mediapipe"

echo "[1/4] GenD (yermandy/GenD_CLIP_L_14)"
"$PYTHON" - "$W" <<'PY'
import sys
from huggingface_hub import hf_hub_download

w = sys.argv[1]
for name in ("config.json", "model.safetensors"):
    hf_hub_download("yermandy/GenD_CLIP_L_14", name, local_dir=f"{w}/gend")
print("[2/4] CLIP ViT-L/14 config files (no weights: GenD's checkpoint already holds them)")
for name in ("config.json", "preprocessor_config.json"):
    hf_hub_download("openai/clip-vit-large-patch14", name, local_dir=f"{w}/clip-vit-large-patch14")
PY

echo "[3/4] MediaPipe Face Landmarker"
MP="$W/mediapipe/face_landmarker.task"
[ -s "$MP" ] || curl -fL --retry 3 -o "$MP" \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

echo "[4/4] SBI EfficientNet-B4 (FF++ c23)"
SBI="$W/sbi/FFc23.tar"
SBI_ID="1X0-NYT8KPursLZZdxduRQju6E52hauV0"
if [ ! -s "$SBI" ]; then
  if ! "$PYTHON" -m gdown "$SBI_ID" -O "$SBI"; then
    rm -f "$SBI"
    cat <<MSG

Could not download the SBI weights automatically (Google Drive often blocks scripted
downloads). Do it by hand:
  1. Open https://drive.google.com/file/d/$SBI_ID/view  (the "c23" link in the SBI README)
  2. Download the file and save it as:  $SBI
The service still starts without it; real mode then reports SBI as missing.
MSG
    exit 2
  fi
fi

echo
echo "Done. Files in $W:"
(cd "$W" && find . -type f ! -name .gitkeep ! -path './.cache/*' -exec ls -lh {} \; | awk '{print $5, $9}')

#!/usr/bin/env bash
# Download model weights for scene-query.
# Run once before starting the application.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models/weights}"
mkdir -p "$MODELS_DIR"

echo "Downloading SAM ViT-H checkpoint..."
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_OUT="$MODELS_DIR/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_OUT" ]; then
    curl -L --progress-bar -o "$SAM_OUT" "$SAM_URL"
    echo "SAM downloaded: $SAM_OUT"
else
    echo "SAM already present: $SAM_OUT"
fi

echo ""
echo "CLIP weights are downloaded automatically by open_clip on first use."
echo ""
echo "Done. Place any additional model checkpoints in: $MODELS_DIR"

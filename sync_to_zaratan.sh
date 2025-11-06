#!/bin/bash
# Sync project to Zaratan using rsync
# Usage: ./sync_to_zaratan.sh

set -euo pipefail

echo "🚀 Syncing text2video project to Zaratan..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from your text2video project directory"
    exit 1
fi

# Sync to Zaratan
echo "📤 Uploading project files to Zaratan..."
rsync -av --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.ckpt' \
    --exclude '*.safetensors' \
    --exclude '*.onnx' \
    --exclude '*.mp4' \
    --exclude '*.webm' \
    --exclude '*.gif' \
    . vvr2211@zaratan.umd.edu:~/text2video/

echo ""
echo "✅ Sync complete!"
echo ""
echo "📋 Next steps:"
echo "1. SSH to Zaratan: ssh vvr2211@zaratan.umd.edu"
echo "2. Run setup script: bash ~/text2video/setup_zaratan.sh"
echo "   OR follow the manual setup in setup_zaratan.sh"

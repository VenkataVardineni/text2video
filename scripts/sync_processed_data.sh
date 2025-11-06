#!/bin/bash
# Sync preprocessed data from local machine to Zaratan
# Usage: ./scripts/sync_processed_data.sh

set -euo pipefail

echo "🚀 Syncing preprocessed data to Zaratan"
echo "========================================"

# Check if processed data exists
if [ ! -d "data/processed" ]; then
    echo "❌ Error: data/processed directory not found"
    echo "   Please run preprocessing first:"
    echo "   python code/download_msrvtt_local.py"
    exit 1
fi

echo "📁 Local data directory:"
du -sh data/processed/* 2>/dev/null | head -5

echo ""
echo "📤 Syncing to Zaratan..."
rsync -av --progress \
    data/processed/ \
    vvr2211@login.zaratan.umd.edu:~/scratch.msml612-fa25/data/

echo ""
echo "✅ Sync complete!"
echo ""
echo "📋 On Zaratan, verify:"
echo "   ls -lh ~/scratch.msml612-fa25/data/manifests/"
echo "   ls -lh ~/scratch.msml612-fa25/data/raw/videos/train/ | head -10"

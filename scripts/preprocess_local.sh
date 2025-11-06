#!/bin/bash
# Preprocess MSR-VTT data locally (no GPU needed)
# Usage: ./scripts/preprocess_local.sh [train|val|test]

set -euo pipefail

SPLIT=${1:-train}

echo "🚀 Preprocessing MSR-VTT data locally (split: $SPLIT)"
echo "====================================================="

# Check if required tools are installed
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "⚠️  ffmpeg not found (needed for video conversion)"; }

# Check if decord is installed (optional - will use ffprobe if not available)
python3 -c "import decord" 2>/dev/null || {
    echo "📦 Attempting to install decord (optional - will use ffprobe if fails)..."
    pip3 install decord==0.6.0 2>/dev/null || {
        echo "   ⚠️  Decord not available - will use ffprobe for video metadata"
        command -v ffprobe >/dev/null || {
            echo "   ⚠️  Please install ffmpeg: brew install ffmpeg"
        }
    }
}

# Check if other dependencies are installed
python3 -c "import pandas, tqdm" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip3 install pandas tqdm
}

# Check if yt-dlp is installed (for downloading)
command -v yt-dlp >/dev/null 2>&1 || {
    echo "📦 Installing yt-dlp..."
    pip3 install yt-dlp
}

# Create output directory
mkdir -p data/processed/{raw/videos/{train,val,test},manifests,logs}

# Set environment variables
export SPLIT=$SPLIT
export MSR_VTT_ROOT=./data/msr-vtt
export OUT_ROOT=./data/processed

echo ""
echo "📥 Processing $SPLIT split..."
python3 code/download_msrvtt_local.py

echo ""
echo "✅ Preprocessing complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Review manifests: cat data/processed/manifests/${SPLIT}.manifest.jsonl | head -3"
echo "   2. Sync to Zaratan: ./scripts/sync_processed_data.sh"
echo "   3. On Zaratan, update manifest paths:"
echo "      sed -i 's|raw/videos|/scratch/data/raw/videos|g' ~/scratch.msml612-fa25/data/manifests/${SPLIT}.manifest.jsonl"

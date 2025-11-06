#!/bin/bash
# Attempt to download MSR-VTT dataset
# Note: Official download requires registration, but we can try common sources

set -euo pipefail

echo "📥 Downloading MSR-VTT Dataset"
echo "==============================="
echo ""

mkdir -p data/msr-vtt/{annotation,videos/all}

# Try to download annotation file from common sources
echo "📋 Attempting to download MSR_VTT.json..."

# Common GitHub mirrors/repos that host MSR-VTT
ANNOTATION_URLS=(
    "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
    "https://github.com/ArrowLuo/CLIP4Clip/raw/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
)

ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"

if [ ! -f "$ANNOTATION_FILE" ]; then
    for url in "${ANNOTATION_URLS[@]}"; do
        echo "Trying: $url"
        if curl -L -f -o "$ANNOTATION_FILE" "$url" 2>/dev/null; then
            echo "✅ Downloaded annotation file from: $url"
            break
        fi
    done
    
    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "⚠️  Could not download annotation file automatically"
        echo ""
        echo "📋 Manual download options:"
        echo "   1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
        echo "   2. Download MSR_VTT.json and place it at:"
        echo "      $ANNOTATION_FILE"
        echo ""
        echo "   Or download from HuggingFace:"
        echo "   https://huggingface.co/datasets/ArrowLuo/CLIP4Clip"
        echo ""
        exit 1
    fi
else
    echo "✅ Annotation file already exists"
fi

# Check annotation file
if [ -f "$ANNOTATION_FILE" ]; then
    echo ""
    echo "📊 Annotation file info:"
    python3 << 'PY'
import json
with open("data/msr-vtt/annotation/MSR_VTT.json") as f:
    data = json.load(f)
    videos = data.get("videos", [])
    print(f"  Total videos: {len(videos)}")
    splits = {}
    for v in videos:
        split = v.get("split", "unknown")
        splits[split] = splits.get(split, 0) + 1
    print(f"  Splits: {splits}")
PY
fi

echo ""
echo "📹 Video download options:"
echo ""
echo "MSR-VTT videos are large (~10GB). Options:"
echo ""
echo "1. Download from official source (requires registration):"
echo "   http://ms-multimedia-challenge.com/2017/dataset"
echo ""
echo "2. Download videos on-demand during preprocessing (from URLs in annotation)"
echo "   This will download videos as needed from YouTube/other sources"
echo ""
echo "3. Use pre-downloaded videos:"
echo "   Place videos in: data/msr-vtt/videos/all/"
echo ""

read -p "Do you want to proceed with preprocessing? (videos will be downloaded on-demand) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "✅ Ready to preprocess!"
    echo "   Run: ./scripts/preprocess_local.sh train"
else
    echo ""
    echo "📋 When ready, run: ./scripts/preprocess_local.sh train"
fi

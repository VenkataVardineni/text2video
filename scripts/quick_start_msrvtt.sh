#!/bin/bash
# Quick start: Try to get MSR-VTT and start preprocessing

echo "🚀 Quick Start: MSR-VTT Preprocessing"
echo "======================================"

# Create directories
mkdir -p data/msr-vtt/annotation
mkdir -p data/processed/{raw/videos/{train,val,test},manifests,logs}

# Try to download annotation from a known source
ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "📥 Attempting to download MSR_VTT.json..."
    
    # Try direct download from a mirror (if available)
    # Note: These URLs may change, adjust as needed
    curl -L -f -o "$ANNOTATION_FILE" \
        "https://github.com/ArrowLuo/CLIP4Clip/raw/main/datasets/MSR-VTT/annotation/MSR_VTT.json" \
        2>/dev/null && echo "✅ Downloaded!" || {
        
        echo "⚠️  Automatic download failed"
        echo ""
        echo "📋 Please download MSR_VTT.json manually:"
        echo "   1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
        echo "   2. Or search for 'MSR-VTT dataset download'"
        echo "   3. Place file at: $ANNOTATION_FILE"
        echo ""
        echo "   Then run: ./scripts/preprocess_local.sh train"
        exit 1
    }
fi

# Check if we have the file now
if [ -f "$ANNOTATION_FILE" ]; then
    echo "✅ Found annotation file"
    echo ""
    echo "📊 Dataset info:"
    python3 << 'PY'
import json
with open("data/msr-vtt/annotation/MSR_VTT.json") as f:
    data = json.load(f)
    videos = data.get("videos", [])
    print(f"  Total videos: {len(videos)}")
    splits = {}
    for v in videos:
        split = v.get("split", "unknown").lower()
        splits[split] = splits.get(split, 0) + 1
    for split in ["train", "val", "test"]:
        print(f"  {split}: {splits.get(split, 0)} videos")
PY
    
    echo ""
    echo "🚀 Starting preprocessing..."
    echo ""
    ./scripts/preprocess_local.sh train
else
    echo "❌ Annotation file not found"
    exit 1
fi

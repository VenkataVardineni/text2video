#!/bin/bash
# Verify MSR-VTT dataset and start preprocessing
# Run this after you've downloaded MSR_VTT.json manually

set -euo pipefail

ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"

echo "🔍 Verifying MSR-VTT Dataset"
echo "============================"
echo ""

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "❌ MSR_VTT.json not found at: $ANNOTATION_FILE"
    echo ""
    echo "📋 Please download it first:"
    echo "   1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
    echo "   2. Register and download MSR_VTT.json"
    echo "   3. Place it at: $ANNOTATION_FILE"
    exit 1
fi

echo "✅ Found annotation file"
echo ""

# Verify it's valid JSON and check size
python3 << 'PY'
import json
from pathlib import Path

file_path = Path("data/msr-vtt/annotation/MSR_VTT.json")

try:
    with open(file_path) as f:
        data = json.load(f)
    
    videos = data.get("videos", [])
    video_count = len(videos)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total videos: {video_count}")
    
    if video_count < 100:
        print(f"\n⚠️  WARNING: Only {video_count} videos found!")
        print("   This looks like a test/sample dataset, not the full MSR-VTT.")
        print("   Full MSR-VTT should have ~10,000 videos.")
    elif video_count >= 10000:
        print(f"\n✅ Full dataset detected! ({video_count} videos)")
    else:
        print(f"\n✅ Dataset looks good ({video_count} videos)")
    
    # Show splits
    splits = {}
    for v in videos:
        split = v.get("split", "unknown").lower()
        splits[split] = splits.get(split, 0) + 1
    
    print(f"\n   Splits:")
    for split in ["train", "val", "test", "train_val"]:
        count = splits.get(split, 0)
        if count > 0:
            print(f"     {split}: {count} videos")
    
    # Show sample
    if videos:
        sample = videos[0]
        print(f"\n   Sample video:")
        print(f"     ID: {sample.get('video_id', 'N/A')}")
        print(f"     Split: {sample.get('split', 'N/A')}")
        print(f"     Captions: {len(sample.get('annotations', []))}")
        print(f"     URL: {sample.get('url', 'N/A')[:60]}...")
    
    # File size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n   File size: {size_mb:.1f} MB")
    
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON file: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit(1)
PY

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
read -p "🚀 Start preprocessing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📦 Preprocessing train split..."
    ./scripts/preprocess_local.sh train
    
    echo ""
    read -p "📦 Preprocess validation split? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/preprocess_local.sh val
    fi
    
    echo ""
    read -p "📦 Preprocess test split? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/preprocess_local.sh test
    fi
    
    echo ""
    echo "✅ Preprocessing complete!"
    echo ""
    echo "📤 To sync to Zaratan:"
    echo "   ./scripts/sync_processed_data.sh"
fi

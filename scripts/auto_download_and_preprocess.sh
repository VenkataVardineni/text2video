#!/bin/bash
# Automatically download MSR-VTT and preprocess it
# Tries multiple sources to get the dataset

set -euo pipefail

echo "🚀 Automatic MSR-VTT Download & Preprocessing"
echo "=============================================="
echo ""

# Create directories
mkdir -p data/msr-vtt/{annotation,videos/all}
mkdir -p data/processed/{raw/videos/{train,val,test},manifests,logs}

ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"

# Function to download annotation
download_annotation() {
    echo "📥 Trying to download MSR_VTT.json from various sources..."
    
    # Try multiple sources
    SOURCES=(
        "https://github.com/ArrowLuo/CLIP4Clip/raw/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
        "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
        "https://github.com/ArrowLuo/CLIP4Clip/raw/master/datasets/MSR-VTT/annotation/MSR_VTT.json"
    )
    
    for url in "${SOURCES[@]}"; do
        echo "   Trying: $url"
        if curl -L -f -s -o "$ANNOTATION_FILE" "$url" 2>/dev/null; then
            if [ -f "$ANNOTATION_FILE" ] && [ -s "$ANNOTATION_FILE" ]; then
                # Verify it's valid JSON
                if python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
                    echo "✅ Successfully downloaded from: $url"
                    return 0
                fi
            fi
        fi
        rm -f "$ANNOTATION_FILE"
    done
    
    return 1
}

# Try to download annotation
if [ ! -f "$ANNOTATION_FILE" ]; then
    if ! download_annotation; then
        echo ""
        echo "⚠️  Automatic download failed. Trying HuggingFace datasets..."
        
        # Try HuggingFace
        python3 << 'PY'
import sys
try:
    import subprocess
    import json
    from pathlib import Path
    
    print("   Installing/checking HuggingFace datasets...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=False)
    
    from datasets import load_dataset
    
    print("   Loading MSR-VTT from HuggingFace...")
    # Try different dataset names
    dataset = None
    for name in ["ArrowLuo/CLIP4Clip", "msr-vtt"]:
        try:
            dataset = load_dataset(name, trust_remote_code=True)
            print(f"   ✅ Found dataset: {name}")
            break
        except:
            continue
    
    if dataset:
        # Try to extract annotation
        # This is dataset-specific, adjust as needed
        print("   Extracting annotation...")
        # Save what we can
        Path("data/msr-vtt/annotation").mkdir(parents=True, exist_ok=True)
        # Note: HuggingFace structure may differ, this is a placeholder
        print("   ⚠️  HuggingFace dataset structure may need conversion")
    else:
        print("   ❌ Could not find MSR-VTT on HuggingFace")
        
except Exception as e:
    print(f"   ⚠️  HuggingFace approach failed: {e}")
PY
        
        # Final check
        if [ ! -f "$ANNOTATION_FILE" ]; then
            echo ""
            echo "❌ Could not automatically download MSR_VTT.json"
            echo ""
            echo "📋 Manual download required:"
            echo "   1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
            echo "   2. Register and download MSR_VTT.json"
            echo "   3. Place it at: $ANNOTATION_FILE"
            echo ""
            echo "   Or try:"
            echo "   wget -O $ANNOTATION_FILE <direct_download_url>"
            exit 1
        fi
    fi
else
    echo "✅ Annotation file already exists"
fi

# Verify annotation file
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "❌ Annotation file not found"
    exit 1
fi

echo ""
echo "📊 Dataset information:"
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
        count = splits.get(split, 0)
        if count > 0:
            print(f"  {split}: {count} videos")
PY

# Check dependencies
echo ""
echo "🔍 Checking dependencies..."

install_deps() {
    echo "📦 Installing dependencies..."
    
    # Check Python packages
    python3 -c "import pandas" 2>/dev/null || pip3 install -q pandas
    python3 -c "import tqdm" 2>/dev/null || pip3 install -q tqdm
    python3 -c "import decord" 2>/dev/null || pip3 install -q decord==0.6.0
    command -v yt-dlp >/dev/null || pip3 install -q yt-dlp
    
    # Check ffmpeg
    if ! command -v ffmpeg >/dev/null; then
        echo "⚠️  ffmpeg not found. Installing..."
        if command -v brew >/dev/null; then
            brew install ffmpeg 2>/dev/null || echo "   Please install ffmpeg manually: brew install ffmpeg"
        else
            echo "   Please install ffmpeg manually"
        fi
    fi
}

install_deps

echo ""
echo "🚀 Starting preprocessing..."
echo "   This will download videos from URLs in the annotation file"
echo "   (This may take a while depending on internet speed)"
echo ""

# Run preprocessing
export SPLIT=train
export MSR_VTT_ROOT=./data/msr-vtt
export OUT_ROOT=./data/processed

python3 code/download_msrvtt_local.py

echo ""
echo "✅ Preprocessing complete!"
echo ""
echo "📋 Summary:"
echo "   Videos: data/processed/raw/videos/train/"
echo "   Manifest: data/processed/manifests/train.manifest.jsonl"
echo ""
echo "📤 To sync to Zaratan:"
echo "   ./scripts/sync_processed_data.sh"

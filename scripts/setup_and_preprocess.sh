#!/bin/bash
# Complete setup and preprocessing workflow
# This script guides you through getting MSR-VTT data and preprocessing it

set -euo pipefail

echo "🚀 MSR-VTT Data Setup & Preprocessing"
echo "====================================="
echo ""

# Step 1: Check/create directories
echo "📁 Setting up directories..."
mkdir -p data/msr-vtt/{annotation,videos/all}
mkdir -p data/processed/{raw/videos/{train,val,test},manifests,logs}
echo "✅ Directories created"
echo ""

# Step 2: Check for annotation file
ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"

if [ -f "$ANNOTATION_FILE" ]; then
    echo "✅ Found MSR_VTT.json"
    
    # Show stats
    echo ""
    echo "📊 Dataset info:"
    python3 << 'PY'
import json
try:
    with open("data/msr-vtt/annotation/MSR_VTT.json") as f:
        data = json.load(f)
        videos = data.get("videos", [])
        print(f"  Total videos: {len(videos)}")
        splits = {}
        for v in videos:
            split = v.get("split", "unknown").lower()
            splits[split] = splits.get(split, 0) + 1
        for split, count in sorted(splits.items()):
            print(f"  {split}: {count} videos")
except Exception as e:
    print(f"  Error reading file: {e}")
PY
    
else
    echo "⚠️  MSR_VTT.json not found"
    echo ""
    echo "📥 To get MSR-VTT annotation file:"
    echo ""
    echo "Option 1: Download from official source"
    echo "   1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
    echo "   2. Register and download MSR_VTT.json"
    echo "   3. Place it at: $ANNOTATION_FILE"
    echo ""
    echo "Option 2: Use wget/curl (if you have a direct URL)"
    echo "   wget -O $ANNOTATION_FILE <URL>"
    echo ""
    echo "Option 3: If you have it elsewhere"
    echo "   cp /path/to/MSR_VTT.json $ANNOTATION_FILE"
    echo ""
    
    read -p "Do you have MSR_VTT.json ready to place? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "📋 Please get MSR_VTT.json first, then run this script again"
        echo "   Or place it manually at: $ANNOTATION_FILE"
        exit 1
    fi
fi

# Step 3: Check dependencies
echo ""
echo "🔍 Checking dependencies..."

MISSING_DEPS=()

command -v python3 >/dev/null || MISSING_DEPS+=("python3")
command -v ffmpeg >/dev/null || MISSING_DEPS+=("ffmpeg")

python3 -c "import pandas" 2>/dev/null || MISSING_DEPS+=("pandas")
python3 -c "import tqdm" 2>/dev/null || MISSING_DEPS+=("tqdm")
python3 -c "import decord" 2>/dev/null || MISSING_DEPS+=("decord")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "⚠️  Missing dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "📦 Installing missing dependencies..."
    
    if [[ " ${MISSING_DEPS[@]} " =~ " ffmpeg " ]]; then
        echo "   Installing ffmpeg..."
        if command -v brew >/dev/null; then
            brew install ffmpeg || echo "   ⚠️  Please install ffmpeg manually: brew install ffmpeg"
        else
            echo "   ⚠️  Please install ffmpeg manually"
        fi
    fi
    
    PIP_DEPS=()
    [[ " ${MISSING_DEPS[@]} " =~ " pandas " ]] && PIP_DEPS+=("pandas")
    [[ " ${MISSING_DEPS[@]} " =~ " tqdm " ]] && PIP_DEPS+=("tqdm")
    [[ " ${MISSING_DEPS[@]} " =~ " decord " ]] && PIP_DEPS+=("decord==0.6.0")
    
    if [ ${#PIP_DEPS[@]} -gt 0 ]; then
        echo "   Installing Python packages: ${PIP_DEPS[*]}"
        pip3 install "${PIP_DEPS[@]}" || {
            echo "   ⚠️  Installation failed. Please install manually:"
            echo "   pip3 install ${PIP_DEPS[*]}"
        }
    fi
    
    # Check yt-dlp for downloading
    command -v yt-dlp >/dev/null || {
        echo "   Installing yt-dlp for video downloads..."
        pip3 install yt-dlp || echo "   ⚠️  yt-dlp installation failed"
    }
else
    echo "✅ All dependencies available"
fi

# Step 4: Start preprocessing
echo ""
echo "🎬 Ready to preprocess!"
echo ""
read -p "Start preprocessing train split? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting preprocessing (this may take a while)..."
    echo ""
    ./scripts/preprocess_local.sh train
else
    echo ""
    echo "📋 To preprocess manually:"
    echo "   ./scripts/preprocess_local.sh train    # Train split"
    echo "   ./scripts/preprocess_local.sh val      # Validation split"
    echo "   ./scripts/preprocess_local.sh test     # Test split"
fi

#!/bin/bash
# Comprehensive script to get complete MSR-VTT dataset
# Tries multiple methods and provides clear instructions

set -euo pipefail

echo "📥 Getting Complete MSR-VTT Dataset"
echo "===================================="
echo ""

ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"
mkdir -p data/msr-vtt/annotation

# Backup test file if it exists
if [ -f "$ANNOTATION_FILE" ]; then
    python3 << 'PY'
import json
with open("data/msr-vtt/annotation/MSR_VTT.json") as f:
    data = json.load(f)
    video_count = len(data.get("videos", []))
    if video_count < 100:
        print(f"⚠️  Current file has only {video_count} videos (likely test data)")
        print("   Will attempt to replace with full dataset")
    else:
        print(f"✅ Already have {video_count} videos - looks like full dataset!")
        exit(0)
PY
    if [ $? -eq 0 ]; then
        exit 0
    fi
    mv "$ANNOTATION_FILE" "${ANNOTATION_FILE}.test.backup"
    echo "   Backed up test file"
fi

echo ""
echo "🔍 Attempting to download complete MSR-VTT dataset..."
echo ""

# Method 1: Try direct download from known academic sources
echo "Method 1: Trying academic mirrors..."

# Try common academic repository URLs
ACADEMIC_URLS=(
    "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/MSR-VTT.zip"
    "https://github.com/ArrowLuo/CLIP4Clip/raw/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
)

for url in "${ACADEMIC_URLS[@]}"; do
    echo "   Trying: $(basename $(dirname $url))"
    if curl -L -f -s -o "$ANNOTATION_FILE" "$url" 2>/dev/null; then
        if python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
            video_count=$(python3 -c "import json; print(len(json.load(open('$ANNOTATION_FILE')).get('videos', [])))")
            if [ "$video_count" -gt 100 ]; then
                echo "   ✅ Success! Found $video_count videos"
                break
            else
                echo "   ⚠️  Only $video_count videos (not full dataset)"
                rm -f "$ANNOTATION_FILE"
            fi
        fi
    fi
done

# Method 2: Try using git to get from repositories
if [ ! -f "$ANNOTATION_FILE" ] || ! python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
    echo ""
    echo "Method 2: Cloning repositories..."
    
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Try cloning CLIP4Clip and extracting
    echo "   Cloning CLIP4Clip repository..."
    if git clone --depth 1 --quiet "https://github.com/ArrowLuo/CLIP4Clip.git" "$TEMP_DIR/clip4clip" 2>/dev/null; then
        # Look for annotation file
        found=$(find "$TEMP_DIR/clip4clip" -name "MSR_VTT.json" -type f | head -1)
        if [ -n "$found" ]; then
            cp "$found" "$ANNOTATION_FILE"
            video_count=$(python3 -c "import json; print(len(json.load(open('$ANNOTATION_FILE')).get('videos', [])))")
            if [ "$video_count" -gt 100 ]; then
                echo "   ✅ Found full dataset with $video_count videos!"
                rm -rf "$TEMP_DIR"
                exit 0
            fi
        fi
        rm -rf "$TEMP_DIR/clip4clip"
    fi
    
    rm -rf "$TEMP_DIR"
fi

# Check if we got it
if [ -f "$ANNOTATION_FILE" ] && python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
    video_count=$(python3 -c "import json; print(len(json.load(open('$ANNOTATION_FILE')).get('videos', [])))")
    
    if [ "$video_count" -gt 100 ]; then
        echo ""
        echo "✅ Successfully obtained MSR-VTT dataset!"
        echo "   Total videos: $video_count"
        python3 << 'PY'
import json
with open("data/msr-vtt/annotation/MSR_VTT.json") as f:
    data = json.load(f)
    videos = data.get("videos", [])
    splits = {}
    for v in videos:
        split = v.get("split", "unknown").lower()
        splits[split] = splits.get(split, 0) + 1
    print("   Splits:")
    for split, count in sorted(splits.items()):
        print(f"     {split}: {count} videos")
PY
        exit 0
    fi
fi

# If we get here, automatic download failed
echo ""
echo "❌ Could not automatically download complete MSR-VTT dataset"
echo ""
echo "📋 Manual Download Instructions:"
echo "=================================="
echo ""
echo "MSR-VTT requires registration. Follow these steps:"
echo ""
echo "1. Visit the official website:"
echo "   http://ms-multimedia-challenge.com/2017/dataset"
echo ""
echo "2. Register for access (free for research purposes)"
echo ""
echo "3. Download the annotation file:"
echo "   - File name: MSR_VTT.json"
echo "   - Should contain ~10,000 videos"
echo ""
echo "4. Place the file at:"
echo "   $ANNOTATION_FILE"
echo ""
echo "5. Then run preprocessing:"
echo "   ./scripts/preprocess_local.sh train"
echo ""
echo "Alternative: If you have the file elsewhere, copy it:"
echo "   cp /path/to/MSR_VTT.json $ANNOTATION_FILE"
echo ""
echo "Or download via wget/curl if you have a direct URL:"
echo "   wget -O $ANNOTATION_FILE <download_url>"
echo ""

exit 1

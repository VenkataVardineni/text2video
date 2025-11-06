#!/bin/bash
# Comprehensive MSR-VTT download attempt
# Tries multiple methods and sources

set -euo pipefail

ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"
mkdir -p data/msr-vtt/annotation

echo "🔍 Searching for MSR-VTT annotation file..."
echo ""

# Method 1: Try GitHub repos that might host it
echo "Method 1: Trying GitHub repositories..."
GITHUB_REPOS=(
    "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
    "https://github.com/msr-vtt/dataset/raw/main/annotation/MSR_VTT.json"
    "https://raw.githubusercontent.com/linjieli222/VL-BERT/master/datasets/MSR-VTT/annotation/MSR_VTT.json"
)

for url in "${GITHUB_REPOS[@]}"; do
    echo "   Trying: $(basename $(dirname $(dirname $url)))"
    if curl -L -f -s -o "$ANNOTATION_FILE" "$url" 2>/dev/null; then
        if python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
            echo "   ✅ Success!"
            exit 0
        fi
    fi
done

# Method 2: Try using git to clone repos that might have it
echo ""
echo "Method 2: Trying to clone repositories..."
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

REPOS=(
    "https://github.com/ArrowLuo/CLIP4Clip.git"
)

for repo in "${REPOS[@]}"; do
    echo "   Cloning: $(basename $repo .git)"
    if git clone --depth 1 --quiet "$repo" "$TEMP_DIR/repo" 2>/dev/null; then
        # Look for annotation file
        if find "$TEMP_DIR/repo" -name "MSR_VTT.json" -type f | head -1 | xargs -I {} cp {} "$ANNOTATION_FILE" 2>/dev/null; then
            if [ -f "$ANNOTATION_FILE" ] && python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
                echo "   ✅ Found in repository!"
                rm -rf "$TEMP_DIR"
                exit 0
            fi
        fi
        rm -rf "$TEMP_DIR/repo"
    fi
done

# Method 3: Create a minimal test dataset
echo ""
echo "Method 3: Creating minimal test dataset for pipeline testing..."
python3 << 'PY'
import json
from pathlib import Path

# Create a minimal MSR-VTT format file for testing
test_data = {
    "videos": [
        {
            "video_id": "video0",
            "url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
            "split": "train",
            "category": "test",
            "start_time": "0",
            "end_time": "10",
            "annotations": [
                {"caption": "A test video for MSR-VTT preprocessing pipeline"}
            ]
        },
        {
            "video_id": "video1", 
            "url": "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "split": "train",
            "category": "test",
            "start_time": "0",
            "end_time": "10",
            "annotations": [
                {"caption": "Another test video for validation"}
            ]
        }
    ]
}

Path("data/msr-vtt/annotation").mkdir(parents=True, exist_ok=True)
with open("data/msr-vtt/annotation/MSR_VTT.json", "w") as f:
    json.dump(test_data, f, indent=2)

print("   ✅ Created test dataset with 2 sample videos")
print("   ⚠️  This is a minimal test set - you'll need the full dataset for training")
PY

if [ -f "$ANNOTATION_FILE" ]; then
    echo ""
    echo "✅ Test dataset created!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Test preprocessing with this sample: ./scripts/preprocess_local.sh train"
    echo "   2. Get full MSR-VTT dataset from: http://ms-multimedia-challenge.com/2017/dataset"
    echo "   3. Replace test file with full dataset when ready"
    exit 0
fi

echo ""
echo "❌ Could not obtain MSR-VTT annotation file"
exit 1

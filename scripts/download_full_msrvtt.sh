#!/bin/bash
# Download complete MSR-VTT dataset
# Tries multiple methods to get the full dataset

set -euo pipefail

echo "📥 Downloading Complete MSR-VTT Dataset"
echo "========================================"
echo ""

mkdir -p data/msr-vtt/{annotation,videos/all}

ANNOTATION_FILE="data/msr-vtt/annotation/MSR_VTT.json"

# Method 1: Try to download from known GitHub repositories
echo "🔍 Method 1: Searching GitHub repositories..."

download_from_github() {
    local repo=$1
    local path=$2
    echo "   Trying: $repo"
    
    # Try raw.githubusercontent.com
    if curl -L -f -s -o "$ANNOTATION_FILE" \
        "https://raw.githubusercontent.com/${repo}/${path}" 2>/dev/null; then
        if python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
            echo "   ✅ Success!"
            return 0
        fi
    fi
    
    # Try GitHub API
    local owner=$(echo "$repo" | cut -d'/' -f1)
    local repo_name=$(echo "$repo" | cut -d'/' -f2)
    local file_path=$(echo "$path" | sed 's|^/||')
    
    if curl -L -f -s \
        "https://api.github.com/repos/${repo}/contents/${file_path}" \
        -H "Accept: application/vnd.github.v3.raw" \
        -o "$ANNOTATION_FILE" 2>/dev/null; then
        if python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
            echo "   ✅ Success via GitHub API!"
            return 0
        fi
    fi
    
    return 1
}

# Try multiple GitHub repos
REPOS=(
    "ArrowLuo/CLIP4Clip:main:datasets/MSR-VTT/annotation/MSR_VTT.json"
    "linjieli222/VL-BERT:master:datasets/MSR-VTT/annotation/MSR_VTT.json"
    "sail-sg/EditAnything:main:datasets/MSR-VTT/annotation/MSR_VTT.json"
)

for repo_info in "${REPOS[@]}"; do
    IFS=':' read -r repo branch path <<< "$repo_info"
    if download_from_github "$repo" "$path"; then
        break
    fi
done

# Method 2: Clone repository and extract
if [ ! -f "$ANNOTATION_FILE" ] || ! python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
    echo ""
    echo "🔍 Method 2: Cloning repositories..."
    
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    CLONE_REPOS=(
        "https://github.com/ArrowLuo/CLIP4Clip.git"
    )
    
    for repo_url in "${CLONE_REPOS[@]}"; do
        repo_name=$(basename "$repo_url" .git)
        echo "   Cloning: $repo_name"
        
        if git clone --depth 1 --quiet "$repo_url" "$TEMP_DIR/$repo_name" 2>/dev/null; then
            # Search for MSR_VTT.json
            found_file=$(find "$TEMP_DIR/$repo_name" -name "MSR_VTT.json" -type f | head -1)
            if [ -n "$found_file" ]; then
                echo "   ✅ Found annotation file!"
                cp "$found_file" "$ANNOTATION_FILE"
                if python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
                    rm -rf "$TEMP_DIR"
                    break
                fi
            fi
            rm -rf "$TEMP_DIR/$repo_name"
        fi
    done
    
    rm -rf "$TEMP_DIR" 2>/dev/null || true
fi

# Verify we have the annotation file
if [ ! -f "$ANNOTATION_FILE" ] || ! python3 -m json.tool "$ANNOTATION_FILE" >/dev/null 2>&1; then
    echo ""
    echo "❌ Could not automatically download MSR_VTT.json"
    echo ""
    echo "📋 Manual Download Required:"
    echo ""
    echo "The MSR-VTT dataset requires registration. Here are your options:"
    echo ""
    echo "Option 1: Official Source (Recommended)"
    echo "   1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
    echo "   2. Register and download MSR_VTT.json"
    echo "   3. Place it at: $ANNOTATION_FILE"
    echo ""
    echo "Option 2: Academic Mirror"
    echo "   Search for 'MSR-VTT dataset download' on:"
    echo "   - Google Scholar"
    echo "   - Academic repositories"
    echo "   - Research paper supplementary materials"
    echo ""
    echo "Option 3: Use HuggingFace (if available)"
    echo "   python3 scripts/download_msrvtt_hf.py"
    echo ""
    exit 1
fi

# Show dataset info
echo ""
echo "✅ Annotation file downloaded!"
echo ""
echo "📊 Dataset Information:"
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
    
    for split in ["train", "val", "test", "train_val"]:
        count = splits.get(split, 0)
        if count > 0:
            print(f"  {split}: {count} videos")
    
    # Show sample
    if videos:
        sample = videos[0]
        print(f"\n  Sample video:")
        print(f"    ID: {sample.get('video_id', 'N/A')}")
        print(f"    Split: {sample.get('split', 'N/A')}")
        print(f"    Captions: {len(sample.get('annotations', []))}")
PY

echo ""
echo "🚀 Ready to preprocess!"
echo "   Run: ./scripts/preprocess_local.sh train"
echo ""

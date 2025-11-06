#!/bin/bash
# Fix manifest paths on Zaratan
# Updates paths from local machine to Zaratan paths
# Usage: Run this on Zaratan after syncing data

set -euo pipefail

MANIFEST_DIR="$HOME/scratch.msml612-fa25/data/manifests"
VIDEOS_DIR="$HOME/scratch.msml612-fa25/data/raw/videos"

echo "🔧 Fixing manifest paths for Zaratan"
echo "====================================="

if [ ! -d "$MANIFEST_DIR" ]; then
    echo "❌ Error: Manifest directory not found: $MANIFEST_DIR"
    exit 1
fi

# Create videos directory if it doesn't exist
mkdir -p "$VIDEOS_DIR/all"

for manifest in "$MANIFEST_DIR"/*.manifest*.jsonl; do
    if [ -f "$manifest" ]; then
        echo "Processing: $(basename $manifest)"
        
        # Backup original
        cp "$manifest" "${manifest}.bak"
        
        # Update paths:
        # 1. Local Kaggle cache paths -> Zaratan paths
        # 2. Relative paths -> absolute Zaratan paths
        python3 << PY
import json
import sys
from pathlib import Path

manifest_file = Path("$manifest")
videos_dir = Path("$VIDEOS_DIR")

items = []
with open(manifest_file) as f:
    for line in f:
        item = json.loads(line)
        old_path = item.get("path", "")
        
        if old_path:
            # Extract video filename
            video_name = Path(old_path).name
            
            # Check if video exists in any subdirectory
            video_path = None
            for subdir in ["all", "train", "val", "test"]:
                candidate = videos_dir / subdir / video_name
                if candidate.exists():
                    video_path = str(candidate)
                    break
            
            # If not found, use /scratch/data/raw/videos/all/ as default
            if not video_path:
                video_path = f"/scratch/data/raw/videos/all/{video_name}"
            
            item["path"] = video_path
        
        items.append(item)

# Write updated manifest
with open(manifest_file, "w") as f:
    for item in items:
        f.write(json.dumps(item) + "\n")

print(f"  ✅ Updated {len(items)} entries")
PY
        
        echo "  ✅ Paths updated"
    fi
done

echo ""
echo "✅ All manifests updated!"
echo ""
echo "📋 Verify:"
echo "   head -1 $MANIFEST_DIR/train.manifest.processed.jsonl | python3 -m json.tool | grep path"

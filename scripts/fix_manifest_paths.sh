#!/bin/bash
# Fix manifest paths after syncing to Zaratan
# Updates relative paths to absolute paths in container
# Usage: Run this on Zaratan after syncing data

set -euo pipefail

MANIFEST_DIR="$HOME/scratch.msml612-fa25/data/manifests"

echo "🔧 Fixing manifest paths for Zaratan"
echo "====================================="

if [ ! -d "$MANIFEST_DIR" ]; then
    echo "❌ Error: Manifest directory not found: $MANIFEST_DIR"
    exit 1
fi

for manifest in "$MANIFEST_DIR"/*.manifest.jsonl; do
    if [ -f "$manifest" ]; then
        echo "Processing: $(basename $manifest)"
        # Update paths: raw/videos/... -> /scratch/data/raw/videos/...
        sed -i.bak 's|"path":"raw/videos|"path":"/scratch/data/raw/videos|g' "$manifest"
        # Also handle null paths
        sed -i.bak 's|"path":null|"path":null|g' "$manifest"
        echo "  ✅ Updated paths"
    fi
done

echo ""
echo "✅ All manifests updated!"
echo ""
echo "📋 Verify:"
echo "   head -1 $MANIFEST_DIR/train.manifest.jsonl | python3 -m json.tool"

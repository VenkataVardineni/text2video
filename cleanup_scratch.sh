#!/bin/bash
# Cleanup script for Zaratan scratch directory
# Run this if you run out of disk space

set -euo pipefail

SCR=~/scratch.text2video

echo "🧹 Cleaning up Zaratan scratch directory"
echo "========================================"
echo ""

# Show current disk usage
echo "💾 Current disk usage:"
df -h ~ | tail -1
echo ""

# Show scratch directory size
if [[ -d "$SCR" ]]; then
    echo "📁 Scratch directory size:"
    du -sh "$SCR" 2>/dev/null || echo "   Unable to calculate"
    echo ""
    
    echo "📋 Contents of scratch directory:"
    ls -lh "$SCR" 2>/dev/null || echo "   Directory empty or inaccessible"
    echo ""
fi

# Clean up old build temp files
echo "🗑️  Cleaning up old build temporary files..."
rm -rf "$SCR/tmp/build-temp-*" 2>/dev/null && echo "✅ Removed build temp files" || echo "   No build temp files found"
rm -rf /tmp/apptainer_build_* 2>/dev/null && echo "✅ Removed /tmp build files" || echo "   No /tmp build files found"
rm -rf "$SCR/tmp/bundle-temp-*" 2>/dev/null && echo "✅ Removed bundle temp files" || echo "   No bundle temp files found"
echo ""

# Option to clean cache (will require re-downloading containers)
read -p "Remove container cache? (will need to re-download containers) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$SCR/appt_cache" 2>/dev/null && echo "✅ Removed container cache" || echo "   No cache found"
    mkdir -p "$SCR/appt_cache"
fi

echo ""
echo "💾 Disk usage after cleanup:"
df -h ~ | tail -1

echo ""
echo "✅ Cleanup complete!"

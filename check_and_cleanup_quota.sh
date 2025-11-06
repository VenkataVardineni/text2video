#!/bin/bash
# Check quota and clean up aggressively
# Run this on Zaratan

echo "🔍 Checking Disk Quota and Usage"
echo "=================================="

echo ""
echo "📊 Quota Information:"
quota -s 2>/dev/null || echo "quota command not available"

echo ""
echo "💾 Disk Usage:"
df -h ~ | tail -1
df -h /tmp 2>/dev/null | tail -1 || echo "/tmp not available"

echo ""
echo "📁 Home directory size:"
du -sh ~ 2>/dev/null | head -1

echo ""
echo "📁 Scratch directory size:"
du -sh ~/scratch.text2video 2>/dev/null || echo "Scratch directory doesn't exist"

echo ""
echo "🧹 Aggressive Cleanup..."
echo "Removing old container cache..."
rm -rf ~/scratch.text2video/appt_cache/* 2>/dev/null && echo "✅ Removed container cache" || echo "   No cache found"

echo "Removing old build files..."
rm -rf ~/scratch.text2video/tmp/* 2>/dev/null && echo "✅ Removed build files" || echo "   No build files"

echo "Removing old Python cache..."
find ~ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null && echo "✅ Removed Python cache" || echo "   No Python cache"

echo "Removing .pyc files..."
find ~ -name "*.pyc" -delete 2>/dev/null && echo "✅ Removed .pyc files" || echo "   No .pyc files"

echo ""
echo "💾 Disk Usage After Cleanup:"
df -h ~ | tail -1
du -sh ~ 2>/dev/null | head -1

echo ""
echo "💡 If still over quota, consider:"
echo "  1. Remove old files: rm -rf ~/old_directory"
echo "  2. Contact HPC support to increase quota"
echo "  3. Use container's built-in packages only (no pip install)"

#!/bin/bash
# Download MSR-VTT dataset
# Note: MSR-VTT requires registration and manual download
# This script helps set up the structure and provides download instructions

set -euo pipefail

echo "📥 MSR-VTT Dataset Setup"
echo "========================"
echo ""

# Create directory structure
mkdir -p data/msr-vtt/{annotation,videos/all,processed}
mkdir -p data/processed/{raw/videos/{train,val,test},manifests,logs}

echo "✅ Created directory structure"
echo ""

# Check if annotation file exists
if [ -f "data/msr-vtt/annotation/MSR_VTT.json" ]; then
    echo "✅ Found MSR_VTT.json annotation file"
else
    echo "⚠️  MSR_VTT.json not found"
    echo ""
    echo "📋 To download MSR-VTT dataset:"
    echo ""
    echo "1. Visit: http://ms-multimedia-challenge.com/2017/dataset"
    echo "   Or: https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/"
    echo ""
    echo "2. Register and download:"
    echo "   - MSR_VTT.json (annotations)"
    echo "   - Videos (all videos in one zip or individual files)"
    echo ""
    echo "3. Place files:"
    echo "   - MSR_VTT.json → data/msr-vtt/annotation/MSR_VTT.json"
    echo "   - Videos → data/msr-vtt/videos/all/"
    echo ""
    echo "4. Or if you have the files elsewhere, create symlinks:"
    echo "   ln -s /path/to/MSR_VTT.json data/msr-vtt/annotation/MSR_VTT.json"
    echo "   ln -s /path/to/videos data/msr-vtt/videos/all"
    echo ""
    
    read -p "Do you have MSR_VTT.json already? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter path to MSR_VTT.json: " json_path
        if [ -f "$json_path" ]; then
            cp "$json_path" data/msr-vtt/annotation/MSR_VTT.json
            echo "✅ Copied annotation file"
        else
            echo "❌ File not found: $json_path"
        fi
    fi
fi

# Check if videos exist
video_count=$(find data/msr-vtt/videos/all -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
if [ "$video_count" -gt 0 ]; then
    echo "✅ Found $video_count video files"
else
    echo "⚠️  No video files found in data/msr-vtt/videos/all/"
    echo "   Videos will be downloaded from URLs in the annotation file"
fi

echo ""
echo "📋 Next steps:"
echo "   1. Ensure MSR_VTT.json is in: data/msr-vtt/annotation/"
echo "   2. (Optional) Place videos in: data/msr-vtt/videos/all/"
echo "   3. Run preprocessing: ./scripts/preprocess_local.sh train"

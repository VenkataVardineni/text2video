#!/usr/bin/env python3
"""
Setup MSR-VTT using Kaggle videos and create annotation structure
"""
import json
import shutil
from pathlib import Path

def main():
    print("🔧 Setting up MSR-VTT from Kaggle dataset")
    print("=" * 50)
    
    # Paths
    kaggle_path = Path.home() / ".cache/kagglehub/datasets/vishnutheepb/msrvtt/versions/1/TrainValVideo"
    videos_dir = Path("data/msr-vtt/videos/all")
    annotation_file = Path("data/msr-vtt/annotation/MSR_VTT.json")
    
    # Create directories
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Count videos
    if kaggle_path.exists():
        videos = list(kaggle_path.glob("*.mp4"))
        print(f"✅ Found {len(videos)} videos in Kaggle dataset")
        
        # Create symlinks (or copy if symlinks don't work)
        print(f"\n📹 Setting up video links...")
        linked = 0
        for video in videos[:10]:  # Test with first 10
            target = videos_dir / video.name
            if not target.exists():
                try:
                    target.symlink_to(video)
                    linked += 1
                except:
                    shutil.copy2(video, target)
                    linked += 1
        
        print(f"   Linked {linked} videos (showing first 10 as test)")
        print(f"   Full path: {kaggle_path}")
        print(f"   Total videos available: {len(videos)}")
    else:
        print(f"❌ Kaggle dataset not found at: {kaggle_path}")
        return False
    
    # Check for annotation file
    if annotation_file.exists():
        with open(annotation_file) as f:
            data = json.load(f)
            video_count = len(data.get("videos", []))
            print(f"\n✅ Annotation file found: {video_count} videos")
            return True
    else:
        print(f"\n⚠️  Annotation file not found")
        print(f"   You need to download MSR_VTT.json from:")
        print(f"   http://ms-multimedia-challenge.com/2017/dataset")
        print(f"\n   Or the videos are ready at:")
        print(f"   {kaggle_path}")
        print(f"   You can use them directly once you have the annotation file")
        return False

if __name__ == "__main__":
    main()

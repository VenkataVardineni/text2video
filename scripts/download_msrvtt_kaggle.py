#!/usr/bin/env python3
"""
Download MSR-VTT dataset from Kaggle using kagglehub
"""
import sys
import json
import shutil
from pathlib import Path

def main():
    print("📥 Downloading MSR-VTT from Kaggle")
    print("===================================")
    print("")
    
    # Install kagglehub if needed
    try:
        import kagglehub
    except ImportError:
        print("📦 Installing kagglehub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
    
    print("📥 Downloading dataset from Kaggle...")
    print("   This may take a while (dataset is large)...")
    print("")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("vishnutheepb/msrvtt")
        print(f"✅ Download complete!")
        print(f"   Path: {path}")
        print("")
        
        # Find MSR_VTT.json in the downloaded directory
        annotation_file = None
        for json_file in Path(path).rglob("MSR_VTT.json"):
            annotation_file = json_file
            break
        
        # Also try other common names
        if not annotation_file:
            for json_file in Path(path).rglob("*.json"):
                if "msr" in json_file.name.lower() or "vtt" in json_file.name.lower():
                    annotation_file = json_file
                    break
        
        # If no annotation file found, try to download it separately
        if not annotation_file:
            print("   ⚠️  No annotation file in Kaggle dataset")
            print("   Videos found, but need annotation file separately")
            print("   Attempting to download annotation from GitHub...")
            
            # Try to download annotation from GitHub
            import urllib.request
            annotation_urls = [
                "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/main/datasets/MSR-VTT/annotation/MSR_VTT.json",
                "https://github.com/ArrowLuo/CLIP4Clip/raw/main/datasets/MSR-VTT/annotation/MSR_VTT.json"
            ]
            
            target_dir = Path("data/msr-vtt/annotation")
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "MSR_VTT.json"
            
            for url in annotation_urls:
                try:
                    print(f"   Trying: {url}")
                    urllib.request.urlretrieve(url, target_file)
                    if target_file.exists() and target_file.stat().st_size > 1000:
                        # Verify it's valid JSON
                        with open(target_file) as f:
                            json.load(f)
                        print(f"   ✅ Downloaded annotation from GitHub!")
                        annotation_file = target_file
                        break
                except Exception as e:
                    print(f"   Failed: {e}")
                    continue
        
        if annotation_file:
            print(f"✅ Found annotation file: {annotation_file}")
            
            # Copy to our expected location
            target_dir = Path("data/msr-vtt/annotation")
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "MSR_VTT.json"
            
            shutil.copy2(annotation_file, target_file)
            print(f"✅ Copied to: {target_file}")
            
            # Verify
            with open(target_file) as f:
                data = json.load(f)
                videos = data.get("videos", [])
                print(f"✅ Verified: {len(videos)} videos found")
                
                # Show splits
                splits = {}
                for v in videos:
                    split = v.get("split", "unknown").lower()
                    splits[split] = splits.get(split, 0) + 1
                
                print(f"\n📊 Dataset splits:")
                for split in ["train", "val", "test", "train_val"]:
                    count = splits.get(split, 0)
                    if count > 0:
                        print(f"   {split}: {count} videos")
            
            # Check for video files
            video_dirs = list(Path(path).rglob("videos"))
            if video_dirs:
                print(f"\n📹 Video directory found: {video_dirs[0]}")
                print(f"   You can copy videos from: {video_dirs[0]}")
                print(f"   To: data/msr-vtt/videos/all/")
            
            return True
        else:
            print("⚠️  Could not find MSR_VTT.json in downloaded files")
            print(f"   Searching in: {path}")
            print(f"   JSON files found:")
            for json_file in Path(path).rglob("*.json"):
                print(f"     - {json_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("")
        print("📋 Troubleshooting:")
        print("   1. Make sure you have Kaggle API credentials set up")
        print("   2. Install kagglehub: pip install kagglehub")
        print("   3. Check internet connection")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("")
        print("🎉 Dataset download complete!")
        print("")
        print("📋 Next steps:")
        print("   1. Verify dataset: ./scripts/verify_and_preprocess_msrvtt.sh")
        print("   2. Or start preprocessing: ./scripts/preprocess_local.sh train")
    else:
        print("")
        print("❌ Download failed. Please check the error above.")
        sys.exit(1)

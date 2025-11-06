#!/usr/bin/env python3
"""
Download complete MSR-VTT dataset using HuggingFace datasets
"""
import sys
import json
from pathlib import Path

def main():
    print("📥 Downloading MSR-VTT from HuggingFace...")
    
    try:
        import subprocess
        print("   Installing/updating datasets library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "huggingface_hub"], check=False)
        
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        
        print("   Searching for MSR-VTT dataset...")
        
        # Try to download annotation file directly from HuggingFace
        try:
            print("   Attempting to download from HuggingFace Hub...")
            annotation_path = hf_hub_download(
                repo_id="ArrowLuo/CLIP4Clip",
                filename="datasets/MSR-VTT/annotation/MSR_VTT.json",
                repo_type="dataset",
                local_dir="./data/msr-vtt",
                local_dir_use_symlinks=False
            )
            
            if Path(annotation_path).exists():
                print(f"   ✅ Downloaded to: {annotation_path}")
                # Copy to expected location
                target = Path("data/msr-vtt/annotation/MSR_VTT.json")
                target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(annotation_path, target)
                print(f"   ✅ Copied to: {target}")
                return True
        except Exception as e:
            print(f"   ⚠️  HuggingFace Hub download failed: {e}")
        
        # Try loading as dataset
        try:
            print("   Trying to load as dataset...")
            dataset = load_dataset("ArrowLuo/CLIP4Clip", trust_remote_code=False)
            print(f"   ✅ Loaded dataset: {list(dataset.keys())}")
            
            # Try to extract annotation
            # This depends on the dataset structure
            if "train" in dataset:
                print(f"   Train split: {len(dataset['train'])} examples")
            
            # Save what we can
            Path("data/msr-vtt/annotation").mkdir(parents=True, exist_ok=True)
            # Note: May need to convert format
            
        except Exception as e:
            print(f"   ⚠️  Dataset loading failed: {e}")
        
        return False
        
    except ImportError:
        print("   ❌ Could not import required libraries")
        print("   Install with: pip install datasets huggingface_hub")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Download complete!")
        print("   File: data/msr-vtt/annotation/MSR_VTT.json")
    else:
        print("\n❌ Automatic download failed")
        print("\n📋 Manual download required:")
        print("   1. Visit: http://ms-multimedia-challenge.com/2017/dataset")
        print("   2. Register and download MSR_VTT.json")
        print("   3. Place at: data/msr-vtt/annotation/MSR_VTT.json")

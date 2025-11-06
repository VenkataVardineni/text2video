#!/usr/bin/env python3
"""
Download MSR-VTT dataset using HuggingFace datasets
"""
import os
from pathlib import Path
import json

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace datasets not installed")
    print("   Install with: pip install datasets")

def main():
    if not HF_AVAILABLE:
        print("\n📦 Installing HuggingFace datasets...")
        import subprocess
        subprocess.run(["pip", "install", "datasets"], check=True)
        from datasets import load_dataset
    
    print("📥 Downloading MSR-VTT from HuggingFace...")
    print("   This may take a while...")
    
    # Try to load MSR-VTT dataset
    try:
        # MSR-VTT might be available through various repos
        # Try common ones
        dataset = None
        
        repos = [
            "ArrowLuo/CLIP4Clip",
            "msr-vtt",
        ]
        
        for repo in repos:
            try:
                print(f"   Trying: {repo}")
                dataset = load_dataset(repo, "msr-vtt", trust_remote_code=True)
                break
            except:
                continue
        
        if dataset is None:
            print("❌ Could not find MSR-VTT on HuggingFace")
            print("\n📋 Alternative: Download manually")
            print("   1. Visit: https://huggingface.co/datasets")
            print("   2. Search for 'MSR-VTT'")
            print("   3. Download MSR_VTT.json")
            return
        
        # Save annotation file
        output_dir = Path("data/msr-vtt/annotation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to MSR-VTT format if needed
        # This depends on the dataset structure
        print("✅ Dataset loaded")
        print(f"   Splits: {list(dataset.keys())}")
        
        # Try to extract annotation
        # Note: Structure may vary, adjust as needed
        print("\n💡 Note: HuggingFace dataset structure may differ")
        print("   You may need to convert it to MSR-VTT format")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📋 Manual download recommended:")
        print("   1. Visit: http://ms-multimedia-challenge.com/2017/dataset")
        print("   2. Download MSR_VTT.json")
        print("   3. Place at: data/msr-vtt/annotation/MSR_VTT.json")

if __name__ == "__main__":
    main()

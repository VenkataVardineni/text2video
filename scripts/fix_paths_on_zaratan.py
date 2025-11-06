#!/usr/bin/env python3
"""
Fix manifest paths on Zaratan
Updates paths from local machine to Zaratan paths
Run this directly on Zaratan
"""
import json
from pathlib import Path

MANIFEST_DIR = Path.home() / "scratch.msml612-fa25/data/manifests"
VIDEOS_DIR = Path.home() / "scratch.msml612-fa25/data/raw/videos"

print("🔧 Fixing manifest paths for Zaratan")
print("=" * 50)

if not MANIFEST_DIR.exists():
    print(f"❌ Error: Manifest directory not found: {MANIFEST_DIR}")
    exit(1)

# Create videos directory if it doesn't exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
(VIDEOS_DIR / "all").mkdir(parents=True, exist_ok=True)

for manifest_file in MANIFEST_DIR.glob("*.manifest*.jsonl"):
    if manifest_file.name.endswith(".bak"):
        continue
    
    print(f"\nProcessing: {manifest_file.name}")
    
    # Backup
    backup_file = manifest_file.with_suffix(manifest_file.suffix + ".bak")
    import shutil
    shutil.copy2(manifest_file, backup_file)
    
    items = []
    updated = 0
    
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
                    candidate = VIDEOS_DIR / subdir / video_name
                    if candidate.exists():
                        # Use container path
                        video_path = f"/scratch/data/raw/videos/{subdir}/{video_name}"
                        updated += 1
                        break
                
                # If not found, use /scratch/data/raw/videos/all/ as default
                if not video_path:
                    video_path = f"/scratch/data/raw/videos/all/{video_name}"
                    updated += 1
                
                item["path"] = video_path
            
            items.append(item)
    
    # Write updated manifest
    with open(manifest_file, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    
    print(f"  ✅ Updated {updated} paths out of {len(items)} entries")

print("\n✅ All manifests updated!")
print(f"\n📋 Verify:")
print(f"   head -1 {MANIFEST_DIR}/train.manifest.processed.jsonl | python3 -m json.tool | grep path")

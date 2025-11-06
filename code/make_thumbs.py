"""
Create thumbnail grids from videos in manifest
Samples 16 frames and creates a 4x4 grid PNG
"""
import os
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import decord as de

de.bridge.set_bridge('native')

MANIFEST = Path(os.environ.get("MANIFEST", "/scratch/data/manifests/train.manifest.jsonl"))
OUTDIR = Path("/scratch/data/processed/thumbnails")
OUTDIR.mkdir(parents=True, exist_ok=True)
MAX_THUMBS = int(os.environ.get("MAX_THUMBS", "10"))


def grid16(frames):
    """Create 4x4 grid from 16 frames"""
    if len(frames) < 16:
        # Pad with last frame if needed
        frames = frames + [frames[-1]] * (16 - len(frames))
    
    H, W, C = frames[0].shape
    g = Image.new("RGB", (W * 4, H * 4))
    k = 0
    for r in range(4):
        for c in range(4):
            if k < len(frames):
                g.paste(Image.fromarray(frames[k]), (c * W, r * H))
            k += 1
    return g


def main():
    count = 0
    with MANIFEST.open() as f:
        for line in f:
            if count >= MAX_THUMBS:
                break
            
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            
            path = rec["path"]
            video_id = rec.get("id", Path(path).stem)
            
            try:
                vr = de.VideoReader(path)
                n = len(vr)
                
                if n < 16:
                    print(f"Skipping {video_id}: too few frames ({n})")
                    continue
                
                # Sample 16 frames uniformly
                idx = np.linspace(0, n - 1, 16, dtype=int)
                frames = [vr[i].asnumpy() for i in idx]
                
                img = grid16(frames)
                img = ImageOps.exif_transpose(img)
                
                out = OUTDIR / f"{video_id}_grid.png"
                img.save(out)
                print(f"✅ Wrote {out}")
                count += 1
                
            except Exception as e:
                print(f"⚠️  Skip {video_id}: {e}")
    
    print(f"\n✅ Created {count} thumbnails in {OUTDIR}")


if __name__ == "__main__":
    main()

# MSR-VTT Dataset Status

## ✅ Complete Setup Verified

**Date**: November 6, 2024  
**Status**: Ready for Training

---

## Dataset Summary

- **Total Videos**: 7,010 videos
- **Total Size**: ~4.26 GB
- **Location on Zaratan**: `~/scratch.msml612-fa25/data/raw/videos/all/`
- **Manifest**: `~/scratch.msml612-fa25/data/manifests/train.manifest.processed.jsonl`
- **Manifest Entries**: 7,010 (all videos successfully processed)

---

## What Was Accomplished

### 1. Dataset Download ✅
- Downloaded MSR-VTT annotation file (`MSR_VTT.json`) from CLIP4Clip GitHub release
- Downloaded 7,010 videos from Kaggle (`vishnutheepb/msrvtt`)
- Videos stored locally in Kaggle cache: `~/.cache/kagglehub/datasets/vishnutheepb/msrvtt/`

### 2. Local Preprocessing ✅
- Created manifest mapping videos to captions
- Extracted metadata for all videos:
  - Duration (seconds)
  - FPS (frames per second)
  - Frame count
  - Resolution
  - Codec information
- Processed manifest saved: `data/processed/manifests/train.manifest.processed.jsonl`

### 3. Zaratan Sync ✅
- Synced all 7,010 videos to Zaratan scratch directory
- Synced processed manifest
- Updated manifest paths to Zaratan location (`/scratch/data/raw/videos/all/`)
- Verified all paths are correct and files exist

---

## File Locations

### On Zaratan:
```
~/scratch.msml612-fa25/data/
├── raw/
│   └── videos/
│       └── all/                    # 7,010 video files
│           ├── video0.mp4
│           ├── video1000.mp4
│           └── ...
└── manifests/
    └── train.manifest.processed.jsonl  # Complete manifest with metadata
```

### Manifest Format:
Each line in `train.manifest.processed.jsonl` is a JSON object:
```json
{
  "id": "video0",
  "captions": ["caption1", "caption2", ...],
  "path": "/scratch/data/raw/videos/all/video0.mp4",
  "split": "train",
  "duration": 10.5,
  "fps": 30.0,
  "num_frames": 315,
  "width": 320,
  "height": 240,
  "codec": "h264",
  "status": "ok"
}
```

---

## Verification Commands

Run these on Zaratan to verify everything is working:

```bash
# Check video count
ls ~/scratch.msml612-fa25/data/raw/videos/all/*.mp4 | wc -l
# Expected: 7010

# Check manifest count
wc -l ~/scratch.msml612-fa25/data/manifests/train.manifest.processed.jsonl
# Expected: 7010

# Verify a sample entry
head -1 ~/scratch.msml612-fa25/data/manifests/train.manifest.processed.jsonl | \
    python3 -m json.tool

# Check if video file exists
ls -lh ~/scratch.msml612-fa25/data/raw/videos/all/video0.mp4
```

---

## Ready for Training

The dataset is fully prepared and ready to use. You can:

1. **Load the dataset** using the manifest:
   ```python
   import json
   from pathlib import Path
   
   manifest = Path("/scratch/data/manifests/train.manifest.processed.jsonl")
   with open(manifest) as f:
       for line in f:
           item = json.loads(line)
           video_path = item["path"]
           captions = item["captions"]
           # Load video and train...
   ```

2. **Use with PyTorch Dataset**:
   - See `code/datasets.py` for example implementation
   - Uses `decord` for efficient video loading
   - Supports frame sampling and transformations

3. **Start training**:
   - All videos are accessible
   - Metadata is complete
   - Paths are correct

---

## Notes

- **Dataset Source**: Kaggle (`vishnutheepb/msrvtt`) - 7,010 videos available
- **Annotation Source**: CLIP4Clip GitHub release (`MSRVTT_data.json`)
- **Preprocessing**: Done locally (no GPU required)
- **Sync Method**: `rsync` from local machine to Zaratan
- **Path Mapping**: Container path `/scratch/data/` maps to host `~/scratch.msml612-fa25/data/`

---

## Next Steps

1. ✅ Dataset ready - **DONE**
2. ⏭️ Implement your text2video model
3. ⏭️ Create training script
4. ⏭️ Start training on Zaratan GPU

---

**Last Updated**: November 6, 2024  
**Status**: ✅ Complete and Verified


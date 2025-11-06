# Local Data Preprocessing Guide

Preprocess MSR-VTT data on your local machine (no GPU needed), then sync to Zaratan.

## Prerequisites

Install required tools locally:

```bash
# Install Python dependencies
pip3 install decord==0.6.0 pandas tqdm yt-dlp pillow

# Install ffmpeg (for video conversion)
# macOS:
brew install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

## Quick Start

### 1. Prepare MSR-VTT Dataset Locally

```bash
# Create directory structure
mkdir -p data/msr-vtt/{annotation,videos/all}

# Place your MSR_VTT.json annotation file:
# data/msr-vtt/annotation/MSR_VTT.json

# (Optional) Place video files if you have them:
# data/msr-vtt/videos/all/video0.mp4
# data/msr-vtt/videos/all/video1.mp4
# ...
```

### 2. Preprocess Data Locally

```bash
# Process train split
./scripts/preprocess_local.sh train

# Process validation split
./scripts/preprocess_local.sh val

# Process test split
./scripts/preprocess_local.sh test
```

Or manually:

```bash
SPLIT=train \
MSR_VTT_ROOT=./data/msr-vtt \
OUT_ROOT=./data/processed \
python3 code/download_msrvtt_local.py
```

### 3. Review Processed Data

```bash
# Check manifest
head -3 data/processed/manifests/train.manifest.jsonl | python3 -m json.tool

# Check video count
ls data/processed/raw/videos/train/ | wc -l

# Check disk usage
du -sh data/processed/
```

### 4. Sync to Zaratan

```bash
# Sync all processed data
./scripts/sync_processed_data.sh

# Or manually:
rsync -av --progress \
    data/processed/ \
    vvr2211@login.zaratan.umd.edu:~/scratch.msml612-fa25/data/
```

### 5. Fix Paths on Zaratan

After syncing, update manifest paths to work in container:

```bash
# SSH to Zaratan
ssh zaratan

# Run fix script
bash ~/text2video/scripts/fix_manifest_paths.sh

# Or manually:
sed -i 's|"path":"raw/videos|"path":"/scratch/data/raw/videos|g' \
    ~/scratch.msml612-fa25/data/manifests/*.manifest.jsonl
```

## Directory Structure

After preprocessing locally:

```
data/
├── msr-vtt/                    # Source data
│   ├── annotation/
│   │   └── MSR_VTT.json
│   └── videos/
│       └── all/
│           ├── video0.mp4
│           └── ...
└── processed/                  # Preprocessed (ready to sync)
    ├── raw/
    │   └── videos/
    │       ├── train/
    │       ├── val/
    │       └── test/
    ├── manifests/
    │   ├── train.manifest.jsonl
    │   ├── val.manifest.jsonl
    │   └── test.manifest.jsonl
    └── logs/
```

After syncing to Zaratan:

```
~/scratch.msml612-fa25/data/
├── raw/videos/{train,val,test}/  # Videos
├── manifests/                    # Manifests (paths fixed)
└── logs/                         # Processing logs
```

## What Gets Processed

1. **Videos**: Downloaded/copied and normalized to H.264/AAC MP4
2. **Manifests**: Created with metadata (frames, fps, duration, captions)
3. **Paths**: Stored as relative paths (fixed after syncing)

## Advantages of Local Preprocessing

✅ **No GPU needed** - Process while waiting for GPU allocation  
✅ **Faster iteration** - Test locally before syncing  
✅ **Bandwidth efficient** - Only sync processed data  
✅ **Parallel processing** - Use all CPU cores locally  

## Troubleshooting

### Missing ffmpeg
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Missing decord
```bash
pip3 install decord==0.6.0
```

### Video download fails
- Check internet connection
- Verify URLs in MSR_VTT.json
- Some videos may be unavailable (will be marked as "download_failed")

### Large dataset
- Process splits separately
- Use `--exclude` with rsync to skip failed downloads
- Check disk space before syncing

## Next Steps

After syncing and fixing paths:

1. **Verify on Zaratan**:
   ```bash
   ~/text2video/run.sh python -c "
   import json
   with open('/scratch/data/manifests/train.manifest.jsonl') as f:
       print(json.loads(f.readline()))
   "
   ```

2. **Create thumbnails** (optional):
   ```bash
   MANIFEST=/scratch/data/manifests/train.manifest.jsonl \
   ~/text2video/run.sh python ~/text2video/code/make_thumbs.py
   ```

3. **Start training**:
   ```python
   from datasets import MSRVTTDataset
   ds = MSRVTTDataset("/scratch/data/manifests/train.manifest.jsonl")
   ```

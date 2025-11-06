# MSR-VTT Dataset Setup Guide

## Dataset Structure

MSR-VTT (Microsoft Research Video to Text) dataset structure:

```
msr-vtt/
├── annotation/
│   └── MSR_VTT.json          # Annotations file
├── videos/
│   └── all/                  # Video files (video_id.mp4)
│       ├── video0.mp4
│       ├── video1.mp4
│       └── ...
└── README.txt
```

## Download MSR-VTT

1. **Official Download**: 
   - Visit: http://ms-multimedia-challenge.com/2017/dataset
   - Download the dataset (videos + annotations)

2. **Place on Zaratan**:
   ```bash
   # On Zaratan, create directory
   mkdir -p ~/text2video/data/msr-vtt
   
   # Upload annotation file
   scp MSR_VTT.json vvr2211@login.zaratan.umd.edu:~/text2video/data/msr-vtt/annotation/
   
   # Upload videos (if you have them locally)
   rsync -av videos/ vvr2211@login.zaratan.umd.edu:~/text2video/data/msr-vtt/videos/all/
   ```

## Quick Start

### 1. Set up directories on Zaratan

```bash
mkdir -p ~/scratch.msml612-fa25/data/{raw,processed,manifests,logs}
mkdir -p ~/scratch.msml612-fa25/data/raw/videos/{train,val,test}
mkdir -p ~/scratch.msml612-fa25/data/processed/thumbnails
```

### 2. Install dependencies

```bash
~/text2video/run.sh pip install --no-cache-dir yt-dlp pandas tqdm decord==0.6.0 opencv-python-headless pillow
```

### 3. Download and create manifest

```bash
# For train split
SPLIT=train \
MSR_VTT_ROOT=~/text2video/data/msr-vtt \
~/text2video/run.sh python ~/text2video/code/download_msrvtt.py

# For validation split
SPLIT=val \
MSR_VTT_ROOT=~/text2video/data/msr-vtt \
~/text2video/run.sh python ~/text2video/code/download_msrvtt.py

# For test split
SPLIT=test \
MSR_VTT_ROOT=~/text2video/data/msr-vtt \
~/text2video/run.sh python ~/text2video/code/download_msrvtt.py
```

### 4. Create thumbnails (optional)

```bash
MANIFEST=/scratch/data/manifests/train.manifest.jsonl \
MAX_THUMBS=10 \
~/text2video/run.sh python ~/text2video/code/make_thumbs.py
```

### 5. Use in training

```python
from datasets import MSRVTTDataset
from torch.utils.data import DataLoader

# Load dataset
train_ds = MSRVTTDataset(
    "/scratch/data/manifests/train.manifest.jsonl",
    num_frames=16,
    size=256
)

# Create dataloader
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Iterate
for batch in train_loader:
    videos = batch["video"]  # (B, T, 3, H, W)
    captions = batch["caption"]  # List of strings
    # ... your training code
```

## Manifest Format

Each line in `manifest.jsonl` is a JSON object:

```json
{
  "id": "video0",
  "url": "https://...",
  "caption": "A person is ...",
  "captions": ["A person is ...", "Someone is ..."],
  "path": "/scratch/data/raw/videos/train/video0.mp4",
  "split": "train",
  "category": "music",
  "num_frames": 150,
  "fps": 30.0,
  "duration": 5.0,
  "status": "ok"
}
```

## Notes

- MSR-VTT has multiple captions per video (typically 20)
- The script uses the first caption by default, but stores all captions
- Videos are normalized to H.264/AAC MP4 format
- If videos are already downloaded locally, the script will copy them
- If only URLs are available, it will attempt to download using yt-dlp

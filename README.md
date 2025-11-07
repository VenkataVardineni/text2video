# Text-to-Video (Interim) — MSR-VTT Subset

A reproducible baseline for text-to-video diffusion on MSR-VTT dataset.

## Reproduce (Zaratan)

### 1. Get GPU Node

```bash
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=24:00:00 --mem=40G
srun --pty bash
```

### 2. Ensure Setup

Make sure `~/text2video/run.sh` is available (container/venv helper).

### 3. Split & Stats

```bash
MANIFEST=/scratch/data/manifests/train.manifest.processed.jsonl \
~/text2video/run.sh python ~/text2video/code/split_manifest.py

MANIFEST=/scratch/data/manifests/train.jsonl \
~/text2video/run.sh python ~/text2video/code/data_card.py
```

### 4. Train

```bash
chmod +x ~/text2video/train_baseline.sh
bash ~/text2video/train_baseline.sh
```

### 5. Evaluate

```bash
VAL_MANIFEST=/scratch/data/manifests/val.jsonl \
FRAMES=8 \
~/text2video/run.sh python ~/text2video/code/eval_clipscore.py
```

### 6. Throughput

```bash
BS=2 FRAMES=8 SIZE=64 \
~/text2video/run.sh python ~/text2video/code/bench_throughput.py
```

## Model

* **3D UNet** with FiLM conditioning (CLIP text embedding).
* **DDPM** noise schedule (cosine); AMP + EMA; preview samples saved as frame grids.

## Data

* MSR-VTT subset, 7,010 videos. Preprocessed manifest includes path, captions, fps, duration, frames.
* Train/val split stratified by duration.

## Notes

* All installs & caches live in `/scratch/venv` and `/scratch/*_cache`.
* Replace the baseline with a latent UNet or add CFG/temporal attention for higher performance.

## Dependencies

Install via:
```bash
~/text2video/run.sh pip install --no-cache-dir einops transformers seaborn matplotlib open_clip_torch
```

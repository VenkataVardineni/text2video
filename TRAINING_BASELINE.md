# Text-to-Video Diffusion Training Baseline

A compact, working baseline for training a text-to-video diffusion model on MSR-VTT.

## Features

- ✅ **3D UNet** with text conditioning (CLIP text encoder → FiLM)
- ✅ **DDPM** trainer in pixel space on downsampled clips (8×64×64)
- ✅ Dataloader wired to processed manifest
- ✅ **EMA**, mixed precision, checkpoints, and sample frame grids
- ✅ Single-GPU training (A100 compatible)

## Quick Start

### 1. Install Dependencies

On Zaratan (inside your container):

```bash
~/text2video/run.sh pip install --no-cache-dir einops transformers
```

### 2. Verify Dataset

Make sure your manifest is ready:

```bash
# Check manifest exists
ls -lh ~/scratch.msml612-fa25/data/manifests/train.manifest.processed.jsonl

# Verify a sample entry
head -1 ~/scratch.msml612-fa25/data/manifests/train.manifest.processed.jsonl | \
    python3 -m json.tool | head -10
```

### 3. Launch Training

**Option A: Use the helper script**

```bash
# Default settings (2000 steps, batch=2, frames=8, size=64)
./scripts/train_t2v_baseline.sh

# Custom settings
STEPS=5000 BS=4 FRAMES=8 SIZE=64 ./scripts/train_t2v_baseline.sh
```

**Option B: Direct command**

```bash
MANIFEST=/scratch/data/manifests/train.manifest.processed.jsonl \
OUTDIR=/scratch/runs/t2v_ddpm_baseline \
STEPS=2000 BS=2 FRAMES=8 SIZE=64 LR=1e-4 \
~/text2video/run.sh python ~/text2video/code/train_t2v_diffusion.py
```

## What to Expect

### Training Output

```
Out: /scratch/runs/t2v_ddpm_baseline | Device: cuda
step 000020 | loss 0.1234
step 000040 | loss 0.0987
...
step 000200 | loss 0.0456
saved /scratch/runs/t2v_ddpm_baseline/ckpt_000200.pt
...
```

### Output Files

- **Checkpoints**: `OUTDIR/ckpt_*.pt` (every 200 steps)
  - Contains: `step`, `model`, `ema` state dicts
- **Samples**: `OUTDIR/samples/preview_step*.png` (every 200 steps)
  - Frame grids showing denoised previews

### Monitoring

```bash
# Watch training progress
tail -f /scratch/runs/t2v_ddpm_baseline/train.log  # if redirected

# Check GPU usage
nvidia-smi

# List checkpoints
ls -lh /scratch/runs/t2v_ddpm_baseline/ckpt_*.pt

# View latest sample
ls -t /scratch/runs/t2v_ddpm_baseline/samples/ | head -1
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MANIFEST` | `/scratch/data/manifests/train.manifest.processed.jsonl` | Path to manifest file |
| `OUTDIR` | `/scratch/runs/t2v_ddpm` | Output directory for checkpoints/samples |
| `STEPS` | `1000` | Number of training steps |
| `BS` | `2` | Batch size |
| `FRAMES` | `8` | Number of frames per video clip |
| `SIZE` | `64` | Frame resolution (SIZE×SIZE) |
| `LR` | `1e-4` | Learning rate |

### Memory Optimization

If you hit VRAM limits, try:

```bash
# Reduce batch size
BS=1 ./scripts/train_t2v_baseline.sh

# Reduce frames
FRAMES=4 ./scripts/train_t2v_baseline.sh

# Reduce resolution
SIZE=48 ./scripts/train_t2v_baseline.sh

# Or combine
BS=1 FRAMES=4 SIZE=48 ./scripts/train_t2v_baseline.sh
```

## Architecture

### Model: UNet3DCond

- **Input**: `(B, T, 3, H, W)` video clips in `[-1, 1]`
- **Conditioning**: CLIP text embeddings (512d) via FiLM
- **Output**: Predicted noise `(B, T, 3, H, W)`
- **Base channels**: 64 (scales to 128, 256)

### Diffusion: DDPM

- **Schedule**: Cosine beta schedule (1000 steps)
- **Loss**: MSE between predicted and actual noise
- **Sampling**: Single-step preview (not full denoising)

### Text Encoder: CLIP

- **Model**: `openai/clip-vit-base-patch32`
- **Output**: Mean-pooled text embeddings (512d)
- **Frozen**: Text encoder is not trained

## Next Steps / Upgrades

When ready to improve:

1. **Latent Diffusion**: Replace pixel-space with VAE latents
   - Use Diffusers' `AutoencoderKL`
   - Train in latent space (faster, better quality)

2. **Cross-Attention UNet**: Replace FiLM with cross-attention
   - Condition with text tokens instead of pooled embedding
   - Better text-video alignment

3. **Better Sampling**: Implement full DDPM sampling
   - Multi-step denoising
   - Cosine or k-arras sampling schedules

4. **Multi-GPU**: Use Accelerate for distributed training
   - Scale to multiple GPUs
   - Support bf16 training

5. **Larger Model**: Increase base channels
   - `base=128` or `base=256` for better quality
   - Requires more VRAM

## Troubleshooting

### "CUDA out of memory"

Reduce batch size, frames, or resolution (see Memory Optimization above).

### "Manifest not found"

Check the manifest path:
```bash
ls -lh ~/scratch.msml612-fa25/data/manifests/train.manifest.processed.jsonl
```

Update `MANIFEST` environment variable if needed.

### "No module named 'einops'"

Install dependencies:
```bash
~/text2video/run.sh pip install --no-cache-dir einops transformers
```

### Slow training

- Reduce `num_workers` in DataLoader (default: 4)
- Check disk I/O (videos loading from scratch)
- Verify GPU is being used (`nvidia-smi`)

## Files

- `code/dataset_msrvtt_diff.py` - Dataset loader with CLIP tokenization
- `code/models_unet3d_film.py` - 3D UNet with FiLM conditioning
- `code/diffusion_min.py` - DDPM utilities
- `code/train_t2v_diffusion.py` - Main training script
- `scripts/train_t2v_baseline.sh` - Launch script

---

**Ready to train!** 🚀


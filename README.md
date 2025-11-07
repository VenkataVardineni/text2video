# Text-to-Video Diffusion on MSR-VTT Dataset

A complete, reproducible implementation of text-conditioned video diffusion using DDPM with 3D UNet and FiLM conditioning. This project demonstrates end-to-end video generation from text prompts on the MSR-VTT dataset.

## 📊 Project Overview

This project implements a text-to-video diffusion model following the DDPM (Denoising Diffusion Probabilistic Models) framework, adapted for temporal video generation with text conditioning. The implementation includes data curation, model design, training infrastructure, and evaluation metrics.

---

## [1] Data Prep/Curation - Time Sequence Data (10 points)

### Dataset Collection

- **Source**: MSR-VTT (Microsoft Research Video to Text) dataset subset
- **Total Videos**: **7,010 videos** (~4.26 GB)
- **Video Format**: H.264 MP4, normalized and verified
- **Annotations**: 10,000 video entries with multiple captions per video (up to 20 captions)

### Data Processing Pipeline

1. **Download & Verification**
   - Videos downloaded from Kaggle (`vishnutheepb/msrvtt`)
   - Annotations from CLIP4Clip GitHub release (`MSRVTT_data.json`)
   - All video paths verified and validated

2. **Metadata Extraction**
   - **Duration**: Extracted for all videos (range: 1-40 seconds)
   - **FPS**: Frame rate analysis (typically 30 FPS)
   - **Frame Count**: Total frames per video
   - **Resolution**: Width/height extraction
   - **Codec**: Video codec verification (H.264)

3. **Data Curation**
   - **Manifest Creation**: JSONL format with complete metadata
   - **Train/Val Split**: 90/10 stratified by duration bins (ensures balanced distribution)
   - **Quality Filtering**: Videos with missing metadata or corruption flagged
   - **Statistics Generation**: Duration and FPS histograms for dataset analysis

### Data Statistics

- **Training Set**: ~6,309 videos (90%)
- **Validation Set**: ~701 videos (10%)
- **Average Duration**: ~10-15 seconds per video
- **Frame Sampling**: 8 frames per video clip (uniform sampling)
- **Resolution**: Downsampled to 64×64 for training

### Data Files

```
data/
├── manifests/
│   ├── train.manifest.processed.jsonl  # 7,010 entries with full metadata
│   ├── train.jsonl                      # Training split
│   ├── val.jsonl                         # Validation split
│   └── stats/                           # Dataset statistics
│       ├── durations.csv
│       ├── fps.csv
│       ├── duration_hist.png
│       └── fps_hist.png
└── raw/videos/all/                      # 7,010 video files
```

**Manifest Format**:
```json
{
  "id": "video0",
  "captions": ["A person is walking...", "Someone walks..."],
  "path": "/scratch/data/raw/videos/all/video0.mp4",
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

## [2] Difficulty of NN Design & Implementation (25 points)

### Model Architecture

**Baseline**: Text-conditioned diffusion model (DDPM) adapted for video generation.

#### Core Components

1. **3D UNet with Temporal Convolutions**
   - **Architecture**: U-Net encoder-decoder with 3D convolutions
   - **Base Channels**: 64 (scales to 128, 256 in deeper layers)
   - **Temporal Hierarchy**: 
     - Encoder: Spatial downsampling (H×W) → Temporal + spatial downsampling (T×H×W)
     - Decoder: Temporal upsampling → Spatial upsampling
   - **Skip Connections**: Preserves fine-grained details across encoder-decoder

2. **FiLM (Feature-wise Linear Modulation) Conditioning**
   - **Text Encoding**: CLIP text encoder (`openai/clip-vit-base-patch32`) → 512D embeddings
   - **FiLM Layers**: Per-block gamma (scale) and beta (shift) modulation
   - **Integration**: Text embeddings fused with timestep embeddings
   - **Adaptive Conditioning**: Each 3D block receives conditional modulation

3. **DDPM Diffusion Process**
   - **Noise Schedule**: Cosine beta schedule (1000 timesteps)
   - **Forward Process**: Progressive noise addition q(x_t | x_0)
   - **Reverse Process**: Learned denoising p(x_{t-1} | x_t, text)
   - **Loss**: MSE between predicted and actual noise

#### Implementation Complexity

- **Temporal 3D Convolutions**: Handles video sequences (B, T, C, H, W)
- **Conditional FiLM Modulation**: Per-layer text conditioning without cross-attention overhead
- **EMA (Exponential Moving Average)**: Stable training with moving average of model weights
- **Mixed Precision Training (AMP)**: Memory-efficient training with automatic mixed precision
- **Preview Sampling**: Real-time visualization of denoising progress

#### Model Complexity Metrics

- **Parameters**: ~15M parameters (base=64)
- **Input**: (B, 8, 3, 64, 64) video clips + text embeddings
- **Output**: Predicted noise (B, 8, 3, 64, 64)
- **Memory**: ~4-6 GB VRAM (batch_size=2, frames=8, size=64)

### Citations & References

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)
- **CLIP**: Radford et al., "Learning Transferable Visual Models from Natural Language Supervision" (ICML 2021)
- **3D UNet**: Adapted from 2D UNet (Ronneberger et al., U-Net, 2015) for temporal data

### Adaptations for Video Generation

- Extended 2D UNet to 3D for temporal modeling
- Integrated FiLM conditioning for text-video alignment
- Implemented cosine noise schedule optimized for video
- Added frame-level sampling and temporal consistency

---

## [3] Working, Clean, Readable Code (20 points)

### Reproducibility

**One-Command Training**:
```bash
bash ~/text2video/train_baseline.sh
```

**Complete Setup**:
- Containerized execution via `run.sh` (Apptainer/Singularity)
- Fixed random seeds (`PYTHONHASHSEED=612`)
- Deterministic operations (`CUBLAS_WORKSPACE_CONFIG`)
- Version tracking (`versions.txt` generated after training)

### Project Structure

```
text2video/
├── code/
│   ├── dataset_msrvtt_diff.py      # Dataset loader with CLIP tokenization
│   ├── models_unet3d_film.py       # 3D UNet with FiLM conditioning
│   ├── diffusion_min.py            # DDPM utilities
│   ├── train_t2v_diffusion.py      # Main training script
│   ├── split_manifest.py           # Train/val split with stratification
│   ├── data_card.py                # Dataset statistics generation
│   ├── eval_clipscore.py           # CLIPScore evaluation
│   └── bench_throughput.py         # Throughput benchmarking
├── configs/
│   └── baseline.sh                 # Configuration file
├── scripts/
│   └── train_t2v_baseline.sh      # One-click training script
├── train_baseline.sh               # Main training entry point
├── run.sh                          # Container/venv helper
└── README.md                       # This file
```

### Code Quality Features

- **Modular Design**: Separate files for dataset, model, training, evaluation
- **Clear Configuration**: Environment variables for all hyperparameters
- **Error Handling**: Robust video loading with fallbacks
- **Documentation**: Inline comments and docstrings
- **Type Hints**: Python type annotations for clarity
- **Clean Outputs**: Organized checkpoint and sample directories

### Reproducibility Checklist

✅ Fixed random seeds (Python, NumPy, PyTorch)  
✅ Deterministic CUDA operations  
✅ Containerized environment (Apptainer/Singularity)  
✅ One-command training script  
✅ Configuration file with all hyperparameters  
✅ Version tracking (pip freeze → versions.txt)  
✅ Clear directory structure  
✅ Comprehensive README  

---

## [4] High Performance (25 points)

### Evaluation Metrics

#### Quantitative Metrics

1. **CLIPScore@8f**
   - **Metric**: CLIP-based text-video alignment score
   - **Method**: Average cosine similarity between video frames and text embeddings
   - **Implementation**: Uses OpenCLIP ViT-B/32
   - **Evaluation**: On validation set (701 videos)

2. **Throughput**
   - **Metric**: Samples/second and iterations/second
   - **Hardware**: A100_1g.5gb slice on Zaratan
   - **Measurement**: 50 forward passes, averaged
   - **Configuration**: Batch size, frames, resolution

3. **Training Efficiency**
   - **VRAM Usage**: ~4-6 GB (batch_size=2, frames=8, size=64)
   - **Wall-clock Time**: Per 1000 steps
   - **Convergence**: Loss curves and checkpoint analysis

#### Qualitative Evaluation

- **Sample Frame Grids**: Visual progress of denoising (saved every 200 steps)
- **Temporal Consistency**: Frame-to-frame coherence
- **Text-Video Alignment**: Visual verification of caption matching

### Performance Benchmarks

**Run evaluation**:
```bash
# CLIPScore evaluation
VAL_MANIFEST=/scratch/data/manifests/val.jsonl FRAMES=8 \
~/text2video/run.sh python ~/text2video/code/eval_clipscore.py

# Throughput benchmark
BS=2 FRAMES=8 SIZE=64 \
~/text2video/run.sh python ~/text2video/code/bench_throughput.py
```

**Expected Results** (fill in after training):
- CLIPScore@8f: _TBD_
- Throughput: _TBD_ samples/sec (BS=2, T=8, size=64)
- Training time: _TBD_ minutes per 1000 steps

### Optimization Techniques

- **Mixed Precision (AMP)**: 2x speedup, reduced memory
- **EMA**: Stable convergence, better final performance
- **Efficient Video Loading**: Decord for fast frame sampling
- **Pin Memory**: Faster GPU transfer
- **Multi-worker DataLoader**: Parallel data loading

---

## 🚀 Quick Start

### Prerequisites

- Access to UMD Zaratan GPU cluster
- Apptainer/Singularity container runtime
- Python 3.10+

### Installation

```bash
# 1. Get GPU allocation
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=24:00:00 --mem=40G
srun --pty bash

# 2. Install dependencies
~/text2video/run.sh pip install --no-cache-dir \
    einops transformers seaborn matplotlib open_clip_torch

# 3. Prepare data
MANIFEST=/scratch/data/manifests/train.manifest.processed.jsonl \
~/text2video/run.sh python ~/text2video/code/split_manifest.py

MANIFEST=/scratch/data/manifests/train.jsonl \
~/text2video/run.sh python ~/text2video/code/data_card.py
```

### Training

```bash
# One-command training
bash ~/text2video/train_baseline.sh

# Or with custom settings
STEPS=5000 BS=4 FRAMES=8 SIZE=64 \
bash ~/text2video/train_baseline.sh
```

### Evaluation

```bash
# CLIPScore
VAL_MANIFEST=/scratch/data/manifests/val.jsonl FRAMES=8 \
~/text2video/run.sh python ~/text2video/code/eval_clipscore.py

# Throughput
BS=2 FRAMES=8 SIZE=64 \
~/text2video/run.sh python ~/text2video/code/bench_throughput.py
```

---

## 📁 Output Files

After training, you'll find:

```
/scratch/runs/t2v_ddpm_baseline/
├── ckpt_000200.pt          # Checkpoints (every 200 steps)
├── ckpt_000400.pt
├── ...
├── samples/
│   ├── preview_step000200.png  # Sample frame grids
│   ├── preview_step000400.png
│   └── ...
└── versions.txt            # Package versions for reproducibility
```

---

## 🔧 Configuration

All hyperparameters are in `configs/baseline.sh`:

```bash
export MANIFEST=/scratch/data/manifests/train.manifest.processed.jsonl
export VAL_MANIFEST=/scratch/data/manifests/val.jsonl
export OUTDIR=/scratch/runs/t2v_ddpm_baseline
export STEPS=2000
export BS=2
export FRAMES=8
export SIZE=64
export LR=1e-4
export PYTHONHASHSEED=612
```

---

## 📚 References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
- Perez, E., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
- Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML*.
- Xu, J., et al. (2016). MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. *CVPR*.

---

## 📝 License

This project is for educational/research purposes. MSR-VTT dataset usage follows Microsoft Research terms.

---

## 👤 Author

VenkataVardineni  
UMD MSML612 - Fall 2024

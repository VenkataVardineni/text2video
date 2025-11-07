# Project Structure & File Descriptions

Complete overview of all important files and folders in the text2video project.

---

## 📁 Root Directory Files

### `README.md`
**Purpose**: Main project documentation  
**Contents**: 
- Complete rubric coverage (Data prep, NN design, Code quality, Performance)
- Quick start guide
- Configuration details
- Evaluation instructions
- References and citations

### `train_baseline.sh`
**Purpose**: One-click training entry point  
**Contents**:
- Sources configuration from `configs/baseline.sh`
- Sets up random seeds for reproducibility
- Executes training script inside container
- Ensures deterministic training with `PYTHONHASHSEED`

### `run.sh`
**Purpose**: Container/venv helper script  
**Contents**:
- Loads Apptainer/Singularity container runtime
- Sets up Python virtual environment in `/scratch/venv`
- Configures cache directories (HuggingFace, PyTorch)
- Mounts host directories to container paths
- Executes user commands in isolated environment

### `report_interim.md`
**Purpose**: Interim report template  
**Contents**:
- Rubric sections for interim submission
- Placeholders for results (CLIPScore, throughput)
- Limitations and next steps discussion

### `DATASET_STATUS.md`
**Purpose**: Dataset setup status documentation  
**Contents**:
- Dataset summary (7,010 videos)
- File locations on Zaratan
- Verification commands
- Manifest format examples

---

## 📁 `code/` - Core Implementation

### `dataset_msrvtt_diff.py`
**Purpose**: PyTorch Dataset for MSR-VTT with diffusion preprocessing  
**Key Classes**:
- `MSRVTTDiffusionDataset`: Loads videos, samples frames, tokenizes captions
- Frame sampling: Uniform sampling of 8 frames per video
- Text tokenization: CLIP tokenizer for text conditioning
- Normalization: Frames normalized to [-1, 1] for diffusion

**Key Functions**:
- `_sample_idx()`: Uniform frame sampling with padding
- `__getitem__()`: Returns video tensor (T,3,H,W) + CLIP tokens

### `models_unet3d_film.py`
**Purpose**: 3D UNet model with FiLM text conditioning  
**Key Classes**:
- `FiLM`: Feature-wise Linear Modulation (gamma/beta scaling)
- `Block3D`: 3D convolutional block with FiLM conditioning
- `UNet3DCond`: Main 3D UNet with encoder-decoder + skip connections

**Architecture**:
- Encoder: 3D convs with spatial + temporal downsampling
- Bottleneck: Middle block for feature refinement
- Decoder: 3D transpose convs with skip connections
- Conditioning: FiLM layers modulate each block with text embeddings

### `diffusion_min.py`
**Purpose**: DDPM diffusion utilities  
**Key Functions**:
- `cosine_beta_schedule()`: Cosine noise schedule (1000 steps)
- `DDPM` class: Forward/reverse diffusion process
  - `q_sample()`: Forward process (add noise)
  - Noise schedule computation (alphas, betas, cumulative products)

### `train_t2v_diffusion.py`
**Purpose**: Main training script  
**Key Components**:
- Data loading: Creates DataLoader with MSRVTTDiffusionDataset
- Model setup: UNet3DCond + EMA model
- Text encoder: Frozen CLIP text encoder
- Training loop: DDPM loss, AMP, EMA updates
- Checkpointing: Saves every 200 steps
- Sampling: Generates preview grids during training

**Key Functions**:
- `get_text_emb()`: Extracts CLIP text embeddings (mean pooling)
- `update_ema()`: Exponential moving average of model weights
- `to_image_grid()`: Saves frame grids for visualization

### `split_manifest.py`
**Purpose**: Creates train/val split with stratification  
**Functionality**:
- Reads full manifest
- Stratifies by duration bins (10 bins: 0-5s, 5-10s, ..., 45-50s)
- 90/10 train/val split within each bin
- Writes `train.jsonl` and `val.jsonl`

### `data_card.py`
**Purpose**: Generates dataset statistics  
**Functionality**:
- Extracts duration and FPS from manifest
- Saves CSV files (durations.csv, fps.csv)
- Generates histograms (duration_hist.png, fps_hist.png)
- Prints summary statistics (mean, std, min, max)

### `eval_clipscore.py`
**Purpose**: CLIPScore evaluation on validation set  
**Functionality**:
- Loads validation manifest
- For each video: samples frames, computes CLIP embeddings
- Computes cosine similarity between video and text embeddings
- Reports average CLIPScore@8f

### `bench_throughput.py`
**Purpose**: Model throughput benchmarking  
**Functionality**:
- Creates dummy inputs (random video + text embeddings)
- Warms up GPU (10 iterations)
- Times 50 forward passes
- Reports samples/sec and iterations/sec

### `datasets.py`
**Purpose**: Original dataset implementations (backward compatibility)  
**Classes**:
- `MSRVTTDataset`: General-purpose video dataset
- `VideoCaptionDataset`: Simple video-caption pairs

### `msrvtt_to_manifest.py`
**Purpose**: Converts MSR-VTT JSON to manifest format  
**Functionality**:
- Parses MSR-VTT annotation JSON
- Maps video IDs to file paths
- Extracts captions (up to 20 per video)
- Writes manifest.jsonl

### `preprocess_msrvtt.py`
**Purpose**: Enriches manifest with video metadata  
**Functionality**:
- Reads manifest
- Probes videos with ffprobe
- Extracts: duration, fps, frame count, resolution, codec
- Writes enriched manifest

### `download_msrvtt_local.py`
**Purpose**: Local video downloader (for preprocessing)  
**Functionality**:
- Downloads videos from URLs (if needed)
- Normalizes to H.264/AAC MP4
- Creates manifest with metadata

### `make_thumbs.py`
**Purpose**: Creates thumbnail grids for visual inspection  
**Functionality**:
- Samples 16 frames from videos
- Creates 4×4 grid PNG images
- Useful for dataset quality checks

### `smoke_train.py`
**Purpose**: Simple GPU test script  
**Functionality**:
- Minimal training loop to verify GPU access
- Tests PyTorch CUDA availability

---

## 📁 `scripts/` - Utility Scripts

### `train_t2v_baseline.sh`
**Purpose**: Training launch script with GPU node detection  
**Functionality**:
- Checks if running on GPU node
- Validates manifest exists
- Sets environment variables
- Launches training with proper paths

### `sync_to_zaratan.sh`
**Purpose**: Syncs local project to Zaratan  
**Functionality**:
- Uses rsync to transfer files
- Excludes .git, __pycache__, LFS files
- Preserves permissions and timestamps

### `sync_processed_data.sh`
**Purpose**: Syncs processed data to Zaratan  
**Functionality**:
- Transfers processed videos and manifests
- Updates paths for Zaratan environment

### `preprocess_local.sh`
**Purpose**: Orchestrates local preprocessing  
**Functionality**:
- Sets up directories
- Runs download/preprocessing scripts
- Handles optional dependencies (decord)

### `fix_paths_on_zaratan.py`
**Purpose**: Updates manifest paths after syncing  
**Functionality**:
- Replaces local paths with Zaratan paths
- Updates `/scratch/data/raw/videos/all/` paths

### `download_full_msrvtt.sh`
**Purpose**: Downloads complete MSR-VTT dataset  
**Functionality**:
- Attempts multiple download sources
- Clones CLIP4Clip repository for annotations

### `get_msrvtt_data.sh`
**Purpose**: Downloads MSR_VTT.json from GitHub  
**Functionality**:
- Fetches annotation file from CLIP4Clip release

### `verify_and_preprocess_msrvtt.sh`
**Purpose**: Verifies dataset and runs preprocessing  
**Functionality**:
- Checks for MSR_VTT.json
- Validates JSON format
- Runs preprocessing pipeline

---

## 📁 `configs/` - Configuration Files

### `baseline.sh`
**Purpose**: Training configuration  
**Environment Variables**:
- `MANIFEST`: Path to training manifest
- `VAL_MANIFEST`: Path to validation manifest
- `OUTDIR`: Output directory for checkpoints
- `STEPS`: Number of training steps
- `BS`: Batch size
- `FRAMES`: Frames per video clip
- `SIZE`: Frame resolution (SIZE×SIZE)
- `LR`: Learning rate
- `PYTHONHASHSEED`: Random seed for reproducibility
- `CUBLAS_WORKSPACE_CONFIG`: CUDA deterministic operations

---

## 📁 `data/` - Data Files

### `msr-vtt/annotation/`
**Contents**:
- `MSR_VTT.json`: Original annotation file (10,000 videos)
- `MSRVTT_data.json`: Alternative format from CLIP4Clip

### `processed/manifests/`
**Contents**:
- `train.manifest.processed.jsonl`: Full processed manifest (7,010 entries)
- `train.jsonl`: Training split
- `val.jsonl`: Validation split
- `stats/`: Dataset statistics (CSV + PNG histograms)

### `README_MSRVTT.md`
**Purpose**: MSR-VTT dataset setup guide  
**Contents**:
- Dataset structure
- Download instructions
- Quick start guide
- Manifest format documentation

---

## 📁 Documentation Files

### `TRAINING_BASELINE.md`
**Purpose**: Training baseline documentation  
**Contents**:
- Model architecture details
- Training workflow
- Configuration options
- Troubleshooting guide

### `WORKFLOW.md`
**Purpose**: Daily workflow guide for Zaratan  
**Contents**:
- Quick start commands
- GPU allocation instructions
- Data importing guide
- Useful commands

### `ZARATAN_SETUP.md`
**Purpose**: Zaratan cluster setup guide  
**Contents**:
- Container setup instructions
- GPU allocation commands
- Environment configuration
- Troubleshooting

### `LOCAL_PREPROCESSING.md`
**Purpose**: Local preprocessing workflow  
**Contents**:
- Prerequisites
- Step-by-step preprocessing
- Syncing to Zaratan
- Path fixing instructions

### `GET_MSRVTT_DATASET.md`
**Purpose**: MSR-VTT dataset download guide  
**Contents**:
- Official download sources
- Alternative sources
- Verification steps
- Expected dataset size

---

## 🔧 Key Workflows

### Training Workflow
1. `configs/baseline.sh` → Sets environment variables
2. `train_baseline.sh` → Sources config, sets seeds, runs training
3. `code/train_t2v_diffusion.py` → Main training loop
4. Uses `code/dataset_msrvtt_diff.py` → Data loading
5. Uses `code/models_unet3d_film.py` → Model forward pass
6. Uses `code/diffusion_min.py` → Noise schedule

### Data Pipeline
1. `code/msrvtt_to_manifest.py` → Creates initial manifest
2. `code/preprocess_msrvtt.py` → Enriches with metadata
3. `code/split_manifest.py` → Creates train/val split
4. `code/data_card.py` → Generates statistics

### Evaluation Pipeline
1. `code/eval_clipscore.py` → CLIPScore on validation set
2. `code/bench_throughput.py` → Throughput measurement
3. Sample grids saved during training → Qualitative evaluation

---

## 📊 File Size & Complexity

**Core Training Files**:
- `train_t2v_diffusion.py`: ~115 lines (main training logic)
- `models_unet3d_film.py`: ~40 lines (model architecture)
- `diffusion_min.py`: ~15 lines (diffusion utilities)
- `dataset_msrvtt_diff.py`: ~70 lines (data loading)

**Total Project**: ~15 Python files, ~5 shell scripts, comprehensive documentation

---

## 🎯 Quick Reference

**To train**: `bash train_baseline.sh`  
**To evaluate**: `code/eval_clipscore.py`  
**To benchmark**: `code/bench_throughput.py`  
**To split data**: `code/split_manifest.py`  
**To generate stats**: `code/data_card.py`

---

**Last Updated**: November 2024


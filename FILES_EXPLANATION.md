# Complete File Explanation

## 📁 Local Files (Your Computer)

### Core Model Files

#### 1. **`models.py`** (5.4KB)
**Purpose**: Defines the Transformer-based video denoiser architecture

**Contents**:
- `TransformerBlock`: Self-attention blocks for spatial relationships
- `CrossAttentionBlock`: Cross-attention blocks for text conditioning
- `VideoTransformer`: Main model class with:
  - 12 layers, 768 hidden dimensions, 12 attention heads
  - Sequence length: 1024 (32×32 for single frame)
  - Positional embeddings, time embeddings, text projections
  - Zero-initialized output layer for stable training

**Key Features**:
- Uses cross-attention for proper text conditioning
- Processes single frames at 32×32 latent resolution
- Compatible with Diffusers library

---

#### 2. **`models_unet3d_film.py`** (81 lines)
**Purpose**: Original UNet3D implementation (replaced by Transformer)

**Contents**:
- `FiLM` (Feature-wise Linear Modulation): Conditions features with text/timestep
- `Block3D`: 3D convolutional blocks with FiLM conditioning
- `UNet3DCond`: 3D UNet with encoder-decoder structure

**Why it was replaced**: 
- Transformer architecture provides better text conditioning
- More scalable and easier to train
- Better cross-attention mechanism

---

### Training & Preprocessing

#### 3. **`train.py`** (7.6KB)
**Purpose**: Main training script for the diffusion model

**Contents**:
- `VideoDataset`: Loads preprocessed latents and text embeddings
- Training loop with:
  - Mixed precision (FP16) training
  - Gradient accumulation (4 steps)
  - DDPM scheduler for noise prediction
  - Checkpoint saving every 5 epochs
- Uses Accelerate library for distributed training support

**Key Configuration**:
- Batch size: 2
- Learning rate: 1e-4
- Total epochs: 100
- Saves checkpoints every 5 epochs

---

#### 4. **`preprocess.py`** (7.0KB)
**Purpose**: Converts raw videos to VAE latents for faster training

**Contents**:
- Loads MSR-VTT videos from `data/msr-vtt/videos/`
- Extracts evenly spaced frames (default: 16 frames per video)
- Encodes frames to VAE latents (32×32) using Stable Diffusion VAE
- Generates CLIP text embeddings for captions
- Saves preprocessed data to:
  - `data/msr-vtt/latents/` - VAE latents
  - `data/msr-vtt/text_embeddings/` - CLIP embeddings

**Why preprocessing**: 
- Training is 10x faster with preprocessed latents
- Reduces disk I/O during training
- Allows training on larger datasets

---

### Inference & Generation

#### 5. **`inference.py`** (9.6KB)
**Purpose**: Generates videos from text prompts

**Contents**:
- `VideoGenerator` class:
  - Loads trained checkpoint
  - Generates videos from scratch using DDIM scheduler
  - Supports Classifier-Free Guidance (CFG)
  - Decodes latents to pixel space using VAE
- `find_latest_checkpoint()`: Finds most recent checkpoint
- Can generate single frames or multi-frame videos

**Usage**: 
```python
gen = VideoGenerator(checkpoint_path="checkpoints/checkpoint_epoch_100.pt")
video = gen.generate(prompt="a cat playing with a ball")
```

---

#### 6. **`app.py`** (5.2KB)
**Purpose**: Gradio web UI for video generation

**Contents**:
- Web interface with:
  - Text prompt input
  - Inference steps slider
  - Guidance scale slider
  - Video output display
- Lazy loading of model (loads once on first use)
- Saves generated videos to `outputs/` directory

**Run**: `python app.py` → Opens web UI at `http://localhost:7860`

---

### Video Refinement Scripts

#### 7. **`refine_video_custom.py`** (8.0KB)
**Purpose**: Refines noisy videos with custom prompts (Vid2Vid)

**Contents**:
- Takes a noisy video and text prompt
- Uses epoch 100 checkpoint to denoise
- Supports custom noise strength (default: 0.3 = 30%)
- Minimal output (only progress lines)

**Usage**:
```bash
python refine_video_custom.py noisy_video.mp4 output.mp4 "dog running on the grass" 0.3
```

**Output**: Shows only:
- "Refining with Epoch 100 model (X steps)..."
- "Step X/Y" progress
- "Decoding X frames..." progress
- "✅ Output saved in GPU: output.mp4"

---

#### 8. **`refine_video_epoch100.py`** (9.1KB)
**Purpose**: Similar to `refine_video_custom.py` but with hardcoded settings

**Contents**:
- Hardcoded checkpoint path: `checkpoints/checkpoint_epoch_100.pt`
- Hardcoded prompt: "dog running on the grass"
- More verbose output (for debugging)

**Note**: `refine_video_custom.py` is preferred for flexibility

---

#### 9. **`add_noise_to_video.py`** (5.8KB)
**Purpose**: Adds noise to a clean video for Vid2Vid refinement

**Contents**:
- Loads input video
- Encodes to VAE latents
- Adds specified amount of noise (0.1 = 10%, 0.3 = 30%, etc.)
- Decodes back to video
- Saves noisy video

**Usage**:
```bash
python add_noise_to_video.py input_video.mp4 noisy_video.mp4 0.3
```

**Workflow**: 
1. Add noise: `add_noise_to_video.py` → creates noisy video
2. Refine: `refine_video_custom.py` → denoises with model + prompt

---

### Configuration & Documentation

#### 10. **`requirements.txt`**
**Purpose**: Python package dependencies

**Key Packages**:
- `torch>=2.0.0`: PyTorch for deep learning
- `diffusers>=0.21.0`: Diffusion model utilities
- `transformers>=4.30.0`: CLIP text encoder
- `accelerate>=0.20.0`: Mixed precision training
- `gradio`: Web UI framework
- `opencv-python`, `imageio`: Video processing

---

#### 11. **`README.md`** (275 lines)
**Purpose**: Project documentation

**Contents**:
- Project overview
- Dataset information (MSR-VTT)
- Architecture details
- Training process
- Evaluation results (CLIP scores)
- Setup instructions

---

#### 12. **`TERMINAL_COMMANDS.md`** (57 lines)
**Purpose**: Quick reference for terminal commands

**Contents**:
- How to connect to GPU VM
- How to generate videos with custom prompts
- How to download generated videos

---

### Data Files

#### 13. **`data/msr-vtt/`**
**Structure**:
```
data/msr-vtt/
├── videos/              # Raw video files
│   └── Video_Generation_with_Low_Quality_Model.mp4
├── annotations/         # Caption files
│   ├── train_9k.json    # Training captions
│   ├── test_1k.json     # Test captions
│   └── train.manifest.jsonl  # Preprocessed manifest
└── latents/             # Preprocessed VAE latents (created by preprocess.py)
└── text_embeddings/     # Preprocessed CLIP embeddings (created by preprocess.py)
```

---

#### 14. **`checkpoints/`** (Local)
**Contents**:
- `checkpoint_epoch_18.pt` (2.0GB)
- `checkpoint_epoch_57.pt` (2.0GB)

**Note**: Only a few checkpoints are kept locally. Full training checkpoints are on GPU VM.

---

## 🖥️ GPU VM Files (a100-golden)

### Python Files (Same as Local)
All the same Python files exist on the VM:
- `models.py`
- `train.py`
- `preprocess.py`
- `inference.py`
- `app.py`
- `refine_video_custom.py`
- `refine_video_epoch100.py`
- `add_noise_to_video.py`

### Additional Files on GPU VM

#### 1. **`checkpoints/`** Directory
**Location**: `~/text2video/checkpoints/`

**Contents**:
- **Full training checkpoints** (epochs 1-100):
  - `checkpoint_epoch_1.pt` through `checkpoint_epoch_100.pt`
  - Each checkpoint: **2.0GB** (model weights)
  - Total: ~200GB of checkpoints

- **Final model**:
  - `final_model.pt` (657MB) - Final trained model

**Why on VM**: 
- Too large to store locally
- Needed for inference on GPU
- Can download specific checkpoints if needed

---

#### 2. **`data/msr-vtt/`** Directory
**Location**: `~/text2video/data/msr-vtt/`

**Contents**:
- **Preprocessed latents**: `latents/` directory
  - Thousands of `.pt` files (one per video)
  - Each file: VAE-encoded video frames (32×32 latents)
  - Used for fast training

- **Text embeddings**: `text_embeddings/` directory
  - CLIP embeddings for each video caption
  - Precomputed to save time during training

- **Original videos**: `videos/` directory
  - Full MSR-VTT dataset (~10,000 videos)
  - Used for preprocessing and inference

**Size**: Several hundred GB total

---

#### 3. **`outputs/`** Directory
**Location**: `~/text2video/outputs/`

**Contents**:
- Generated videos from inference
- Refined videos from Vid2Vid
- Temporary video files

**Usage**: 
- Videos generated by `app.py` are saved here
- Refined videos from `refine_video_custom.py` are saved here

---

#### 4. **`venv/`** Directory
**Location**: `~/text2video/venv/`

**Contents**:
- Python virtual environment
- All installed packages (torch, diffusers, etc.)
- GPU-enabled PyTorch with CUDA support

**Activation**: `source venv/bin/activate`

---

#### 5. **`input_video.mp4`** & **`noisy_video.mp4`**
**Location**: `~/text2video/`

**Contents**:
- `input_video.mp4`: Uploaded video for Vid2Vid refinement
- `noisy_video.mp4`: Noisy version created by `add_noise_to_video.py`

**Workflow**:
1. Upload video → `input_video.mp4`
2. Add noise → `noisy_video.mp4`
3. Refine with prompt → `output.mp4`

---

## 📊 File Size Summary

### Local Files:
- Python scripts: ~50KB total
- Checkpoints (2 files): ~4GB
- Data: ~5MB (sample video only)

### GPU VM Files:
- Python scripts: ~50KB
- Checkpoints (100 files): ~200GB
- Preprocessed data: ~500GB+
- Virtual environment: ~5GB

---

## 🔄 File Synchronization

**Local → VM**: 
- Upload scripts: `gcloud compute scp`
- Upload videos: `gcloud compute scp`

**VM → Local**:
- Download generated videos: `gcloud compute scp`
- Download specific checkpoints: `gcloud compute scp`

**Note**: Full dataset and all checkpoints stay on VM due to size.

---

## 🎯 Key Workflows

### 1. Training Workflow:
```
preprocess.py → Creates latents/ and text_embeddings/
train.py → Trains model, saves checkpoints/
```

### 2. Generation Workflow:
```
inference.py → Generates from scratch
app.py → Web UI for generation
```

### 3. Vid2Vid Workflow:
```
add_noise_to_video.py → Creates noisy video
refine_video_custom.py → Refines with prompt
```

---

## 📝 Notes

- **Monkey Patches**: All scripts include `torch.load` patches for PyTorch 2.6+ compatibility
- **Mixed Precision**: Training uses FP16 for faster training and lower memory
- **Checkpoint Format**: Each checkpoint is 2.0GB (model weights + optimizer state)
- **VAE**: Uses Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`)
- **Text Encoder**: Uses CLIP (`openai/clip-vit-base-patch32`)


# Text-to-Video Diffusion Model

A transformer-based text-to-video diffusion model trained on the MSR-VTT dataset. This project implements a video generation system that can create videos from text prompts using a diffusion process with cross-attention for text conditioning.

## Project Overview

This project implements a text-to-video generation model using:
- **Architecture**: Transformer-based denoiser with cross-attention for text conditioning
- **Dataset**: MSR-VTT (Microsoft Research Video to Text)
- **Training**: Diffusion model (DDPM) with 100 epochs
- **Resolution**: 32x32 latent space (256x256 pixel output)
- **Framework**: PyTorch with Diffusers and Transformers libraries

## What Was Done

### 1. Dataset Preparation
- Downloaded MSR-VTT dataset (videos and captions)
- Preprocessed videos: converted to VAE latents (32x32) for faster training
- Generated text embeddings using CLIP text encoder
- Created manifest files for training data

### 2. Model Architecture
- **VideoTransformer**: Transformer-based denoiser with:
  - Self-attention blocks for spatial relationships
  - Cross-attention blocks for text conditioning
  - 12 layers, 768 hidden dimensions, 12 attention heads
  - Sequence length: 1024 (32×32 for single frame)
  - Zero-initialized output layer for stable training

### 3. Training
- Trained for 100 epochs on MSR-VTT dataset
- Mixed precision training (FP16) with gradient accumulation
- Batch size: 1, Gradient accumulation: 8
- Learning rate: 1e-4
- Optimizer: AdamW with weight decay 0.01
- Checkpoints saved every epoch

### 4. Evaluation
- Generated videos from scratch using epoch 100 checkpoint
- Calculated CLIP scores for text-video alignment
- Average CLIP Score: 0.1999 (range: 0.1704 - 0.2131)
- Throughput: ~0.38 frames/second

## Project Structure

```
text2video/
├── models.py          # VideoTransformer model architecture
├── train.py           # Training script
├── preprocess.py      # Data preprocessing (video → latents)
├── inference.py       # Video generation from text prompts
├── app.py             # Gradio UI for interactive generation
├── requirements.txt   # Python dependencies
├── checkpoints/       # Saved model checkpoints
└── data/              # Dataset files
    └── msr-vtt/
        ├── videos/           # Video files
        ├── latents/          # Preprocessed VAE latents
        ├── text_embeddings/  # CLIP text embeddings
        └── annotations/      # Caption files
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA A100 or similar)
- 20GB+ GPU memory for training
- 100GB+ disk space for dataset

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd text2video
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download MSR-VTT dataset:**
   - Download videos from: https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
   - Extract to `data/msr-vtt/videos/`
   - Download captions/annotations and place in `data/msr-vtt/annotations/`

## Usage

### 1. Preprocessing

Convert videos to VAE latents and generate text embeddings:

```bash
python preprocess.py
```

This will:
- Load videos from `data/msr-vtt/videos/`
- Encode frames to VAE latents (32x32)
- Generate CLIP text embeddings
- Save to `data/msr-vtt/latents/` and `data/msr-vtt/text_embeddings/`

### 2. Training

Train the model from scratch:

```bash
python train.py
```

Or resume from a checkpoint:
```python
# Edit train.py and set:
RESUME_FROM = "checkpoints/checkpoint_epoch_X.pt"
```

Training configuration (in `train.py`):
- `NUM_EPOCHS = 100`
- `BATCH_SIZE = 1`
- `GRAD_ACCUM = 8`
- `LEARNING_RATE = 1e-4`
- `SAVE_EVERY = 1` (save after each epoch)

### 3. Inference

Generate videos from text prompts:

```bash
python inference.py --prompt "A cat playing with a ball" --checkpoint checkpoints/checkpoint_epoch_100.pt --output output.mp4
```

Or use the interactive UI:

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### 4. Example Prompts

Top 3 performing prompts (by CLIP score):
- "A dog running in the grass" (CLIP: 0.2131) ⭐ Best
- "A man talking to the camera" (CLIP: 0.2123)
- "A person playing a guitar" (CLIP: 0.2039)

## Model Architecture Details

### VideoTransformer

The model uses a transformer architecture with:

1. **Input Processing:**
   - Latent input: [Batch, 4, 32, 32] (VAE latent space)
   - Flattened to sequence: [Batch, 1024, 4]
   - Projected to hidden dimension: [Batch, 1024, 768]

2. **Embeddings:**
   - Positional embeddings: Learned, size [1, 1024, 768]
   - Time embeddings: Sinusoidal, projected to 768
   - Text embeddings: CLIP embeddings [Batch, 77, 512] → projected to 768

3. **Transformer Blocks:**
   - Alternating self-attention and cross-attention blocks
   - Self-attention: Spatial relationships within frames
   - Cross-attention: Text conditioning (query from spatial, key/value from text)
   - Feed-forward networks with GELU activation

4. **Output:**
   - Projected back to latent space: [Batch, 1024, 4]
   - Reshaped to: [Batch, 4, 32, 32]

### Training Details

- **Scheduler**: DDPMScheduler (1000 timesteps)
- **Noise Schedule**: Scaled linear (beta_start=0.00085, beta_end=0.012)
- **Loss**: MSE between predicted and actual noise
- **Mixed Precision**: FP16 with gradient scaling
- **Gradient Clipping**: Max norm 1.0

## Results

### Quantitative Metrics

**Top 3 CLIP Scores (text-video alignment):**
1. "A dog running in the grass": 0.2131
2. "A man talking to the camera": 0.2123
3. "A person playing a guitar": 0.2039

- **Average CLIP Score (Top 3)**: 0.2098
- **Generation Speed**: ~0.38 frames/second
- **Video Duration**: 5-17 seconds (configurable)
- **Resolution**: 256×256 pixels (from 32×32 latents)

### Model Checkpoints

Best performing checkpoint: **Epoch 100**
- File size: ~2.0GB
- Training loss: Decreased from ~0.86 to ~0.31 over 100 epochs
- Text conditioning: Working (verified with color prompts)

## Technical Notes

### Memory Optimization

- Batch size: 1 (to fit in GPU memory)
- Gradient accumulation: 8 (effective batch size = 8)
- Gradient checkpointing: Enabled for memory efficiency
- FP16 mixed precision: Reduces memory usage

### Known Limitations

1. **Low Resolution**: 32×32 latent space limits detail
2. **Single Frame Training**: Model trained on individual frames, not temporal sequences
3. **Abstract Outputs**: Early training produces abstract/blurry results
4. **Text Conditioning**: Works for simple prompts (colors, objects), less effective for complex scenes

### Future Improvements

- Increase latent resolution to 64×64 or higher
- Add temporal consistency loss
- Train on video sequences (not just frames)
- Implement classifier-free guidance during training
- Add video-to-video refinement capability

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size or gradient accumulation
- Enable gradient checkpointing
- Use smaller model (reduce hidden_dim or num_layers)

### Green Screen Artifacts
- Decode VAE on CPU with FP32 (already implemented in inference.py)
- Ensure latents are in correct range before decoding

### ModuleNotFoundError
- Ensure virtual environment is activated
- Install all requirements: `pip install -r requirements.txt`
- Check PYTHONPATH if importing models fails

## Citation

If you use this code, please cite:
- MSR-VTT dataset
- Diffusers library (HuggingFace)
- CLIP model (OpenAI)

## License

[Specify your license here]

## Contact

[Your contact information]


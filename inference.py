"""
Inference script for text-to-video generation (pure text-to-video, no video input)
Generates videos from text prompts ONLY - builds from complete scratch using pure noise.
NO video input is used - each frame is generated independently from the text prompt.
"""
import torch
import torch.nn.functional as F
import math
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from models import VideoTransformer
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import os

# --- MONKEY PATCH FOR PYTORCH 2.6+ COMPATIBILITY ---
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' in kwargs:
        del kwargs['weights_only']
    return original_load(*args, **kwargs, weights_only=False)
torch.load = safe_load
# ---------------------------------------------------

# Configuration
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
LATENT_HEIGHT = 32
LATENT_WIDTH = 32
SEQUENCE_LENGTH = 1024  # 32 * 32 = 1024
IMAGE_SIZE = 256  # Output image size
NUM_FRAMES = 8  # Number of frames to generate for video


def get_time_embedding(timesteps, dim=320):
    """Create sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1))
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class VideoGenerator:
    """Text-to-Image/Video Generator"""
    def __init__(self, checkpoint_path=None):
        self.device = DEVICE
        print(f"Loading models on {self.device}...")
        
        # Load VAE for decoding
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        # Keep VAE in float32 for better precision (convert latents to float32 before decoding)
        self.vae.eval()
        
        # Load text encoder
        print("Loading text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.text_encoder.eval()
        
        # Load model (new architecture with sequence_length)
        print("Loading video transformer...")
        self.model = VideoTransformer(
            num_layers=12,
            hidden_dim=768,
            num_heads=12,
            sequence_length=SEQUENCE_LENGTH
        ).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Checkpoint loaded!")
        else:
            print("⚠️  No checkpoint provided, using random weights")
        
        # Convert model to float16 for inference (matches training dtype)
        self.model = self.model.half()
        self.model.eval()
        
        # Create scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        
        print("✅ All models loaded!")
    
    @torch.no_grad()
    def generate_frame(self, prompt, num_inference_steps=NUM_INFERENCE_STEPS, vae_cpu=None):
        """
        Generate a single frame from text prompt ONLY (no video input)
        Generates from complete scratch using zero initialization.
        
        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps
        
        Returns:
            frame: numpy array of shape [height, width, 3] (RGB, 0-255)
        """
        # Encode text
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]  # [1, 77, 512]
        
        # Initialize latents with ZEROS (generating from scratch, no video input)
        # This is a text-to-video generation - we start from zero initialization
        latents = torch.zeros(
            (1, 4, LATENT_HEIGHT, LATENT_WIDTH),
            device=self.device,
            dtype=torch.float16
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            timestep = t.unsqueeze(0).to(self.device)
            
            # Get time embedding
            time_emb = get_time_embedding(timestep).to(self.device)
            time_emb = time_emb.to(latents.dtype)
            text_embeddings = text_embeddings.to(latents.dtype)
            
            # Predict noise
            noise_pred = self.model(
                latents,
                time_emb,
                text_embeddings
            )
            
            # Denoise
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image on CPU in full precision (fixes green screen issue)
        # Use provided VAE CPU instance or create one
        if vae_cpu is None:
            vae_cpu = self.vae.to("cpu").float()
        
        latents_cpu = (latents / 0.18215).to("cpu").float()
        with torch.no_grad():
            frame = vae_cpu.decode(latents_cpu).sample  # [1, 3, H, W]
        frame = (frame + 1.0) / 2.0  # Normalize to [0, 1]
        frame = torch.clamp(frame, 0, 1)
        frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        frame = (frame * 255).astype(np.uint8)
        
        # Resize to target size
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        
        return frame
    
    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE):
        """
        Generate video (multiple frames) from text prompt ONLY (no video input)
        Builds video from complete scratch using zero initialization.
        Each frame is generated independently from the text prompt.
        
        Args:
            prompt: Text description of the video
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (not used in current model, kept for compatibility)
        
        Returns:
            video: numpy array of shape [num_frames, height, width, 3] (RGB, 0-255)
        """
        print(f"Generating {NUM_FRAMES} frames for: '{prompt}'")
        
        # Move VAE to CPU for stable decoding (fixes green screen issue)
        vae_original_device = next(self.vae.parameters()).device
        vae_cpu = self.vae.to("cpu").float()
        
        try:
            frames = []
            for i in range(NUM_FRAMES):
                print(f"  Generating frame {i+1}/{NUM_FRAMES}...")
                frame = self.generate_frame(prompt, num_inference_steps, vae_cpu)
                frames.append(frame)
            
            video = np.stack(frames, axis=0)  # [num_frames, H, W, 3]
            print(f"✅ Generated video: {video.shape}")
            
            return video
        finally:
            # Move VAE back to original device
            self.vae = vae_cpu.to(vae_original_device)
    
    def save_video(self, video, output_path, fps=8):
        """Save video as MP4 file"""
        height, width = video.shape[1], video.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in video:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved to: {output_path}")


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint file"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    checkpoints = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        # Try final_model.pt
        final_model = checkpoint_path / "final_model.pt"
        if final_model.exists():
            return str(final_model)
        return None
    
    # Sort by epoch number
    def get_epoch(f):
        try:
            return int(f.stem.split('_')[-1])
        except:
            return 0
    
    latest = max(checkpoints, key=get_epoch)
    return str(latest)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate video from text prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="generated_video.mp4", help="Output video path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    
    args = parser.parse_args()
    
    # Find checkpoint if not provided
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint()
        if args.checkpoint:
            print(f"Using latest checkpoint: {args.checkpoint}")
        else:
            print("⚠️  No checkpoint found! Using random weights.")
    
    # Generate
    generator = VideoGenerator(checkpoint_path=args.checkpoint)
    video = generator.generate(args.prompt, num_inference_steps=args.steps)
    generator.save_video(video, args.output)

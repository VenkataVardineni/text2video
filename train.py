"""
Training script for Video Transformer Diffusion Model
Uses preprocessed latents for fast training
"""
import torch

# --- MONKEY PATCH FOR PYTORCH 2.6+ COMPATIBILITY ---
# This bypasses the "weights_only" check that crashes older loading scripts
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' in kwargs:
        del kwargs['weights_only']
    return original_load(*args, **kwargs, weights_only=False)
torch.load = safe_load
# ---------------------------------------------------

import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from models import VideoTransformer
from pathlib import Path
import os
import json
import random
from tqdm import tqdm
import wandb

# Configuration
LATENT_DIR = "data/msr-vtt/latents"
TEXT_DIR = "data/msr-vtt/text_embeddings"
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
SAVE_EVERY = 5
LOG_EVERY = 100
USE_WANDB = False  # Disabled for non-interactive training

# Training parameters
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "fp16"


class VideoDataset(torch.utils.data.Dataset):
    """Dataset for loading preprocessed video latents and text embeddings"""
    def __init__(self, latent_dir, text_dir):
        self.latent_dir = Path(latent_dir)
        self.text_dir = Path(text_dir)
        
        # Get all available latents
        self.video_ids = [
            f.stem for f in self.latent_dir.glob("*.pt")
            if (self.text_dir / f"{f.stem}.pt").exists()
        ]
        
        print(f"Found {len(self.video_ids)} valid video-text pairs")
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load latent: [num_frames, 4, 32, 32]
        latent = torch.load(self.latent_dir / f"{video_id}.pt")
        
        # Load text embedding: [1, 77, 768]
        text_emb = torch.load(self.text_dir / f"{video_id}.pt")
        
        return {
            "latent": latent,
            "text_embedding": text_emb.squeeze(0)  # [77, 768]
        }


def train():
    """Main training function"""
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    
    if accelerator.is_main_process:
        print("🚀 Starting Video Transformer Training")
        print("=" * 50)
        if USE_WANDB:
            wandb.init(
                project="text2video-transformer",
                config={
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS,
                    "mixed_precision": MIXED_PRECISION,
                }
            )
    
    # Load Models
    if accelerator.is_main_process:
        print("Loading models...")
    
    model = VideoTransformer(
        num_layers=12,
        hidden_dim=768,
        num_heads=12,
        latent_height=32,
        latent_width=32,
        num_frames=16,
        text_embed_dim=512  # CLIP ViT-Base-Patch32 outputs 512, not 768
    )
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )
    
    # Create dataset and dataloader
    dataset = VideoDataset(LATENT_DIR, TEXT_DIR)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    if accelerator.is_main_process:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Get batch data
                latents = batch["latent"].to(accelerator.device)  # [B, F, C, H, W]
                text_embeddings = batch["text_embedding"].to(accelerator.device)  # [B, 77, 512]
                
                # Convert to format expected by model: [B, C, F, H, W]
                latents = latents.permute(0, 2, 1, 3, 4)
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                noise_pred = model(
                    noisy_latents,
                    timesteps,
                    text_embeddings
                )
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % LOG_EVERY == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                if USE_WANDB:
                    wandb.log({"loss": avg_loss, "step": global_step})
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            if USE_WANDB:
                wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            if accelerator.is_main_process:
                checkpoint_dir = Path("checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_epoch_loss,
                    },
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )
                print(f"✅ Saved checkpoint: checkpoint_epoch_{epoch+1}.pt")
    
    # Final save
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.state_dict(),
            "checkpoints/final_model.pt"
        )
        print("✅ Training complete! Final model saved.")
        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    train()


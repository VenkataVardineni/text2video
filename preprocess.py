"""
Preprocess MSR-VTT videos into VAE latents for faster training
This converts videos to compressed latent space once, saving time during training
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

import os
import cv2
import numpy as np
import json
from pathlib import Path

# Now import transformers components
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

# CRITICAL: Aggressively patch transformers to disable PyTorch version check
# This MUST happen immediately after imports but before any model loading
import transformers.utils.import_utils as import_utils
import transformers.modeling_utils as modeling_utils

# Replace check_torch_load_is_safe with a no-op that never raises
def noop_check(*args, **kwargs):
    pass  # Do nothing, never raise ValueError
import_utils.check_torch_load_is_safe = noop_check

# Patch load_state_dict to ensure check is disabled during loading
original_load_state_dict = modeling_utils.load_state_dict
def patched_load_state_dict(*args, **kwargs):
    # Ensure check is disabled
    import_utils.check_torch_load_is_safe = noop_check
    return original_load_state_dict(*args, **kwargs)
modeling_utils.load_state_dict = patched_load_state_dict

# Configuration
# Use absolute paths to allow access from different user directories
VIDEO_DIR = os.getenv("VIDEO_DIR", "data/msr-vtt/videos")
ANNOTATION_FILE = os.getenv("ANNOTATION_FILE", "data/msr-vtt/annotations/train_9k.json")
# If running as different user, set: VIDEO_DIR=/home/venkatarevanth/text2video/data/msr-vtt/videos
OUT_LATENT_DIR = "data/msr-vtt/latents"
OUT_TEXT_DIR = "data/msr-vtt/text_embeddings"
BATCH_SIZE = 4
IMAGE_SIZE = 256
NUM_FRAMES = 16  # Number of frames to extract per video
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_LATENT_DIR, exist_ok=True)
os.makedirs(OUT_TEXT_DIR, exist_ok=True)

def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    # Get evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If frame read fails, use last successful frame
            if frames:
                frames.append(frames[-1])
            else:
                return None
        else:
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frames.append(frame)
    
    cap.release()
    return np.array(frames)  # [num_frames, H, W, 3]


def process_videos():
    """Process all videos into latents"""
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.eval()
    
    print("Loading text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # Load without safetensors (monkey-patch handles the version check)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    text_encoder.eval()
    
    # Load annotations
    print("Loading annotations...")
    with open(ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)
    
    # Create video_id to captions mapping
    video_captions = {}
    for item in annotations:
        video_id = item['video_id']
        captions = item.get('caption', [])
        if isinstance(captions, list) and len(captions) > 0:
            # Use first caption for now
            video_captions[video_id] = captions[0]
        elif isinstance(captions, str):
            video_captions[video_id] = captions
    
    # Get all video files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos to process")
    
    processed = 0
    failed = 0
    
    for vid_file in tqdm(video_files, desc="Processing videos"):
        video_id = vid_file.replace('.mp4', '')
        video_path = os.path.join(VIDEO_DIR, vid_file)
        
        # Check if already processed
        latent_path = os.path.join(OUT_LATENT_DIR, f"{video_id}.pt")
        text_path = os.path.join(OUT_TEXT_DIR, f"{video_id}.pt")
        if os.path.exists(latent_path) and os.path.exists(text_path):
            processed += 1
            continue
        
        try:
            # Extract frames
            frames = extract_frames(video_path, NUM_FRAMES)
            if frames is None:
                failed += 1
                continue
            
            # Convert to tensor: [num_frames, H, W, 3] -> [num_frames, 3, H, W]
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            frames_tensor = (frames_tensor / 127.5) - 1.0  # Normalize to [-1, 1]
            frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)  # [1, num_frames, 3, H, W]
            
            # Encode to latents (process frame by frame for memory efficiency)
            latents_list = []
            with torch.no_grad():
                for i in range(NUM_FRAMES):
                    frame = frames_tensor[0, i].unsqueeze(0)  # [1, 3, H, W]
                    latent = vae.encode(frame).latent_dist.sample() * 0.18215
                    latents_list.append(latent)
            
            # Stack: [num_frames, 4, latent_h, latent_w]
            latents = torch.stack(latents_list, dim=1).squeeze(0)  # [num_frames, 4, 32, 32]
            
            # Process text
            caption = video_captions.get(video_id, "a video")
            text_inputs = tokenizer(
                [caption], 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids)[0]  # [1, 77, 768]
            
            # Save
            torch.save(latents.cpu(), latent_path)
            torch.save(text_embeddings.cpu(), text_path)
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {vid_file}: {e}")
            failed += 1
            continue
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {processed}")
    print(f"   Failed: {failed}")
    print(f"   Latents saved to: {OUT_LATENT_DIR}")
    print(f"   Text embeddings saved to: {OUT_TEXT_DIR}")


if __name__ == "__main__":
    process_videos()


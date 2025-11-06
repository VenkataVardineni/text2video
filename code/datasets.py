"""
PyTorch Dataset for MSR-VTT Video Captioning
"""
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import decord as de
from PIL import Image
import torchvision.transforms as T

de.bridge.set_bridge('native')


class MSRVTTDataset(Dataset):
    """
    MSR-VTT Video Caption Dataset
    
    Args:
        manifest_path: Path to manifest.jsonl file
        num_frames: Number of frames to sample per video
        size: Target size for frame resizing
        temporal_stride: Stride for temporal sampling (1 = consecutive frames)
        random_sample: If True, randomly sample frames; else uniformly sample
    """
    
    def __init__(
        self,
        manifest_path: str,
        num_frames: int = 16,
        size: int = 256,
        temporal_stride: int = 1,
        random_sample: bool = True
    ):
        self.items = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    self.items.append(rec)
        
        self.num_frames = num_frames
        self.size = size
        self.temporal_stride = temporal_stride
        self.random_sample = random_sample
        
        # Image transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size),
            T.CenterCrop(size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        rec = self.items[idx]
        video_path = rec["path"]
        
        # Load video
        vr = de.VideoReader(video_path)
        n = len(vr)
        
        # Sample frames
        if n >= self.num_frames * self.temporal_stride:
            if self.random_sample:
                max_start = n - (self.num_frames * self.temporal_stride)
                start_idx = random.randint(0, max_start)
                indices = [start_idx + i * self.temporal_stride for i in range(self.num_frames)]
            else:
                # Uniform sampling
                indices = np.linspace(0, n - 1, self.num_frames, dtype=int)
        else:
            # Video is shorter than needed, pad by repeating frames
            indices = list(range(n))
            while len(indices) < self.num_frames:
                indices.extend(range(n))
            indices = indices[:self.num_frames]
        
        # Get frames
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8
        
        # Convert to tensor and apply transforms
        frame_tensors = []
        for frame in frames:
            img = Image.fromarray(frame)
            frame_tensors.append(self.transform(img))
        
        frames_tensor = torch.stack(frame_tensors)  # (T, 3, H, W)
        
        # Get caption (use first caption or random if multiple available)
        if isinstance(rec.get("captions"), list) and len(rec["captions"]) > 0:
            caption = random.choice(rec["captions"])
        else:
            caption = rec.get("caption", "")
        
        return {
            "video": frames_tensor,  # (T, 3, H, W)
            "caption": caption,
            "video_id": rec.get("id", ""),
            "num_frames": n,
            "fps": rec.get("fps", 30.0)
        }


class VideoCaptionDataset(Dataset):
    """
    Generic Video Caption Dataset (backward compatibility)
    """
    def __init__(self, manifest_path, num_frames=16, size=256):
        self.items = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    self.items.append(rec)
        self.num_frames = num_frames
        self.size = size
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size),
            T.CenterCrop(size)
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        vr = de.VideoReader(rec["path"])
        n = len(vr)
        
        import numpy as np
        if n >= self.num_frames:
            idxs = np.linspace(0, n-1, self.num_frames, dtype=int)
        else:
            idxs = np.pad(np.arange(n), (0, self.num_frames-n), mode="edge")
        
        frames = vr.get_batch(idxs).asnumpy()  # (T,H,W,3) uint8
        
        frames_tensor = torch.stack([
            self.transform(Image.fromarray(f)) for f in frames
        ])  # (T,3,H,W)
        
        text = rec.get("caption", "")
        return frames_tensor, text

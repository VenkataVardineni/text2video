from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import decord as de
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

de.bridge.set_bridge('native')

class MSRVTTDiffusionDataset(Dataset):
    def __init__(self, manifest_path: str, num_frames: int = 8, size: int = 64,
                 min_duration: float = 1.0, max_duration: float = 40.0,
                 tokenizer_name: str = "openai/clip-vit-base-patch32", max_text_len: int = 32):
        self.manifest = Path(manifest_path)
        self.items: List[Dict[str,Any]] = []
        with self.manifest.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") != "ok": 
                    continue
                d = rec.get("duration", None)
                if d is not None and not (min_duration <= d <= max_duration):
                    continue
                self.items.append(rec)
        self.num_frames = num_frames
        self.size = size
        self.frame_tf = T.Compose([
            T.ToTensor(),  # [0,1]
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            lambda x: x*2.0 - 1.0,  # [-1,1]
        ])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len
        self.rng = random.Random(1234)

    def __len__(self): return len(self.items)

    def _sample_idx(self, n, k):
        if n<=k:
            idx=np.arange(n)
            return np.pad(idx,(0,k-n),'edge') if n<k else idx
        return np.linspace(0,n-1,k,dtype=int)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        path = rec["path"]
        caps = rec.get("captions") or rec.get("caption") or [""]
        text = caps[0] if isinstance(caps, list) else str(caps)

        vr = de.VideoReader(path)
        n = len(vr)
        ids = self._sample_idx(len(vr), self.num_frames)
        batch = vr.get_batch(ids).asnumpy()  # (T,H,W,3) uint8
        frames = [self.frame_tf(Image.fromarray(fr)) for fr in batch]  # list of (3,H,W)
        vid = torch.stack(frames, dim=0)  # (T,3,H,W)

        toks = self.tok(text, max_length=self.max_text_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        # flatten token dict -> tensors
        for k in list(toks.keys()):
            toks[k] = toks[k].squeeze(0)

        return {"video": vid, "text": text, "tokens": toks, "path": path}


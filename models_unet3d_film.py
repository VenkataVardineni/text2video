from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, in_ch, emb_dim):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, in_ch)
        self.to_beta  = nn.Linear(emb_dim, in_ch)
    def forward(self, x, emb):
        # x: (B,C,T,H,W), emb: (B, E)
        gamma = self.to_gamma(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = self.to_beta(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class Block3D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act   = nn.SiLU()
        self.film1 = FiLM(out_ch, emb_dim) if emb_dim else None
        self.film2 = FiLM(out_ch, emb_dim) if emb_dim else None
    def forward(self, x, emb=None):
        x = self.conv1(x); x = self.norm1(x); x = self.act(x)
        if self.film1 and emb is not None: x = self.film1(x, emb)
        x = self.conv2(x); x = self.norm2(x); x = self.act(x)
        if self.film2 and emb is not None: x = self.film2(x, emb)
        return x

class UNet3DCond(nn.Module):
    def __init__(self, in_ch=3, base=64, emb_dim=512):
        super().__init__()
        # enc
        self.b1 = Block3D(in_ch, base, emb_dim)
        self.d1 = nn.Conv3d(base, base, 4, stride=(1,2,2), padding=1)  # down H,W
        self.b2 = Block3D(base, base*2, emb_dim)
        self.d2 = nn.Conv3d(base*2, base*2, 4, stride=(2,2,2), padding=1)  # down T,H,W
        self.b3 = Block3D(base*2, base*4, emb_dim)
        # mid
        self.mid = Block3D(base*4, base*4, emb_dim)
        # dec
        self.u2 = nn.ConvTranspose3d(base*4, base*2, 4, stride=(2,2,2), padding=1)
        self.b2d= Block3D(base*4, base*2, emb_dim)
        self.u1 = nn.ConvTranspose3d(base*2, base, 4, stride=(1,2,2), padding=1)
        self.b1d= Block3D(base*2, base, emb_dim)
        self.out= nn.Conv3d(base, in_ch, 1)

        # simple timestep embedding
        self.tproj = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        # text projection -> emb_dim (we will pass CLIP pooled embedding already)
        self.txt_proj = nn.Sequential(nn.Linear(512, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x, t, txt):
        # x: (B,T,3,H,W) → (B,3,T,H,W)
        x = x.transpose(1,2).contiguous()
        temb = self.tproj(t[:,None])         # (B,emb)
        zemb = self.txt_proj(txt) + temb     # fuse

        e1 = self.b1(x, zemb)
        x  = self.d1(e1)
        e2 = self.b2(x, zemb)
        x  = self.d2(e2)
        x  = self.b3(x, zemb)

        x  = self.mid(x, zemb)

        x  = self.u2(x)
        x  = torch.cat([x, e2], dim=1)
        x  = self.b2d(x, zemb)
        x  = self.u1(x)
        x  = torch.cat([x, e1], dim=1)
        x  = self.b1d(x, zemb)
        x  = self.out(x)
        # back to (B,T,3,H,W)
        x  = x.transpose(1,2).contiguous()
        return x


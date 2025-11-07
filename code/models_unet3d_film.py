from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, c, d): super().__init__(); self.g=nn.Linear(d,c); self.b=nn.Linear(d,c)
    def forward(self,x,e): g=self.g(e)[...,None,None,None]; b=self.b(e)[...,None,None,None]; return x*(1+g)+b

class Block3D(nn.Module):
    def __init__(self,i,o,d=None):
        super().__init__()
        self.c1=nn.Conv3d(i,o,3,1,1); self.c2=nn.Conv3d(o,o,3,1,1)
        self.n1=nn.GroupNorm(8,o); self.n2=nn.GroupNorm(8,o); self.a=nn.SiLU()
        self.f1=FiLM(o,d) if d else None; self.f2=FiLM(o,d) if d else None
    def forward(self,x,e=None):
        x=self.c1(x); x=self.n1(x); x=self.a(x); x=self.f1(x,e) if self.f1 else x
        x=self.c2(x); x=self.n2(x); x=self.a(x); x=self.f2(x,e) if self.f2 else x
        return x

class UNet3DCond(nn.Module):
    def __init__(self,in_ch=3,base=64,emb=512):
        super().__init__()
        self.b1=Block3D(in_ch,base,emb); self.d1=nn.Conv3d(base,base,4,(1,2,2),1)
        self.b2=Block3D(base,base*2,emb); self.d2=nn.Conv3d(base*2,base*2,4,2,1)
        self.b3=Block3D(base*2,base*4,emb); self.mid=Block3D(base*4,base*4,emb)
        self.u2=nn.ConvTranspose3d(base*4,base*2,4,2,1); self.b2d=Block3D(base*4,base*2,emb)
        self.u1=nn.ConvTranspose3d(base*2,base,4,(1,2,2),1); self.b1d=Block3D(base*2,base,emb)
        self.out=nn.Conv3d(base,in_ch,1)
        self.tproj=nn.Sequential(nn.Linear(1,emb),nn.SiLU(),nn.Linear(emb,emb))
        self.txt_proj=nn.Sequential(nn.Linear(512,emb),nn.SiLU(),nn.Linear(emb,emb))
    def forward(self,x,t,txt):
        x=x.transpose(1,2).contiguous()
        e=self.tproj(t[:,None])+self.txt_proj(txt)
        e1=self.b1(x,e); x=self.d1(e1)
        e2=self.b2(x,e); x=self.d2(e2)
        x=self.b3(x,e); x=self.mid(x,e)
        x=self.u2(x); x=torch.cat([x,e2],1); x=self.b2d(x,e)
        x=self.u1(x); x=torch.cat([x,e1],1); x=self.b1d(x,e)
        x=self.out(x); return x.transpose(1,2).contiguous()


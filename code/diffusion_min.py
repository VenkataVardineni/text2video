from __future__ import annotations
import torch, math

def cosine_beta_schedule(T=1000,s=0.008):
    t=torch.linspace(0,T,T+1); a=torch.cos(((t/T)+s)/(1+s)*math.pi*0.5)**2; a=a/a[0]
    b=1-(a[1:]/a[:-1]); return b.clamp(1e-5,0.999)
class DDPM:
    def __init__(self,T=1000,device='cuda'):
        self.T=T; self.betas=cosine_beta_schedule(T).to(device)
        self.alphas=1-self.betas; self.ab=torch.cumprod(self.alphas,0)
        self.sqrt_ab=torch.sqrt(self.ab); self.sqrt_1m=torch.sqrt(1-self.ab)
    def q_sample(self,x0,t,eps):
        a=self.sqrt_ab[t].view(-1,1,1,1,1); b=self.sqrt_1m[t].view(-1,1,1,1,1)
        return a*x0 + b*eps


from __future__ import annotations
import torch, math

def cosine_beta_schedule(T=1000, s=0.008):
    steps = T
    t = torch.linspace(0, T, steps+1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((t/T)+s)/(1+s) * math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)

class DDPM:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        # x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * noise
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1,1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1,1)
        return a*x0 + b*noise


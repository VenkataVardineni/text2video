import os, time, torch
B=int(os.environ.get("BS","2")); T=int(os.environ.get("FRAMES","8")); H=W=int(os.environ.get("SIZE","64"))
x=torch.randn(B,T,3,H,W, device="cuda")
from models_unet3d_film import UNet3DCond
m=UNet3DCond().cuda().eval(); txt=torch.randn(B,512,device="cuda"); t=torch.rand(B,device="cuda")
for _ in range(10): m(x,t,txt); torch.cuda.synchronize()
t0=time.time()
for _ in range(50): m(x,t,txt)
torch.cuda.synchronize()
dt=time.time()-t0
print(f"samples/sec ~ {B*50/dt:.2f} | iters/sec ~ {50/dt:.2f}")


import torch, torch.nn as nn, torch.optim as optim, os, pathlib
out = pathlib.Path("/scratch/runs"); out.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

model = nn.Sequential(nn.Linear(1024,2048), nn.ReLU(), nn.Linear(2048,10)).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(3):
    x = torch.randn(512,1024, device=device)
    y = torch.randint(0,10,(512,), device=device)
    opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    torch.save({"step": step, "model": model.state_dict()}, out / f"ckpt_{step}.pt")
    print(f"step {step} | loss {loss.item():.4f}")

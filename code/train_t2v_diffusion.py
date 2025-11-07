from __future__ import annotations
import os, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from transformers import CLIPTextModel, AutoTokenizer
from dataset_msrvtt_diff import MSRVTTDiffusionDataset
from models_unet3d_film import UNet3DCond
from diffusion_min import DDPM

def to_image_grid(vid, out, step, tag, n=8):
    # vid: (B,T,3,H,W) in [-1,1] → save first sample grid of T frames
    v = vid[0].clamp(-1,1); v = (v+1)/2  # [0,1]
    save_dir = Path(out)/"samples"; save_dir.mkdir(parents=True, exist_ok=True)
    # put frames in rows (T per grid)
    # concatenate frames horizontally
    imgs = [v[t] for t in range(min(v.shape[0], n))]
    grid = torch.cat(imgs, dim=-1)  # (3,H,W*n)
    save_image(grid, save_dir/f"{tag}_step{step:06d}.png")

def main():
    manifest = os.environ.get("MANIFEST","/scratch/data/manifests/train.manifest.processed.jsonl")
    outdir   = Path(os.environ.get("OUTDIR","/scratch/runs/t2v_ddpm"))
    steps    = int(os.environ.get("STEPS","1000"))
    bsz      = int(os.environ.get("BS","2"))
    frames   = int(os.environ.get("FRAMES","8"))
    size     = int(os.environ.get("SIZE","64"))
    lr       = float(os.environ.get("LR","1e-4"))
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir.mkdir(parents=True, exist_ok=True)
    print("Out:", outdir, "| Device:", device)

    # data
    ds = MSRVTTDiffusionDataset(manifest, num_frames=frames, size=size)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # text encoder (frozen CLIP text encoder → 512d pooled embedding)
    tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    for p in txt_enc.parameters(): p.requires_grad_(False)

    # model, opt, ema
    model = UNet3DCond().to(device)
    ema   = UNet3DCond().to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ddpm  = DDPM(T=1000, device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    mse    = nn.MSELoss()

    def get_text_emb(tokens):
        out = txt_enc(**tokens.to(device), output_hidden_states=False, return_dict=True)
        # CLIPTextModel last hidden → pooled: take [EOS] (last token) representation
        last = out.last_hidden_state  # (B, L, 512)
        # gather EOS positions (clip uses padding on right)
        # simple pool: take tokens' mean over non-pad
        attn_mask = tokens["attention_mask"].to(device).unsqueeze(-1)  # (B,L,1)
        emb = (last * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)
        return emb  # (B,512)

    def update_ema(mu=0.999):
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.data.mul_(mu).add_(p.data, alpha=1-mu)

    it = iter(dl)
    for step in range(1, steps+1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        x0 = batch["video"].to(device, non_blocking=True)  # (B,T,3,H,W)
        tokens = batch["tokens"]                            # dict of ids, mask
        txt = get_text_emb(tokens)

        B = x0.size(0)
        t = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, t, noise)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            eps_pred = model(x_t, t.float()/ddpm.T, txt)   # predict noise
            loss = mse(eps_pred, noise)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        update_ema()

        if step % 20 == 0:
            print(f"step {step:06d} | loss {loss.item():.4f}")

        if step % 200 == 0:
            # save ckpt + sample (denoise single step preview: x0_hat = (x_t - b*eps)/a^0.5)
            ckpt = outdir/f"ckpt_{step:06d}.pt"
            torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict()}, ckpt)
            print("saved", ckpt)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                a = ddpm.sqrt_ab[t][:,None,None,None,None]
                b = ddpm.sqrt_1m[t][:,None,None,None,None]
                eps = ema(x_t, t.float()/ddpm.T, txt)
                x0_hat = (x_t - b*eps)/a
                to_image_grid(x0_hat.detach().cpu(), outdir, step, tag="preview")

    print("✅ done")

if __name__ == "__main__":
    main()


import os, json, torch, numpy as np
from pathlib import Path
import decord as de
try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    print("Warning: open_clip_torch not available. Install with: pip install open_clip_torch")
from PIL import Image
de.bridge.set_bridge('native')
MAN = Path(os.environ.get("VAL_MANIFEST","/scratch/data/manifests/val.jsonl"))
NFR = int(os.environ.get("FRAMES","8"))
device = "cuda" if torch.cuda.is_available() else "cpu"
if HAS_OPEN_CLIP:
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    def clipscore(path, text):
        vr = de.VideoReader(path); n=len(vr); idx = np.linspace(0, n-1, NFR, dtype=int)
        ims = torch.cat([preprocess(Image.fromarray(vr[i].asnumpy())).unsqueeze(0) for i in idx]).to(device)
        with torch.no_grad():
            im_emb = model.encode_image(ims); txt = tokenizer([text]).to(device); txt_emb = model.encode_text(txt)
            im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True); txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            return (im_emb @ txt_emb.T).squeeze(-1).mean().item()
    scores=[]; k=0
    for L in MAN.open():
        r=json.loads(L); 
        if r.get("status")!="ok": continue
        cap=(r.get("captions") or [""])[0]
        scores.append(clipscore(r["path"], cap)); k+=1
        if k%50==0: print(k, np.mean(scores))
    print("CLIPScore@{}f: {:.4f} over {} videos".format(NFR, float(np.mean(scores)), len(scores)))
else:
    print("Error: open_clip_torch not installed. Cannot compute CLIPScore.")


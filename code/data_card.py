import json, os, numpy as np
from pathlib import Path
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: seaborn/matplotlib not available, skipping plots")

MAN = Path(os.environ.get("MANIFEST","/scratch/data/manifests/train.jsonl"))
OUT = Path("/scratch/data/manifests/stats"); OUT.mkdir(parents=True, exist_ok=True)
dur,fps=[],[]
for L in MAN.open():
    r=json.loads(L); 
    if r.get("status")!="ok": continue
    if r.get("duration"): dur.append(float(r["duration"]))
    if r.get("fps"): fps.append(float(r["fps"]))
np.savetxt(OUT/"durations.csv", np.array(dur), delimiter=",")
np.savetxt(OUT/"fps.csv", np.array(fps), delimiter=",")
if HAS_PLOTTING:
    sns.histplot(dur, bins=40); plt.title("Duration (s)"); plt.savefig(OUT/"duration_hist.png", dpi=160); plt.close()
    sns.histplot(fps, bins=40); plt.title("FPS"); plt.savefig(OUT/"fps_hist.png", dpi=160); plt.close()
print("Wrote:", OUT)
print(f"Duration: mean={np.mean(dur):.2f}s, std={np.std(dur):.2f}s, min={np.min(dur):.2f}s, max={np.max(dur):.2f}s")
print(f"FPS: mean={np.mean(fps):.2f}, std={np.std(fps):.2f}, min={np.min(fps):.2f}, max={np.max(fps):.2f}")


import json, random, os
from pathlib import Path
random.seed(612)
MAN = Path(os.environ.get("MANIFEST","/scratch/data/manifests/train.manifest.processed.jsonl"))
OUT = Path("/scratch/data/manifests"); OUT.mkdir(parents=True, exist_ok=True)
train, val = [], []
bybin={}
for L in MAN.open():
    r=json.loads(L)
    if r.get("status")!="ok": continue
    d=float(r.get("duration",10.0))
    b=int(min(9, max(0, d//5)))
    bybin.setdefault(b,[]).append(r)
for b,items in bybin.items():
    random.shuffle(items)
    k=max(1,int(0.1*len(items)))
    val+=items[:k]; train+=items[k:]
for name,lst in [("train",train),("val",val)]:
    with (OUT/f"{name}.jsonl").open("w") as f:
        for r in lst: f.write(json.dumps(r)+"\n")
    print(name,len(lst))


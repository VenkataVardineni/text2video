import json, os, re
from pathlib import Path
from tqdm import tqdm

# --------- Config via env ---------
default_json = Path(__file__).parent.parent / "data/msr-vtt/annotation/MSR_VTT.json"
JSON = Path(os.environ.get("MSRVTT_JSON", str(default_json)))
# Default: try local Kaggle cache, then scratch
default_videos = Path.home() / ".cache/kagglehub/datasets/vishnutheepb/msrvtt/versions/1/TrainValVideo"
if not default_videos.exists():
    default_videos = Path("/scratch/data/raw/videos")
VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", str(default_videos)))  # set this to where your videos are
SPLIT = os.environ.get("SPLIT", "train")
# Output: use local processed dir if not on Zaratan
if Path("/scratch").exists():
    OUT = Path("/scratch/data/manifests") / f"{SPLIT}.manifest.jsonl"
else:
    OUT = Path("data/processed/manifests") / f"{SPLIT}.manifest.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

def find_video_path(vid: str):
    """
    Try common file name patterns:
      video1234.(mp4|webm|mkv)
    Also try nested subdirs.
    """
    stems = [vid, f"{vid}.mp4", f"{vid}.webm", f"{vid}.mkv"]
    # common layout: /scratch/data/raw/videos/[train|val|test]/video1234.mp4 or flat
    cand_dirs = [VIDEOS_DIR, VIDEOS_DIR / "train", VIDEOS_DIR / "val", VIDEOS_DIR / "test", VIDEOS_DIR / "all"]
    for d in cand_dirs:
        for s in stems:
            p = d / (s if s.endswith(('.mp4','.webm','.mkv')) else f"{s}.mp4")
            if p.exists(): return p
        # also search by stem to be safe
        for ext in (".mp4",".webm",".mkv"):
            p = d / f"{vid}{ext}"
            if p.exists(): return p
    return None

def load_items_from_msr_vtt(json_path: Path):
    """
    Support two popular formats:
      A) 'videos' + 'sentences' lists (official style)
      B) flat dict or list like {'video1234': [...captions...]} (CLIP4Clip-style)
    Return: list of (video_id, [captions])
    """
    data = json.loads(json_path.read_text())
    items = []

    # try A: official-like
    if isinstance(data, dict) and "videos" in data and "sentences" in data:
        # sentences: list of dicts with 'video_id' and 'caption'
        caps = {}
        for s in data["sentences"]:
            vid = s.get("video_id") or s.get("videoid") or s.get("videoID")
            if not vid: continue
            caps.setdefault(vid, []).append(s.get("caption","").strip())
        # videos list contains the ids
        for v in data["videos"]:
            vid = v.get("video_id") or v.get("id") or v.get("videoID")
            if not vid: continue
            items.append((vid, caps.get(vid, [])))
        return items

    # try B: mapping dict {video_id: [caps...]} or list of {video, caption}
    if isinstance(data, dict):
        ok = True
        tmp = []
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                tmp.append((str(k), [str(x).strip() for x in v]))
            else:
                ok = False; break
        if ok: return tmp

    if isinstance(data, list):
        # list of {video_id:..., caption:...}
        tmp = {}
        for r in data:
            if not isinstance(r, dict): continue
            vid = r.get("video_id") or r.get("video") or r.get("id")
            cap = r.get("caption") or r.get("sentence")
            if vid and cap:
                tmp.setdefault(str(vid), []).append(str(cap).strip())
        if tmp: return list(tmp.items())

    raise SystemExit(f"Unrecognized MSR-VTT json format: {json_path}")

def main():
    pairs = load_items_from_msr_vtt(JSON)
    if not pairs:
        raise SystemExit("No items parsed from MSR_VTT.json")

    n_found = 0
    with OUT.open("w") as f:
        for vid, caps in tqdm(pairs, desc="Build manifest"):
            # standardize id like 'video1234'
            m = re.search(r'(\d+)$', vid)
            vstd = f"video{m.group(1)}" if m else vid
            p = find_video_path(vstd)
            rec = {
                "id": vstd,
                "captions": caps[:20],      # keep up to 20 captions per official spec
                "split": SPLIT,
                "path": str(p) if p else None,
                "status": "ok" if p and Path(p).exists() else "missing"
            }
            if rec["status"] == "ok":
                n_found += 1
            f.write(json.dumps(rec) + "\n")
    print(f"✅ wrote {OUT}")
    print(f"   videos matched: {n_found} / {len(pairs)}")

if __name__ == "__main__":
    main()

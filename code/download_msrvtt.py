"""
MSR-VTT Dataset Downloader and Manifest Builder
Downloads MSR-VTT videos and creates training manifest.jsonl
"""
import os
import sys
import json
import subprocess
import shlex
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm

# -------- config --------
SPLIT = os.environ.get("SPLIT", "train")
MSR_VTT_ROOT = Path(os.environ.get("MSR_VTT_ROOT", str(Path.home() / "text2video/data/msr-vtt")))
OUT_ROOT = Path("/scratch/data")  # inside container: ~/scratch.msml612-fa25/data
RAW_DIR = OUT_ROOT / "raw" / "videos" / SPLIT
MANIFEST = OUT_ROOT / "manifests" / f"{SPLIT}.manifest.jsonl"
LOG_DIR = OUT_ROOT / "logs"
TIMEOUT_S = 600
MAX_TRIES = 3
TARGET_EXT = ".mp4"
# -----------------------

def run(cmd: str, timeout=TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run shell command"""
    return subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout
    )

def ensure_mp4(path: Path) -> Path:
    """Convert to H.264/AAC mp4 if needed"""
    if path.suffix.lower() == ".mp4" and path.exists():
        # Quick check if it's valid mp4
        try:
            cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{path}"'
            res = run(cmd, timeout=10)
            if res.returncode == 0 and b"h264" in res.stdout.lower():
                return path
        except:
            pass
    
    out = path.with_suffix(".mp4")
    cmd = f'ffmpeg -y -i "{path}" -c:v libx264 -preset veryfast -crf 23 -c:a aac -movflags +faststart "{out}"'
    try:
        res = run(cmd, timeout=TIMEOUT_S)
        if res.returncode == 0 and out.exists():
            if path != out:
                path.unlink(missing_ok=True)
            return out
    except Exception as e:
        print(f"Warning: conversion failed for {path}: {e}")
    
    return path  # fallback

def probe_with_decord(path: Path) -> Dict[str, Any]:
    """Probe video with decord to get metadata"""
    try:
        import decord as de
        de.bridge.set_bridge('native')
        vr = de.VideoReader(str(path))
        n = len(vr)
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
        dur = n / max(fps, 1e-6)
        return dict(num_frames=n, fps=fps, duration=dur)
    except Exception as e:
        return dict(num_frames=0, fps=None, duration=None, error=str(e))

def load_msrvtt_annotations(annotation_file: Path) -> List[Dict]:
    """Load MSR-VTT annotations from JSON file"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # MSR-VTT format: {"videos": [{"video_id": "...", "category": "...", "url": "...", "start_time": "...", "end_time": "...", "split": "...", "annotations": [{"caption": "..."}]}]}
    videos = []
    for video in data.get("videos", []):
        video_id = video.get("video_id", "")
        split = video.get("split", "")
        url = video.get("url", "")
        
        # Get all captions
        captions = []
        for ann in video.get("annotations", []):
            captions.append(ann.get("caption", ""))
        
        videos.append({
            "video_id": video_id,
            "split": split,
            "url": url,
            "captions": captions,
            "category": video.get("category", ""),
            "start_time": video.get("start_time", ""),
            "end_time": video.get("end_time", "")
        })
    
    return videos

def download_video_from_url(url: str, output_path: Path) -> bool:
    """Download video from URL using yt-dlp"""
    if not url or url == "":
        return False
    
    for attempt in range(1, MAX_TRIES + 1):
        try:
            # Try yt-dlp first (for YouTube URLs)
            if "youtube.com" in url or "youtu.be" in url:
                cmd = f'yt-dlp -f "mp4/best" -o "{output_path}" "{url}"'
            else:
                # For direct URLs, use wget/curl
                cmd = f'wget -O "{output_path}" "{url}"'
            
            res = run(cmd, timeout=TIMEOUT_S)
            if output_path.exists() and output_path.stat().st_size > 1000:
                return True
            
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)
    
    return False

def copy_local_video(source_path: Path, dest_path: Path) -> bool:
    """Copy video from local MSR-VTT dataset"""
    try:
        import shutil
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        return dest_path.exists() and dest_path.stat().st_size > 1000
    except Exception as e:
        print(f"Copy failed: {e}")
        return False

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find MSR-VTT annotation file
    annotation_file = MSR_VTT_ROOT / "annotation" / "MSR_VTT.json"
    if not annotation_file.exists():
        # Try alternative locations
        alt_locations = [
            MSR_VTT_ROOT / "MSR_VTT.json",
            MSR_VTT_ROOT / "annotations.json",
            Path("/scratch/data/msr-vtt/annotation/MSR_VTT.json"),
        ]
        for alt in alt_locations:
            if alt.exists():
                annotation_file = alt
                break
        else:
            raise FileNotFoundError(
                f"MSR-VTT annotation file not found. Expected: {annotation_file}\n"
                f"Please download MSR-VTT dataset and place annotation file at: {annotation_file}"
            )
    
    print(f"Loading annotations from: {annotation_file}")
    videos = load_msrvtt_annotations(annotation_file)
    
    # Filter by split
    split_videos = [v for v in videos if v["split"].lower() == SPLIT.lower()]
    print(f"Found {len(split_videos)} videos for split '{SPLIT}'")
    
    # Check if videos directory exists locally
    local_videos_dir = MSR_VTT_ROOT / "videos" / "all"
    has_local_videos = local_videos_dir.exists() and any(local_videos_dir.glob("*.mp4"))
    
    with open(MANIFEST, "w") as mf:
        for video in tqdm(split_videos, desc=f"Processing {SPLIT}"):
            video_id = video["video_id"]
            captions = video["captions"]
            
            # Use first caption (or you can use all captions)
            caption = captions[0] if captions else ""
            
            out_path = RAW_DIR / f"{video_id}{TARGET_EXT}"
            
            # Try to get video
            video_found = False
            
            # Option 1: Check if already downloaded
            if out_path.exists() and out_path.stat().st_size > 1000:
                video_found = True
            # Option 2: Copy from local MSR-VTT dataset
            elif has_local_videos:
                local_video = local_videos_dir / f"{video_id}.mp4"
                if local_video.exists():
                    video_found = copy_local_video(local_video, out_path)
            # Option 3: Download from URL
            elif video.get("url"):
                video_found = download_video_from_url(video["url"], out_path)
            
            if not video_found:
                rec = dict(
                    id=video_id,
                    url=video.get("url", ""),
                    caption=caption,
                    captions=captions,
                    path=None,
                    split=SPLIT,
                    status="download_failed"
                )
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                continue
            
            # Normalize container/codec
            final_path = ensure_mp4(out_path)
            
            # Probe video
            meta = probe_with_decord(final_path)
            status = "ok" if meta.get("num_frames", 0) > 0 else "corrupt"
            
            rec = dict(
                id=video_id,
                url=video.get("url", ""),
                caption=caption,
                captions=captions,  # Store all captions
                path=str(final_path),
                split=SPLIT,
                category=video.get("category", ""),
                **meta,
                status=status
            )
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
    
    print(f"✅ Wrote manifest: {MANIFEST}")
    print(f"   Videos: {RAW_DIR}")
    print(f"   Total entries: {len(split_videos)}")

if __name__ == "__main__":
    main()

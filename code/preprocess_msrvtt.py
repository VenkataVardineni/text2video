#!/usr/bin/env python3
"""
Preprocess MSR-VTT videos: verify, extract metadata, normalize formats
"""
import json
import subprocess
import shlex
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

def probe_with_ffprobe(path: Path) -> Dict[str, Any]:
    """Probe video with ffprobe to get metadata"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                # Get duration from format or stream
                duration = float(data.get('format', {}).get('duration', 0))
                fps_str = video_stream.get('r_frame_rate', '30/1')
                fps_parts = fps_str.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
                
                # Get frame count if available
                nb_frames = video_stream.get('nb_frames')
                if nb_frames:
                    num_frames = int(nb_frames)
                else:
                    # Estimate from duration and fps
                    num_frames = int(duration * fps) if duration > 0 else 0
                
                # Get resolution
                width = video_stream.get('width', 0)
                height = video_stream.get('height', 0)
                
                # Get codec info
                codec = video_stream.get('codec_name', 'unknown')
                
                return dict(
                    num_frames=num_frames,
                    fps=fps,
                    duration=duration,
                    width=width,
                    height=height,
                    codec=codec,
                    status="ok"
                )
        
        return dict(status="error", error="ffprobe failed")
    except Exception as e:
        return dict(status="error", error=str(e))

def normalize_video(input_path: Path, output_path: Path) -> bool:
    """Normalize video to H.264/AAC MP4 format"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            str(output_path)
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minute timeout
        )
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"  Warning: normalization failed: {e}")
        return False

def main():
    import os
    
    # Config
    manifest_path = Path(os.environ.get(
        "MANIFEST_PATH",
        "data/processed/manifests/train.manifest.jsonl"
    ))
    output_dir = Path(os.environ.get(
        "OUTPUT_DIR",
        "data/processed/raw/videos/train"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    normalize = os.environ.get("NORMALIZE", "false").lower() == "true"
    
    print("🚀 MSR-VTT Preprocessing")
    print("=" * 50)
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")
    print(f"Normalize videos: {normalize}")
    print()
    
    # Load manifest
    items = []
    with open(manifest_path) as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"📊 Processing {len(items)} videos...")
    print()
    
    # Process each video
    processed = []
    errors = []
    
    for item in tqdm(items, desc="Preprocessing"):
        video_id = item.get("id", "unknown")
        video_path_str = item.get("path", "")
        
        if not video_path_str:
            item["status"] = "missing"
            item["error"] = "Video path not found"
            errors.append(video_id)
            processed.append(item)
            continue
        
        video_path = Path(video_path_str)
        
        if not video_path.exists():
            item["status"] = "missing"
            item["error"] = "Video file not found"
            errors.append(video_id)
            processed.append(item)
            continue
        
        # Probe video metadata
        metadata = probe_with_ffprobe(video_path)
        
        if metadata.get("status") != "ok":
            item["status"] = "error"
            item["error"] = metadata.get("error", "Unknown error")
            errors.append(video_id)
            processed.append(item)
            continue
        
        # Update item with metadata
        item.update({
            "num_frames": metadata.get("num_frames", 0),
            "fps": metadata.get("fps", 30.0),
            "duration": metadata.get("duration", 0),
            "width": metadata.get("width", 0),
            "height": metadata.get("height", 0),
            "codec": metadata.get("codec", "unknown"),
            "status": "ok"
        })
        
        # Normalize if requested
        if normalize:
            output_path = output_dir / f"{video_id}.mp4"
            if not output_path.exists() or output_path.stat().st_size == 0:
                if normalize_video(video_path, output_path):
                    item["path"] = str(output_path)
                    item["normalized"] = True
                else:
                    item["normalized"] = False
                    item["warning"] = "Normalization failed, using original"
            else:
                item["path"] = str(output_path)
                item["normalized"] = True
        
        processed.append(item)
    
    # Write updated manifest
    output_manifest = manifest_path.parent / f"{manifest_path.stem}.processed.jsonl"
    with open(output_manifest, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")
    
    # Statistics
    ok_count = sum(1 for item in processed if item.get("status") == "ok")
    error_count = len(errors)
    
    print()
    print("✅ Preprocessing complete!")
    print(f"   Processed: {len(processed)}")
    print(f"   Success: {ok_count}")
    print(f"   Errors: {error_count}")
    print(f"   Output manifest: {output_manifest}")
    
    if ok_count > 0:
        # Show sample stats
        ok_items = [item for item in processed if item.get("status") == "ok"]
        avg_duration = sum(item.get("duration", 0) for item in ok_items) / len(ok_items)
        avg_fps = sum(item.get("fps", 0) for item in ok_items) / len(ok_items)
        print()
        print("📊 Statistics:")
        print(f"   Average duration: {avg_duration:.2f}s")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Total frames: {sum(item.get('num_frames', 0) for item in ok_items):,}")

if __name__ == "__main__":
    main()

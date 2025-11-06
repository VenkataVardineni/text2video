#!/bin/bash
# Quick fix setup script - uses /tmp for building
# Run this directly on Zaratan compute node

set -euo pipefail

echo "🚀 Quick Setup Fix - Using /tmp for container building"
echo "======================================================"

SCR=~/scratch.text2video
mkdir -p "$SCR"/{appt_cache,runs}

# Force use of /tmp for building
BUILD_TMP="/tmp/apptainer_build_$$"
mkdir -p "$BUILD_TMP"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
export APPTAINER_MKSQUASHFS_PROCS=1
export SINGULARITY_MKSQUASHFS_PROCS=1

echo "📁 Using /tmp for building: $BUILD_TMP"
echo "📁 Cache directory: $SCR/appt_cache"

# Check disk space
echo ""
echo "💾 Disk space:"
df -h /tmp | tail -1
df -h ~ | tail -1

# Clean up old files
echo ""
echo "🧹 Cleaning up..."
rm -rf ~/scratch.text2video/tmp/build-temp-* 2>/dev/null || true
rm -rf ~/scratch.text2video/tmp/bundle-temp-* 2>/dev/null || true
rm -rf /tmp/apptainer_build_* 2>/dev/null || true
echo "✅ Cleanup complete"

# Load Apptainer/Singularity
echo ""
echo "📦 Loading container runtime..."
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)
echo "✅ Using: $ap"

# Test container (this will build in /tmp)
echo ""
echo "🧪 Testing container (building in /tmp)..."
if ! "$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  python - <<'PY'
import torch
print("✅ Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
PY
then
    echo ""
    echo "❌ Container test failed. Trying alternative approach..."
    echo "   Attempting to use a smaller container..."
    exit 1
fi

# Create venv
echo ""
echo "🐍 Creating virtual environment..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc 'python -m venv /scratch/venv && source /scratch/venv/bin/activate && pip install -U pip wheel setuptools'

# Install deps
echo ""
echo "📥 Installing dependencies..."
if [ -f ~/text2video/requirements.txt ]; then
    "$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
      docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
      bash -lc 'source /scratch/venv/bin/activate && pip install -r ~/text2video/requirements.txt'
else
    "$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
      docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
      bash -lc 'source /scratch/venv/bin/activate && pip install "sympy==1.13.1" "typing-extensions>=4.8.0"'
fi

# Create helper script
echo ""
echo "📝 Creating helper script..."
cat > ~/text2video/run_in_container.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)
SCR=~/scratch.text2video
mkdir -p "$SCR"/{appt_cache}
export APPTAINER_CACHEDIR="$SCR/appt_cache"
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime "$@"
SH
chmod +x ~/text2video/run_in_container.sh

# Cleanup
rm -rf "$BUILD_TMP" 2>/dev/null || true

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Quick reference:"
echo "  Project: ~/text2video"
echo "  Storage: $SCR (mounted as /scratch)"
echo "  Python:  /scratch/venv"
echo ""
echo "🚀 Run your code:"
echo "  ~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python ~/text2video/your_script.py'"

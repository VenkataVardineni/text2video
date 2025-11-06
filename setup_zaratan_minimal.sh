#!/bin/bash
# Minimal setup - installs only essential packages to avoid quota issues
# Run this on Zaratan compute node

set -euo pipefail

echo "🚀 Minimal Setup - Essential packages only"
echo "=========================================="

SCR=~/scratch.text2video
mkdir -p "$SCR"/{appt_cache,runs}

# Use /tmp for building
BUILD_TMP="/tmp/apptainer_build_$$"
mkdir -p "$BUILD_TMP"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
export APPTAINER_MKSQUASHFS_PROCS=1

echo "📁 Using /tmp for building: $BUILD_TMP"
echo "💾 Checking disk space:"
df -h /tmp | tail -1
df -h ~ | tail -1
du -sh "$SCR" 2>/dev/null || echo "Scratch directory empty"

# Clean up
echo ""
echo "🧹 Cleaning up..."
rm -rf ~/scratch.text2video/tmp/* 2>/dev/null || true
rm -rf /tmp/apptainer_build_* 2>/dev/null || true

# Load container runtime
echo ""
echo "📦 Loading container runtime..."
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)
echo "✅ Using: $ap"

# Test container
echo ""
echo "🧪 Testing container..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  python -c "import torch; print('✅ Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Create venv with minimal space
echo ""
echo "🐍 Creating minimal virtual environment..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc '
python -m venv /scratch/venv --without-pip
source /scratch/venv/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip wheel setuptools --no-cache-dir
'

# Install ONLY essential packages (minimal set)
echo ""
echo "📥 Installing ESSENTIAL packages only (minimal set)..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc '
source /scratch/venv/bin/activate
# Install only core packages needed for PyTorch
pip install --no-cache-dir "sympy==1.13.1" "typing-extensions>=4.8.0"
# Install transformers and diffusers (core for text2video)
pip install --no-cache-dir transformers diffusers accelerate
# Install only essential video/image processing
pip install --no-cache-dir pillow imageio
# Skip heavy packages like opencv-python, jupyter, matplotlib for now
echo "✅ Essential packages installed"
python -c "import torch, transformers, diffusers; print(\"✅ Core packages working\")"
'

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
echo "🎉 Minimal setup complete!"
echo ""
echo "📋 Installed packages:"
echo "  - PyTorch (from container)"
echo "  - transformers, diffusers, accelerate"
echo "  - pillow, imageio"
echo "  - sympy, typing-extensions"
echo ""
echo "💡 To install additional packages later:"
echo "  ~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && pip install --no-cache-dir package_name'"
echo ""
echo "🚀 Run your code:"
echo "  ~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python ~/text2video/your_script.py'"

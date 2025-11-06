#!/bin/bash
# Setup using container's built-in packages only (no pip install)
# This avoids all quota issues

set -euo pipefail

echo "🚀 Setup using container packages only (no pip install)"
echo "======================================================="

SCR=~/scratch.text2video
mkdir -p "$SCR"/{appt_cache,runs}

BUILD_TMP="/tmp/apptainer_build_$$"
mkdir -p "$BUILD_TMP"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
export APPTAINER_MKSQUASHFS_PROCS=1

echo "📁 Using /tmp for building"
df -h /tmp | tail -1

rm -rf ~/scratch.text2video/tmp/* 2>/dev/null || true
rm -rf /tmp/apptainer_build_* 2>/dev/null || true

module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)

echo ""
echo "🧪 Testing container with built-in packages..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  python - <<'PY'
import torch
print("✅ Torch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))

# Check what's available
try:
    import torchvision
    print("✅ torchvision available")
except:
    print("⚠️  torchvision not available")

try:
    import numpy
    print("✅ numpy available")
except:
    print("⚠️  numpy not available")
PY

# Create helper script
echo ""
echo "📝 Creating helper script..."
cat > ~/text2video/run_in_container.sh <<'SH'
#!/usr/bin/env bash
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)
SCR=~/scratch.text2video
export APPTAINER_CACHEDIR="$SCR/appt_cache"
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime "$@"
SH
chmod +x ~/text2video/run_in_container.sh

rm -rf "$BUILD_TMP" 2>/dev/null || true

echo ""
echo "🎉 Setup complete (using container's built-in packages)!"
echo ""
echo "📋 Available packages:"
echo "  - PyTorch (built-in)"
echo "  - torchvision (likely built-in)"
echo "  - numpy, scipy (likely built-in)"
echo ""
echo "⚠️  Note: transformers, diffusers need to be installed"
echo "   But you can install them on-demand when needed:"
echo "   ~/text2video/run_in_container.sh pip install --user --no-cache-dir transformers"
echo ""
echo "🚀 Run your code:"
echo "  ~/text2video/run_in_container.sh python ~/text2video/your_script.py"

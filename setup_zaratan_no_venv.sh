#!/bin/bash
# Setup without venv - uses container's Python directly
# This avoids quota issues by not creating a separate venv

set -euo pipefail

echo "🚀 Setup without venv - Using container Python directly"
echo "======================================================="

SCR=~/scratch.text2video
mkdir -p "$SCR"/{appt_cache,runs}

# Use /tmp for building
BUILD_TMP="/tmp/apptainer_build_$$"
mkdir -p "$BUILD_TMP"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
export APPTAINER_MKSQUASHFS_PROCS=1

echo "📁 Using /tmp for building"
echo "💾 Disk space:"
df -h /tmp | tail -1
df -h ~ | tail -1

# Clean up
rm -rf ~/scratch.text2video/tmp/* 2>/dev/null || true
rm -rf /tmp/apptainer_build_* 2>/dev/null || true

# Load container runtime
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)

echo ""
echo "🧪 Testing container..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  python -c "import torch; print('✅ Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Install packages to user site-packages (no venv needed)
echo ""
echo "📥 Installing packages to container Python (user site)..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc '
# Install to user site-packages (avoids quota issues)
pip install --user --no-cache-dir "sympy==1.13.1" "typing-extensions>=4.8.0"
pip install --user --no-cache-dir transformers diffusers accelerate
pip install --user --no-cache-dir pillow imageio
python -c "import torch, transformers, diffusers; print(\"✅ Packages installed and working\")"
'

# Create helper script
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

rm -rf "$BUILD_TMP" 2>/dev/null || true

echo ""
echo "🎉 Setup complete (no venv - using container Python directly)!"
echo ""
echo "🚀 Run your code:"
echo "  ~/text2video/run_in_container.sh python ~/text2video/your_script.py"
echo ""
echo "💡 To install more packages:"
echo "  ~/text2video/run_in_container.sh pip install --user --no-cache-dir package_name"

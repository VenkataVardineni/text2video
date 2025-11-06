#!/bin/bash
# Zaratan GPU Setup Script using Apptainer/Singularity
# Run this on Zaratan after syncing your project
# Usage: bash setup_zaratan.sh

set -euo pipefail

echo "🚀 Setting up text2video on Zaratan GPU Cluster"
echo "================================================"

# Check if we're on Zaratan
if [[ ! "$(hostname)" =~ zaratan ]]; then
    echo "⚠️  Warning: This script should be run on Zaratan"
    echo "   Current hostname: $(hostname)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set up scratch directory
SCR=~/scratch.text2video
mkdir -p "$SCR"/{tmp,appt_cache,runs}
export APPTAINER_TMPDIR="$SCR/tmp"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
export APPTAINER_MKSQUASHFS_PROCS=1
ulimit -n 4096 2>/dev/null || true

echo "📁 Scratch directory: $SCR"

# Load Apptainer/Singularity
echo "📦 Loading container runtime..."
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)
echo "✅ Using container runner: $ap"

# Test GPU and container
echo ""
echo "🧪 Testing GPU and PyTorch container..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  python - <<'PY'
import torch
print("✅ Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
PY

# Create virtual environment inside container
echo ""
echo "🐍 Creating virtual environment..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc '
python -m venv /scratch/venv
source /scratch/venv/bin/activate
pip install -U pip wheel setuptools
echo "✅ Virtual environment created"
'

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
if [ -f ~/text2video/requirements.txt ]; then
    echo "   Found requirements.txt, installing..."
    "$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
      docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
      bash -lc '
source /scratch/venv/bin/activate
pip install -r ~/text2video/requirements.txt
python - << "PY"
import sys, torch
print("✅ Python:", sys.version.split()[0])
print("✅ Torch:", torch.__version__, "| CUDA avail:", torch.cuda.is_available())
PY
'
else
    echo "   No requirements.txt found, installing minimal deps..."
    "$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
      docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
      bash -lc '
source /scratch/venv/bin/activate
pip install "sympy==1.13.1" "typing-extensions>=4.8.0"
'
fi

# Create run_in_container.sh helper
echo ""
echo "📝 Creating helper script..."
cat > ~/text2video/run_in_container.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)
SCR=~/scratch.text2video
mkdir -p "$SCR"/{tmp,appt_cache}
export APPTAINER_TMPDIR="$SCR/tmp"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime "$@"
SH
chmod +x ~/text2video/run_in_container.sh
echo "✅ Helper script created: ~/text2video/run_in_container.sh"

# Test training step
echo ""
echo "🧪 Testing training step..."
"$ap" exec --nv -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc '
source /scratch/venv/bin/activate
python - << "PY"
import torch, torch.nn as nn, torch.optim as optim
d="cuda" if torch.cuda.is_available() else "cpu"
net=nn.Sequential(nn.Linear(1024,2048), nn.ReLU(), nn.Linear(2048,10)).to(d)
opt=optim.AdamW(net.parameters(), 1e-3)
x,y=torch.randn(256,1024,device=d), torch.randint(0,10,(256,),device=d)
loss=nn.CrossEntropyLoss()(net(x), y); loss.backward(); opt.step()
print("✅ Training step OK on", d)
PY
'

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Quick reference:"
echo "  Project location: ~/text2video"
echo "  Fast storage:     $SCR (mounted as /scratch in container)"
echo "  Python env:       /scratch/venv (inside container)"
echo ""
echo "🚀 Daily workflow:"
echo "  1. Local: ./sync_to_zaratan.sh"
echo "  2. Remote: ssh vvr2211@zaratan.umd.edu"
echo "  3. Get GPU: salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=02:00:00 --mem=40G"
echo "  4. Run: srun --pty bash"
echo "  5. Execute: ~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python ~/text2video/your_script.py'"

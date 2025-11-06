#!/bin/bash
# Script to update run.sh on Zaratan
# Copy and paste this entire script on Zaratan

cat > ~/text2video/run.sh << 'RUNSH_END'
#!/usr/bin/env bash
set -euo pipefail

# 1) load container runtime
module load apptainer 2>/dev/null || module load singularity
ap=$(command -v apptainer || command -v singularity)

# 2) host paths + caches (scratch)
SCR="$HOME/scratch.msml612-fa25"
mkdir -p "$SCR"/{tmp,appt_cache,hf_cache,torch_cache}
export APPTAINER_TMPDIR="$SCR/tmp"
export APPTAINER_CACHEDIR="$SCR/appt_cache"
export HF_HOME=/scratch/hf_cache
export TORCH_HOME=/scratch/torch_cache
export APPTAINER_MKSQUASHFS_PROCS=1

# 3) exec inside container (docker:// for now, can use SIF later)
exec "$ap" exec --nv \
  -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
  docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime bash -lc '
    # ensure venv exists in /scratch and activate
    if [ ! -d /scratch/venv ]; then
      python -m venv /scratch/venv
      . /scratch/venv/bin/activate
      pip install -U pip wheel setuptools
      # base deps that play nice with torch 2.5.1
      pip install "sympy==1.13.1" "typing-extensions>=4.8.0"
    else
      . /scratch/venv/bin/activate
    fi
    # pass through env caches inside container
    export HF_HOME=/scratch/hf_cache
    export TORCH_HOME=/scratch/torch_cache
    # run the user command in the venv
    exec "$@"
  ' "$@"
RUNSH_END

chmod +x ~/text2video/run.sh
echo "✅ run.sh updated!"

# Create directories
mkdir -p ~/text2video/code ~/scratch.msml612-fa25/{data,runs,weights}
echo "✅ Directories created"

# Test
echo ""
echo "🧪 Testing GPU access..."
~/text2video/run.sh python -c "import torch; print('✅ Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  Install packages: ~/text2video/run.sh pip install --no-cache-dir transformers diffusers accelerate"
echo "  Run scripts: ~/text2video/run.sh python ~/text2video/code/your_script.py"

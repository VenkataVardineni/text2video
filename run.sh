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

# 3) exec inside your SIF (or swap to docker://... if you prefer no SIF)
SIF="$HOME/text2video/pytorch2.5.1-cu121.sif"
if [ ! -f "$SIF" ]; then
  # Fallback to docker:// if SIF doesn't exist
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
      '"$*"'
    '
else
  exec "$ap" exec --nv \
    -B "$HOME":"$HOME" -B "$SCR":/scratch -W /scratch \
    "$SIF" bash -lc '
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
      '"$*"'
    '
fi

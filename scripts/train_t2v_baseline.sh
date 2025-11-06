#!/bin/bash
# Launch text-to-video diffusion training baseline
# Usage: ./scripts/train_t2v_baseline.sh [STEPS] [BATCH_SIZE] [FRAMES] [SIZE]
#
# Note: Make sure you're on a GPU node (gpu-*). If on login node, run:
#   salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=24:00:00 --mem=40G
#   srun --pty bash

set -euo pipefail

# Check if we're on a GPU node
if [[ ! "$(hostname)" =~ gpu- ]]; then
    echo "⚠️  Warning: You don't appear to be on a GPU node (hostname: $(hostname))"
    echo "   If you're on a login node, get GPU allocation first:"
    echo "   salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=24:00:00 --mem=40G"
    echo "   srun --pty bash"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default parameters (can be overridden via environment)
STEPS=${STEPS:-2000}
BS=${BS:-2}
FRAMES=${FRAMES:-8}
SIZE=${SIZE:-64}
LR=${LR:-1e-4}

# Paths
MANIFEST=${MANIFEST:-/scratch/data/manifests/train.manifest.processed.jsonl}
OUTDIR=${OUTDIR:-/scratch/runs/t2v_ddpm_baseline}

echo "🚀 Starting Text-to-Video Diffusion Training"
echo "=============================================="
echo "Manifest: $MANIFEST"
echo "Output:   $OUTDIR"
echo "Steps:    $STEPS"
echo "Batch:    $BS"
echo "Frames:   $FRAMES"
echo "Size:     ${SIZE}x${SIZE}"
echo "LR:       $LR"
echo ""

# Verify manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo "❌ Error: Manifest not found at $MANIFEST"
    echo "   Update MANIFEST environment variable or check path"
    exit 1
fi

# Run training
MANIFEST="$MANIFEST" \
OUTDIR="$OUTDIR" \
STEPS="$STEPS" \
BS="$BS" \
FRAMES="$FRAMES" \
SIZE="$SIZE" \
LR="$LR" \
~/text2video/run.sh python ~/text2video/code/train_t2v_diffusion.py

echo ""
echo "✅ Training complete!"
echo "   Checkpoints: $OUTDIR/ckpt_*.pt"
echo "   Samples:     $OUTDIR/samples/preview_*.png"


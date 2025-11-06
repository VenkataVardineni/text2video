#!/bin/bash
# Script to get an interactive GPU session on Zaratan

echo "🚀 Connecting to GPU on Zaratan..."
echo ""
echo "Option 1: Interactive GPU Session (for testing/development)"
echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash"
echo ""
echo "Option 2: Submit GPU Job (for batch processing)"
echo "  sbatch zaratan_gpu_job.sh"
echo ""
echo "Option 3: Check GPU availability"
echo "  sinfo -p gpu"
echo ""
echo "After connecting, run these commands to verify GPU:"
echo "  module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')\""
echo ""
echo "📚 To check job status: squeue -u \$USER"


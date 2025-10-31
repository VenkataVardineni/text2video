#!/bin/bash
#SBATCH --job-name=text2video
#SBATCH --output=text2video_%j.out
#SBATCH --error=text2video_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=your_account_name

# Load required modules
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2

# Activate virtual environment (if using one)
# source ~/venv/text2video/bin/activate

# Set working directory
cd ~/text2video

# Test GPU availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run your text2video application
# python main.py --input "Your text prompt here" --output "output_video.mp4"

# Example for different frameworks:
# python runway_gen2.py --prompt "Your text prompt here"
# python stable_video_diffusion.py --prompt "Your text prompt here"

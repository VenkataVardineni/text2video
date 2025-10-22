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
module load python/3.9
module load cuda/11.8
module load gcc/9.2.0

# Activate virtual environment (if using one)
# source ~/venv/text2video/bin/activate

# Set working directory
cd ~/text2video

# Install dependencies (run once)
# pip install -r requirements.txt

# Run your text2video application
python main.py --input "Your text prompt here" --output "output_video.mp4"

# Example for different frameworks:
# python runway_gen2.py --prompt "Your text prompt here"
# python stable_video_diffusion.py --prompt "Your text prompt here"

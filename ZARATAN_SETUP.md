# Zaratan GPU Cluster Setup Guide

This guide helps you set up and run your text2video project on UMD's Zaratan GPU cluster using Apptainer/Singularity containers.

## 🚀 Quick Start

### Step 1: Sync Project to Zaratan (Local Machine)

```bash
# From your local project directory
./sync_to_zaratan.sh
```

Or manually:
```bash
rsync -av --delete . vvr2211@login.zaratan.umd.edu:~/text2video/
```

### Step 2: Setup on Zaratan

**⚠️ Important:** Container building may fail on login nodes. Run setup on a compute node:

```bash
# SSH to Zaratan
ssh zaratan
# (or: ssh vvr2211@login.zaratan.umd.edu)

# Get a compute node allocation first (recommended)
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=01:00:00 --mem=20G
srun --pty bash

# Now run setup script
bash ~/text2video/setup_zaratan.sh
```

**Alternative:** If you want to try on login node (may fail):
```bash
bash ~/text2video/setup_zaratan.sh
# Script will warn you and ask for confirmation
```

### Step 3: Get GPU Allocation and Run

```bash
# Request GPU allocation
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=02:00:00 --mem=40G

# Get interactive shell on GPU node
srun --pty bash

# Run your code in container
~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python ~/text2video/your_script.py'
```

## 📁 Directory Structure

- **Project location**: `~/text2video` (mounted inside container at same path)
- **Fast storage**: `~/scratch.text2video` (mounted as `/scratch` in container)
- **Python environment**: `/scratch/venv` (inside container)

## 🔧 Daily Workflow

### 1. Sync Changes (Local)
```bash
./sync_to_zaratan.sh
```

### 2. Connect and Run (Remote)
```bash
ssh zaratan
# (or: ssh vvr2211@login.zaratan.umd.edu)
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=02:00:00 --mem=40G
srun --pty bash
```

### 3. Execute Your Code
```bash
# Using the helper script
~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python ~/text2video/train.py'

# Or run interactively
~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python'
```

## 📊 Useful Commands

### Check GPU Status
```bash
sinfo -p gpu              # Check GPU partition availability
squeue -u $USER           # Check your job status
nvidia-smi                # Check GPU usage (on compute node)
```

### Container Commands
```bash
# Run Python script
~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && python script.py'

# Interactive Python shell
~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && ipython'

# Install new package
~/text2video/run_in_container.sh bash -lc 'source /scratch/venv/bin/activate && pip install package_name'
```

## 🐛 Troubleshooting

### Container Issues
- If Apptainer fails, try: `module load singularity`
- Check scratch directory: `ls -la ~/scratch.text2video`

### GPU Not Available
- Verify allocation: `squeue -u $USER`
- Check GPU: `nvidia-smi -L` (on compute node)
- Test in container: `~/text2video/run_in_container.sh python -c "import torch; print(torch.cuda.is_available())"`

### Module Errors
- List available modules: `module avail python`
- Load container runtime: `module load apptainer` or `module load singularity`

## 📚 Resources

- **Zaratan Documentation**: https://hpcc.umd.edu/hpcc/help/basics.html
- **HPC Help Desk**: 301-405-1500
- **Slurm Documentation**: https://slurm.schedmd.com/

## 🔐 Security Note

Remember to:
- Never commit passwords or API keys
- Use SSH keys for authentication
- Keep your scratch directory clean

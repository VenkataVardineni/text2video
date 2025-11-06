# Zaratan GPU Workflow Guide

## 🚀 Rock-Solid Workflow

This guide provides a complete workflow for running text2video projects on UMD Zaratan GPU cluster.

## 0️⃣ One-Time Setup

### Update Helper Script

The `run.sh` script automatically:
- Creates/uses venv in `/scratch/venv` (not `~/.local`)
- Sets up cache directories for HuggingFace and PyTorch
- Keeps all data in scratch (avoids quota issues)

**Already set up!** The script is ready to use.

## 1️⃣ Quick Tests

### Test GPU Access

```bash
~/text2video/run.sh python - <<'PY'
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

### Install Packages

```bash
# Install core ML libraries (into /scratch/venv)
~/text2video/run.sh pip install --no-cache-dir transformers diffusers accelerate safetensors

# Install video processing
~/text2video/run.sh pip install --no-cache-dir opencv-python imageio imageio-ffmpeg pillow

# Install utilities
~/text2video/run.sh pip install --no-cache-dir numpy scipy matplotlib tqdm
```

**Note:** If you hit quota issues, install packages one-by-one:
```bash
~/text2video/run.sh pip install --no-cache-dir transformers
~/text2video/run.sh pip install --no-cache-dir diffusers
```

### Clean Cache (if needed)

```bash
# Free up space by clearing HuggingFace cache
rm -rf ~/scratch.msml612-fa25/hf_cache/*
```

## 2️⃣ Project Structure

```
~/text2video/
├── run.sh              # Main helper script
├── code/               # Your Python scripts
│   └── smoke_train.py  # Example training script
└── ...

~/scratch.msml612-fa25/
├── venv/              # Python virtual environment
├── data/              # Datasets
├── runs/              # Training outputs, checkpoints
├── weights/           # Model weights
├── hf_cache/         # HuggingFace cache
└── torch_cache/       # PyTorch cache
```

### Create Directories

```bash
mkdir -p ~/text2video/code ~/scratch.msml612-fa25/{data,runs,weights}
```

## 3️⃣ Run Training Scripts

### Example: Smoke Test

```bash
# Run the example training script
~/text2video/run.sh python ~/text2video/code/smoke_train.py
```

**Expected output:**
- `Device: cuda | GPU: NVIDIA A100...`
- Training steps with loss values
- Checkpoints saved to `~/scratch.msml612-fa25/runs/ckpt_*.pt`

### Run Your Own Scripts

```bash
# Run any Python script
~/text2video/run.sh python ~/text2video/code/your_script.py

# With command-line arguments
~/text2video/run.sh python ~/text2video/code/train.py --epochs 10 --batch-size 32

# Interactive Python
~/text2video/run.sh python
```

## 4️⃣ Jupyter Notebook Setup (Optional)

### Terminal 1: Start Jupyter on Zaratan

```bash
# SSH and get GPU
ssh zaratan
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=02:00:00 --mem=40G
srun --pty bash

# Start Jupyter inside container
~/text2video/run.sh bash -lc '
  pip install -q jupyterlab ipykernel
  jupyter lab --no-browser --ip=127.0.0.1 --port=8899
'
```

### Terminal 2: Tunnel from Local Machine

```bash
# Replace <node> with actual compute node hostname (from hostname command)
ssh -N -L 8899:localhost:8899 -J vvr2211@zaratan.umd.edu vvr2211@<node>
```

### Access Jupyter

1. Open `http://localhost:8899` in your browser
2. Paste the token from Terminal 1 output

## 5️⃣ Daily Workflow

### Step 1: Sync Code (Local Machine)

```bash
# From your local project directory
rsync -av --delete . vvr2211@login.zaratan.umd.edu:~/text2video/
```

Or use the sync script:
```bash
./sync_to_zaratan.sh
```

### Step 2: Connect and Get GPU (Zaratan)

```bash
# SSH to Zaratan
ssh zaratan

# Request GPU allocation
salloc -p gpu --gres=gpu:a100_1g.5gb:1 --time=02:00:00 --mem=40G

# Get interactive shell on compute node
srun --pty bash
```

### Step 3: Run Your Code

```bash
# Run training script
~/text2video/run.sh python ~/text2video/code/smoke_train.py

# Or your own script
~/text2video/run.sh python ~/text2video/code/your_script.py
```

## 6️⃣ Data Importing

When ready to import datasets:

1. **Place data in scratch:**
   ```bash
   # On Zaratan
   mkdir -p ~/scratch.msml612-fa25/data
   # Copy your dataset here
   ```

2. **Access from container:**
   ```python
   # Inside your Python script
   data_path = "/scratch/data/your_dataset"
   # The container mounts ~/scratch.msml612-fa25 as /scratch
   ```

3. **Run with data:**
   ```bash
   ~/text2video/run.sh python ~/text2video/code/train.py --data-dir /scratch/data
   ```

## 📋 Useful Commands

### Check GPU Status
```bash
nvidia-smi                    # On compute node
squeue -u $USER               # Check job status
sinfo -p gpu                  # Check GPU partition
```

### Manage Cache
```bash
# Check cache sizes
du -sh ~/scratch.msml612-fa25/*

# Clear HuggingFace cache
rm -rf ~/scratch.msml612-fa25/hf_cache/*

# Clear PyTorch cache
rm -rf ~/scratch.msml612-fa25/torch_cache/*
```

### Install Packages On-Demand
```bash
# Install single package
~/text2video/run.sh pip install --no-cache-dir package_name

# Install from requirements.txt
~/text2video/run.sh pip install --no-cache-dir -r ~/text2video/requirements.txt
```

## 🐛 Troubleshooting

### Quota Issues
- Install packages one at a time
- Clear caches: `rm -rf ~/scratch.msml612-fa25/hf_cache/*`
- Contact HPC support: hpcc-help@umd.edu

### Container Issues
- Check container: `~/text2video/run.sh python -c "import torch; print(torch.__version__)"`
- Rebuild if needed: The script auto-creates venv on first run

### GPU Not Available
- Verify allocation: `squeue -u $USER`
- Check GPU: `nvidia-smi` (on compute node)
- Test: `~/text2video/run.sh python -c "import torch; print(torch.cuda.is_available())"`

## 📚 Resources

- **Zaratan Documentation**: https://hpcc.umd.edu/hpcc/help/basics.html
- **HPC Help Desk**: 301-405-1500 or hpcc-help@umd.edu
- **Slurm Documentation**: https://slurm.schedmd.com/

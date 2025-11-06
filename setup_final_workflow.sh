#!/bin/bash
# Final workflow setup - run this on Zaratan after syncing
# This sets up the directory structure and updates run.sh

set -euo pipefail

echo "🚀 Setting up final workflow structure"
echo "======================================"

# Create project directories
echo "📁 Creating directories..."
mkdir -p ~/text2video/code
mkdir -p ~/scratch.msml612-fa25/{data,runs,weights,tmp,appt_cache,hf_cache,torch_cache}
echo "✅ Directories created"

# Update run.sh if it exists, otherwise create it
if [ -f ~/text2video/run.sh ]; then
    echo "📝 Updating run.sh..."
else
    echo "📝 Creating run.sh..."
fi

# The run.sh should already be synced, but verify it exists
if [ ! -f ~/text2video/run.sh ]; then
    echo "⚠️  Warning: run.sh not found. Please sync project files first."
    exit 1
fi

chmod +x ~/text2video/run.sh
echo "✅ run.sh is executable"

# Test the setup
echo ""
echo "🧪 Testing setup..."
echo "Testing GPU access..."
~/text2video/run.sh python - <<'PY'
import torch
print("✅ Torch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
PY

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Install packages: ~/text2video/run.sh pip install --no-cache-dir transformers diffusers accelerate"
echo "2. Run smoke test: ~/text2video/run.sh python ~/text2video/code/smoke_train.py"
echo "3. Check WORKFLOW.md for detailed guide"
echo ""
echo "📁 Project structure:"
echo "  Code: ~/text2video/code/"
echo "  Data: ~/scratch.msml612-fa25/data/"
echo "  Runs: ~/scratch.msml612-fa25/runs/"
echo "  Weights: ~/scratch.msml612-fa25/weights/"

# Terminal Commands for Video Generation

## Step 1: Connect to VM
```bash
gcloud compute ssh a100-golden --zone=us-central1-a
```

## Step 2: Navigate to project directory
```bash
cd ~/text2video
source venv/bin/activate
```

## Step 3: Generate video with your custom prompt (on VM)
```bash
# Generate video with your prompt on GPU
python refine_video_custom.py noisy_video.mp4 output_video.mp4 "your prompt here" 0.3
```

**Examples:**
```bash
python refine_video_custom.py noisy_video.mp4 output.mp4 "dog running on the grass" 0.3

```

## Step 4: Download the generated video (from your LOCAL terminal - exit VM first)
```bash
# Exit the VM first (type 'exit' or press Ctrl+D)
# Then from your local machine, download the output video
# Replace 'output.mp4' with the filename you used in Step 3
gcloud compute scp --zone=us-central1-a a100-golden:~/text2video/output.mp4 ./
```

---

## Quick One-Liner Examples

### Full workflow with custom prompt (all on GPU VM):
```bash
# 1. Connect and setup
gcloud compute ssh a100-golden --zone=us-central1-a
cd ~/text2video && source venv/bin/activate

# 2. Generate video with your prompt (on GPU)
python refine_video_custom.py noisy_video.mp4 my_output.mp4 "your custom prompt here" 0.3
```

---

## Tips:
- Use quotes around your prompt if it contains spaces
- The output video will be saved in the `~/text2video/` directory on the VM
- Videos are generated at 256x256 resolution, 8 FPS
- **Important**: Step 4 (download) must be run from your LOCAL terminal, not from inside the VM
- Make sure the filename in Step 4 matches the output filename you used in Step 3


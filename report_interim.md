# Interim Report — Text-to-Video on MSR-VTT (Subset)

## 1. Data prep / curation (10)

- **Dataset**: MSR-VTT subset (7,010 videos).
- **Processing**: verified paths; extracted fps, duration, frames, resolution; normalized to H.264 MP4.
- **Train/Val Split**: 90/10 stratified by duration bins.
- **Summaries**: duration & FPS histograms (figures), sample thumbnails/clips.

## 2. NN design & implementation (25)

- **Baseline**: text-conditioned diffusion (DDPM) with 3D UNet and FiLM text conditioning (CLIP text encoder).
- **Complexity**: temporal 3D conv hierarchy, conditional FiLM modulation, EMA, AMP, preview sampling.
- **Potential Enhancements** (planned): classifier-free guidance (CFG), temporal attention at bottleneck, latent-space UNet via VAE.
- **Citations**: (add exact references to the baseline ideas / libraries you followed).

## 3. Code quality & reproducibility (20)

- **Containerized** execution (`run.sh`).
- **One-command** training (`train_baseline.sh`), fixed seeds, config in `configs/baseline.sh`, `versions.txt`.
- Clean project structure and clear outputs: checkpoints + samples.

## 4. Performance (25)

- **Metric**: CLIPScore@8f on validation set.
- **Throughput**: samples/sec on A100_1g.5gb slice; VRAM & wall-clock per 1k steps.
- **Qualitative**: sample frame grids showing denoising progress.

## Results (fill in)

- CLIPScore@8f: …
- Throughput: … samples/sec (BS=?, T=?, size=?)
- Qual: attach 2–3 sample grids.

## Limitations & Next Steps

- Discuss limits of pixel-space at 64×64 and advantages of moving to latent space + CFG.


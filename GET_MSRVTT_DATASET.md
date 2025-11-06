# How to Get Complete MSR-VTT Dataset

## Quick Summary

MSR-VTT is a **proprietary dataset** that requires registration. Here's how to get it:

## Step-by-Step Guide

### 1. Download MSR_VTT.json

**Official Source (Recommended):**
1. Visit: **http://ms-multimedia-challenge.com/2017/dataset**
2. Click "Register" or "Download"
3. Fill out the registration form (free for research)
4. Download `MSR_VTT.json` annotation file
5. Place it at: `data/msr-vtt/annotation/MSR_VTT.json`

**Alternative Sources:**
- Check your university's dataset repository
- Ask your research advisor/lab for access
- Check if your institution has a license

### 2. Verify the Dataset

```bash
# Verify you have the complete dataset
./scripts/verify_and_preprocess_msrvtt.sh
```

This will check:
- ✅ File is valid JSON
- ✅ Contains ~10,000 videos (full dataset)
- ✅ Has train/val/test splits
- ✅ Shows statistics

### 3. Preprocess the Dataset

Once verified, the script will offer to preprocess:

```bash
# Or manually:
./scripts/preprocess_local.sh train    # ~6,513 videos
./scripts/preprocess_local.sh val      # ~497 videos  
./scripts/preprocess_local.sh test      # ~2,990 videos
```

### 4. Sync to Zaratan

```bash
./scripts/sync_processed_data.sh
```

## Expected Dataset Size

- **Full MSR-VTT**: ~10,000 videos
  - Train: ~6,513 videos
  - Val: ~497 videos
  - Test: ~2,990 videos
- **Annotation file**: ~50-100 MB
- **Videos**: ~10-20 GB (if downloading all)

## Current Status

You currently have a **test dataset** with 2 videos. This is useful for:
- ✅ Testing the preprocessing pipeline
- ✅ Verifying everything works
- ✅ Developing code

But you'll need the **full dataset** for actual training.

## Troubleshooting

### "File not found"
- Make sure you downloaded `MSR_VTT.json` (not a zip file)
- Check the file path: `data/msr-vtt/annotation/MSR_VTT.json`

### "Only 2 videos found"
- You still have the test dataset
- Replace with the full dataset from official source

### "Invalid JSON"
- The file might be corrupted
- Re-download from official source

## Next Steps After Getting Full Dataset

1. **Verify**: `./scripts/verify_and_preprocess_msrvtt.sh`
2. **Preprocess**: Script will guide you through it
3. **Sync**: `./scripts/sync_processed_data.sh`
4. **Train**: Start training on Zaratan!

## Need Help?

- **Official Website**: http://ms-multimedia-challenge.com/2017/dataset
- **Paper**: Search "MSR-VTT: A Large Video Description Dataset"
- **Contact**: Dataset maintainers via official website

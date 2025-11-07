export MANIFEST=/scratch/data/manifests/train.manifest.processed.jsonl
export VAL_MANIFEST=/scratch/data/manifests/val.jsonl
export OUTDIR=/scratch/runs/t2v_ddpm_baseline
export STEPS=2000
export BS=2
export FRAMES=8
export SIZE=64
export LR=1e-4
export PYTHONHASHSEED=612
export CUBLAS_WORKSPACE_CONFIG=:4096:8


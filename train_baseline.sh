#!/usr/bin/env bash
set -euo pipefail
source ~/text2video/configs/baseline.sh
~/text2video/run.sh bash -lc '
set -e
export PYTHONHASHSEED='"$PYTHONHASHSEED"'
python - <<PY
import torch, random, numpy as np, os
seed=int(os.environ["PYTHONHASHSEED"])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
PY
python ~/text2video/code/train_t2v_diffusion.py
'


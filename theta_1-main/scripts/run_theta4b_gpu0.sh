#!/bin/bash
# GPU 0: THETA 4B training - FCPB/zero_shot + mental_health/zero_shot
set -euo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPU 0: THETA 4B Training ==="

log "THETA 4B: FCPB / zero_shot"
bash $S/04_train_theta.sh --dataset FCPB --model_size 4B --mode zero_shot \
  --num_topics 20 --epochs 30 --batch_size 16 --gpu 0 --language en
log "  Done: FCPB / zero_shot"

log "THETA 4B: mental_health / zero_shot"
bash $S/04_train_theta.sh --dataset mental_health --model_size 4B --mode zero_shot \
  --num_topics 20 --epochs 30 --batch_size 16 --gpu 0 --language en
log "  Done: mental_health / zero_shot"

log "=== GPU 0 THETA 4B Complete ==="

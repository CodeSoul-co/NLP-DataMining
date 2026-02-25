#!/bin/bash
# GPU 1: THETA 4B training - mental_health/supervised + eval for completed models
set -euo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPU 1: THETA 4B Training ==="

log "THETA 4B: mental_health / supervised"
bash $S/04_train_theta.sh --dataset mental_health --model_size 4B --mode supervised \
  --num_topics 20 --epochs 30 --batch_size 16 --gpu 1 --language en
log "  Done: mental_health / supervised"

log "=== GPU 1 THETA 4B Complete ==="

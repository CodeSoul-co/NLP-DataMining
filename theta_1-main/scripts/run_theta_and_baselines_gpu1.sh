#!/bin/bash
# GPU 1: THETA 4B (socialTwitter) + Baselines (prodlda/etm/ctm/bertopic)
set -euo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPU 1 Start ==="

# --- THETA 4B ---
log "THETA: mental_health / zero_shot"
bash $S/04_train_theta.sh --dataset mental_health --model_size 4B --mode zero_shot --num_topics 20 --epochs 50 --gpu 1 --language en
log "THETA: mental_health / supervised"
bash $S/04_train_theta.sh --dataset mental_health --model_size 4B --mode supervised --num_topics 20 --epochs 50 --gpu 1 --language en
log "THETA: socialTwitter / zero_shot"
bash $S/04_train_theta.sh --dataset socialTwitter --model_size 4B --mode zero_shot --num_topics 20 --epochs 50 --gpu 1 --language en
log "THETA: socialTwitter / supervised"
bash $S/04_train_theta.sh --dataset socialTwitter --model_size 4B --mode supervised --num_topics 20 --epochs 50 --gpu 1 --language en

# --- Baselines ---
for model in prodlda etm; do
  for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    log "Baseline: $model / $ds (GPU 1)"
    bash $S/05_train_baseline.sh --dataset $ds --models $model --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 1 --language en || log "  FAILED: $model/$ds"
  done
done

for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
  log "Baseline: ctm / $ds (GPU 1)"
  bash $S/05_train_baseline.sh --dataset $ds --models ctm --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 1 --language en || log "  FAILED: ctm/$ds"
done

for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
  log "Baseline: bertopic / $ds (GPU 1)"
  bash $S/05_train_baseline.sh --dataset $ds --models bertopic --vocab_size 5000 --gpu 1 --language en || log "  FAILED: bertopic/$ds"
done

log "=== GPU 1 ALL DONE ==="

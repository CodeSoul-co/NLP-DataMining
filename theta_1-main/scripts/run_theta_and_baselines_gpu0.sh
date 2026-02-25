#!/bin/bash
# GPU 0: THETA 4B (mental_health) + Baselines (lda/hdp/btm/nvdm/gsm)
set -euo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPU 0 Start ==="

# --- THETA 4B ---
log "THETA: FCPB / zero_shot"
bash $S/04_train_theta.sh --dataset FCPB --model_size 4B --mode zero_shot --num_topics 20 --epochs 50 --gpu 0 --language en
log "THETA: FCPB / unsupervised"
bash $S/04_train_theta.sh --dataset FCPB --model_size 4B --mode unsupervised --num_topics 20 --epochs 50 --gpu 0 --language en
log "THETA: germanCoal / zero_shot"
bash $S/04_train_theta.sh --dataset germanCoal --model_size 4B --mode zero_shot --num_topics 20 --epochs 50 --gpu 0 --language en
log "THETA: germanCoal / unsupervised"
bash $S/04_train_theta.sh --dataset germanCoal --model_size 4B --mode unsupervised --num_topics 20 --epochs 50 --gpu 0 --language en

# --- Baselines ---
for model in lda btm; do
  for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    log "Baseline: $model / $ds"
    bash $S/05_train_baseline.sh --dataset $ds --models $model --num_topics 20 --vocab_size 5000 --language en || log "  FAILED: $model/$ds"
  done
done

for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
  log "Baseline: hdp / $ds"
  bash $S/05_train_baseline.sh --dataset $ds --models hdp --vocab_size 5000 --language en || log "  FAILED: hdp/$ds"
done

for model in nvdm gsm; do
  for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    log "Baseline: $model / $ds (GPU 0)"
    bash $S/05_train_baseline.sh --dataset $ds --models $model --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 0 --language en || log "  FAILED: $model/$ds"
  done
done

log "=== GPU 0 ALL DONE ==="

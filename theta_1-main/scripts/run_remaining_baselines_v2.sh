#!/bin/bash
# Run remaining baselines on GPU 1 (GPU 0 has stale memory)
# Missing: FCPB(bertopic), hatespeech(gsm,prodlda,ctm,bertopic), mental_health(all 9)
set -uo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
export JOBLIB_TEMP_FOLDER=/root/autodl-tmp/tmp_joblib
export TMPDIR=/root/autodl-tmp/tmp_joblib
mkdir -p /root/autodl-tmp/tmp_joblib
S=scripts
GPU=${1:-1}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- hatespeech: gsm, prodlda, ctm, bertopic ---
log "=== hatespeech: gsm,prodlda ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models gsm,prodlda \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech gsm,prodlda error"

log "=== hatespeech: ctm ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models ctm \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech ctm error"

log "=== hatespeech: bertopic ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech bertopic error"

# --- FCPB: bertopic ---
log "=== FCPB: bertopic ==="
timeout 7200 bash $S/05_train_baseline.sh --dataset FCPB --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB bertopic timeout/error"

# --- mental_health: all 9 (largest dataset ~1M docs) ---
log "=== mental_health: lda,hdp ==="
bash $S/05_train_baseline.sh --dataset mental_health --models lda,hdp \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health trad error"

log "=== mental_health: etm,nvdm,gsm,prodlda ==="
bash $S/05_train_baseline.sh --dataset mental_health --models etm,nvdm,gsm,prodlda \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health neural error"

log "=== mental_health: ctm ==="
bash $S/05_train_baseline.sh --dataset mental_health --models ctm \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health ctm error"

log "=== mental_health: bertopic ==="
timeout 7200 bash $S/05_train_baseline.sh --dataset mental_health --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health bertopic timeout/error"

# Cleanup
rm -rf /root/autodl-tmp/tmp_joblib
log "=== ALL REMAINING BASELINES COMPLETE ==="

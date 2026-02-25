#!/bin/bash
# Run all missing baseline models (excluding BTM)
# Fixed: JOBLIB_TEMP_FOLDER on larger disk to avoid /tmp overflow
set -uo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
export JOBLIB_TEMP_FOLDER=/root/autodl-tmp/tmp_joblib
export TMPDIR=/root/autodl-tmp/tmp_joblib
export TEMP=/root/autodl-tmp/tmp_joblib
export TMP=/root/autodl-tmp/tmp_joblib
mkdir -p /root/autodl-tmp/tmp_joblib
S=scripts
GPU=${1:-0}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# FCPB: lda, bertopic missing
log "=== FCPB: lda ==="
bash $S/05_train_baseline.sh --dataset FCPB --models lda \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB lda error"

log "=== FCPB: bertopic ==="
bash $S/05_train_baseline.sh --dataset FCPB --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB bertopic error"

# hatespeech: all 9 missing (no btm)
log "=== hatespeech: lda,hdp ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models lda,hdp \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech trad error"

log "=== hatespeech: etm,nvdm,gsm,prodlda ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models etm,nvdm,gsm,prodlda \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech neural error"

log "=== hatespeech: ctm ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models ctm \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech ctm error"

log "=== hatespeech: bertopic ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech bertopic error"

# mental_health: all 9 missing (no btm, largest dataset ~1M docs)
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
bash $S/05_train_baseline.sh --dataset mental_health --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health bertopic error"

# Cleanup joblib temp
rm -rf /root/autodl-tmp/tmp_joblib
log "=== ALL MISSING BASELINES (no BTM) COMPLETE ==="

#!/bin/bash
# Run all missing baseline models
set -uo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
GPU=${1:-0}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# socialTwitter: only btm missing
log "=== socialTwitter: btm ==="
bash $S/05_train_baseline.sh --dataset socialTwitter --models btm \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: socialTwitter btm error"

# FCPB: lda,hdp,btm,bertopic missing
log "=== FCPB: lda ==="
bash $S/05_train_baseline.sh --dataset FCPB --models lda \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB lda error"

log "=== FCPB: btm ==="
bash $S/05_train_baseline.sh --dataset FCPB --models btm \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB btm error"

log "=== FCPB: bertopic ==="
bash $S/05_train_baseline.sh --dataset FCPB --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB bertopic error"

# hatespeech: all 10 missing
log "=== hatespeech: lda,hdp ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models lda,hdp \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech trad error"

log "=== hatespeech: btm ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models btm \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech btm error"

log "=== hatespeech: etm,nvdm,gsm,prodlda ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models etm,nvdm,gsm,prodlda \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech neural error"

log "=== hatespeech: ctm ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models ctm \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech ctm error"

log "=== hatespeech: bertopic ==="
bash $S/05_train_baseline.sh --dataset hatespeech --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech bertopic error"

# mental_health: all 10 missing (largest dataset, 1M docs)
log "=== mental_health: lda,hdp ==="
bash $S/05_train_baseline.sh --dataset mental_health --models lda,hdp \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health trad error"

log "=== mental_health: btm ==="
bash $S/05_train_baseline.sh --dataset mental_health --models btm \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health btm error"

log "=== mental_health: etm,nvdm,gsm,prodlda ==="
bash $S/05_train_baseline.sh --dataset mental_health --models etm,nvdm,gsm,prodlda \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health neural error"

log "=== mental_health: ctm ==="
bash $S/05_train_baseline.sh --dataset mental_health --models ctm \
  --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health ctm error"

log "=== mental_health: bertopic ==="
bash $S/05_train_baseline.sh --dataset mental_health --models bertopic \
  --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: mental_health bertopic error"

log "=== ALL MISSING BASELINES COMPLETE ==="

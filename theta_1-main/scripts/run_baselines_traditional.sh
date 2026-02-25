#!/bin/bash
# Run traditional baseline models (no GPU needed) on all 5 datasets
set -uo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
log() { echo "[$(date +%H:%M:%S)] $*"; }

DATASETS="germanCoal socialTwitter FCPB hatespeech mental_health"

for ds in $DATASETS; do
    log "=== $ds: lda,hdp,btm ==="
    bash $S/05_train_baseline.sh --dataset $ds --models lda,hdp,btm \
      --num_topics 20 --with-viz --language en 2>&1 || log "  WARN: $ds traditional had errors"
    log "=== $ds traditional DONE ==="
done

log "=== ALL TRADITIONAL BASELINES COMPLETE ==="

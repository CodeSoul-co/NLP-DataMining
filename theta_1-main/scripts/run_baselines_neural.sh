#!/bin/bash
# Run neural baseline models on all 5 datasets
# These need GPU - run after THETA training finishes
set -uo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
GPU=${1:-0}
log() { echo "[$(date +%H:%M:%S)] $*"; }

DATASETS="germanCoal socialTwitter FCPB hatespeech mental_health"

for ds in $DATASETS; do
    log "=== $ds: etm,nvdm,gsm,prodlda ==="
    bash $S/05_train_baseline.sh --dataset $ds --models etm,nvdm,gsm,prodlda \
      --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: $ds neural-1 had errors"

    log "=== $ds: ctm ==="
    bash $S/05_train_baseline.sh --dataset $ds --models ctm \
      --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: $ds ctm had errors"

    log "=== $ds: bertopic ==="
    bash $S/05_train_baseline.sh --dataset $ds --models bertopic \
      --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: $ds bertopic had errors"

    log "=== $ds neural DONE ==="
done

log "=== ALL NEURAL BASELINES COMPLETE ==="

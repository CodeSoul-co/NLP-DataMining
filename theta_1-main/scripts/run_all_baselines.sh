#!/bin/bash
# Run all baseline models on all 5 datasets
# Traditional models (no GPU): lda, hdp, btm (STM requires covariates, skipped automatically)
# Neural models (GPU): etm, ctm, nvdm, gsm, prodlda, bertopic
set -euo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
GPU=${1:-0}
log() { echo "[$(date +%H:%M:%S)] $*"; }

DATASETS="germanCoal socialTwitter FCPB hatespeech mental_health"
TRADITIONAL="lda,hdp,btm"
NEURAL="etm,ctm,nvdm,gsm,prodlda"

for ds in $DATASETS; do
    log "=== $ds: Traditional models ==="
    bash $S/05_train_baseline.sh --dataset $ds --models $TRADITIONAL \
      --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: $ds traditional had errors"

    log "=== $ds: Neural models ==="
    bash $S/05_train_baseline.sh --dataset $ds --models $NEURAL \
      --num_topics 20 --epochs 100 --batch_size 64 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: $ds neural had errors"

    log "=== $ds: BERTopic ==="
    bash $S/05_train_baseline.sh --dataset $ds --models bertopic \
      --num_topics 20 --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: $ds bertopic had errors"

    log "=== $ds DONE ==="
done

log "=== ALL BASELINES COMPLETE ==="

#!/bin/bash
# Re-run visualization for already-completed baselines that only have metrics_summary.png
# This uses --skip-train to only re-evaluate and re-visualize
set -uo pipefail
cd /root/autodl-tmp
export PYTHONUNBUFFERED=1
S=scripts
GPU=${1:-0}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# germanCoal: all 9 done but viz may be incomplete
log "=== germanCoal: re-viz all ==="
for m in lda hdp etm nvdm gsm prodlda ctm bertopic; do
    log "  Re-viz germanCoal/$m"
    bash $S/05_train_baseline.sh --dataset germanCoal --models $m \
      --num_topics 20 --skip-train --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: germanCoal $m viz error"
done

# socialTwitter: all 9 done
log "=== socialTwitter: re-viz all ==="
for m in lda hdp etm nvdm gsm prodlda ctm bertopic; do
    log "  Re-viz socialTwitter/$m"
    bash $S/05_train_baseline.sh --dataset socialTwitter --models $m \
      --num_topics 20 --skip-train --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: socialTwitter $m viz error"
done

# FCPB: 8 done (no bertopic)
log "=== FCPB: re-viz completed models ==="
for m in lda hdp etm nvdm gsm prodlda ctm; do
    log "  Re-viz FCPB/$m"
    bash $S/05_train_baseline.sh --dataset FCPB --models $m \
      --num_topics 20 --skip-train --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: FCPB $m viz error"
done

# hatespeech: 4 done (lda,hdp,etm,nvdm)
log "=== hatespeech: re-viz completed models ==="
for m in lda hdp etm nvdm; do
    log "  Re-viz hatespeech/$m"
    bash $S/05_train_baseline.sh --dataset hatespeech --models $m \
      --num_topics 20 --skip-train --with-viz --language en --gpu $GPU 2>&1 || log "  WARN: hatespeech $m viz error"
done

log "=== RE-VIZ COMPLETE ==="

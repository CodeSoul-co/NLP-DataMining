#!/bin/bash
# =============================================================================
# GPU 1: THETA 4B (mental_health/socialTwitter)
#        + Baselines (prodlda/etm/ctm/bertopic) × 5 datasets
# NOTE: Start this AFTER Phase 1 (data prep) on GPU 0 is done.
#       Watch gpu0 log for "Phase 1 Complete" before starting this.
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPTS="/root/autodl-tmp/scripts"
# Find the log dir created by GPU 0
LOG_DIR=$(ls -dt /root/autodl-tmp/logs/train_* 2>/dev/null | head -1)
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="/root/autodl-tmp/logs/train_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPU 1 Pipeline Started ==="
log "Logs: $LOG_DIR"

ALL_DS="FCPB germanCoal hatespeech mental_health socialTwitter"

# =============================================================================
# Phase 2: THETA 4B — GPU 1 handles mental_health, socialTwitter
# =============================================================================
log "=== Phase 2: THETA 4B (GPU 1) ==="

declare -A DS_THETA_MODES=(
    [mental_health]="zero_shot supervised"
    [socialTwitter]="zero_shot supervised"
)

for ds in mental_health socialTwitter; do
    for mode in ${DS_THETA_MODES[$ds]}; do
        log "THETA 4B: $ds / $mode"
        if bash "$SCRIPTS/04_train_theta.sh" --dataset "$ds" --model_size 4B --mode "$mode" \
            --num_topics 20 --epochs 50 --gpu 1 --language en \
            > "$LOG_DIR/theta_${ds}_${mode}.log" 2>&1; then
            log "  ✓ $ds/$mode done"
        else
            log "  ✗ $ds/$mode FAILED"
        fi
    done
done

log "=== Phase 2 (GPU 1) Complete ==="

# =============================================================================
# Phase 3: Baselines on GPU 1 — prodlda/etm/ctm/bertopic
# =============================================================================
log "=== Phase 3: Baselines (GPU 1) ==="

for model in prodlda etm; do
    for ds in $ALL_DS; do
        log "$model — $ds (GPU 1)"
        if bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
            --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 1 --language en \
            > "$LOG_DIR/bl_${model}_${ds}.log" 2>&1; then
            log "  ✓"
        else
            log "  ✗ FAILED"
        fi
    done
done

# CTM
for ds in $ALL_DS; do
    log "ctm — $ds (GPU 1)"
    if bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models ctm \
        --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 1 --language en \
        > "$LOG_DIR/bl_ctm_${ds}.log" 2>&1; then
        log "  ✓"
    else
        log "  ✗ FAILED"
    fi
done

# BERTopic (auto topics)
for ds in $ALL_DS; do
    log "bertopic — $ds (GPU 1)"
    if bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models bertopic \
        --vocab_size 5000 --gpu 1 --language en \
        > "$LOG_DIR/bl_bertopic_${ds}.log" 2>&1; then
        log "  ✓"
    else
        log "  ✗ FAILED"
    fi
done

log "=========================================="
log "GPU 1 Pipeline COMPLETE"
log "Logs: $LOG_DIR"
log "=========================================="

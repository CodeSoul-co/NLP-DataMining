#!/bin/bash
# =============================================================================
# GPU 0: Data Prep (all 5 datasets) + THETA 4B (FCPB/germanCoal/hatespeech)
#        + Baselines (lda/hdp/btm/nvdm/gsm) × 5 datasets
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPTS="/root/autodl-tmp/scripts"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/root/autodl-tmp/logs/train_${TS}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPU 0 Pipeline Started ==="
log "Logs: $LOG_DIR"

# =============================================================================
# Phase 1: Baseline data prep — one CTM run per dataset (BOW + SBERT + W2V)
# =============================================================================
log "=== Phase 1: Baseline Data Prep ==="

declare -A DS_LANG=(
    [FCPB]=english
    [germanCoal]=german
    [hatespeech]=english
    [mental_health]=english
    [socialTwitter]=english
)

for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    lang=${DS_LANG[$ds]}
    log "Preparing $ds (lang=$lang) — full: BOW + SBERT + Word2Vec"
    if bash "$SCRIPTS/03_prepare_data.sh" --dataset "$ds" --model ctm --vocab_size 5000 --language "$lang" \
        > "$LOG_DIR/prep_${ds}.log" 2>&1; then
        log "  ✓ $ds done"
    else
        log "  ✗ $ds FAILED — see $LOG_DIR/prep_${ds}.log"
    fi
done

log "=== Phase 1 Complete ==="

# =============================================================================
# Phase 2: THETA 4B — GPU 0 handles FCPB, germanCoal, hatespeech
# =============================================================================
log "=== Phase 2: THETA 4B (GPU 0) ==="

declare -A DS_THETA_MODES=(
    [FCPB]="zero_shot unsupervised"
    [germanCoal]="zero_shot unsupervised"
    [hatespeech]="zero_shot supervised"
)

for ds in FCPB germanCoal hatespeech; do
    for mode in ${DS_THETA_MODES[$ds]}; do
        log "THETA 4B: $ds / $mode"
        if bash "$SCRIPTS/04_train_theta.sh" --dataset "$ds" --model_size 4B --mode "$mode" \
            --num_topics 20 --epochs 50 --gpu 0 --language en \
            > "$LOG_DIR/theta_${ds}_${mode}.log" 2>&1; then
            log "  ✓ $ds/$mode done"
        else
            log "  ✗ $ds/$mode FAILED"
        fi
    done
done

log "=== Phase 2 Complete ==="

# =============================================================================
# Phase 3: Baselines on GPU 0 — traditional (CPU) + nvdm/gsm (GPU 0)
# =============================================================================
log "=== Phase 3: Baselines (GPU 0) ==="

ALL_DS="FCPB germanCoal hatespeech mental_health socialTwitter"

# Traditional (CPU) models
for model in lda btm; do
    for ds in $ALL_DS; do
        log "$model — $ds"
        if bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
            --num_topics 20 --vocab_size 5000 --language en \
            > "$LOG_DIR/bl_${model}_${ds}.log" 2>&1; then
            log "  ✓"
        else
            log "  ✗ FAILED"
        fi
    done
done

# HDP (no num_topics)
for ds in $ALL_DS; do
    log "hdp — $ds"
    if bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models hdp \
        --vocab_size 5000 --language en \
        > "$LOG_DIR/bl_hdp_${ds}.log" 2>&1; then
        log "  ✓"
    else
        log "  ✗ FAILED"
    fi
done

# Neural models on GPU 0
for model in nvdm gsm; do
    for ds in $ALL_DS; do
        log "$model — $ds (GPU 0)"
        if bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
            --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 0 --language en \
            > "$LOG_DIR/bl_${model}_${ds}.log" 2>&1; then
            log "  ✓"
        else
            log "  ✗ FAILED"
        fi
    done
done

log "=========================================="
log "GPU 0 Pipeline COMPLETE"
log "Logs: $LOG_DIR"
log "=========================================="

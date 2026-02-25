#!/bin/bash
# =============================================================================
# Full Training Pipeline — GPU 1 portion
# 5 datasets × (THETA 4B for mental_health/socialTwitter + ProdLDA/ETM/CTM/BERTopic)
# Parameters: vocab_size=5000, num_topics=20, epochs=50
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

SCRIPTS="/root/autodl-tmp/scripts"
# Use same log dir pattern — find the latest one created by GPU 0 script
LOG_DIR=$(ls -dt /root/autodl-tmp/logs/training_* 2>/dev/null | head -1)
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="/root/autodl-tmp/logs/training_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Training Pipeline — GPU 1"
echo "Log directory: $LOG_DIR"
echo "Started at: $(date)"
echo "=============================================="

DATASETS="FCPB germanCoal hatespeech mental_health socialTwitter"

# =============================================================================
# Step 2: THETA 4B Training — GPU 1 portion (mental_health, socialTwitter)
# =============================================================================
echo ""
echo ">>> Step 2: THETA 4B Training (GPU 1: mental_health, socialTwitter)"
echo "=============================================="

# mental_health: zero_shot + supervised
for mode in zero_shot supervised; do
    echo "[$(date +%H:%M:%S)] THETA 4B — mental_health / $mode"
    bash "$SCRIPTS/04_train_theta.sh" --dataset mental_health --model_size 4B --mode $mode \
        --num_topics 20 --epochs 50 --gpu 1 --language en \
        > "$LOG_DIR/theta_mental_health_${mode}.log" 2>&1 || echo "  [WARN] THETA mental_health/$mode failed"
    echo "  Done."
done

# socialTwitter: zero_shot + supervised
for mode in zero_shot supervised; do
    echo "[$(date +%H:%M:%S)] THETA 4B — socialTwitter / $mode"
    bash "$SCRIPTS/04_train_theta.sh" --dataset socialTwitter --model_size 4B --mode $mode \
        --num_topics 20 --epochs 50 --gpu 1 --language en \
        > "$LOG_DIR/theta_socialTwitter_${mode}.log" 2>&1 || echo "  [WARN] THETA socialTwitter/$mode failed"
    echo "  Done."
done

echo ""
echo ">>> Step 2 (GPU 1) COMPLETE at $(date)"
echo ""

# =============================================================================
# Step 3: Baseline Training — GPU 1 portion
# (ProdLDA, ETM, CTM, BERTopic)
# =============================================================================
echo ">>> Step 3: Baseline Training (GPU 1)"
echo "=============================================="

# Neural BOW models on GPU 1
for model in prodlda etm; do
    echo ""
    echo "--- $model (GPU 1) ---"
    for ds in $DATASETS; do
        echo "[$(date +%H:%M:%S)] $model — $ds"
        bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
            --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 1 --language en \
            > "$LOG_DIR/baseline_${model}_${ds}.log" 2>&1 || echo "  [WARN] $model/$ds failed"
        echo "  Done."
    done
done

# CTM (GPU 1)
echo ""
echo "--- ctm (GPU 1) ---"
for ds in $DATASETS; do
    echo "[$(date +%H:%M:%S)] ctm — $ds"
    bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models ctm \
        --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 1 --language en \
        > "$LOG_DIR/baseline_ctm_${ds}.log" 2>&1 || echo "  [WARN] ctm/$ds failed"
    echo "  Done."
done

# BERTopic (GPU 1, auto topics)
echo ""
echo "--- bertopic (GPU 1) ---"
for ds in $DATASETS; do
    echo "[$(date +%H:%M:%S)] bertopic — $ds"
    bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models bertopic \
        --vocab_size 5000 --gpu 1 --language en \
        > "$LOG_DIR/baseline_bertopic_${ds}.log" 2>&1 || echo "  [WARN] bertopic/$ds failed"
    echo "  Done."
done

echo ""
echo "=============================================="
echo "GPU 1 Pipeline COMPLETE at $(date)"
echo "Logs: $LOG_DIR"
echo "=============================================="

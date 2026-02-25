#!/bin/bash
# =============================================================================
# Full Training Pipeline: 5 datasets × 12 methods (THETA 4B + 11 baselines)
# Parameters: vocab_size=5000, num_topics=20, epochs=50
# GPU Assignment: This script uses GPU 0. Run run_all_5ds_training_gpu1.sh on GPU 1.
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

SCRIPTS="/root/autodl-tmp/scripts"
LOG_DIR="/root/autodl-tmp/logs/training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Training Pipeline — GPU 0"
echo "Log directory: $LOG_DIR"
echo "Started at: $(date)"
echo "=============================================="

DATASETS="FCPB germanCoal hatespeech mental_health socialTwitter"

# =============================================================================
# Step 1: Baseline data preparation (BOW for all, sequential, uses CPU mostly)
# =============================================================================
echo ""
echo ">>> Step 1: Baseline data preparation (BOW)"
echo "=============================================="
for ds in $DATASETS; do
    lang=english
    [ "$ds" = "germanCoal" ] && lang=german
    echo "[$(date +%H:%M:%S)] Preparing BOW for $ds..."
    bash "$SCRIPTS/03_prepare_data.sh" --dataset "$ds" --model lda --vocab_size 5000 --language "$lang" \
        > "$LOG_DIR/prep_bow_${ds}.log" 2>&1 || echo "  [WARN] BOW prep failed for $ds"
    echo "  Done."
done

# CTM data (BOW + SBERT)
echo ""
echo ">>> Step 1b: CTM data preparation (SBERT)"
echo "=============================================="
for ds in $DATASETS; do
    lang=english
    [ "$ds" = "germanCoal" ] && lang=german
    echo "[$(date +%H:%M:%S)] Preparing CTM data for $ds..."
    bash "$SCRIPTS/03_prepare_data.sh" --dataset "$ds" --model ctm --vocab_size 5000 --language "$lang" \
        > "$LOG_DIR/prep_ctm_${ds}.log" 2>&1 || echo "  [WARN] CTM prep failed for $ds"
    echo "  Done."
done

# BERTopic data
echo ""
echo ">>> Step 1c: BERTopic data preparation"
echo "=============================================="
for ds in $DATASETS; do
    lang=english
    [ "$ds" = "germanCoal" ] && lang=german
    echo "[$(date +%H:%M:%S)] Preparing BERTopic data for $ds..."
    bash "$SCRIPTS/03_prepare_data.sh" --dataset "$ds" --model bertopic --vocab_size 5000 --language "$lang" \
        > "$LOG_DIR/prep_bertopic_${ds}.log" 2>&1 || echo "  [WARN] BERTopic prep failed for $ds"
    echo "  Done."
done

echo ""
echo ">>> Step 1 COMPLETE at $(date)"
echo ""

# =============================================================================
# Step 2: THETA 4B Training — GPU 0 portion (FCPB, germanCoal, hatespeech)
# =============================================================================
echo ">>> Step 2: THETA 4B Training (GPU 0: FCPB, germanCoal, hatespeech)"
echo "=============================================="

# FCPB: zero_shot + unsupervised
for mode in zero_shot unsupervised; do
    echo "[$(date +%H:%M:%S)] THETA 4B — FCPB / $mode"
    bash "$SCRIPTS/04_train_theta.sh" --dataset FCPB --model_size 4B --mode $mode \
        --num_topics 20 --epochs 50 --gpu 0 --language en \
        > "$LOG_DIR/theta_FCPB_${mode}.log" 2>&1 || echo "  [WARN] THETA FCPB/$mode failed"
    echo "  Done."
done

# germanCoal: zero_shot + unsupervised
for mode in zero_shot unsupervised; do
    echo "[$(date +%H:%M:%S)] THETA 4B — germanCoal / $mode"
    bash "$SCRIPTS/04_train_theta.sh" --dataset germanCoal --model_size 4B --mode $mode \
        --num_topics 20 --epochs 50 --gpu 0 --language en \
        > "$LOG_DIR/theta_germanCoal_${mode}.log" 2>&1 || echo "  [WARN] THETA germanCoal/$mode failed"
    echo "  Done."
done

# hatespeech: zero_shot + supervised
for mode in zero_shot supervised; do
    echo "[$(date +%H:%M:%S)] THETA 4B — hatespeech / $mode"
    bash "$SCRIPTS/04_train_theta.sh" --dataset hatespeech --model_size 4B --mode $mode \
        --num_topics 20 --epochs 50 --gpu 0 --language en \
        > "$LOG_DIR/theta_hatespeech_${mode}.log" 2>&1 || echo "  [WARN] THETA hatespeech/$mode failed"
    echo "  Done."
done

echo ""
echo ">>> Step 2 (GPU 0) COMPLETE at $(date)"
echo ""

# =============================================================================
# Step 3: Baseline Training — GPU 0 portion
# (LDA, HDP, STM, BTM = CPU; NVDM, GSM = GPU 0)
# =============================================================================
echo ">>> Step 3: Baseline Training (GPU 0)"
echo "=============================================="

# Traditional models (CPU)
for model in lda hdp btm; do
    echo ""
    echo "--- $model ---"
    for ds in $DATASETS; do
        echo "[$(date +%H:%M:%S)] $model — $ds"
        EXTRA=""
        [ "$model" = "hdp" ] || [ "$model" = "bertopic" ] && EXTRA="" || EXTRA="--num_topics 20"
        # hdp doesn't take num_topics
        if [ "$model" = "hdp" ]; then
            bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
                --vocab_size 5000 --language en \
                > "$LOG_DIR/baseline_${model}_${ds}.log" 2>&1 || echo "  [WARN] $model/$ds failed"
        else
            bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
                --num_topics 20 --vocab_size 5000 --language en \
                > "$LOG_DIR/baseline_${model}_${ds}.log" 2>&1 || echo "  [WARN] $model/$ds failed"
        fi
        echo "  Done."
    done
done

# Neural BOW models on GPU 0
for model in nvdm gsm; do
    echo ""
    echo "--- $model (GPU 0) ---"
    for ds in $DATASETS; do
        echo "[$(date +%H:%M:%S)] $model — $ds"
        bash "$SCRIPTS/05_train_baseline.sh" --dataset "$ds" --models "$model" \
            --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 0 --language en \
            > "$LOG_DIR/baseline_${model}_${ds}.log" 2>&1 || echo "  [WARN] $model/$ds failed"
        echo "  Done."
    done
done

echo ""
echo "=============================================="
echo "GPU 0 Pipeline COMPLETE at $(date)"
echo "Logs: $LOG_DIR"
echo "=============================================="

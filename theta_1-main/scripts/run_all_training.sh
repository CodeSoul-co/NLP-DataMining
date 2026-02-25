#!/bin/bash
# =============================================================================
# Master Training Script: THETA 4B + All Baselines on 5 Datasets
# =============================================================================
# Pipeline:
#   Phase 1: Prepare BOW for THETA 4B (convert .npz→.npy, symlink from 0.6B)
#   Phase 2: Train THETA 4B (all modes per dataset) with visualization
#   Phase 3: Prepare baseline data for all 5 datasets (BOW + SBERT)
#   Phase 4: Train 10 baseline models on all 5 datasets (2 GPUs parallel)
#   Phase 5: Generate combined evaluation CSV (4B + 0.6B + baselines)
#
# 2x RTX 4090 D (GPU 0 + GPU 1)
# =============================================================================

PROJ_ROOT="/root/autodl-tmp"
RESULT_DIR="$PROJ_ROOT/result"
LOG_DIR="$RESULT_DIR/training_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)

# Dataset config: name:language:modes_for_4B
DATASETS=(
    "FCPB:english:zero_shot,unsupervised"
    "germanCoal:german:zero_shot,unsupervised"
    "hatespeech:english:zero_shot,supervised"
    "mental_health:english:zero_shot,supervised"
    "socialTwitter:english:zero_shot,supervised"
)

NUM_TOPICS=20
THETA_EPOCHS=100
BASELINE_EPOCHS=100
VOCAB_SIZE=5000

echo "============================================================"
echo " Master Training Script — $(date)"
echo "============================================================"
echo " Datasets:  FCPB, germanCoal, hatespeech, mental_health, socialTwitter"
echo " THETA 4B:  zero_shot + supervised/unsupervised per dataset"
echo " Baselines: lda,hdp,btm,nvdm,gsm,prodlda,etm,ctm,bertopic (STM requires covariates)"
echo " GPUs:      0, 1"
echo " K=$NUM_TOPICS  THETA_epochs=$THETA_EPOCHS  Baseline_epochs=$BASELINE_EPOCHS"
echo "============================================================"
echo ""

# Helper: log with timestamp
log() { echo "[$(date +%H:%M:%S)] $*"; }

# =============================================================================
# PHASE 1: Prepare BOW for THETA 4B
# =============================================================================
log "===== PHASE 1: Prepare BOW for THETA 4B ====="

# Convert .npz (sparse) to .npy (dense) and ensure all required BOW files exist
python3 << 'PHASE1_PY'
import os, numpy as np, scipy.sparse as sp, shutil, json

RESULT = "/root/autodl-tmp/result"
DATASETS = ["FCPB", "germanCoal", "hatespeech", "mental_health", "socialTwitter"]

for ds in DATASETS:
    src_bow_dir = os.path.join(RESULT, "0.6B", ds, "bow")
    dst_bow_dir = os.path.join(RESULT, "4B", ds, "bow")
    
    # Check if 4B BOW already has bow_matrix.npy
    dst_npy = os.path.join(dst_bow_dir, "bow_matrix.npy")
    if os.path.exists(dst_npy):
        print(f"[SKIP] {ds}: bow_matrix.npy already exists in 4B")
        continue
    
    os.makedirs(dst_bow_dir, exist_ok=True)
    
    # Convert .npz -> .npy
    src_npz = os.path.join(src_bow_dir, "bow_matrix.npz")
    src_npy = os.path.join(src_bow_dir, "bow_matrix.npy")
    
    if os.path.exists(src_npy):
        print(f"[COPY] {ds}: Copying bow_matrix.npy from 0.6B")
        shutil.copy2(src_npy, dst_npy)
    elif os.path.exists(src_npz):
        print(f"[CONVERT] {ds}: Converting bow_matrix.npz -> .npy")
        bow_sparse = sp.load_npz(src_npz)
        bow_dense = bow_sparse.toarray()
        np.save(dst_npy, bow_dense)
        print(f"  Shape: {bow_dense.shape}, saved to {dst_npy}")
    else:
        print(f"[ERROR] {ds}: No BOW found in 0.6B!")
        continue
    
    # Copy vocab files
    for fname in ["vocab.txt", "vocab.json", "vocab_embeddings.npy"]:
        src = os.path.join(src_bow_dir, fname)
        dst = os.path.join(dst_bow_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
    
    # If vocab.json doesn't exist, create from vocab.txt
    vocab_json = os.path.join(dst_bow_dir, "vocab.json")
    vocab_txt = os.path.join(dst_bow_dir, "vocab.txt")
    if not os.path.exists(vocab_json) and os.path.exists(vocab_txt):
        with open(vocab_txt, 'r') as f:
            vocab = [line.strip() for line in f if line.strip()]
        with open(vocab_json, 'w') as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"  Created vocab.json ({len(vocab)} words)")

print("\nPhase 1a complete: BOW ready for all 5 datasets in 4B")
PHASE1_PY

# Phase 1b: Create legacy-style embedding paths for 4B
# The training script expects: result/4B/{ds}/{mode}/embeddings/embeddings.npy
# But 4B embeddings are at: result/4B/{ds}/embedding/{mode}_embeddings_*.npy
log "Phase 1b: Creating legacy embedding paths for 4B"

python3 << 'PHASE1B_PY'
import os, glob, shutil

RESULT = "/root/autodl-tmp/result"
DS_MODES = {
    "FCPB": ["zero_shot", "unsupervised"],
    "germanCoal": ["zero_shot", "unsupervised"],
    "hatespeech": ["zero_shot", "supervised"],
    "mental_health": ["zero_shot", "supervised"],
    "socialTwitter": ["zero_shot", "supervised"],
}

for ds, modes in DS_MODES.items():
    emb_dir = os.path.join(RESULT, "4B", ds, "embedding")
    if not os.path.isdir(emb_dir):
        print(f"[WARN] {ds}: No embedding dir at {emb_dir}")
        continue
    
    for mode in modes:
        legacy_dir = os.path.join(RESULT, "4B", ds, mode, "embeddings")
        legacy_file = os.path.join(legacy_dir, "embeddings.npy")
        
        if os.path.exists(legacy_file):
            print(f"[SKIP] {ds}/{mode}: embeddings.npy already exists")
            continue
        
        # Find the source embedding file
        pattern = os.path.join(emb_dir, f"{mode}_embeddings_*.npy")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[WARN] {ds}/{mode}: No embedding file matching {pattern}")
            continue
        
        src = matches[-1]  # Use latest
        os.makedirs(legacy_dir, exist_ok=True)
        
        # Symlink instead of copy (save disk space for large files)
        os.symlink(src, legacy_file)
        print(f"[LINK] {ds}/{mode}: {os.path.basename(src)} -> embeddings.npy")
        
        # Also link labels if they exist
        label_pattern = os.path.join(emb_dir, f"{mode}_labels_*.npy")
        label_matches = sorted(glob.glob(label_pattern))
        if label_matches:
            label_dst = os.path.join(legacy_dir, "labels.npy")
            if not os.path.exists(label_dst):
                os.symlink(label_matches[-1], label_dst)

print("\nPhase 1b complete: Legacy embedding paths ready")
PHASE1B_PY

echo ""

# =============================================================================
# PHASE 2: Train THETA 4B on all datasets
# =============================================================================
log "===== PHASE 2: Train THETA 4B ====="

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r DS LANG MODES <<< "$entry"
    IFS=',' read -ra MODE_LIST <<< "$MODES"
    
    for MODE in "${MODE_LIST[@]}"; do
        LOG_FILE="$LOG_DIR/${DS}_theta_4B_${MODE}_${TS}.log"
        log "[THETA 4B] $DS / $MODE -> $LOG_FILE"
        
        bash "$PROJ_ROOT/scripts/04_train_theta.sh" \
            --dataset "$DS" --model_size 4B --mode "$MODE" \
            --num_topics $NUM_TOPICS --epochs $THETA_EPOCHS \
            --batch_size 64 --gpu 0 --language en \
            2>&1 | tee "$LOG_FILE"
        
        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -ne 0 ]; then
            log "[WARN] THETA 4B $DS/$MODE failed (exit=$EXIT_CODE), continuing..."
        else
            log "[OK] THETA 4B $DS/$MODE done"
        fi
    done
done
echo ""

# =============================================================================
# PHASE 3: Prepare baseline data
# =============================================================================
log "===== PHASE 3: Prepare baseline data ====="

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r DS LANG MODES <<< "$entry"
    
    BASELINE_DATA_DIR="$RESULT_DIR/baseline/$DS/data"
    
    # 3a. BOW-only data (shared by lda,hdp,btm,nvdm,gsm,prodlda,etm)
    BOW_EXP=$(find "$BASELINE_DATA_DIR" -maxdepth 1 -name "exp_*" -not -name "*ctm*" -not -name "*dtm*" -not -name "*bertopic*" -type d 2>/dev/null | head -1)
    if [ -n "$BOW_EXP" ] && [ -f "$BOW_EXP/bow_matrix.npy" ]; then
        log "[SKIP] $DS: BOW baseline data exists ($BOW_EXP)"
    else
        log "[PREP] $DS: Preparing BOW baseline data"
        bash "$PROJ_ROOT/scripts/03_prepare_data.sh" \
            --dataset "$DS" --model lda --vocab_size $VOCAB_SIZE --language "$LANG" \
            2>&1 | tee "$LOG_DIR/${DS}_prep_bow_${TS}.log"
    fi
    
    # 3b. CTM data (BOW + SBERT)
    CTM_EXP=$(find "$BASELINE_DATA_DIR" -maxdepth 1 -name "*ctm*" -type d 2>/dev/null | head -1)
    if [ -n "$CTM_EXP" ] && [ -f "$CTM_EXP/sbert_embeddings.npy" ]; then
        log "[SKIP] $DS: CTM data exists ($CTM_EXP)"
    else
        log "[PREP] $DS: Preparing CTM data (SBERT)"
        bash "$PROJ_ROOT/scripts/03_prepare_data.sh" \
            --dataset "$DS" --model ctm --vocab_size $VOCAB_SIZE --language "$LANG" --gpu 0 \
            2>&1 | tee "$LOG_DIR/${DS}_prep_ctm_${TS}.log"
    fi
    
    # 3c. BERTopic reuses CTM's SBERT data (no separate prep needed)
done
echo ""

# =============================================================================
# PHASE 4: Train baseline models (2 GPUs parallel)
# =============================================================================
log "===== PHASE 4: Train baseline models ====="

train_baseline() {
    local DS=$1
    local MODELS=$2
    local GPU=$3
    local TAG=$4
    local LOG_FILE="$LOG_DIR/${DS}_bl_${TAG}_${TS}.log"
    
    log "  [BL] $DS / $MODELS (GPU $GPU) -> $LOG_FILE"
    
    bash "$PROJ_ROOT/scripts/05_train_baseline.sh" \
        --dataset "$DS" --models "$MODELS" \
        --num_topics $NUM_TOPICS --epochs $BASELINE_EPOCHS \
        --batch_size 64 --gpu "$GPU" --language en --with-viz \
        2>&1 | tee "$LOG_FILE"
    
    return ${PIPESTATUS[0]}
}

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r DS LANG MODES <<< "$entry"
    
    log "--- $DS: Training baselines ---"
    
    # GPU 0: BOW-only traditional models (lda,hdp,btm — no GPU needed but run here)
    # GPU 1: BOW-only neural models (nvdm,gsm,prodlda — need GPU)
    train_baseline "$DS" "lda,hdp,btm" 0 "traditional" &
    PID1=$!
    train_baseline "$DS" "nvdm,gsm,prodlda" 1 "neural_bow" &
    PID2=$!
    wait $PID1; log "  [OK] $DS traditional done"
    wait $PID2; log "  [OK] $DS neural_bow done"
    
    # GPU 0: ETM  |  GPU 1: CTM
    train_baseline "$DS" "etm" 0 "etm" &
    PID1=$!
    train_baseline "$DS" "ctm" 1 "ctm" &
    PID2=$!
    wait $PID1; log "  [OK] $DS ETM done"
    wait $PID2; log "  [OK] $DS CTM done"
    
    # GPU 0: BERTopic (single, uses SBERT)
    train_baseline "$DS" "bertopic" 0 "bertopic"
    log "  [OK] $DS BERTopic done"
    
    log "--- $DS: All baselines complete ---"
    echo ""
done

# =============================================================================
# PHASE 5: Generate combined evaluation CSV
# =============================================================================
log "===== PHASE 5: Generate combined CSV ====="

python3 << 'PHASE5_PY'
import os, json, csv, re
from pathlib import Path

RESULT_DIR = "/root/autodl-tmp/result"
OUTPUT_CSV = os.path.join(RESULT_DIR, "all_evaluation_metrics.csv")

COLUMNS = [
    "dataset", "model", "model_size", "mode", "num_topics",
    "TD", "iRBO", "NPMI", "C_V", "UMass", "Exclusivity", "PPL"
]

rows = []

def safe_float(v):
    """Convert to float string, return '' if not possible."""
    try:
        f = float(v)
        if f != f:  # NaN
            return ""
        return str(round(f, 6))
    except:
        return str(v) if v else ""

def extract_metrics(metrics):
    """Extract 7 metrics from various JSON formats."""
    if "topic_diversity" in metrics and isinstance(metrics["topic_diversity"], dict):
        return {
            "TD": safe_float(metrics["topic_diversity"].get("td", "")),
            "iRBO": safe_float(metrics["topic_diversity"].get("irbo", "")),
            "NPMI": safe_float(metrics["topic_coherence"].get("npmi_avg", "")),
            "C_V": safe_float(metrics["topic_coherence"].get("cv_avg", "")),
            "UMass": safe_float(metrics["topic_coherence"].get("umass_avg", "")),
            "Exclusivity": safe_float(metrics.get("topic_exclusivity", {}).get("avg", "")),
            "PPL": safe_float(metrics.get("perplexity", "")),
        }
    else:
        return {
            "TD": safe_float(metrics.get("td", metrics.get("topic_diversity", ""))),
            "iRBO": safe_float(metrics.get("irbo", "")),
            "NPMI": safe_float(metrics.get("npmi", metrics.get("npmi_avg", ""))),
            "C_V": safe_float(metrics.get("cv", metrics.get("cv_avg", ""))),
            "UMass": safe_float(metrics.get("umass", metrics.get("umass_avg", ""))),
            "Exclusivity": safe_float(metrics.get("exclusivity", metrics.get("topic_exclusivity", ""))),
            "PPL": safe_float(metrics.get("perplexity", metrics.get("ppl", ""))),
        }

# -------------------------------------------------------
# 1. Load existing 0.6B THETA results from CSV
# -------------------------------------------------------
csv_06b = os.path.join(RESULT_DIR, "0.6B", "THETA_evaluation_metrics.csv")
if os.path.exists(csv_06b):
    print(f"Loading 0.6B from {csv_06b}")
    with open(csv_06b, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset": row.get("dataset", ""),
                "model": "THETA",
                "model_size": "0.6B",
                "mode": row.get("mode", ""),
                "num_topics": "20",
                "TD": safe_float(row.get("topic_diversity_td", "")),
                "iRBO": safe_float(row.get("topic_diversity_irbo", "")),
                "NPMI": safe_float(row.get("topic_coherence_npmi_avg", "")),
                "C_V": safe_float(row.get("topic_coherence_cv_avg", "")),
                "UMass": safe_float(row.get("topic_coherence_umass_avg", "")),
                "Exclusivity": safe_float(row.get("topic_exclusivity_avg", "")),
                "PPL": safe_float(row.get("perplexity", "")),
            })
    print(f"  Loaded {len(rows)} rows")

# -------------------------------------------------------
# 2. Load 4B THETA results
# -------------------------------------------------------
print("\nScanning 4B THETA results...")
for ds_dir in sorted(Path(RESULT_DIR, "4B").iterdir()):
    if not ds_dir.is_dir() or ds_dir.name.startswith('.'):
        continue
    dataset = ds_dir.name
    
    # New exp structure: models/exp_*/
    models_dir = ds_dir / "models"
    if models_dir.exists():
        for exp_dir in sorted(models_dir.glob("exp_*")):
            for mf in list(exp_dir.glob("evaluation/metrics*.json")) + list(exp_dir.glob("metrics*.json")):
                try:
                    metrics = json.load(open(mf))
                    config_file = exp_dir / "config.json"
                    mode = ""
                    if config_file.exists():
                        config = json.load(open(config_file))
                        mode = config.get("mode", "")
                    m = extract_metrics(metrics)
                    m.update({"dataset": dataset, "model": "THETA", "model_size": "4B",
                              "mode": mode, "num_topics": str(metrics.get("num_topics", "20"))})
                    rows.append(m)
                    print(f"  [4B] {dataset}/{exp_dir.name}: OK")
                except Exception as e:
                    print(f"  [4B] Error {mf}: {e}")
    
    # Legacy structure: {mode}/evaluation/
    for mode_name in ('zero_shot', 'unsupervised', 'supervised'):
        eval_dir = ds_dir / mode_name / "evaluation"
        if eval_dir.exists():
            for mf in eval_dir.glob("metrics*.json"):
                try:
                    metrics = json.load(open(mf))
                    m = extract_metrics(metrics)
                    m.update({"dataset": dataset, "model": "THETA", "model_size": "4B",
                              "mode": mode_name, "num_topics": str(metrics.get("num_topics", "20"))})
                    rows.append(m)
                    print(f"  [4B legacy] {dataset}/{mode_name}: OK")
                except Exception as e:
                    print(f"  [4B legacy] Error {mf}: {e}")

# -------------------------------------------------------
# 3. Load baseline results
# -------------------------------------------------------
print("\nScanning baseline results...")
baseline_dir = Path(RESULT_DIR, "baseline")
if baseline_dir.exists():
    for ds_dir in sorted(baseline_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith('.'):
            continue
        dataset = ds_dir.name
        models_dir = ds_dir / "models"
        if not models_dir.exists():
            continue
        
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            
            for exp_dir in sorted(model_dir.glob("exp_*")):
                metrics_files = (
                    list(exp_dir.glob("metrics_k*.json")) +
                    list(exp_dir.glob("evaluation/metrics*.json")) +
                    list(exp_dir.glob("model/metrics_k*.json"))
                )
                for mf in metrics_files:
                    try:
                        metrics = json.load(open(mf))
                        m = extract_metrics(metrics)
                        num_topics = metrics.get("num_topics", "")
                        if not num_topics:
                            match = re.search(r'k(\d+)', mf.name)
                            if match:
                                num_topics = match.group(1)
                        display_name = model_name.upper()
                        if model_name == "bertopic":
                            display_name = "BERTopic"
                        m.update({"dataset": dataset, "model": display_name, "model_size": "-",
                                  "mode": "-", "num_topics": str(num_topics)})
                        rows.append(m)
                        print(f"  [BL] {dataset}/{model_name}: OK")
                    except Exception as e:
                        print(f"  [BL] Error {mf}: {e}")

# -------------------------------------------------------
# 4. Write combined CSV
# -------------------------------------------------------
rows.sort(key=lambda r: (r["dataset"], r["model"], r.get("model_size", ""), r.get("mode", "")))

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"\n{'='*110}")
print(f"Combined CSV: {OUTPUT_CSV}  ({len(rows)} rows)")
print(f"{'='*110}")
print(f"{'Dataset':<16} {'Model':<10} {'Size':<5} {'Mode':<14} {'K':<4} "
      f"{'TD':<8} {'iRBO':<8} {'NPMI':<8} {'C_V':<8} {'UMass':<10} {'Excl':<8} {'PPL':<10}")
print("-"*110)
for r in rows:
    def f(v, w=8):
        try: return f"{float(v):<{w}.4f}"
        except: return f"{str(v):<{w}}"
    print(f"{r['dataset']:<16} {r['model']:<10} {r['model_size']:<5} {r['mode']:<14} {str(r['num_topics']):<4} "
          f"{f(r['TD'])} {f(r['iRBO'])} {f(r['NPMI'])} {f(r['C_V'])} "
          f"{f(r['UMass'],10)} {f(r['Exclusivity'])} {f(r['PPL'],10)}")

PHASE5_PY

echo ""
log "============================================================"
log " ALL TRAINING COMPLETE — $(date)"
log "============================================================"
log " THETA 4B:  $RESULT_DIR/4B/{dataset}/models/"
log " Baselines: $RESULT_DIR/baseline/{dataset}/models/"
log " Combined:  $RESULT_DIR/all_evaluation_metrics.csv"
log " Logs:      $LOG_DIR/"

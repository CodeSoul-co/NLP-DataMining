#!/bin/bash
# THETA 完整处理流程脚本
# 支持多用户并发处理
#
# 用法: ./run_full_pipeline.sh <JOB_ID> <DATASET> [<TEXT_COL>] [<TIME_COL>] [<NUM_TOPICS>]
#
# 示例: ./run_full_pipeline.sh job_20260128_001 hatespeech 内容 发布时间 20

set -e  # 遇到错误立即退出

# ============ 参数接收 ============
JOB_ID=$1
DATASET=$2
TEXT_COL=${3:-""}
TIME_COL=${4:-""}
NUM_TOPICS=${5:-0}

# 检查必要参数
if [ -z "$JOB_ID" ] || [ -z "$DATASET" ]; then
    echo "错误: 缺少必要参数"
    echo "用法: ./run_full_pipeline.sh <JOB_ID> <DATASET> [<TEXT_COL>] [<TIME_COL>] [<NUM_TOPICS>]"
    exit 1
fi

# ============ 环境变量 ============
export THETA_BASE=${THETA_BASE:-"/root/autodl-tmp"}
export THETA_MODEL_SIZE=${THETA_MODEL_SIZE:-"0.6B"}

# ============ 路径配置 ============
# BASE_DIR 从环境变量 THETA_BASE 获取，支持 DLC 环境自动适配
BASE_DIR="${THETA_BASE}"
LOG_DIR="$BASE_DIR/result/job_$JOB_ID"
LOG_FILE="$LOG_DIR/log.txt"

# 创建结果目录
mkdir -p "$LOG_DIR"

# ============ 日志函数 ============
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

update_status() {
    local status=$1
    local message=$2
    local status_file="$BASE_DIR/job_status/${JOB_ID}.json"
    
    # 读取现有状态文件
    if [ -f "$status_file" ]; then
        local temp=$(mktemp)
        jq ".status = \"$status\" | .message = \"$message\" | .updated_at = \"$(date -Iseconds)\"" "$status_file" > "$temp"
        mv "$temp" "$status_file"
    else
        # 创建新状态文件
        echo "{\"job_id\": \"$JOB_ID\", \"status\": \"$status\", \"message\": \"$message\", \"updated_at\": \"$(date -Iseconds)\"}" > "$status_file"
    fi
}

# ============ 开始执行 ============
log "=========================================="
log "任务开始: $JOB_ID"
log "数据集: $DATASET"
log "文本列: ${TEXT_COL:-未指定}"
log "时间列: ${TIME_COL:-未指定}"
log "主题数: $NUM_TOPICS"
log "=========================================="

update_status "running" "任务已开始"

# -------- Stage 1: 数据预处理 - BOW生成 --------
log "[Stage 1/5] 运行数据预处理 - BOW生成..."
update_status "running" "正在生成BOW矩阵"

# 检查数据文件
DATA_DIR="$BASE_DIR/data/job_$JOB_ID"
if [ ! -d "$DATA_DIR" ]; then
    log "⚠️ 数据目录不存在: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# 检查是否已有BOW
BOW_DIR="$BASE_DIR/result/job_$JOB_ID/$THETA_MODEL_SIZE/bow"
if [ -f "$BOW_DIR/bow_matrix.npz" ] && [ -f "$BOW_DIR/vocab.txt" ]; then
    log "⚠️ 检测到BOW已存在，跳过 Stage 1"
else
    # 运行BOW生成
    python "$BASE_DIR/THETA_aliyun/ETM/prepare_data.py" \
        --dataset "$DATASET" \
        --job_id "$JOB_ID" \
        --model theta \
        --model_size "$THETA_MODEL_SIZE" \
        --mode zero_shot \
        --vocab_size 10000 \
        --bow-only \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "❌ ERROR: BOW生成失败"
        update_status "failed" "BOW生成失败"
        exit 1
    fi
    log "✓ BOW生成完成"
fi

# -------- Stage 2: 数据预处理 - 嵌入生成 --------
log "[Stage 2/5] 运行数据预处理 - 嵌入生成..."
update_status "running" "正在生成嵌入向量"

# 检查是否已有嵌入
EMB_DIR="$BASE_DIR/result/job_$JOB_ID/$THETA_MODEL_SIZE/zero_shot/embeddings"
if [ -f "$EMB_DIR/embeddings.npy" ]; then
    log "⚠️ 检测到嵌入已存在，跳过 Stage 2"
else
    # 运行嵌入生成
    python "$BASE_DIR/THETA_aliyun/ETM/prepare_data.py" \
        --dataset "$DATASET" \
        --job_id "$JOB_ID" \
        --model theta \
        --model_size "$THETA_MODEL_SIZE" \
        --mode zero_shot \
        --vocab_size 10000 \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "❌ ERROR: 嵌入生成失败"
        update_status "failed" "嵌入生成失败"
        exit 1
    fi
    log "✓ 嵌入生成完成"
fi

# -------- Stage 3: ETM模型训练 --------
log "[Stage 3/5] 运行ETM模型训练..."
update_status "running" "正在训练ETM模型"

# 检查是否已有模型
MODEL_DIR="$BASE_DIR/result/job_$JOB_ID/$THETA_MODEL_SIZE/zero_shot/model"
if [ -f "$MODEL_DIR/theta.npy" ] && [ -f "$MODEL_DIR/beta.npy" ]; then
    log "⚠️ 检测到模型已存在，跳过 Stage 3"
else
    # 准备工作目录
    mkdir -p /workspace
    cp -r "$BASE_DIR/THETA_aliyun/ETM" /workspace/
    cd /workspace/ETM
    
    # 运行ETM训练
    python run_pipeline.py \
        --dataset "$DATASET" \
        --job_id "$JOB_ID" \
        --models theta \
        --model_size "$THETA_MODEL_SIZE" \
        --mode zero_shot \
        --num_topics "$NUM_TOPICS" \
        --epochs 50 \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "❌ ERROR: ETM训练失败"
        update_status "failed" "ETM训练失败"
        exit 1
    fi
    log "✓ ETM训练完成"
fi

# -------- Stage 4: 可视化生成 --------
log "[Stage 4/5] 生成可视化图表..."
update_status "running" "正在生成可视化图表"

# 检查是否已有可视化
VIZ_DIR="$BASE_DIR/result/job_$JOB_ID/visualization"
if [ -d "$VIZ_DIR" ] && [ "$(ls -A $VIZ_DIR)" ]; then
    log "⚠️ 检测到可视化已存在，跳过 Stage 4"
else
    # 创建可视化目录
    mkdir -p "$VIZ_DIR"
    
    # 运行可视化生成
    cd /workspace/ETM
    python visualization/run_visualization.py \
        --dataset "$DATASET" \
        --job_id "$JOB_ID" \
        --model_size "$THETA_MODEL_SIZE" \
        --mode zero_shot \
        --output_dir "$VIZ_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "❌ ERROR: 可视化生成失败"
        update_status "failed" "可视化生成失败"
        exit 1
    fi
    log "✓ 可视化生成完成"
fi

# -------- Stage 5: 生成结果JSON --------
log "[Stage 5/5] 生成结果JSON..."
update_status "running" "正在生成结果JSON"

# 生成analysis_result.json
RESULT_JSON="$BASE_DIR/result/job_$JOB_ID/analysis_result.json"
if [ -f "$RESULT_JSON" ]; then
    log "⚠️ 检测到结果JSON已存在，跳过 Stage 5"
else
    # 生成结果JSON
    python "$BASE_DIR/THETA_aliyun/scripts/generate_result_json.py" \
        --dataset "$DATASET" \
        --job_id "$JOB_ID" \
        --model_size "$THETA_MODEL_SIZE" \
        --mode zero_shot \
        --output "$RESULT_JSON" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "❌ ERROR: 结果JSON生成失败"
        update_status "failed" "结果JSON生成失败"
        exit 1
    fi
    log "✓ 结果JSON生成完成"
fi

# -------- 完成 --------
log "=========================================="
log "任务完成: $JOB_ID"
log "结果路径: $BASE_DIR/result/job_$JOB_ID/"
log "=========================================="

update_status "completed" "任务已完成"

exit 0

#!/bin/bash
# =============================================================================
# THETA Environment Setup Script
# =============================================================================
# This script is sourced by all other scripts to set up environment variables.
# It automatically detects PROJECT_ROOT from the script location and loads .env
#
# Usage (in other scripts):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/env_setup.sh"
# =============================================================================

# Detect PROJECT_ROOT from script location
# This works regardless of where the script is called from
if [ -z "$PROJECT_ROOT" ]; then
    # Get the directory where this script is located
    ENV_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # PROJECT_ROOT is the parent of scripts/
    export PROJECT_ROOT="$(cd "$ENV_SETUP_DIR/.." && pwd)"
fi

# Load .env file if it exists
# Priority: external export > .env > auto-detected defaults
# Only set variables from .env if they are NOT already set in the environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # Remove leading/trailing whitespace from key
        key=$(echo "$key" | xargs)
        # Skip if key is empty after trimming
        [[ -z "$key" ]] && continue
        # Only set if not already defined in environment
        if [ -z "${!key+x}" ]; then
            # Remove surrounding quotes from value if present
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
            export "$key=$value"
        fi
    done < "$PROJECT_ROOT/.env"
fi

# =============================================================================
# Core Directory Variables (with defaults)
# =============================================================================

# Source directory (contains models and embedding)
export SRC_DIR="${SRC_DIR:-$PROJECT_ROOT/src}"

# Models module directory (formerly ETM)
export MODELS_DIR="${MODELS_DIR:-$SRC_DIR/models}"

# ETM_DIR alias for backward compatibility with existing scripts
export ETM_DIR="${ETM_DIR:-$MODELS_DIR}"

# Embedding module directory
export EMBEDDING_DIR="${EMBEDDING_DIR:-$SRC_DIR/embedding}"

# Agent module directory
export AGENT_DIR="${AGENT_DIR:-$PROJECT_ROOT/agent}"

# Scripts directory
export SCRIPTS_DIR="${SCRIPTS_DIR:-$PROJECT_ROOT/scripts}"

# =============================================================================
# Data Directory Variables
# =============================================================================

# Workspace directory (for user data)
export WORKSPACE_DIR="${WORKSPACE_DIR:-$PROJECT_ROOT/workspace}"

# Data directory (cleaned datasets)
export DATA_DIR="${DATA_DIR:-$WORKSPACE_DIR/data}"

# Raw data directory
export RAW_DATA_DIR="${RAW_DATA_DIR:-$DATA_DIR/raw_data}"

# =============================================================================
# Output Directory Variables
# =============================================================================

# Result directory (model outputs, embeddings, BOW, etc.)
export RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/result}"

# HuggingFace cache directory
export HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"

# =============================================================================
# Model Directory Variables
# =============================================================================

# Base directory for embedding models (Qwen, SBERT, etc.)
export EMBEDDING_MODELS_DIR="${EMBEDDING_MODELS_DIR:-$PROJECT_ROOT/embedding_models}"

# Qwen model paths (based on model size)
export QWEN_MODEL_0_6B="${QWEN_MODEL_0_6B:-$EMBEDDING_MODELS_DIR/qwen3_embedding_0.6B}"
export QWEN_MODEL_4B="${QWEN_MODEL_4B:-$EMBEDDING_MODELS_DIR/qwen3_embedding_4B}"
export QWEN_MODEL_8B="${QWEN_MODEL_8B:-$EMBEDDING_MODELS_DIR/qwen3_embedding_8B}"

# SBERT model path
export SBERT_MODEL_PATH="${SBERT_MODEL_PATH:-$MODELS_DIR/model/baselines/sbert/sentence-transformers/all-MiniLM-L6-v2}"

# =============================================================================
# Helper Functions
# =============================================================================

# Get Qwen model path based on model size
get_qwen_model_path() {
    local model_size="$1"
    case "$model_size" in
        0.6B) echo "$QWEN_MODEL_0_6B" ;;
        4B)   echo "$QWEN_MODEL_4B" ;;
        8B)   echo "$QWEN_MODEL_8B" ;;
        *)    echo "$QWEN_MODEL_0_6B" ;;  # Default to 0.6B
    esac
}

# Get result directory for a specific model type
get_result_dir() {
    local model_type="$1"  # "theta" or "baseline"
    local model_size="$2"  # e.g., "0.6B" (only for theta)
    
    if [ "$model_type" = "theta" ]; then
        echo "$RESULT_DIR/${model_size:-0.6B}"
    else
        echo "$RESULT_DIR/baseline"
    fi
}

# Print environment info (for debugging)
print_env_info() {
    echo "=========================================="
    echo "THETA Environment Configuration"
    echo "=========================================="
    echo "PROJECT_ROOT:         $PROJECT_ROOT"
    echo "SRC_DIR:              $SRC_DIR"
    echo "MODELS_DIR:           $MODELS_DIR"
    echo "EMBEDDING_DIR:        $EMBEDDING_DIR"
    echo "AGENT_DIR:            $AGENT_DIR"
    echo "DATA_DIR:             $DATA_DIR"
    echo "WORKSPACE_DIR:        $WORKSPACE_DIR"
    echo "RESULT_DIR:           $RESULT_DIR"
    echo "HF_CACHE_DIR:         $HF_CACHE_DIR"
    echo "EMBEDDING_MODELS_DIR: $EMBEDDING_MODELS_DIR"
    echo "=========================================="
}

# Always print env info when sourced (can be disabled with THETA_QUIET=1)
if [ -z "$THETA_QUIET" ]; then
    print_env_info
fi

# Export helper functions
export -f get_qwen_model_path
export -f get_result_dir
export -f print_env_info

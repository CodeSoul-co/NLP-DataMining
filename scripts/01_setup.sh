#!/bin/bash
# =============================================================================
# THETA Setup Script
# =============================================================================
# Install dependencies and download pre-trained data from HuggingFace
#
# This script performs the following:
#   1. Install Python dependencies from requirements.txt
#   2. Install agent dependencies (openai, python-dotenv)
#   3. Download pre-trained embeddings and BOW from HuggingFace
#
# Pre-trained Data:
#   Repository: https://huggingface.co/CodeSoulco/THETA
#   Contents:
#     - Pre-computed Qwen embeddings for benchmark datasets
#     - BOW matrices and vocabularies
#     - LoRA fine-tuned weights (optional)
#
# Usage:
#   ./01_setup.sh
# =============================================================================

set -e

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

echo "=========================================="
echo "THETA Setup Script"
echo "=========================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo ""

# Install Python dependencies
echo "[1/3] Installing Python dependencies..."
cd "$ETM_DIR"
pip install -r requirements.txt -q

# Install openai for agent
echo "[2/3] Installing agent dependencies..."
pip install openai python-dotenv -q

# Download pre-trained data from HuggingFace (if not exists locally)
echo "[3/3] Checking pre-trained data..."
if [ ! -d "$RESULT_DIR/0.6B" ] || [ -z "$(ls -A "$RESULT_DIR/0.6B" 2>/dev/null)" ]; then
    echo "Downloading pre-trained data from HuggingFace..."
    echo "Repository: https://huggingface.co/CodeSoulco/THETA"
    
    # Install huggingface_hub if not installed
    pip install huggingface_hub -q
    
    # Download using huggingface-cli
    python -c "
from huggingface_hub import snapshot_download
import os

hf_cache_dir = os.environ.get('HF_CACHE_DIR', '$HF_CACHE_DIR')

# Download pre-trained embeddings and results
try:
    snapshot_download(
        repo_id='CodeSoulco/THETA',
        local_dir=os.path.join(hf_cache_dir, 'THETA'),
        repo_type='model'
    )
    print('Downloaded successfully!')
except Exception as e:
    print(f'Download failed: {e}')
    print('You can manually download from: https://huggingface.co/CodeSoulco/THETA')
"
else
    echo "Pre-trained data already exists, skipping download."
fi

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="

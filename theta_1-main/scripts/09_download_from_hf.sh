#!/bin/bash
# Download Pre-trained Data from HuggingFace
# Download embeddings, BOW, and LoRA weights from CodeSoulco/THETA

set -e

echo "=========================================="
echo "Download from HuggingFace"
echo "=========================================="
echo "Repository: https://huggingface.co/CodeSoulco/THETA"
echo ""

# Install huggingface_hub if not installed
pip install huggingface_hub -q

# Download
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
import os
import shutil

print('Downloading from HuggingFace...')
print('This may take a while depending on your network speed.')
print('')

try:
    # Download the entire repository
    local_dir = snapshot_download(
        repo_id='CodeSoulco/THETA',
        local_dir='/root/autodl-tmp/hf_cache/THETA',
        repo_type='model'
    )
    
    print(f'Downloaded to: {local_dir}')
    print('')
    print('Contents:')
    for item in os.listdir(local_dir):
        item_path = os.path.join(local_dir, item)
        if os.path.isdir(item_path):
            print(f'  📁 {item}/')
        else:
            size = os.path.getsize(item_path) / (1024*1024)
            print(f'  📄 {item} ({size:.1f} MB)')
    
    print('')
    print('To use the downloaded data:')
    print('  - Embeddings: Copy to /root/autodl-tmp/result/0.6B/{dataset}/{mode}/embeddings/')
    print('  - BOW data: Copy to /root/autodl-tmp/result/0.6B/{dataset}/bow/')
    print('  - LoRA weights: Load using the model loading functions')
    print('')
    print('Download completed successfully!')
    
except Exception as e:
    print(f'Error: {e}')
    print('')
    print('Please manually download from:')
    print('  https://huggingface.co/CodeSoulco/THETA')
    exit(1)
"

echo ""
echo "=========================================="
echo "Download completed!"
echo "=========================================="

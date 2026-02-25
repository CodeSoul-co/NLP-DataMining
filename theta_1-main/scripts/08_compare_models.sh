#!/bin/bash
# =============================================================================
# THETA Model Comparison Script
# =============================================================================
# Compare evaluation metrics across multiple trained models
#
# This script reads metrics from trained models and generates a comparison table
#
# Usage:
#   ./08_compare_models.sh --dataset <name> --models <model_list> [options]
#
# Examples:
#   ./08_compare_models.sh --dataset edu_data --models lda,prodlda,ctm --num_topics 20
#   ./08_compare_models.sh --dataset edu_data --models lda,hdp,nvdm,gsm,prodlda
# =============================================================================

set -e

# Default values
DATASET=""
MODELS=""
NUM_TOPICS=20
OUTPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --models <model_list> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset      Dataset name (required)"
            echo "  --models       Comma-separated model list (required)"
            echo "  --num_topics   Number of topics (default: 20)"
            echo "  --output       Output CSV file (optional)"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset edu_data --models lda,prodlda,ctm --num_topics 20"
            echo "  $0 --dataset edu_data --models lda,hdp,nvdm,gsm,prodlda"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ] || [ -z "$MODELS" ]; then
    echo "Error: --dataset and --models are required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

echo "=========================================="
echo "THETA Model Comparison"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Models:     $MODELS"
echo "Num Topics: $NUM_TOPICS"
echo ""

cd /root/autodl-tmp/ETM

# Run comparison
python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp/ETM')
import json
import os
from pathlib import Path

dataset = '$DATASET'
models = '$MODELS'.split(',')
num_topics = $NUM_TOPICS
output_file = '$OUTPUT_FILE' if '$OUTPUT_FILE' else None

result_dir = Path('/root/autodl-tmp/result/baseline') / dataset

print('='*80)
print(f'Model Comparison: {dataset} (K={num_topics})')
print('='*80)
print()

# Header
header = ['Model', 'TD', 'iRBO', 'NPMI', 'C_V', 'UMass', 'Exclusivity', 'PPL']
print(f'{header[0]:<12} {header[1]:>8} {header[2]:>8} {header[3]:>8} {header[4]:>8} {header[5]:>8} {header[6]:>12} {header[7]:>10}')
print('-'*80)

results = []
for model in models:
    model = model.strip()
    model_dir = result_dir / model
    
    # Handle CTM special case
    if model == 'ctm':
        model_dir = result_dir / 'ctm_zeroshot'
    
    metrics_path = model_dir / f'metrics_k{num_topics}.json'
    
    if not metrics_path.exists():
        print(f'{model:<12} [Metrics not found]')
        continue
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    td = metrics.get('topic_diversity_td', 0)
    irbo = metrics.get('topic_diversity_irbo', 0)
    npmi = metrics.get('topic_coherence_npmi_avg', 0)
    cv = metrics.get('topic_coherence_cv_avg', 0)
    umass = metrics.get('topic_coherence_umass_avg', 0)
    excl = metrics.get('topic_exclusivity_avg', 0)
    ppl = metrics.get('perplexity', 0)
    
    print(f'{model:<12} {td:>8.4f} {irbo:>8.4f} {npmi:>8.4f} {cv:>8.4f} {umass:>8.4f} {excl:>12.4f} {ppl:>10.2f}')
    
    results.append({
        'model': model,
        'td': td, 'irbo': irbo, 'npmi': npmi, 'cv': cv,
        'umass': umass, 'exclusivity': excl, 'ppl': ppl
    })

print('-'*80)
print()

# Find best models
if results:
    best_td = max(results, key=lambda x: x['td'])
    best_npmi = max(results, key=lambda x: x['npmi'])
    best_ppl = min(results, key=lambda x: x['ppl'] if x['ppl'] > 0 else float('inf'))
    
    print('Best Models:')
    print(f'  - Best TD (Topic Diversity): {best_td[\"model\"]} ({best_td[\"td\"]:.4f})')
    print(f'  - Best NPMI (Coherence):     {best_npmi[\"model\"]} ({best_npmi[\"npmi\"]:.4f})')
    print(f'  - Best PPL (Perplexity):     {best_ppl[\"model\"]} ({best_ppl[\"ppl\"]:.2f})')

# Save to CSV if output file specified
if output_file and results:
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'td', 'irbo', 'npmi', 'cv', 'umass', 'exclusivity', 'ppl'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\\nResults saved to: {output_file}')
"

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "=========================================="

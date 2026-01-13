# Topic Model Analysis Agents

This directory contains a comprehensive LangChain multi-agent system for topic model analysis, designed to work with the `/root/autodl-tmp/` directory structure.

## Agents Overview

### Core Processing Agents

| Agent | Responsibility | Input | Output |
|-------|---------------|-------|--------|
| **DataCleaningAgent** | Data validation and configuration setup | `data/{job_id}/data.csv` | `result/{job_id}/config.yaml` |
| **BowAgent** | Generate BOW representation and vocabulary | Validated CSV | `ETM/outputs/vocab/` + `ETM/outputs/bow/` |
| **EmbeddingAgent** | Generate document and vocabulary embeddings | Vocabulary + BOW | `embedding/outputs/zero_shot/` |
| **ETMAgent** | Run Embedded Topic Model | Embeddings + BOW | `ETM/outputs/theta/` + `ETM/outputs/beta/` |
| **VisualizationAgent** | Generate charts and word clouds | ETM outputs | `visualization/outputs/{job_id}/` |

### QA Agents

| Agent | Responsibility | Input | Output |
|-------|---------------|-------|--------|
| **TextQAAgent** | Answer text-based questions about analysis | `result/{job_id}/analysis_result.json` | Text answer |
| **VisionQAAgent** | Answer visual questions about charts | Chart paths + optional images | Visual analysis |

### Orchestrator

| Agent | Responsibility | Methods |
|-------|---------------|---------|
| **OrchestratorAgent** | Single entry point, manages dependency chain | `run_full()`, `handle_query()` |

## Dependency Chain

```
DataCleaning → (Embedding + Bow) → ETM → Visualization → Report → Packager
```

**Auto-completion**: If any dependency is missing, the orchestrator automatically runs the required preceding steps.

## Directory Structure

```
/root/autodl-tmp/  (or THETA_ROOT env var)
├── data/
│   └── {job_id}/
│       └── data.csv                    # Input data
├── ETM/
│   └── outputs/
│       ├── vocab/{job_id}_vocab.json   # Vocabulary
│       ├── bow/{job_id}_bow.npz        # BOW matrix (N×V)
│       ├── theta/{job_id}_theta.npy    # Document-topic (N×K)
│       ├── beta/{job_id}_beta.npy      # Topic-word (K×V)
│       ├── alpha/{job_id}_alpha.npy    # Topic embeddings (K×1024)
│       └── topic_words/{job_id}_topics.json
├── embedding/
│   └── outputs/zero_shot/
│       ├── {job_id}_embeddings.npy     # Document embeddings (N×1024)
│       └── {job_id}_vocab_emb.npy      # Vocabulary embeddings (V×1024)
├── visualization/
│   └── outputs/{job_id}/
│       ├── wordcloud_topic_{i}.png     # Word clouds
│       ├── topic_distribution.png
│       ├── heatmap_doc_topic.png
│       ├── coherence_curve.png
│       └── topic_similarity.png
├── result/
│   └── {job_id}/
│       ├── analysis_result.json        # Main result file
│       ├── report.docx                 # Generated report
│       ├── theta.csv                    # Exported theta matrix
│       ├── beta.csv                     # Exported beta matrix
│       ├── metrics.json                 # Evaluation metrics
│       ├── config.yaml                  # Configuration snapshot
│       └── log.txt                      # Processing log
└── scripts/
    ├── run_engine_a.py                  # BOW generation
    ├── run_engine_b.py                  # Embedding generation
    ├── run_engine_c.py                  # ETM training
    └── run_visualization.py             # Visualization generation
```

## Usage Examples

### Full Pipeline Execution

```python
from agents import OrchestratorAgent

# Initialize orchestrator
orchestrator = OrchestratorAgent(base_dir="/root/autodl-tmp")

# Run complete pipeline
result = orchestrator.run_full("job_20250106_001")
print(f"Status: {result['status']}")
```

### Query-Based Processing

```python
# Answer question about analysis
qa_result = orchestrator.handle_query(
    job_id="job_20250106_001",
    question="What are the main topics and their proportions?"
)
print(qa_result['answer'])

# Visual question
viz_result = orchestrator.handle_query(
    job_id="job_20250106_001", 
    question="What patterns do you see in the topic distribution chart?"
)
print(viz_result['answer'])
```

## analysis_result.json Schema

```json
{
  "job_id": "string",
  "status": "success|failed|processing",
  "completed_at": "ISO datetime",
  "duration_seconds": "number",
  "metrics": {
    "coherence_score": "float",
    "diversity_score": "float", 
    "optimal_k": "integer"
  },
  "topics": [
    {
      "id": "integer",
      "name": "string",
      "keywords": ["string"],
      "proportion": "float",
      "wordcloud_url": "/api/download/{job_id}/wordcloud_topic_{i}.png"
    }
  ],
  "charts": {
    "topic_distribution": "/api/download/{job_id}/topic_distribution.png",
    "heatmap": "/api/download/{job_id}/heatmap_doc_topic.png",
    "coherence_curve": "/api/download/{job_id}/coherence_curve.png",
    "topic_similarity": "/api/download/{job_id}/topic_similarity.png"
  },
  "downloads": {
    "report": "/api/download/{job_id}/report.docx",
    "theta_csv": "/api/download/{job_id}/theta.csv",
    "beta_csv": "/api/download/{job_id}/beta.csv"
  }
}
```

## API Download Paths

The system supports the following download endpoints:

- `/api/download/{job_id}/report.docx` - Generated report
- `/api/download/{job_id}/theta.csv` - Document-topic matrix
- `/api/download/{job_id}/beta.csv` - Topic-word matrix
- `/api/download/{job_id}/wordcloud_topic_{i}.png` - Topic word cloud
- `/api/download/{job_id}/topic_distribution.png` - Topic distribution chart
- `/api/download/{job_id}/heatmap_doc_topic.png` - Document-topic heatmap
- `/api/download/{job_id}/coherence_curve.png` - Coherence analysis
- `/api/download/{job_id}/topic_similarity.png` - Topic similarity matrix

## Shape Validation Rules

All agents enforce strict shape validation:

- **embeddings**: (N, E) float32 - E depends on embedding model (1024 for Qwen, configurable for simple mode)
- **vocab_emb**: (V, E) float32 - must match document embedding dimension
- **bow**: (N, V) matching vocabulary length
- **theta**: (N, K) document-topic distributions
- **beta**: (K, V) topic-word distributions
- **alpha**: (K, E) topic embeddings

**Validation failures** are logged to `result/{job_id}/log.txt` and set `analysis_result.status="failed"`.

## Embedding Modes

The system supports two embedding modes:

### 1. Qwen Mode (Recommended for Production)
Uses Qwen3-Embedding model for high-quality semantic embeddings:
```python
orchestrator = OrchestratorAgent(
    base_dir="/root/autodl-tmp",
    embedding_model_path="/path/to/qwen3_embedding_0.6B"
)
```

### 2. Simple Mode (For Testing/Demo)
Uses random embeddings when Qwen model is not available:
```python
orchestrator = OrchestratorAgent(base_dir="/root/autodl-tmp")
# No embedding_model_path = uses simple random embeddings
```

## Common Error Scenarios

### Shape Inconsistency
```
Error: Shape mismatch: vocab length 5000 != BOW columns 4500
Solution: Check data preprocessing in run_engine_a.py
```

### Missing Files
```
Error: File not found: ETM/outputs/theta/job_001_theta.npy
Solution: Ensure ETM processing completed successfully
```

### Training Convergence Issues
```
Error: ETM training failed to converge
Solution: Check learning rate, epochs, or data quality
```

## Configuration

### Environment Variables
- `THETA_ROOT`: Override default base directory (default: `/root/autodl-tmp`)

### LLM Configuration
```python
llm_config = {
    "provider": "qwen",  # or "openai"
    "api_key": "your-api-key",
    "base_url": "api-endpoint",
    "vision_enabled": True  # For VisionQAAgent
}
```

## Logging

All agents write to `result/{job_id}/log.txt` with timestamps:
```
[2025-01-13 10:30:15] [INFO] OrchestratorAgent: Starting full pipeline for job_id: job_001
[2025-01-13 10:30:16] [INFO] DataCleaningAgent: Data validation successful. Shape: (1000, 1)
[2025-01-13 10:30:20] [ERROR] BowAgent: Script execution failed: Permission denied
```

## Integration Notes

- **FastAPI Integration**: Use OrchestratorAgent as the backend service
- **Batch Processing**: Supports multiple concurrent job processing
- **Error Recovery**: Failed jobs can be resumed from last successful step
- **Resource Management**: Automatic cleanup of temporary files
- **Monitoring**: Real-time progress tracking via log files

## Dependencies

Install required packages:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn wordcloud pillow
pip install fastapi uvicorn pydantic requests pyyaml
```

Optional for LLM integration:
```bash
pip install openai dashscope  # For OpenAI/Qwen APIs
```

# THETA - ETM Topic Model Agent System

A LangGraph-based Agent System for ETM (Embedded Topic Model) pipeline management with a modern React frontend.

## Architecture

```
langgraph_agent/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── agents/         # LangGraph Agent & Nodes
│   │   │   ├── etm_agent.py    # Main agent with graph definition
│   │   │   └── nodes.py        # Pipeline nodes (preprocess, embed, train, eval, viz)
│   │   ├── api/            # REST & WebSocket endpoints
│   │   │   ├── routes.py       # REST API routes
│   │   │   └── websocket.py    # Real-time updates
│   │   ├── core/           # Configuration & utilities
│   │   ├── schemas/        # Pydantic models
│   │   └── services/       # Business logic
│   ├── requirements.txt
│   └── run.py              # Server entry point
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # UI components (Layout, Sidebar, Header)
│   │   ├── pages/          # Page components (Chat, Projects, Data, Results, Viz)
│   │   ├── hooks/          # Custom hooks (useWebSocket)
│   │   ├── services/       # API client
│   │   └── styles/         # Tailwind CSS
│   └── package.json
└── README.md
```

## Features

- **LangGraph Pipeline**: 5-node DAG for ETM training
  - Preprocess: BOW matrix & vocabulary generation
  - Embedding: Document embedding loading
  - Training: ETM model training with real-time progress
  - Evaluation: Topic coherence & diversity metrics
  - Visualization: Topic word clouds, similarity heatmaps, pyLDAvis

- **Real-time Updates**: WebSocket-based status streaming
- **Checkpointing**: Resume failed tasks from last checkpoint
- **Modern UI**: Gemini-style chat interface with sidebar navigation

## Quick Start

### Prerequisites

- Python 3.10+ with `jiqun` conda environment
- Node.js 18+
- GPU 1 available (GPU 0 is reserved)

### Backend Setup

```bash
# Activate environment
source activate jiqun

# Install dependencies
cd /root/autodl-tmp/langgraph_agent/backend
pip install -r requirements.txt

# Start server (uses GPU 1)
python run.py --port 8000
```

### Frontend Setup

```bash
cd /root/autodl-tmp/langgraph_agent/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### Access

- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/api/ws

## API Endpoints

### Tasks
- `POST /api/tasks` - Create new training task
- `GET /api/tasks` - List all tasks
- `GET /api/tasks/{task_id}` - Get task status
- `DELETE /api/tasks/{task_id}` - Cancel task

### Data
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{name}` - Get dataset info

### Results
- `GET /api/results` - List all results
- `GET /api/results/{dataset}/{mode}` - Get result details
- `GET /api/results/{dataset}/{mode}/topic-words` - Get topic words
- `GET /api/results/{dataset}/{mode}/visualizations` - List visualizations

### WebSocket
- `/api/ws` - General updates
- `/api/ws/task/{task_id}` - Task-specific updates

## Usage Example

### Via Chat Interface

1. Open http://localhost:3000
2. Type: "Train on socialTwitter"
3. Watch real-time progress in the chat

### Via API

```python
import requests

# Start training
response = requests.post("http://localhost:8000/api/tasks", json={
    "dataset": "socialTwitter",
    "mode": "zero_shot",
    "num_topics": 20,
    "epochs": 50
})
task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/tasks/{task_id}").json()
print(status)
```

## Configuration

Key settings in `backend/app/core/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| GPU_ID | 1 | GPU to use (avoid GPU 0) |
| RESULT_DIR | /root/autodl-tmp/result | Results output directory |
| DATA_DIR | /root/autodl-tmp/data | Datasets directory |
| QWEN_MODEL_PATH | /root/autodl-tmp/qwen3_embedding_0.6B | Embedding model |

## LangGraph Pipeline

```
┌─────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐    ┌──────────────┐
│ Preprocess  │───▶│ Embedding │───▶│ Training │───▶│ Evaluation │───▶│ Visualization│───▶ END
│ (BOW+Vocab) │    │  (Load)   │    │  (ETM)   │    │ (Metrics)  │    │   (Plots)    │
└─────────────┘    └───────────┘    └──────────┘    └────────────┘    └──────────────┘
                         │
                         ▼ (on error)
                    ┌─────────┐
                    │  Error  │───▶ END
                    └─────────┘
```

## Notes

- Always use GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
- Do not modify `/root/autodl-tmp/pc` or `/root/autodl-tmp/imagenet`
- Results are saved to `/root/autodl-tmp/result/{dataset}/{mode}/`
- Each run creates timestamped files (no overwriting)

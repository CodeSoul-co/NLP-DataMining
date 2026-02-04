# THETA Cloud Deployment Guide

This document explains how to deploy the THETA topic modeling project to Alibaba Cloud.

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Alibaba Cloud Deployment Architecture          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │   OSS   │────▶│   PAI-DLC   │────▶│      PAI-EAS        │   │
│  │ (Data)  │     │ (Offline Training) │  │ (Online Inference Service) │  │
│  └─────────┘     └─────────────┘     └─────────────────────┘   │
│       │                                        │                │
│       │                                        ▼                │
│       │                              ┌─────────────────────┐   │
│       └─────────────────────────────▶│       ECS           │   │
│                                      │   (Backend API Service)     │   │
│                                      └─────────────────────┘   │
│                                                │                │
│                                                ▼                │
│                                      ┌─────────────────────┐   │
│                                      │      Frontend App     │   │
│                                      └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Module Overview

| Module | Cloud Resource | Code | Description |
|------|--------|----------|------|
| Data Storage | OSS | data/, result/ | Store datasets, embeddings, and model outputs |
| Offline Training | PAI-DLC | embedding/, ETM/, scripts/ | GPU training jobs |
| Online Inference | PAI-EAS | ETM/inference_server.py | HTTP inference service |
| Backend API | ECS | ETM/api_server.py | Product API endpoints |

## 3. Deployment Steps

### 3.1 Prepare OSS Storage

1. Create an OSS Bucket:
```bash
# Bucket name: theta-bucket
# Region: same region as PAI (e.g., cn-hangzhou)
```

2. Upload data:
```bash
# Upload datasets
ossutil cp -r ./data oss://theta-bucket/data/

# Upload pretrained models
ossutil cp -r ./qwen3_embedding_0.6B oss://theta-bucket/models/qwen3_embedding_0.6B/

# Upload generated embeddings (if any)
ossutil cp -r ./result oss://theta-bucket/result/
```

3. OSS directory structure:
```
oss://theta-bucket/
├── data/
│   ├── hatespeech/
│   ├── socialTwitter/
│   └── ...
├── models/
│   └── qwen3_embedding_0.6B/
└── result/
    └── 0.6B/
        ├── hatespeech/
        │   ├── zero_shot/
        │   │   ├── embeddings/
        │   │   ├── model/
        │   │   └── ...
        │   └── bow/
        └── ...
```

### 3.2 PAI-DLC Training Job

1. Package code:
```bash
cd /root/autodl-tmp/THETA
zip -r theta_code.zip ETM/ embedding/ scripts/
```

2. Create a DLC job in the PAI console:
   - **Code source**: upload theta_code.zip
   - **Image**: registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch-training:1.12-gpu-py39-cu113
   - **Resources**: 1 x V100 (or A10)
   - **Startup command**:
```bash
export THETA_BASE=oss://theta-bucket
export THETA_MODEL_SIZE=0.6B
bash scripts/cloud_train.sh hatespeech zero_shot 20 50
```

3. Environment variables:
```bash
THETA_BASE=oss://theta-bucket
THETA_MODEL_SIZE=0.6B
CUDA_VISIBLE_DEVICES=0
```

### 3.3 PAI-EAS Inference Service

1. Create an EAS service:
   - **Service name**: theta-inference
   - **Model source**: OSS
   - **Model path**: oss://theta-bucket/result/0.6B/hatespeech/zero_shot/model/
   - **Image**: custom image (with dependencies)
   - **Startup command**:
```bash
export THETA_BASE=oss://theta-bucket
python ETM/inference_server.py \
    --model_dir oss://theta-bucket/result/0.6B/hatespeech/zero_shot/model \
    --port 8080
```

2. Service configuration:
   - **Instance type**: ecs.gn6i-c4g1.xlarge (1 x T4)
   - **Instance count**: 1 (scale as needed)
   - **Port**: 8080

3. Test service:
```bash
curl -X POST http://<eas-endpoint>/infer \
    -H "Content-Type: application/json" \
    -d '{"texts": ["This is a test document"]}'
```

### 3.4 ECS Backend API

1. Create an ECS instance:
   - **Instance type**: ecs.c6.large (2 vCPU, 4 GB)
   - **OS**: Ubuntu 20.04

2. Deploy API service:
```bash
# Install dependencies
pip install -r ETM/requirements.txt

# Start service (cloud mode)
python ETM/api_server.py \
    --mode cloud \
    --eas_url http://<eas-endpoint>/infer \
    --port 5000
```

3. Production deployment with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 "ETM.api_server:app"
```

## 4. Environment Variables

| Variable | Description | Local | Cloud |
|--------|------|--------|--------|
| THETA_BASE | Base path | /root/autodl-tmp | oss://theta-bucket |
| THETA_MODEL_SIZE | Model size | 0.6B | 0.6B |
| OSS_ENDPOINT | OSS endpoint | - | oss-cn-hangzhou.aliyuncs.com |
| OSS_BUCKET | Bucket name | - | theta-bucket |
| CUDA_VISIBLE_DEVICES | GPU device | 0 | 0 |

## 5. API Documentation

### 5.1 Inference Service (PAI-EAS)

**POST /infer**
```json
// Request
{
    "texts": ["doc1", "doc2", ...],
    "top_k_words": 10
}

// Response
{
    "success": true,
    "theta": [[0.1, 0.2, ...], ...],
    "topics": {
        "0": ["word1", "word2", ...],
        "1": ["word1", "word2", ...]
    },
    "dominant_topics": [0, 1, ...],
    "topic_labels": ["word1 | word2 | word3", ...]
}
```

### 5.2 Backend API (ECS)

**POST /analyze** - Analyze topics for texts
```json
// Request
{
    "texts": ["doc1", "doc2", ...]
}

// Response
{
    "success": true,
    "theta": [...],
    "topics": {...},
    "dominant_topics": [...]
}
```

**GET /datasets** - Get available datasets
```json
{
    "datasets": [
        {"name": "hatespeech", "language": "english", ...}
    ]
}
```

**GET /topics/{dataset}/{mode}** - Get topic words
```json
{
    "topics": {"0": ["word1", ...], ...},
    "num_topics": 20
}
```

**GET /options** - Get configuration options (for frontend dropdown)
```json
{
    "num_topics": [5, 10, 15, 20, ...],
    "vocab_size": [1000, 2000, ...],
    "embedding_modes": ["zero_shot", "supervised", "unsupervised"]
}
```

## 6. Local Testing

Before deploying to the cloud, you can test locally:

```bash
# 1. Set local mode
export THETA_BASE=/root/autodl-tmp

# 2. Start inference service
cd /root/autodl-tmp/THETA/ETM
python inference_server.py \
    --model_dir /root/autodl-tmp/result/0.6B/hatespeech/zero_shot/model \
    --port 8080

# 3. Test inference
curl -X POST http://localhost:8080/infer \
    -H "Content-Type: application/json" \
    -d '{"texts": ["This is a test"]}'

# 4. Start API service (in another terminal)
python api_server.py \
    --mode local \
    --model_dir /root/autodl-tmp/result/0.6B/hatespeech/zero_shot/model \
    --port 5000
```

## 7. File List

Files required for cloud deployment:

```
THETA/
├── ETM/
│   ├── cloud_config.py      # Cloud path configuration
│   ├── config.py            # Configuration management (updated to support OSS)
│   ├── inference_server.py  # Inference service (PAI-EAS)
│   ├── api_server.py        # API service (ECS)
│   ├── pipeline_api.py      # Pipeline API
│   ├── main.py              # Training entry
│   ├── requirements.txt     # Dependency list
│   ├── model/               # Model code
│   ├── bow/                 # BOW generation
│   └── ...
├── embedding/               # Embedding generation code
├── scripts/
│   ├── cloud_train.sh       # Cloud training script
│   ├── cloud_inference.sh   # Cloud inference startup script
│   └── ...
└── DEPLOY.md                # This document
```

## 8. FAQ

### Q1: Failed to read OSS path
PAI-DLC natively supports `oss://` paths. Make sure:
- The job has OSS read/write permissions
- The path format is correct: `oss://bucket-name/path/to/file`

### Q2: Model loading is slow
- The first load requires downloading models from OSS
- Consider using EAS model caching

### Q3: GPU out of memory
- Reduce `batch_size`
- Use a smaller model (0.6B)

### Q4: High inference latency
- Check EAS instance type
- Consider using GPU instances
- Enable EAS autoscaling

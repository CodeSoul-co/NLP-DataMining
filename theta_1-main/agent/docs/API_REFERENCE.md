# THETA Agent API Reference

API documentation for frontend developers to integrate with the THETA topic model analysis system.

## Base URL

```
http://localhost:8000
```

For production deployment on Alibaba Cloud, replace with your server URL.

---

## Authentication

Currently no authentication required. For production, add API key authentication.

---

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Start a new analysis job |
| `/jobs/{job_id}/status` | GET | Get job status |
| `/jobs/{job_id}/result` | GET | Get full analysis results |
| `/jobs/{job_id}/topics` | GET | Get topic list |
| `/chat` | POST | Simple Q&A chat |
| `/api/chat/v2` | POST | Multi-turn conversation |
| `/api/interpret/metrics` | POST | Interpret metrics |
| `/api/interpret/topics` | POST | Interpret topics |
| `/api/interpret/summary` | POST | Generate summary |
| `/api/vision/analyze` | POST | Analyze image with Qwen3 Vision |
| `/api/vision/analyze-chart` | POST | Analyze chart from job results |
| `/api/vision/analyze-job-charts` | POST | Analyze all charts from a job |
| `/api/download/{job_id}/{filename}` | GET | Download result file |

---

## Detailed API Documentation

### 1. Start Analysis Job

**POST** `/analyze`

Start a new topic model analysis job.

**Request:**
```http
POST /analyze
Content-Type: multipart/form-data

file: <CSV or Excel file>
config: {"num_topics": 10, "language": "zh"}
```

**Request Body (multipart/form-data):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | CSV or Excel file with text data |
| config | JSON string | No | Analysis configuration |

**Config Options:**
```json
{
  "num_topics": 10,        // Number of topics (default: 10)
  "language": "zh",        // Language: "zh" or "en"
  "text_column": "text",   // Column name containing text
  "epochs": 100            // Training epochs (default: 100)
}
```

**Response:**
```json
{
  "job_id": "job_20260203_001",
  "status": "started",
  "message": "Analysis job started"
}
```

**Frontend Example (JavaScript):**
```javascript
async function startAnalysis(file, config) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('config', JSON.stringify(config));
  
  const response = await fetch('/analyze', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}
```

---

### 2. Get Job Status

**GET** `/jobs/{job_id}/status`

Check the status of an analysis job.

**Response:**
```json
{
  "job_id": "job_20260203_001",
  "status": "processing",
  "progress": 60,
  "current_step": "Training ETM model...",
  "started_at": "2026-02-03T10:30:00",
  "estimated_remaining": "2 minutes"
}
```

**Status Values:**
| Status | Description |
|--------|-------------|
| `started` | Job has been created |
| `processing` | Job is running |
| `completed` | Job finished successfully |
| `failed` | Job failed with error |

**Frontend Example (Polling):**
```javascript
async function pollStatus(jobId, onProgress, onComplete) {
  const response = await fetch(`/jobs/${jobId}/status`);
  const data = await response.json();
  
  if (data.status === 'completed') {
    onComplete(data);
  } else if (data.status === 'failed') {
    throw new Error(data.error);
  } else {
    onProgress(data.progress, data.current_step);
    setTimeout(() => pollStatus(jobId, onProgress, onComplete), 5000);
  }
}
```

---

### 3. Get Analysis Results

**GET** `/jobs/{job_id}/result`

Get full analysis results.

**Response:**
```json
{
  "job_id": "job_20260203_001",
  "status": "completed",
  "completed_at": "2026-02-03T10:35:00",
  "metrics": {
    "topic_coherence_npmi_avg": 0.15,
    "topic_diversity": 0.85,
    "perplexity": 450.5
  },
  "topics": [
    {
      "id": 0,
      "keywords": ["technology", "innovation", "AI", "development", "future"],
      "weight": 0.12
    },
    {
      "id": 1,
      "keywords": ["business", "market", "growth", "strategy", "company"],
      "weight": 0.10
    }
  ],
  "charts": {
    "topic_distribution": "/api/download/job_20260203_001/topic_distribution.png",
    "wordcloud_0": "/api/download/job_20260203_001/wordcloud_0.png"
  },
  "files": {
    "report": "/api/download/job_20260203_001/report.docx"
  }
}
```

---

### 4. Chat with Agent (Simple)

**POST** `/chat`

Simple Q&A about analysis results.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "question": "What does Topic 1 discuss?"
}
```

**Response:**
```json
{
  "answer": "Based on the keywords, Topic 1 primarily discusses technology and innovation...",
  "job_id": "job_20260203_001"
}
```

---

### 5. Chat with Agent (Multi-turn)

**POST** `/api/chat/v2`

Multi-turn conversation with context memory.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "message": "What does Topic 1 discuss?",
  "session_id": "session_001"  // Optional, auto-generated if not provided
}
```

**Response:**
```json
{
  "answer": "Based on the keywords, Topic 1 primarily discusses technology and innovation...",
  "session_id": "session_001",
  "job_id": "job_20260203_001"
}
```

**Frontend Example (Chat Interface):**
```javascript
class ChatClient {
  constructor(jobId) {
    this.jobId = jobId;
    this.sessionId = null;
  }
  
  async sendMessage(message) {
    const response = await fetch('/api/chat/v2', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id: this.jobId,
        message: message,
        session_id: this.sessionId
      })
    });
    
    const data = await response.json();
    this.sessionId = data.session_id;  // Store for next message
    return data.answer;
  }
}

// Usage
const chat = new ChatClient('job_20260203_001');
const answer1 = await chat.sendMessage('What does Topic 1 discuss?');
const answer2 = await chat.sendMessage('How does it relate to Topic 2?');  // Remembers context
```

---

### 6. Interpret Metrics

**POST** `/api/interpret/metrics`

Get human-readable interpretation of evaluation metrics.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "language": "zh"  // "zh" or "en"
}
```

**Response:**
```json
{
  "interpretations": [
    {
      "metric": "topic_coherence_npmi_avg",
      "name": "Topic Coherence (NPMI)",
      "value": 0.15,
      "quality": "good",
      "interpretation": "Topics have meaningful word associations. The model has identified coherent themes."
    },
    {
      "metric": "topic_diversity",
      "name": "Topic Diversity",
      "value": 0.85,
      "quality": "excellent",
      "interpretation": "Topics are well differentiated from each other with minimal overlap."
    }
  ],
  "overall_quality": "good",
  "summary": "The model shows good overall quality with coherent and diverse topics."
}
```

---

### 7. Interpret Topics

**POST** `/api/interpret/topics`

Get semantic interpretation of discovered topics.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "language": "zh"
}
```

**Response:**
```json
{
  "topics": [
    {
      "id": 0,
      "keywords": ["technology", "innovation", "AI"],
      "interpretation": "This topic focuses on technological advancement and artificial intelligence development.",
      "suggested_label": "Technology & AI"
    },
    {
      "id": 1,
      "keywords": ["business", "market", "growth"],
      "interpretation": "This topic discusses business strategies and market growth.",
      "suggested_label": "Business Strategy"
    }
  ]
}
```

---

### 8. Generate Summary

**POST** `/api/interpret/summary`

Generate an executive summary of the analysis.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "language": "zh"
}
```

**Response:**
```json
{
  "summary": "## Analysis Summary\n\nThe topic model analysis identified 10 distinct topics...",
  "key_findings": [
    "Technology and AI is the dominant topic (12% of documents)",
    "Business strategy topics show strong coherence",
    "Customer feedback topics reveal service quality concerns"
  ],
  "recommendations": [
    "Focus on technology-related content for engagement",
    "Address customer service issues identified in Topic 5"
  ]
}
```

---

### 9. Vision Analysis - Analyze Image

**POST** `/api/vision/analyze`

Analyze any image using Qwen3 Vision API.

**Request:**
```json
{
  "image_url": "https://example.com/image.jpg",
  "question": "What is shown in this image?",
  "language": "zh"
}
```

**Response:**
```json
{
  "success": true,
  "image_url": "https://example.com/image.jpg",
  "question": "What is shown in this image?",
  "answer": "The image shows a bar chart displaying topic distribution..."
}
```

---

### 10. Vision Analysis - Analyze Chart

**POST** `/api/vision/analyze-chart`

Analyze a specific chart from job results.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "chart_name": "wordcloud_0.png",
  "analysis_type": "wordcloud",
  "language": "zh"
}
```

**Analysis Types:**
| Type | Description |
|------|-------------|
| `general` | General chart analysis |
| `wordcloud` | Word cloud specific analysis |
| `distribution` | Distribution chart analysis |
| `heatmap` | Heatmap/correlation analysis |

**Response:**
```json
{
  "success": true,
  "analysis_type": "wordcloud",
  "chart_path": "/root/autodl-tmp/result/job_20260203_001/wordcloud_0.png",
  "analysis": "This word cloud shows the key terms for Topic 0. The most prominent words are..."
}
```

---

### 11. Vision Analysis - Analyze All Job Charts

**POST** `/api/vision/analyze-job-charts`

Analyze all charts from a job at once.

**Request:**
```json
{
  "job_id": "job_20260203_001",
  "language": "zh"
}
```

**Response:**
```json
{
  "job_id": "job_20260203_001",
  "charts_analyzed": 5,
  "results": [
    {
      "success": true,
      "filename": "wordcloud_0.png",
      "analysis_type": "wordcloud",
      "analysis": "..."
    },
    {
      "success": true,
      "filename": "topic_distribution.png",
      "analysis_type": "distribution",
      "analysis": "..."
    }
  ]
}
```

**Frontend Example:**
```javascript
async function analyzeJobCharts(jobId) {
  const response = await fetch('/api/vision/analyze-job-charts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, language: 'zh' })
  });
  return await response.json();
}
```

---

### 12. Download Files

**GET** `/api/download/{job_id}/{filename}`

Download result files (reports, charts, etc.).

**Available Files:**
| Filename | Description |
|----------|-------------|
| `report.docx` | Generated Word report |
| `topic_distribution.png` | Topic distribution chart |
| `wordcloud_{n}.png` | Word cloud for topic n |
| `metrics.json` | Raw metrics data |
| `topic_words.json` | Topic keywords data |

**Frontend Example:**
```javascript
function downloadReport(jobId) {
  window.open(`/api/download/${jobId}/report.docx`, '_blank');
}
```

---

## Error Handling

All endpoints return errors in this format:

```json
{
  "error": true,
  "message": "Job not found",
  "code": "JOB_NOT_FOUND"
}
```

**Common Error Codes:**
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `JOB_FAILED` | 500 | Analysis job failed |
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `LLM_ERROR` | 500 | LLM API call failed |

**Frontend Error Handling:**
```javascript
async function apiCall(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  
  if (data.error) {
    throw new Error(data.message);
  }
  
  return data;
}
```

---

## WebSocket (Future)

For real-time updates, WebSocket support is planned:

```javascript
// Future implementation
const ws = new WebSocket('ws://localhost:8000/ws/jobs/{job_id}');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateProgress(data.progress);
};
```

---

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| `/analyze` | 10 requests/minute |
| `/chat`, `/api/chat/v2` | 60 requests/minute |
| Other endpoints | 100 requests/minute |

---

## CORS

CORS is enabled for all origins by default. For production, configure allowed origins in the server settings.

---

## Quick Start for Frontend Developers

```javascript
// 1. Start analysis
const { job_id } = await fetch('/analyze', {
  method: 'POST',
  body: formData
}).then(r => r.json());

// 2. Poll for completion
while (true) {
  const status = await fetch(`/jobs/${job_id}/status`).then(r => r.json());
  if (status.status === 'completed') break;
  await new Promise(r => setTimeout(r, 5000));
}

// 3. Get results
const results = await fetch(`/jobs/${job_id}/result`).then(r => r.json());

// 4. Chat with agent
const chat = await fetch('/api/chat/v2', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ job_id, message: 'Summarize the findings' })
}).then(r => r.json());
```

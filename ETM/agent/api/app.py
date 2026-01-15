"""
Topic-Aware Agent API Interface

Provides RESTful API interface for interacting with the Topic-Aware Agent.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Add project root directory to path
# 从 api/app.py 向上找到项目根目录 (THETA/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).parents[2]))

# Import Agent components
# 尝试两种导入方式以兼容不同的运行环境
try:
    from ETM.agent.core.topic_aware_agent import TopicAwareAgent
    from ETM.agent.utils.config import AgentConfig
except ImportError:
    # 如果从项目根目录导入失败，尝试相对导入
    from agent.core.topic_aware_agent import TopicAwareAgent
    from agent.utils.config import AgentConfig


# Define request and response models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    use_tools: bool = True
    use_topic_enhancement: bool = True


class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str
    message: str
    dominant_topics: List[List[float]]
    topic_shifted: bool


class DocumentRequest(BaseModel):
    """Document addition request model"""
    text: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Document addition response model"""
    document_id: int
    topic_dist: List[float]


class TopicRequest(BaseModel):
    """Topic request model"""
    topic_id: int
    top_k: int = 10


class TopicWordsResponse(BaseModel):
    """Topic words response model"""
    topic_id: int
    words: List[Dict[str, Any]]


class CreateTaskRequest(BaseModel):
    """创建任务请求模型"""
    dataset: str
    mode: str
    num_topics: Optional[int] = 50
    vocab_size: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    hidden_dim: Optional[int] = None
    dev_mode: Optional[bool] = True


# Create FastAPI application
app = FastAPI(
    title="ETM Agent API",
    description="Topic-Aware Agent API Interface",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent instance
agent = None


# Dependency: Get Agent instance
def get_agent():
    global agent
    if agent is None:
        # Load configuration from environment variables
        config = AgentConfig.from_env()
        
        # Use default values if not set in environment variables
        if not config.etm_model_path:
            config.etm_model_path = os.environ.get(
                "ETM_MODEL_PATH",
                "/root/autodl-tmp/ETM/outputs/models/etm_model.pt"
            )
        
        if not config.vocab_path:
            config.vocab_path = os.environ.get(
                "VOCAB_PATH",
                "/root/autodl-tmp/ETM/outputs/engine_a/vocab.json"
            )
        
        # Initialize Agent
        agent = TopicAwareAgent(config, dev_mode=True)
    
    return agent


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent=Depends(get_agent)):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process user input
        result = agent.process(
            user_input=request.message,
            session_id=session_id,
            use_tools=request.use_tools,
            use_topic_enhancement=request.use_topic_enhancement
        )
        
        # Build response
        response = {
            "session_id": session_id,
            "message": result["content"],
            "dominant_topics": [[topic_id, float(weight)] for topic_id, weight in result["dominant_topics"]],
            "topic_shifted": result["topic_shifted"]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add document endpoint
@app.post("/documents", response_model=DocumentResponse)
def add_document(request: DocumentRequest, agent=Depends(get_agent)):
    try:
        # Create document
        document = {
            "text": request.text,
            "metadata": request.metadata or {}
        }
        
        # Get topic distribution for the text
        topic_dist = agent.get_topic_distribution(request.text)
        
        # Add to knowledge base
        document_id = agent.add_document(document, topic_dist=topic_dist)
        
        # Build response
        response = {
            "document_id": document_id,
            "topic_dist": topic_dist.tolist()
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query knowledge base endpoint
@app.post("/query")
def query_knowledge(query: str = Body(..., embed=True), top_k: int = 5, agent=Depends(get_agent)):
    try:
        # Query knowledge base
        results = agent.query_knowledge(query, top_k=top_k)
        
        # Build response
        response = [
            {
                "document_id": doc_id,
                "score": float(score),
                "content": document
            }
            for doc_id, score, document in results
        ]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get topic words endpoint
@app.get("/topics/{topic_id}/words", response_model=TopicWordsResponse)
def get_topic_words(topic_id: int, top_k: int = 10, agent=Depends(get_agent)):
    try:
        # Get topic words
        topic_words = agent.get_topic_words(topic_id, top_k=top_k)
        
        # Build response
        response = {
            "topic_id": topic_id,
            "words": [
                {
                    "word": word,
                    "weight": float(weight)
                }
                for word, weight in topic_words
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get text topic distribution endpoint
@app.post("/analyze")
def analyze_text(text: str = Body(..., embed=True), agent=Depends(get_agent)):
    try:
        # Get topic distribution
        topic_dist = agent.get_topic_distribution(text)
        
        # Get dominant topics
        dominant_topics = agent.get_dominant_topics(topic_dist)
        
        # Build response
        response = {
            "topic_dist": topic_dist.tolist(),
            "dominant_topics": [
                {
                    "topic_id": topic_id,
                    "weight": float(weight),
                    "words": [
                        {
                            "word": word,
                            "weight": float(w)
                        }
                        for word, w in agent.get_topic_words(topic_id, top_k=5)
                    ]
                }
                for topic_id, weight in dominant_topics
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# 数据集和结果管理端点
# =====================================================

# Mock 数据集信息（实际应从文件系统或数据库读取）
mock_datasets = [
    {
        "name": "socialTwitter",
        "path": "data/socialTwitter",
        "size": 1024000,
        "columns": ["text", "label"],
        "has_labels": True,
        "language": "english"
    },
    {
        "name": "hatespeech",
        "path": "data/hatespeech",
        "size": 512000,
        "columns": ["text", "label"],
        "has_labels": True,
        "language": "english"
    }
]

# Mock 结果信息
mock_results = [
    {
        "dataset": "socialTwitter",
        "mode": "zero_shot",
        "timestamp": "2024-01-15T10:30:00",
        "num_topics": 20,
        "metrics": {
            "coherence": 0.45,
            "diversity": 0.72,
            "quality": 0.58
        }
    }
]

@app.get("/api/datasets")
def get_datasets():
    """获取所有数据集"""
    return mock_datasets

@app.get("/api/datasets/{name}")
def get_dataset(name: str):
    """获取单个数据集信息"""
    dataset = next((d for d in mock_datasets if d["name"] == name), None)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {name} not found")
    return dataset

@app.get("/api/results")
def get_results():
    """获取所有分析结果"""
    return mock_results

@app.get("/api/results/{dataset}/{mode}")
def get_result(dataset: str, mode: str):
    """获取特定数据集和模式的结果"""
    result = next((r for r in mock_results if r["dataset"] == dataset and r["mode"] == mode), None)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result for {dataset}/{mode} not found")
    return result

@app.get("/api/results/{dataset}/{mode}/metrics")
def get_result_metrics(dataset: str, mode: str):
    """获取评估指标"""
    result = next((r for r in mock_results if r["dataset"] == dataset and r["mode"] == mode), None)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result for {dataset}/{mode} not found")
    return result.get("metrics", {})

@app.get("/api/results/{dataset}/{mode}/topic-words")
def get_topic_words_api(dataset: str, mode: str, top_k: int = 10):
    """获取主题词"""
    # Mock 主题词数据
    return {
        "topic_0": ["social", "media", "twitter", "post", "share"],
        "topic_1": ["news", "breaking", "update", "report", "story"],
        "topic_2": ["politics", "government", "election", "vote", "policy"],
        "topic_3": ["technology", "innovation", "digital", "tech", "software"],
        "topic_4": ["health", "medical", "covid", "vaccine", "hospital"]
    }

@app.get("/api/results/{dataset}/{mode}/visualizations")
def get_visualizations(dataset: str, mode: str):
    """获取可视化列表"""
    return [
        {"name": "topic_distribution", "type": "bar", "path": f"/visualizations/{dataset}/{mode}/topic_dist.png"},
        {"name": "word_cloud", "type": "image", "path": f"/visualizations/{dataset}/{mode}/wordcloud.png"},
        {"name": "topic_heatmap", "type": "heatmap", "path": f"/visualizations/{dataset}/{mode}/heatmap.png"}
    ]

# =====================================================
# Task management endpoints
# =====================================================
# 简单的任务存储（实际应该使用数据库）
tasks_storage: Dict[str, Dict[str, Any]] = {}

@app.get("/api/tasks")
def get_tasks():
    """获取所有任务"""
    try:
        return list(tasks_storage.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    """获取单个任务"""
    try:
        if task_id not in tasks_storage:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return tasks_storage[task_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks")
async def create_task(request: CreateTaskRequest):
    """创建新任务并启动模拟训练"""
    try:
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "dataset": request.dataset,
            "mode": request.mode,
            "num_topics": request.num_topics or 50,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        tasks_storage[task_id] = task
        
        # 启动后台任务模拟训练进度
        asyncio.create_task(simulate_training(task_id))
        
        return task
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def simulate_training(task_id: str):
    """模拟训练进度"""
    try:
        # 等待一小段时间后开始
        await asyncio.sleep(1)
        
        if task_id not in tasks_storage:
            return
        
        # 更新状态为处理中
        tasks_storage[task_id]["status"] = "processing"
        tasks_storage[task_id]["progress"] = 5
        tasks_storage[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 模拟训练步骤
        steps = [
            ("加载数据集", 10),
            ("预处理文本", 20),
            ("构建词汇表", 30),
            ("初始化模型", 40),
            ("训练 Epoch 1/5", 50),
            ("训练 Epoch 2/5", 60),
            ("训练 Epoch 3/5", 70),
            ("训练 Epoch 4/5", 80),
            ("训练 Epoch 5/5", 90),
            ("生成主题词", 95),
            ("保存结果", 100),
        ]
        
        for step_name, progress in steps:
            await asyncio.sleep(2)  # 每步等待2秒
            
            if task_id not in tasks_storage:
                return
            
            # 检查是否被取消
            if tasks_storage[task_id]["status"] == "cancelled":
                return
            
            tasks_storage[task_id]["progress"] = progress
            tasks_storage[task_id]["current_step"] = step_name
            tasks_storage[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 训练完成
        if task_id in tasks_storage and tasks_storage[task_id]["status"] != "cancelled":
            tasks_storage[task_id]["status"] = "completed"
            tasks_storage[task_id]["progress"] = 100
            tasks_storage[task_id]["completed_at"] = datetime.now().isoformat()
            tasks_storage[task_id]["updated_at"] = datetime.now().isoformat()
            
            # 添加模拟的结果数据
            tasks_storage[task_id]["metrics"] = {
                "coherence": 0.456,
                "diversity": 0.789,
                "perplexity": 123.45
            }
            tasks_storage[task_id]["topic_words"] = {
                "topic_0": ["数据", "分析", "模型", "训练", "学习"],
                "topic_1": ["用户", "界面", "交互", "设计", "体验"],
                "topic_2": ["系统", "服务", "接口", "请求", "响应"],
            }
    except Exception as e:
        if task_id in tasks_storage:
            tasks_storage[task_id]["status"] = "failed"
            tasks_storage[task_id]["error_message"] = str(e)
            tasks_storage[task_id]["updated_at"] = datetime.now().isoformat()

@app.delete("/api/tasks/{task_id}")
def cancel_task(task_id: str):
    """取消任务"""
    try:
        if task_id not in tasks_storage:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        tasks_storage[task_id]["status"] = "cancelled"
        return {"message": f"Task {task_id} cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time task updates
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time task updates and notifications.
    """
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connected successfully"
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle subscription requests
                if data.get("type") == "subscribe" and data.get("task_id"):
                    task_id = data["task_id"]
                    await websocket.send_json({
                        "type": "subscribed",
                        "task_id": task_id,
                        "message": f"Subscribed to task {task_id}"
                    })
                
                # Handle ping/pong for keepalive
                elif data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": str(uuid.uuid4())
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Main function
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Start server with WebSocket support
    # 明确指定使用 websockets（如果可用）
    try:
        import websockets
        ws_protocol = "websockets"
    except ImportError:
        try:
            import wsproto
            ws_protocol = "wsproto"
        except ImportError:
            ws_protocol = "auto"
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        ws=ws_protocol,
        log_level="info"
    )

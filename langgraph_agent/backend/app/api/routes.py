"""
API Routes
REST endpoints for the THETA Agent System
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

from ..schemas.agent import TaskRequest, TaskResponse, TaskStatus
from ..schemas.data import DatasetInfo, ResultInfo, VisualizationInfo, ProjectInfo, MetricsResponse
from ..agents.etm_agent import etm_agent
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", tags=["health"])
async def root():
    """Health check endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health", tags=["health"])
async def health_check():
    """Detailed health check"""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_id": settings.GPU_ID,
        "etm_dir_exists": settings.ETM_DIR.exists(),
        "data_dir_exists": settings.DATA_DIR.exists(),
        "result_dir_exists": settings.RESULT_DIR.exists()
    }


@router.get("/project", response_model=ProjectInfo, tags=["project"])
async def get_project_info():
    """Get project overview information"""
    import torch
    
    datasets = settings.get_available_datasets()
    results = settings.get_available_results()
    active_tasks = len([t for t in etm_agent.get_all_tasks().values() 
                       if t.get("status") == "running"])
    
    recent_results = []
    for r in results[:5]:
        recent_results.append(ResultInfo(
            dataset=r["dataset"],
            mode=r["mode"],
            timestamp="",
            path=r["path"]
        ))
    
    return ProjectInfo(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        datasets_count=len(datasets),
        results_count=len(results),
        active_tasks=active_tasks,
        gpu_available=torch.cuda.is_available(),
        gpu_id=settings.GPU_ID,
        recent_results=recent_results
    )


@router.get("/datasets", response_model=List[DatasetInfo], tags=["data"])
async def list_datasets():
    """List all available datasets"""
    datasets = []
    
    if not settings.DATA_DIR.exists():
        return datasets
    
    for dataset_dir in settings.DATA_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            continue
        
        info = DatasetInfo(
            name=dataset_dir.name,
            path=str(dataset_dir),
            has_labels=False
        )
        
        for csv_file in csv_files:
            if "text_only" in csv_file.name or "cleaned" in csv_file.name:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file, nrows=5)
                    info.size = len(pd.read_csv(csv_file))
                    info.columns = df.columns.tolist()
                    info.has_labels = any(col in df.columns for col in ['label', 'Label', 'labels'])
                except Exception:
                    pass
                break
        
        datasets.append(info)
    
    return datasets


@router.get("/datasets/{dataset_name}", response_model=DatasetInfo, tags=["data"])
async def get_dataset_info(dataset_name: str):
    """Get detailed information about a specific dataset"""
    dataset_dir = settings.DATA_DIR / dataset_name
    
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No CSV files found in dataset '{dataset_name}'")
    
    import pandas as pd
    
    csv_file = csv_files[0]
    for f in csv_files:
        if "text_only" in f.name or "cleaned" in f.name:
            csv_file = f
            break
    
    df = pd.read_csv(csv_file)
    
    return DatasetInfo(
        name=dataset_name,
        path=str(dataset_dir),
        size=len(df),
        columns=df.columns.tolist(),
        has_labels=any(col in df.columns for col in ['label', 'Label', 'labels']),
        language="english"
    )


@router.get("/results", response_model=List[ResultInfo], tags=["results"])
async def list_results():
    """List all training results"""
    results = []
    
    for result in settings.get_available_results():
        result_path = Path(result["path"])
        model_dir = result_path / "model"
        eval_dir = result_path / "evaluation"
        viz_dir = result_path / "visualization"
        
        info = ResultInfo(
            dataset=result["dataset"],
            mode=result["mode"],
            timestamp="",
            path=result["path"],
            has_model=model_dir.exists() and any(model_dir.glob("*.pt")),
            has_theta=(model_dir / "theta_*.npy").parent.exists() if model_dir.exists() else False,
            has_beta=(model_dir / "beta_*.npy").parent.exists() if model_dir.exists() else False,
            has_topic_words=any(model_dir.glob("topic_words_*.json")) if model_dir.exists() else False,
            has_visualizations=viz_dir.exists() and any(viz_dir.glob("*.png"))
        )
        
        if model_dir.exists():
            theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
            if theta_files:
                info.timestamp = theta_files[0].stem.replace("theta_", "")
        
        if eval_dir.exists():
            metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
            if metrics_files:
                try:
                    with open(metrics_files[0]) as f:
                        info.metrics = json.load(f)
                except Exception:
                    pass
        
        results.append(info)
    
    return results


@router.get("/results/{dataset}/{mode}", response_model=ResultInfo, tags=["results"])
async def get_result_info(dataset: str, mode: str):
    """Get detailed information about a specific result"""
    result_path = settings.get_result_path(dataset, mode)
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Result not found for {dataset}/{mode}")
    
    model_dir = result_path / "model"
    eval_dir = result_path / "evaluation"
    viz_dir = result_path / "visualization"
    
    info = ResultInfo(
        dataset=dataset,
        mode=mode,
        timestamp="",
        path=str(result_path),
        has_model=any(model_dir.glob("*.pt")) if model_dir.exists() else False,
        has_theta=any(model_dir.glob("theta_*.npy")) if model_dir.exists() else False,
        has_beta=any(model_dir.glob("beta_*.npy")) if model_dir.exists() else False,
        has_topic_words=any(model_dir.glob("topic_words_*.json")) if model_dir.exists() else False,
        has_visualizations=any(viz_dir.glob("*.png")) if viz_dir.exists() else False
    )
    
    if model_dir.exists():
        theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
        if theta_files:
            info.timestamp = theta_files[0].stem.replace("theta_", "")
            
            history_file = model_dir / f"training_history_{info.timestamp}.json"
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    info.epochs_trained = history.get("epochs_trained")
    
    if eval_dir.exists():
        metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
        if metrics_files:
            with open(metrics_files[0]) as f:
                info.metrics = json.load(f)
    
    return info


@router.get("/results/{dataset}/{mode}/metrics", response_model=MetricsResponse, tags=["results"])
async def get_result_metrics(dataset: str, mode: str):
    """Get detailed metrics for a result"""
    result_path = settings.get_result_path(dataset, mode)
    eval_dir = result_path / "evaluation"
    
    if not eval_dir.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
    if not metrics_files:
        raise HTTPException(status_code=404, detail="No metrics files found")
    
    with open(metrics_files[0]) as f:
        metrics = json.load(f)
    
    timestamp = metrics_files[0].stem.replace("metrics_", "")
    
    return MetricsResponse(
        dataset=dataset,
        mode=mode,
        timestamp=timestamp,
        topic_coherence_avg=metrics.get("topic_coherence_avg"),
        topic_coherence_per_topic=metrics.get("topic_coherence_per_topic"),
        topic_diversity_td=metrics.get("topic_diversity_td"),
        topic_diversity_irbo=metrics.get("topic_diversity_irbo"),
        additional=metrics
    )


@router.get("/results/{dataset}/{mode}/topic-words", tags=["results"])
async def get_topic_words(dataset: str, mode: str, top_k: int = Query(default=10, ge=1, le=50)):
    """Get topic words for a result"""
    result_path = settings.get_result_path(dataset, mode)
    model_dir = result_path / "model"
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model results not found")
    
    topic_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
    if not topic_files:
        raise HTTPException(status_code=404, detail="Topic words not found")
    
    with open(topic_files[0]) as f:
        raw_data = json.load(f)
    
    # 处理两种格式：
    # 格式1: 字典 {"0": ["word1", "word2", ...], "1": [...]}
    # 格式2: 数组 [[0, [["word1", 0.5], ["word2", 0.3]]], [1, [...]]]
    if isinstance(raw_data, dict):
        topic_words = raw_data
    elif isinstance(raw_data, list):
        # 转换数组格式为字典格式
        topic_words = {}
        for item in raw_data:
            if isinstance(item, list) and len(item) >= 2:
                topic_id = str(item[0])
                words_data = item[1]
                # words_data 可能是 [["word", score], ...] 或 ["word", ...]
                if words_data and isinstance(words_data[0], list):
                    topic_words[topic_id] = [w[0] for w in words_data[:top_k]]
                else:
                    topic_words[topic_id] = words_data[:top_k]
    else:
        topic_words = {}
    
    # 如果需要截断
    if top_k < 20 and isinstance(topic_words, dict):
        topic_words = {k: (v[:top_k] if isinstance(v, list) else v) for k, v in topic_words.items()}
    
    return topic_words


@router.get("/results/{dataset}/{mode}/visualizations", response_model=List[VisualizationInfo], tags=["results"])
async def list_visualizations(dataset: str, mode: str):
    """List all visualizations for a result"""
    result_path = settings.get_result_path(dataset, mode)
    viz_dir = result_path / "visualization"
    
    if not viz_dir.exists():
        return []
    
    visualizations = []
    for viz_file in viz_dir.iterdir():
        if viz_file.is_file():
            file_type = "image" if viz_file.suffix in [".png", ".jpg", ".jpeg"] else \
                       "html" if viz_file.suffix == ".html" else "other"
            
            visualizations.append(VisualizationInfo(
                name=viz_file.name,
                path=str(viz_file),
                type=file_type,
                size=viz_file.stat().st_size
            ))
    
    return visualizations


@router.get("/results/{dataset}/{mode}/visualizations/{filename}", tags=["results"])
async def get_visualization(dataset: str, mode: str, filename: str):
    """Get a specific visualization file"""
    result_path = settings.get_result_path(dataset, mode)
    viz_path = result_path / "visualization" / filename
    
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(viz_path)


@router.post("/tasks", response_model=TaskResponse, tags=["tasks"])
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new ETM pipeline task"""
    dataset_dir = settings.DATA_DIR / request.dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found")
    
    # 在非模拟模式下检查 embeddings
    if not settings.SIMULATION_MODE:
        embeddings_dir = settings.get_result_path(request.dataset, request.mode) / "embeddings"
        emb_file = embeddings_dir / f"{request.dataset}_{request.mode}_embeddings.npy"
        if not emb_file.exists():
            raise HTTPException(
                status_code=400, 
                detail=f"Embeddings not found for {request.dataset}/{request.mode}. Please generate embeddings first."
            )
    
    from ..schemas.agent import AgentState
    import uuid
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    initial_state = {
        "task_id": task_id,
        "dataset": request.dataset,
        "mode": request.mode,
        "num_topics": request.num_topics,
        "status": "pending",
        "current_step": "preprocess",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    if settings.SIMULATION_MODE:
        # 模拟模式：使用模拟训练
        async def run_simulated_task():
            await simulate_training_pipeline(task_id, request)
        background_tasks.add_task(run_simulated_task)
    else:
        # 真实模式：运行实际的 LangGraph pipeline
        async def run_task():
            await etm_agent.run_pipeline(request)
        background_tasks.add_task(run_task)
    
    # 存储初始状态
    etm_agent.active_tasks[task_id] = initial_state
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        current_step="preprocess",
        progress=0,
        created_at=initial_state["created_at"],
        updated_at=initial_state["updated_at"]
    )


async def simulate_training_pipeline(task_id: str, request: TaskRequest):
    """模拟训练流水线（用于开发和演示）"""
    import asyncio
    
    steps = [
        ("preprocess", "加载数据集", 5, 10),
        ("preprocess", "构建词汇表", 10, 15),
        ("preprocess", "生成BOW矩阵", 15, 20),
        ("embedding", "加载文档嵌入", 20, 25),
        ("embedding", "验证嵌入维度", 25, 30),
        ("training", "初始化ETM模型", 30, 35),
        ("training", "训练 Epoch 1/10", 35, 40),
        ("training", "训练 Epoch 2/10", 40, 45),
        ("training", "训练 Epoch 3/10", 45, 50),
        ("training", "训练 Epoch 4/10", 50, 55),
        ("training", "训练 Epoch 5/10", 55, 60),
        ("training", "训练 Epoch 6/10", 60, 65),
        ("training", "训练 Epoch 7/10", 65, 70),
        ("training", "训练 Epoch 8/10", 70, 75),
        ("training", "训练 Epoch 9/10", 75, 80),
        ("training", "训练 Epoch 10/10", 80, 85),
        ("evaluation", "计算主题一致性", 85, 90),
        ("evaluation", "计算主题多样性", 90, 92),
        ("visualization", "生成主题词云", 92, 95),
        ("visualization", "生成主题分布图", 95, 98),
        ("visualization", "生成交互式可视化", 98, 100),
    ]
    
    try:
        # 更新为运行中状态
        etm_agent.active_tasks[task_id]["status"] = "running"
        etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logs = []
        for step_name, message, progress_start, progress_end in steps:
            await asyncio.sleep(1)  # 每步等待1秒
            
            if task_id not in etm_agent.active_tasks:
                return
            
            # 检查是否被取消
            if etm_agent.active_tasks[task_id].get("status") == "cancelled":
                return
            
            etm_agent.active_tasks[task_id]["current_step"] = step_name
            etm_agent.active_tasks[task_id]["progress"] = progress_end
            etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            logs.append({
                "step": step_name,
                "status": "completed",
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            etm_agent.active_tasks[task_id]["logs"] = logs
        
        # 训练完成
        if task_id in etm_agent.active_tasks and etm_agent.active_tasks[task_id].get("status") != "cancelled":
            etm_agent.active_tasks[task_id]["status"] = "completed"
            etm_agent.active_tasks[task_id]["progress"] = 100
            etm_agent.active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # 添加模拟的结果数据
            num_topics = request.num_topics or 20
            etm_agent.active_tasks[task_id]["metrics"] = {
                "topic_coherence_avg": 0.456 + (num_topics / 100),
                "topic_diversity_td": 0.789 - (num_topics / 200),
                "topic_diversity_irbo": 0.85,
                "perplexity": 123.45 - num_topics
            }
            
            # 生成模拟的主题词
            topic_words = {}
            sample_words = [
                ["数据", "分析", "模型", "训练", "学习", "算法", "预测", "特征", "样本", "结果"],
                ["用户", "界面", "交互", "设计", "体验", "功能", "操作", "反馈", "流程", "优化"],
                ["系统", "服务", "接口", "请求", "响应", "处理", "调用", "返回", "状态", "错误"],
                ["网络", "通信", "协议", "连接", "传输", "安全", "加密", "认证", "授权", "访问"],
                ["文本", "语言", "语义", "词汇", "句子", "段落", "文档", "摘要", "分类", "聚类"],
            ]
            for i in range(min(num_topics, 20)):
                topic_words[str(i)] = sample_words[i % len(sample_words)]
            
            etm_agent.active_tasks[task_id]["topic_words"] = topic_words
            etm_agent.active_tasks[task_id]["visualization_paths"] = [
                f"/static/results/{request.dataset}/{request.mode}/visualization/topic_words.png",
                f"/static/results/{request.dataset}/{request.mode}/visualization/topic_similarity.png",
                f"/static/results/{request.dataset}/{request.mode}/visualization/doc_topics.png",
            ]
            
    except Exception as e:
        if task_id in etm_agent.active_tasks:
            etm_agent.active_tasks[task_id]["status"] = "failed"
            etm_agent.active_tasks[task_id]["error_message"] = str(e)
            etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()


@router.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks():
    """List all tasks"""
    tasks = []
    for task_id, state in etm_agent.get_all_tasks().items():
        # 处理时间字段（可能是 datetime 对象或字符串）
        created_at = state.get("created_at", datetime.now())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = state.get("updated_at", datetime.now())
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        completed_at = state.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        
        tasks.append(TaskResponse(
            task_id=task_id,
            status=TaskStatus(state.get("status", "pending")),
            current_step=state.get("current_step"),
            progress=_calculate_progress(state),
            metrics=state.get("metrics"),
            created_at=created_at,
            updated_at=updated_at,
            completed_at=completed_at,
            error_message=state.get("error_message")
        ))
    return tasks


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str):
    """Get task status"""
    state = etm_agent.get_task_status(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 处理时间字段（可能是 datetime 对象或字符串）
    created_at = state.get("created_at", datetime.now())
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    updated_at = state.get("updated_at", datetime.now())
    if isinstance(updated_at, str):
        updated_at = datetime.fromisoformat(updated_at)
    
    completed_at = state.get("completed_at")
    if completed_at and isinstance(completed_at, str):
        completed_at = datetime.fromisoformat(completed_at)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus(state.get("status", "pending")),
        current_step=state.get("current_step"),
        progress=_calculate_progress(state),
        metrics=state.get("metrics"),
        topic_words=state.get("topic_words"),
        visualization_paths=state.get("visualization_paths"),
        created_at=created_at,
        updated_at=updated_at,
        completed_at=completed_at,
        error_message=state.get("error_message")
    )


@router.delete("/tasks/{task_id}", tags=["tasks"])
async def cancel_task(task_id: str):
    """Cancel a running task"""
    success = await etm_agent.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")
    return {"message": "Task cancelled", "task_id": task_id}


def _calculate_progress(state: dict) -> float:
    """Calculate task progress percentage"""
    # 如果直接有 progress 字段（模拟模式），直接使用
    if "progress" in state and state["progress"] is not None:
        return float(state["progress"])
    
    # 否则根据完成的步骤计算
    steps = ["preprocess", "embedding", "training", "evaluation", "visualization"]
    completed_flags = [
        state.get("preprocess_completed", False),
        state.get("embedding_completed", False),
        state.get("training_completed", False),
        state.get("evaluation_completed", False),
        state.get("visualization_completed", False)
    ]
    
    completed = sum(completed_flags)
    
    if state.get("status") == "completed":
        return 100.0
    elif state.get("status") == "failed":
        return completed / len(steps) * 100
    else:
        return completed / len(steps) * 100


# ==========================================
# Embedding & Preprocessing API Endpoints
# ==========================================

# In-memory storage for preprocessing jobs
_preprocessing_jobs: dict = {}


from pydantic import BaseModel
from typing import Literal


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing job"""
    embedding_model: str = "Qwen-Embedding-0.6B"
    min_df: int = 5
    max_df_ratio: float = 0.7
    max_vocab_size: int = 50000
    language: str = "multi"
    device: str = "auto"


class PreprocessingRequest(BaseModel):
    """Request to start preprocessing"""
    dataset: str
    text_column: str = "text"
    config: Optional[PreprocessingConfig] = None


class PreprocessingStatus(BaseModel):
    """Status of a preprocessing job"""
    job_id: str
    dataset: str
    status: Literal["pending", "bow_generating", "bow_completed", "embedding_generating", "embedding_completed", "completed", "failed"]
    progress: float = 0.0
    current_stage: Optional[str] = None
    message: Optional[str] = None
    
    # Output paths
    bow_path: Optional[str] = None
    embedding_path: Optional[str] = None
    vocab_path: Optional[str] = None
    
    # Statistics
    num_documents: int = 0
    vocab_size: int = 0
    embedding_dim: int = 0
    bow_sparsity: float = 0.0
    
    # Timing
    bow_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    
    # Error
    error_message: Optional[str] = None
    
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get("/preprocessing/models", tags=["preprocessing"])
async def list_embedding_models():
    """List available embedding models"""
    return {
        "models": [
            {
                "id": "Qwen-Embedding-0.6B",
                "name": "Qwen3 Embedding 0.6B",
                "dim": 1024,
                "description": "Fast, good quality - Recommended for most use cases",
                "available": True
            },
            {
                "id": "Qwen-Embedding-1.8B",
                "name": "Qwen3 Embedding 1.8B",
                "dim": 2048,
                "description": "Better quality, slower",
                "available": False
            },
            {
                "id": "Qwen-Embedding-7B",
                "name": "Qwen3 Embedding 7B",
                "dim": 4096,
                "description": "Best quality, requires high-end GPU",
                "available": False
            }
        ],
        "default": "Qwen-Embedding-0.6B"
    }


@router.post("/preprocessing/start", response_model=PreprocessingStatus, tags=["preprocessing"])
async def start_preprocessing(request: PreprocessingRequest, background_tasks: BackgroundTasks):
    """Start a preprocessing job (BOW + Embedding generation)"""
    import uuid
    
    # Validate dataset exists
    dataset_dir = settings.DATA_DIR / request.dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found")
    
    # Find CSV file
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No CSV files found in dataset '{request.dataset}'")
    
    csv_file = csv_files[0]
    for f in csv_files:
        if "text_only" in f.name or "cleaned" in f.name:
            csv_file = f
            break
    
    # Generate job ID
    job_id = f"preproc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Create job status
    job_status = {
        "job_id": job_id,
        "dataset": request.dataset,
        "csv_path": str(csv_file),
        "text_column": request.text_column,
        "config": request.config.dict() if request.config else {},
        "status": "pending",
        "progress": 0.0,
        "current_stage": None,
        "message": "Job created, waiting to start",
        "bow_path": None,
        "embedding_path": None,
        "vocab_path": None,
        "num_documents": 0,
        "vocab_size": 0,
        "embedding_dim": 0,
        "bow_sparsity": 0.0,
        "bow_time_seconds": 0.0,
        "embedding_time_seconds": 0.0,
        "total_time_seconds": 0.0,
        "error_message": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    _preprocessing_jobs[job_id] = job_status
    
    # Start background processing
    if settings.SIMULATION_MODE:
        background_tasks.add_task(_simulate_preprocessing, job_id)
    else:
        background_tasks.add_task(_run_preprocessing, job_id)
    
    return PreprocessingStatus(**{k: v for k, v in job_status.items() if k in PreprocessingStatus.__fields__})


@router.get("/preprocessing/{job_id}", response_model=PreprocessingStatus, tags=["preprocessing"])
async def get_preprocessing_status(job_id: str):
    """Get status of a preprocessing job"""
    if job_id not in _preprocessing_jobs:
        raise HTTPException(status_code=404, detail=f"Preprocessing job '{job_id}' not found")
    
    job_status = _preprocessing_jobs[job_id]
    return PreprocessingStatus(**{k: v for k, v in job_status.items() if k in PreprocessingStatus.__fields__})


@router.get("/preprocessing", response_model=List[PreprocessingStatus], tags=["preprocessing"])
async def list_preprocessing_jobs():
    """List all preprocessing jobs"""
    return [
        PreprocessingStatus(**{k: v for k, v in job.items() if k in PreprocessingStatus.__fields__})
        for job in _preprocessing_jobs.values()
    ]


@router.delete("/preprocessing/{job_id}", tags=["preprocessing"])
async def cancel_preprocessing(job_id: str):
    """Cancel a preprocessing job"""
    if job_id not in _preprocessing_jobs:
        raise HTTPException(status_code=404, detail=f"Preprocessing job '{job_id}' not found")
    
    job_status = _preprocessing_jobs[job_id]
    if job_status["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    job_status["status"] = "failed"
    job_status["error_message"] = "Cancelled by user"
    job_status["updated_at"] = datetime.now().isoformat()
    
    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/preprocessing/check/{dataset}", tags=["preprocessing"])
async def check_preprocessing_status(dataset: str):
    """Check if a dataset has been preprocessed (has BOW and embeddings)"""
    result_dir = settings.RESULT_DIR / dataset / "embedding"
    
    has_bow = False
    has_embeddings = False
    bow_path = None
    embedding_path = None
    vocab_path = None
    
    if result_dir.exists():
        bow_files = list(result_dir.glob("*_bow.npz"))
        emb_files = list(result_dir.glob("*_embeddings.npy"))
        vocab_files = list(result_dir.glob("*_vocab.json"))
        
        has_bow = len(bow_files) > 0
        has_embeddings = len(emb_files) > 0
        
        if bow_files:
            bow_path = str(bow_files[0])
        if emb_files:
            embedding_path = str(emb_files[0])
        if vocab_files:
            vocab_path = str(vocab_files[0])
    
    return {
        "dataset": dataset,
        "has_bow": has_bow,
        "has_embeddings": has_embeddings,
        "ready_for_training": has_bow and has_embeddings,
        "bow_path": bow_path,
        "embedding_path": embedding_path,
        "vocab_path": vocab_path
    }


async def _simulate_preprocessing(job_id: str):
    """Simulate preprocessing for demo purposes"""
    import asyncio
    
    job = _preprocessing_jobs[job_id]
    
    try:
        # Stage 1: BOW Generation (40% of total)
        job["status"] = "bow_generating"
        job["current_stage"] = "bow"
        
        for i in range(10):
            await asyncio.sleep(0.5)
            job["progress"] = (i + 1) * 4  # 0-40%
            job["message"] = f"Generating BOW matrix... {(i + 1) * 10}%"
            job["updated_at"] = datetime.now().isoformat()
        
        job["status"] = "bow_completed"
        job["bow_time_seconds"] = 5.0
        job["vocab_size"] = 5000
        job["bow_sparsity"] = 0.95
        job["bow_path"] = f"/result/{job['dataset']}/embedding/{job['dataset']}_bow.npz"
        job["vocab_path"] = f"/result/{job['dataset']}/embedding/{job['dataset']}_vocab.json"
        
        # Stage 2: Embedding Generation (60% of total)
        job["status"] = "embedding_generating"
        job["current_stage"] = "embedding"
        
        for i in range(15):
            await asyncio.sleep(0.5)
            job["progress"] = 40 + (i + 1) * 4  # 40-100%
            job["message"] = f"Generating embeddings... {(i + 1) * 100 // 15}%"
            job["updated_at"] = datetime.now().isoformat()
        
        job["status"] = "completed"
        job["embedding_time_seconds"] = 7.5
        job["embedding_dim"] = 1024
        job["num_documents"] = 1000
        job["embedding_path"] = f"/result/{job['dataset']}/embedding/{job['dataset']}_embeddings.npy"
        job["total_time_seconds"] = 12.5
        job["progress"] = 100.0
        job["message"] = "Preprocessing completed successfully"
        job["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["updated_at"] = datetime.now().isoformat()


async def _run_preprocessing(job_id: str):
    """Run actual preprocessing using EmbeddingProcessor"""
    import asyncio
    
    job = _preprocessing_jobs[job_id]
    
    try:
        # Import the processor
        import sys
        sys.path.insert(0, str(settings.ETM_DIR.parent))
        from ETM.preprocessing import EmbeddingProcessor, ProcessingConfig
        
        # Create config
        config_dict = job.get("config", {})
        config = ProcessingConfig(
            embedding_model=config_dict.get("embedding_model", "Qwen-Embedding-0.6B"),
            min_df=config_dict.get("min_df", 5),
            max_df_ratio=config_dict.get("max_df_ratio", 0.7),
            max_vocab_size=config_dict.get("max_vocab_size", 50000),
            language=config_dict.get("language", "multi"),
            device=config_dict.get("device", "auto")
        )
        
        # Progress callback
        def progress_callback(stage: str, progress: float, message: str):
            job["current_stage"] = stage
            job["message"] = message
            job["updated_at"] = datetime.now().isoformat()
            
            if stage == "bow":
                job["status"] = "bow_generating"
                job["progress"] = progress * 40  # 0-40%
            elif stage == "embedding":
                if progress < 0.1:
                    job["status"] = "embedding_generating"
                job["progress"] = 40 + progress * 60  # 40-100%
            elif stage == "complete":
                job["status"] = "completed"
                job["progress"] = 100.0
        
        # Create processor
        processor = EmbeddingProcessor(
            config=config,
            progress_callback=progress_callback,
            dev_mode=True
        )
        
        # Output directory
        output_dir = settings.RESULT_DIR / job["dataset"] / "embedding"
        
        # Run processing
        result = processor.process(
            csv_path=job["csv_path"],
            text_column=job["text_column"],
            output_dir=str(output_dir),
            dataset_name=job["dataset"]
        )
        
        # Update job with results
        job["bow_path"] = result.bow_path
        job["embedding_path"] = result.embedding_path
        job["vocab_path"] = result.vocab_path
        job["num_documents"] = result.num_documents
        job["vocab_size"] = result.vocab_size
        job["embedding_dim"] = result.embedding_dim
        job["bow_sparsity"] = result.bow_sparsity
        job["bow_time_seconds"] = result.bow_time_seconds
        job["embedding_time_seconds"] = result.embedding_time_seconds
        job["total_time_seconds"] = result.total_time_seconds
        
        if result.status.value == "completed":
            job["status"] = "completed"
            job["progress"] = 100.0
            job["message"] = "Preprocessing completed successfully"
        else:
            job["status"] = "failed"
            job["error_message"] = result.error_message
        
        job["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["updated_at"] = datetime.now().isoformat()
        logger.error(f"Preprocessing failed: {e}\n{traceback.format_exc()}")

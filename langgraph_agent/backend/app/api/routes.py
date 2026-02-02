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


@router.get("/datasets/{dataset_name}/columns", tags=["data"])
async def get_dataset_columns(dataset_name: str):
    """
    Get column information for a dataset with auto-detected column types.
    
    Returns columns categorized by their likely purpose:
    - text_columns: Columns likely containing text content
    - time_columns: Columns likely containing timestamps
    - dimension_columns: Columns likely containing categorical dimensions (region, type, etc.)
    - label_columns: Columns likely containing labels
    - other_columns: Other columns
    
    Frontend can use this to let users confirm or modify column mappings.
    """
    import pandas as pd
    
    dataset_dir = settings.DATA_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No CSV files found")
    
    csv_file = csv_files[0]
    for f in csv_files:
        if "text_only" in f.name or "cleaned" in f.name:
            csv_file = f
            break
    
    df = pd.read_csv(csv_file, nrows=100)
    
    # Auto-detect column types
    text_keywords = ['text', 'content', 'body', 'title', 'description', '内容', '文本', '标题']
    time_keywords = ['time', 'date', 'year', 'month', 'day', 'timestamp', '时间', '日期', '年份', '年', '月']
    dim_keywords = ['region', 'province', 'city', 'area', 'category', 'type', 'department', 'source',
                    '省', '市', '地区', '区域', '类型', '部门', '来源', '分类']
    label_keywords = ['label', 'class', 'category', 'tag', '标签', '类别']
    
    columns_info = {
        'all_columns': df.columns.tolist(),
        'text_columns': [],
        'time_columns': [],
        'dimension_columns': [],
        'label_columns': [],
        'other_columns': [],
        'auto_detected': {
            'text_column': None,
            'time_column': None,
            'dimension_column': None
        }
    }
    
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().head(5).tolist()
        
        # Check column type
        is_text = any(k in col_lower for k in text_keywords)
        is_time = any(k in col_lower for k in time_keywords)
        is_dim = any(k in col_lower for k in dim_keywords)
        is_label = any(k in col_lower for k in label_keywords)
        
        # Also check data type and content
        if df[col].dtype == 'object':
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 50:  # Long text
                is_text = True
        
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'sample_values': [str(v)[:100] for v in sample_values],
            'unique_count': int(df[col].nunique()),
            'null_count': int(df[col].isnull().sum())
        }
        
        if is_text:
            columns_info['text_columns'].append(col_info)
            if not columns_info['auto_detected']['text_column']:
                columns_info['auto_detected']['text_column'] = col
        elif is_time:
            columns_info['time_columns'].append(col_info)
            if not columns_info['auto_detected']['time_column']:
                columns_info['auto_detected']['time_column'] = col
        elif is_dim:
            columns_info['dimension_columns'].append(col_info)
            if not columns_info['auto_detected']['dimension_column']:
                columns_info['auto_detected']['dimension_column'] = col
        elif is_label:
            columns_info['label_columns'].append(col_info)
        else:
            columns_info['other_columns'].append(col_info)
    
    return columns_info


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
        topic_words = json.load(f)
    
    if top_k < 20:
        topic_words = {k: v[:top_k] for k, v in topic_words.items()}
    
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


@router.get("/results/{dataset}/{mode}/temporal-analysis", tags=["results"])
async def get_temporal_analysis(
    dataset: str, 
    mode: str, 
    freq: str = Query(default="M", description="Time frequency: Y=year, M=month, Q=quarter"),
    top_k: int = Query(default=10, ge=1, le=20, description="Number of top topics to include")
):
    """
    Get temporal analysis data for frontend visualization.
    
    Returns:
    - document_volume: Document count over time (Tab 2 Chart A)
    - topic_evolution: Topic strength over time (Tab 2 Chart B)
    - topic_trends: Rising/falling topic trends
    """
    import numpy as np
    import pandas as pd
    
    result_path = settings.get_result_path(dataset, mode)
    model_dir = result_path / "model"
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model results not found")
    
    # Load theta and topic_words
    theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
    topic_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
    
    if not theta_files or not topic_files:
        raise HTTPException(status_code=404, detail="Required files not found")
    
    theta = np.load(str(theta_files[0]))
    with open(topic_files[0]) as f:
        topic_words_raw = json.load(f)
    
    topic_words = [(int(k), [(w, float(p)) for w, p in words]) for k, words in topic_words_raw.items()]
    
    # Try to load timestamps from data
    data_dir = settings.DATA_DIR / dataset
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        return {
            "available": False,
            "message": "📊 时序分析功能",
            "reason": "当前数据集未包含原始数据文件。",
            "suggestions": [
                "请将原始CSV数据文件放入 data/{dataset}/ 目录",
                "确保数据文件包含时间列（如 publish_date, timestamp, year 等）"
            ],
            "actions": [
                {"label": "跳过", "action": "skip"},
                {"label": "去添加", "action": "add_data"}
            ]
        }
    
    df = pd.read_csv(csv_files[0])
    time_cols = [c for c in df.columns if any(t in c.lower() for t in ['time', 'date', 'year', '时间', '日期', '年份'])]
    
    if not time_cols:
        available_cols = df.columns.tolist()
        return {
            "available": False,
            "message": "📊 时序分析功能",
            "reason": "当前数据集未包含时间列。",
            "suggestions": [
                "在原始数据中添加时间列（如 publish_date）",
                "重新上传数据并在"列映射"中选择时间列"
            ],
            "available_columns": available_cols,
            "actions": [
                {"label": "跳过", "action": "skip"},
                {"label": "去添加", "action": "add_column"}
            ]
        }
    
    if len(df) != len(theta):
        raise HTTPException(status_code=400, detail="Data length mismatch with model results")
    
    from visualization.temporal_analysis import TemporalTopicAnalyzer
    
    analyzer = TemporalTopicAnalyzer(
        theta=theta,
        timestamps=df[time_cols[0]].values,
        topic_words=topic_words
    )
    
    return analyzer.get_visualization_data_for_frontend(freq=freq, top_k_topics=top_k)


@router.get("/results/{dataset}/{mode}/dimension-analysis", tags=["results"])
async def get_dimension_analysis(
    dataset: str, 
    mode: str,
    top_k_topics: int = Query(default=10, ge=1, le=20),
    top_k_dimensions: int = Query(default=20, ge=1, le=50)
):
    """
    Get dimension/spatial analysis data for frontend visualization.
    
    Returns:
    - heatmap_data: Topic distribution by dimension (Tab 4)
    - dimension_stats: Dimension statistics
    """
    import numpy as np
    import pandas as pd
    
    result_path = settings.get_result_path(dataset, mode)
    model_dir = result_path / "model"
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model results not found")
    
    # Load theta and topic_words
    theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
    topic_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
    
    if not theta_files or not topic_files:
        raise HTTPException(status_code=404, detail="Required files not found")
    
    theta = np.load(str(theta_files[0]))
    with open(topic_files[0]) as f:
        topic_words_raw = json.load(f)
    
    topic_words = [(int(k), [(w, float(p)) for w, p in words]) for k, words in topic_words_raw.items()]
    
    # Try to load dimension column from data
    data_dir = settings.DATA_DIR / dataset
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        return {
            "available": False,
            "message": "📊 维度分析功能",
            "reason": "当前数据集未包含原始数据文件。",
            "suggestions": [
                "请将原始CSV数据文件放入 data/{dataset}/ 目录",
                "确保数据文件包含维度列（如 province, region, category 等）"
            ],
            "actions": [
                {"label": "跳过", "action": "skip"},
                {"label": "去添加", "action": "add_data"}
            ]
        }
    
    df = pd.read_csv(csv_files[0])
    dim_cols = [c for c in df.columns if any(d in c.lower() for d in ['region', 'province', 'city', 'category', 'type', '省', '市', '地区', '类型', '部门'])]
    
    if not dim_cols:
        available_cols = df.columns.tolist()
        return {
            "available": False,
            "message": "📊 维度分析功能",
            "reason": "当前数据集未包含维度列（如省份、地区、类型等）。",
            "suggestions": [
                "在原始数据中添加维度列（如 province, category）",
                "重新上传数据并在"列映射"中选择维度列"
            ],
            "available_columns": available_cols,
            "actions": [
                {"label": "跳过", "action": "skip"},
                {"label": "去添加", "action": "add_column"}
            ]
        }
    
    if len(df) != len(theta):
        raise HTTPException(status_code=400, detail="Data length mismatch with model results")
    
    from visualization.dimension_analysis import DimensionAnalyzer
    
    analyzer = DimensionAnalyzer(
        theta=theta,
        dimension_values=df[dim_cols[0]].values,
        topic_words=topic_words,
        dimension_name=dim_cols[0]
    )
    
    return analyzer.get_visualization_data_for_frontend(
        top_k_topics=top_k_topics,
        top_k_dimensions=top_k_dimensions
    )


@router.get("/results/{dataset}/{mode}/topic-overview", tags=["results"])
async def get_topic_overview(
    dataset: str, 
    mode: str,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of keywords per topic")
):
    """
    Get topic overview data for frontend visualization (Tab 1).
    
    Returns:
    - topics: List of topics with id, keywords, and proportion
    - total_documents: Total number of documents
    """
    import numpy as np
    
    result_path = settings.get_result_path(dataset, mode)
    model_dir = result_path / "model"
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model results not found")
    
    # Load theta and topic_words
    theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
    topic_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
    
    if not theta_files or not topic_files:
        raise HTTPException(status_code=404, detail="Required files not found")
    
    theta = np.load(str(theta_files[0]))
    with open(topic_files[0]) as f:
        topic_words = json.load(f)
    
    # Calculate topic proportions
    topic_proportions = theta.mean(axis=0)
    
    # Build response
    topics = []
    for topic_id, words in topic_words.items():
        topic_idx = int(topic_id)
        keywords = [{"word": w, "weight": float(p)} for w, p in words[:top_k]]
        
        topics.append({
            "id": topic_idx,
            "name": f"Topic {topic_idx}",  # Can be replaced with LLM-generated name
            "keywords": keywords,
            "proportion": float(topic_proportions[topic_idx]),
            "document_count": int((theta.argmax(axis=1) == topic_idx).sum()),
            "wordcloud_path": f"topic_words_{dataset}_{mode}_topic{topic_idx}.png"
        })
    
    # Sort by proportion
    topics.sort(key=lambda x: x["proportion"], reverse=True)
    
    return {
        "topics": topics,
        "total_documents": len(theta),
        "num_topics": len(topics)
    }


@router.post("/tasks", response_model=TaskResponse, tags=["tasks"])
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new ETM pipeline task"""
    dataset_dir = settings.DATA_DIR / request.dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found")
    
    embeddings_dir = settings.get_result_path(request.dataset, request.mode) / "embeddings"
    emb_file = embeddings_dir / f"{request.dataset}_{request.mode}_embeddings.npy"
    if not emb_file.exists():
        raise HTTPException(
            status_code=400, 
            detail=f"Embeddings not found for {request.dataset}/{request.mode}. Please generate embeddings first."
        )
    
    from ..schemas.agent import AgentState
    initial_state = {
        "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "dataset": request.dataset,
        "mode": request.mode,
        "status": "pending",
        "current_step": "preprocess",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    async def run_task():
        await etm_agent.run_pipeline(request)
    
    background_tasks.add_task(run_task)
    
    return TaskResponse(
        task_id=initial_state["task_id"],
        status=TaskStatus.PENDING,
        current_step="preprocess",
        progress=0,
        created_at=initial_state["created_at"],
        updated_at=initial_state["updated_at"]
    )


@router.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks():
    """List all tasks"""
    tasks = []
    for task_id, state in etm_agent.get_all_tasks().items():
        tasks.append(TaskResponse(
            task_id=task_id,
            status=TaskStatus(state.get("status", "pending")),
            current_step=state.get("current_step"),
            progress=_calculate_progress(state),
            metrics=state.get("metrics"),
            created_at=datetime.fromisoformat(state.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(state.get("updated_at", datetime.now().isoformat())),
            completed_at=datetime.fromisoformat(state["completed_at"]) if state.get("completed_at") else None,
            error_message=state.get("error_message")
        ))
    return tasks


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str):
    """Get task status"""
    state = etm_agent.get_task_status(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus(state.get("status", "pending")),
        current_step=state.get("current_step"),
        progress=_calculate_progress(state),
        metrics=state.get("metrics"),
        topic_words=state.get("topic_words"),
        visualization_paths=state.get("visualization_paths"),
        created_at=datetime.fromisoformat(state.get("created_at", datetime.now().isoformat())),
        updated_at=datetime.fromisoformat(state.get("updated_at", datetime.now().isoformat())),
        completed_at=datetime.fromisoformat(state["completed_at"]) if state.get("completed_at") else None,
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

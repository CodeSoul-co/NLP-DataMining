"""
API Routes
REST endpoints for the THETA Agent System
"""

import os
import json
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body, UploadFile, File, Form, Depends
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..schemas.agent import (
    TaskRequest, TaskResponse, TaskStatus, ChatRequest, ChatResponse,
    SuggestionsRequest, SuggestionsResponse, SuggestionItem
)
from ..schemas.data import DatasetInfo, ResultInfo, VisualizationInfo, ProjectInfo, MetricsResponse
from ..agents.etm_agent import etm_agent, create_initial_state
from ..services.chat_service import chat_service
from ..services.task_store import task_store
from ..core.config import settings
from ..core.logging import get_logger
from ..core.etm_paths import check_etm_modules
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..models.user import User
from ..models.user_dataset import UserDataset
from ..services.auth_service import get_current_active_user
from ..core.database import async_session_maker

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


@router.post("/restart", tags=["health"])
async def restart_service(background_tasks: BackgroundTasks):
    """Restart the backend service"""
    try:
        # 获取 backend 目录路径
        # routes.py -> api -> app -> backend
        backend_dir = Path(__file__).parent.parent.parent
        
        # 重启脚本路径（放在 backend 目录下）
        restart_script = backend_dir / "restart_backend.sh"
        
        # 重启命令
        restart_cmd = f"""#!/bin/bash
cd {backend_dir}
pkill -f 'uvicorn app.main:app'
sleep 2
nohup /root/miniconda3/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > {backend_dir}/server.log 2>&1 &
"""
        
        # 写入重启脚本
        with open(restart_script, 'w') as f:
            f.write(restart_cmd)
        os.chmod(restart_script, 0o755)
        
        # 在后台任务中执行重启（延迟执行，确保响应先返回）
        def do_restart():
            import time
            time.sleep(1)  # 等待响应返回
            subprocess.Popen(
                ["/bin/bash", str(restart_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        background_tasks.add_task(do_restart)
        
        logger.info("Service restart requested")
        return {
            "status": "restarting",
            "message": "服务正在重启中，请稍候刷新页面...",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to restart service: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"重启服务失败: {str(e)}"
        )


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


@router.get("/etm/health", tags=["health"])
async def etm_health_check():
    """Check ETM module availability and status"""
    try:
        results = check_etm_modules()
        all_ok = all(
            module.get("status") == "ok" 
            for module in results["modules"].values()
        )
        
        return {
            "status": "ok" if all_ok else "partial",
            "etm_dir": results["etm_dir"],
            "etm_dir_exists": results["etm_dir_exists"],
            "modules": results["modules"],
            "summary": {
                "total_modules": len(results["modules"]),
                "working_modules": sum(
                    1 for m in results["modules"].values() 
                    if m.get("status") == "ok"
                ),
                "failed_modules": [
                    name for name, info in results["modules"].items()
                    if info.get("status") == "error"
                ]
            }
        }
    except Exception as e:
        logger.error(f"ETM health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "etm_dir": str(settings.ETM_DIR),
            "etm_dir_exists": settings.ETM_DIR.exists()
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


def _build_dataset_info(dataset_dir: Path) -> Optional[DatasetInfo]:
    """Build DatasetInfo from a dataset directory."""
    csv_files = list(dataset_dir.glob("*.csv"))
    all_files = list(dataset_dir.iterdir())
    if not all_files:
        return None
    info = DatasetInfo(
        name=dataset_dir.name,
        path=str(dataset_dir),
        has_labels=False,
        size=len(all_files)
    )
    if csv_files:
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
    return info


@router.get("/datasets", response_model=List[DatasetInfo], tags=["data"])
async def list_datasets(
    current_user: User = Depends(get_current_active_user),
):
    """List datasets owned by the current user. Requires authentication."""
    datasets = []
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset.dataset_name).where(UserDataset.user_id == current_user.id)
        )
        owned_names = {row[0] for row in result.fetchall()}
    if not settings.DATA_DIR.exists():
        return datasets
    for name in owned_names:
        dataset_dir = settings.DATA_DIR / name
        if not dataset_dir.is_dir() or name.startswith("."):
            continue
        info = _build_dataset_info(dataset_dir)
        if info:
            datasets.append(info)
    return datasets


class UploadResponse(BaseModel):
    """Response for file upload"""
    success: bool
    message: str
    dataset_name: str
    file_count: int
    total_size: int
    files: List[str]


@router.post("/datasets/upload", response_model=UploadResponse, tags=["data"])
async def upload_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    current_user: User = Depends(get_current_active_user),
):
    """
    Upload data files to create a new dataset.
    
    Accepts any file format for raw data upload (PDF, DOCX, Excel, CSV, TXT, JSON, etc.).
    Files will be saved to DATA_DIR/{dataset_name}/
    
    Note: Raw data files can be processed through the data cleaning module to convert to CSV format.
    """
    import shutil
    
    # Validate dataset name
    if not dataset_name or not dataset_name.strip():
        raise HTTPException(status_code=400, detail="Dataset name is required")
    
    # Sanitize dataset name (remove special characters)
    safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-', ' ')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    
    # Create dataset directory (user-scoped)
    dataset_dir = settings.DATA_DIR / str(current_user.id) / safe_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dataset directory created/verified: {dataset_dir}")
    
    uploaded_files = []
    total_size = 0
    
    for file in files:
        # Accept any file format (raw data can be any format)
        # Data cleaning module will convert them to CSV format if needed
        if file.filename:
            # Save file
            file_path = dataset_dir / file.filename
            try:
                content = await file.read()
                total_size += len(content)
                
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # Verify file was written
                if file_path.exists():
                    actual_size = file_path.stat().st_size
                    logger.info(f"Successfully uploaded file: {file_path} ({actual_size} bytes)")
                    if actual_size != len(content):
                        logger.warning(f"File size mismatch: expected {len(content)}, got {actual_size}")
                else:
                    logger.error(f"File was not created: {file_path}")
                    raise HTTPException(status_code=500, detail=f"Failed to save file: {file.filename}")
                
                uploaded_files.append(file.filename)
            except Exception as e:
                logger.error(f"Failed to save file {file.filename} to {file_path}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to save file: {file.filename}")
    
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")
    
    # Record ownership for user isolation
    async with async_session_maker() as session:
        try:
            ud = UserDataset(user_id=current_user.id, dataset_name=safe_name)
            session.add(ud)
            await session.commit()
        except IntegrityError:
            await session.rollback()
            pass  # Already owned by this user
    
    return UploadResponse(
        success=True,
        message=f"Successfully uploaded {len(uploaded_files)} file(s) to dataset '{safe_name}'",
        dataset_name=safe_name,
        file_count=len(uploaded_files),
        total_size=total_size,
        files=uploaded_files
    )


@router.delete("/datasets/{dataset_name}", tags=["data"])
async def delete_dataset(
    dataset_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """Delete a dataset and all its files. Requires authentication and ownership."""
    import shutil
    
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset).where(
                UserDataset.user_id == current_user.id,
                UserDataset.dataset_name == dataset_name
            )
        )
        ud = result.scalar_one_or_none()
    if not ud:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found or access denied")
    
    dataset_dir = settings.DATA_DIR / str(current_user.id) / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    try:
        shutil.rmtree(dataset_dir)
        from sqlalchemy import delete
        async with async_session_maker() as session:
            await session.execute(
                delete(UserDataset).where(
                    UserDataset.user_id == current_user.id,
                    UserDataset.dataset_name == dataset_name
                )
            )
            await session.commit()

        # 同步删除 OSS 云端文件（容错：失败不影响整体结果）
        try:
            from .data_api import _get_oss_bucket
            import oss2 as _oss2
            bucket = _get_oss_bucket()
            if bucket:
                oss_prefixes = [
                    f"data/{current_user.id}/{dataset_name}/",
                    f"result/baseline/{dataset_name}/",
                    f"result/0.6B/{dataset_name}/",
                    f"result/{dataset_name}/",
                ]
                for prefix in oss_prefixes:
                    keys_to_delete = [
                        obj.key
                        for obj in _oss2.ObjectIterator(bucket, prefix=prefix)
                        if not (obj.size == 0 and obj.key.endswith("/"))
                    ]
                    if keys_to_delete:
                        # batch_delete_objects 每次最多 1000 个
                        for i in range(0, len(keys_to_delete), 1000):
                            bucket.batch_delete_objects(keys_to_delete[i:i + 1000])
                        logger.info(f"OSS: deleted {len(keys_to_delete)} objects under {prefix}")
        except Exception as oss_err:
            logger.warning(f"OSS cleanup for dataset '{dataset_name}' failed (non-fatal): {oss_err}")

        logger.info(f"Deleted dataset: {dataset_name}")
        return {"success": True, "message": f"Dataset '{dataset_name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.get("/datasets/{dataset_name}", response_model=DatasetInfo, tags=["data"])
async def get_dataset_info(
    dataset_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get detailed information about a specific dataset. Requires authentication and ownership."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset).where(
                UserDataset.user_id == current_user.id,
                UserDataset.dataset_name == dataset_name
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found or access denied")
    
    dataset_dir = settings.DATA_DIR / str(current_user.id) / dataset_name
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


@router.get("/datasets/{dataset_name}/preview", tags=["data"])
async def get_dataset_preview(
    dataset_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get CSV columns and first 5 rows for column selection. Requires authentication and ownership."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset).where(
                UserDataset.user_id == current_user.id,
                UserDataset.dataset_name == dataset_name
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found or access denied")
    
    dataset_dir = settings.DATA_DIR / str(current_user.id) / dataset_name
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
    
    df = pd.read_csv(csv_file, nrows=5)
    return {
        "columns": df.columns.tolist(),
        "rows": df.fillna("").astype(str).values.tolist()
    }


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


# model-comparison 必须放在 /results/{dataset}/{mode} 之前，否则 model-comparison 会被当作 mode 匹配
_METRIC_COLS = [
    ("topic_diversity_td", "TD", "up", ["TOPIC_DIVERSITY_TD"]),
    ("topic_diversity_irbo", "iRBO", "up", ["TOPIC_DIVERSITY_IRBO"]),
    ("topic_coherence_npmi_avg", "NPMI", "up", ["topic_coherence_npmi", "TOPIC_COHERENCE_NPMI"]),
    ("topic_coherence_cv_avg", "C_V", "up", ["topic_coherence_cvc_v", "TOPIC_COHERENCE_CVC_V"]),
    ("topic_coherence_umass_avg", "UMass", "neutral", ["topic_coherence_umass", "TOPIC_COHERENCE_UMASS"]),
    ("topic_exclusivity_avg", "Exclusivity", "up", ["topic_exclusivity", "TOPIC_EXCLUSIVITY"]),
    ("perplexity", "PPL", "down", ["PPL"]),
]


def _get_metric_val(metrics: Dict[str, Any], main_key: str, alt_keys: List[str]) -> Optional[float]:
    v = metrics.get(main_key)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    for k in alt_keys:
        v = metrics.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


@router.get("/results/{dataset}/model-comparison", tags=["results"])
async def get_model_comparison(
    dataset: str,
    current_user: User = Depends(get_current_active_user),
):
    """跨模型/模式对比：聚合 THETA 各 mode + baseline 模型（若有）的评估指标"""
    rows: List[Dict[str, Any]] = []
    for mode in ("zero_shot", "unsupervised", "supervised"):
        result_path = settings.get_user_result_path(current_user.id, dataset, mode)
        eval_dir = result_path / "evaluation"
        metrics = None
        if eval_dir.exists():
            metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
            if metrics_files:
                with open(metrics_files[0]) as f:
                    metrics = json.load(f)
        if not metrics:
            for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=20):
                if t.get("mode") == mode and t.get("metrics"):
                    metrics = t["metrics"]
                    break
        if metrics:
            row = {"model": f"THETA ({mode})", "mode": mode}
            for key, _, _, alts in _METRIC_COLS:
                v = _get_metric_val(metrics, key, alts)
                if v is not None:
                    row[key] = v
            rows.append(row)
    baseline_dir = settings.RESULT_DIR / "baseline" / dataset
    if baseline_dir.exists():
        for model_dir in baseline_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                if model_name == "ctm":
                    metrics_path = model_dir / "metrics_k20.json"
                else:
                    for k in [20, 50, 10]:
                        metrics_path = model_dir / f"metrics_k{k}.json"
                        if metrics_path.exists():
                            break
                    else:
                        metrics_path = next(model_dir.glob("metrics_*.json"), None)
                if metrics_path and metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    row = {"model": model_name, "mode": None}
                    for key, _, _, alts in _METRIC_COLS:
                        v = _get_metric_val(metrics, key, alts)
                        if v is not None:
                            row[key] = v
                    rows.append(row)
    return {"dataset": dataset, "rows": rows, "columns": [{"key": c[0], "label": c[1], "direction": c[2]} for c in _METRIC_COLS]}


@router.get("/results/{dataset}/{mode}", response_model=ResultInfo, tags=["results"])
async def get_result_info(
    dataset: str,
    mode: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get detailed information about a specific result"""
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    
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
async def get_result_metrics(
    dataset: str,
    mode: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get detailed metrics for a result. Falls back to task_store when disk files don't exist (e.g. simulation)."""
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    eval_dir = result_path / "evaluation"

    # 1. Try disk first
    if eval_dir.exists():
        metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
        if metrics_files:
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

    # 2. Fallback: 从 task_store 查找已完成任务的 metrics（模拟训练等场景）
    tasks = task_store.get_tasks_list(status="completed", dataset=dataset, limit=1)
    for t in tasks:
        if t.get("mode") == mode and t.get("metrics"):
            m = t["metrics"]
            return MetricsResponse(
                dataset=dataset,
                mode=mode,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                topic_coherence_avg=m.get("topic_coherence_avg"),
                topic_coherence_per_topic=m.get("topic_coherence_per_topic"),
                topic_diversity_td=m.get("topic_diversity_td"),
                topic_diversity_irbo=m.get("topic_diversity_irbo"),
                additional=m
            )

    # 3. Fallback: 从 OSS 下载 metrics（DLC 训练结果在 OSS 上）
    try:
        from .data_api import _fetch_oss_result_data
        oss_data = _fetch_oss_result_data(dataset, mode)
        if oss_data.get("metrics"):
            m = oss_data["metrics"]
            # 保存到 task_store 以便下次直接读取
            for t in tasks:
                if t.get("mode") == mode:
                    task_store.update_task(t["task_id"], {
                        "metrics": m,
                        "topic_words": oss_data.get("topic_words"),
                    })
                    break
            return MetricsResponse(
                dataset=dataset,
                mode=mode,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                topic_coherence_avg=m.get("topic_coherence_avg"),
                topic_coherence_per_topic=m.get("topic_coherence_per_topic"),
                topic_diversity_td=m.get("topic_diversity_td"),
                topic_diversity_irbo=m.get("topic_diversity_irbo"),
                additional=m
            )
    except Exception as e:
        logger.warning(f"OSS metrics fallback failed for {dataset}/{mode}: {e}")

    raise HTTPException(status_code=404, detail="Evaluation results not found")


def _parse_topic_words(raw_data: Any, top_k: int) -> Dict[str, List[str]]:
    """Parse topic words from raw JSON (dict or list format)."""
    if isinstance(raw_data, dict):
        topic_words = raw_data
    elif isinstance(raw_data, list):
        topic_words = {}
        for item in raw_data:
            if isinstance(item, list) and len(item) >= 2:
                topic_id = str(item[0])
                words_data = item[1]
                if words_data and isinstance(words_data[0], list):
                    topic_words[topic_id] = [w[0] for w in words_data[:top_k]]
                else:
                    topic_words[topic_id] = words_data[:top_k]
    else:
        topic_words = {}
    if top_k < 20 and isinstance(topic_words, dict):
        topic_words = {k: (v[:top_k] if isinstance(v, list) else v) for k, v in topic_words.items()}
    return topic_words


@router.get("/results/{dataset}/{mode}/topic-words", tags=["results"])
async def get_topic_words(
    dataset: str,
    mode: str,
    top_k: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
):
    """Get topic words for a result. Falls back to task_store when disk files don't exist (e.g. simulation)."""
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    model_dir = result_path / "model"

    if model_dir.exists():
        topic_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
        if topic_files:
            with open(topic_files[0]) as f:
                raw_data = json.load(f)
            return _parse_topic_words(raw_data, top_k)

    tasks = task_store.get_tasks_list(status="completed", dataset=dataset, limit=1)
    for t in tasks:
        if t.get("mode") == mode and t.get("topic_words"):
            tw = t["topic_words"]
            if top_k < 20 and isinstance(tw, dict):
                tw = {k: (v[:top_k] if isinstance(v, list) else v) for k, v in tw.items()}
            return tw

    # 3. Fallback: 从 OSS 下载 topic_words
    try:
        from .data_api import _fetch_oss_result_data
        oss_data = _fetch_oss_result_data(dataset, mode)
        if oss_data.get("topic_words"):
            tw = oss_data["topic_words"]
            # 保存到 task_store
            for t in tasks:
                if t.get("mode") == mode:
                    task_store.update_task(t["task_id"], {
                        "topic_words": tw,
                        "metrics": oss_data.get("metrics"),
                    })
                    break
            return _parse_topic_words(tw, top_k)
    except Exception as e:
        logger.warning(f"OSS topic_words fallback failed for {dataset}/{mode}: {e}")

    raise HTTPException(status_code=404, detail="Topic words not found")


@router.get("/results/{dataset}/{mode}/visualizations", response_model=List[VisualizationInfo], tags=["results"])
async def list_visualizations(
    dataset: str,
    mode: str,
    current_user: User = Depends(get_current_active_user),
):
    """List all visualizations for a result (recursive, returns relative paths)"""
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    viz_dir = result_path / "visualization"
    
    if not viz_dir.exists():
        return []
    
    visualizations = []
    for viz_file in viz_dir.rglob("*"):
        if viz_file.is_file():
            file_type = "image" if viz_file.suffix in [".png", ".jpg", ".jpeg"] else \
                       "html" if viz_file.suffix == ".html" else "other"
            rel_name = str(viz_file.relative_to(viz_dir))  # e.g. "topics/topic_0/word_importance.png"
            visualizations.append(VisualizationInfo(
                name=rel_name,
                path=str(viz_file),
                type=file_type,
                size=viz_file.stat().st_size
            ))
    
    return visualizations


@router.get("/results/{dataset}/{mode}/visualizations/{filepath:path}", tags=["results"])
async def get_visualization(
    dataset: str,
    mode: str,
    filepath: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific visualization file (supports nested paths like topics/topic_0/word_importance.png)"""
    import re
    if ".." in filepath:
        raise HTTPException(status_code=400, detail="Invalid path")
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    viz_path = (result_path / "visualization" / filepath).resolve()
    root = result_path.resolve()
    if not str(viz_path).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not viz_path.exists() or not viz_path.is_file():
        # On-demand generate topics/topic_N/word_importance.png
        match = re.match(r"topics/topic_(\d+)/word_importance\.png", filepath)
        if match:
            from ..agents.nodes import _generate_topic_word_importance
            result_path.mkdir(parents=True, exist_ok=True)
            viz_dir = result_path / "visualization"
            viz_dir.mkdir(parents=True, exist_ok=True)
            # 1. Try model files on disk
            model_dir = result_path / "model"
            if model_dir.exists():
                try:
                    from visualization.topic_visualizer import load_etm_results
                    result_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
                    if result_files:
                        timestamp = result_files[0].stem.replace("theta_", "")
                        results = load_etm_results(str(model_dir), timestamp)
                        _generate_topic_word_importance(viz_dir, results['topic_words'], results['beta'])
                        if viz_path.exists():
                            return FileResponse(viz_path)
                except Exception:
                    pass
            # 2. Fallback: task_store (simulation mode, no disk files)
            tasks = task_store.get_tasks_list(status="completed", dataset=dataset, limit=5)
            for t in tasks:
                if t.get("mode") == mode and t.get("topic_words"):
                    try:
                        _generate_topic_word_importance(viz_dir, t["topic_words"], None)
                        if viz_path.exists():
                            return FileResponse(viz_path)
                    except Exception:
                        pass
                    break
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(viz_path)


@router.get("/results/{dataset}/{mode}/visualization-data", tags=["results"])
async def get_visualization_data(
    dataset: str,
    mode: str,
    data_type: str = Query(..., description="Type of data: topic_distribution, doc_topic_distribution, topic_similarity"),
    current_user: User = Depends(get_current_active_user),
):
    """Get visualization data for interactive charts. Returns empty structure when no results exist."""
    import numpy as np
    
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    model_dir = result_path / "model"
    
    if not model_dir.exists():
        if data_type == "topic_distribution":
            return {"topics": [], "proportions": [], "topic_words": {}}
        if data_type == "doc_topic_distribution":
            return {"documents": [], "distributions": [], "num_topics": 0}
        if data_type == "topic_similarity":
            return {"topics": [], "similarity_matrix": [], "topic_words": {}}
        return {"topics": [], "proportions": [], "topic_words": {}}
    
    # Find latest theta and beta files
    theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
    beta_files = sorted(model_dir.glob("beta_*.npy"), reverse=True)
    topic_words_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
    
    if not theta_files or not beta_files:
        if data_type == "topic_distribution":
            return {"topics": [], "proportions": [], "topic_words": {}}
        if data_type == "doc_topic_distribution":
            return {"documents": [], "distributions": [], "num_topics": 0}
        if data_type == "topic_similarity":
            return {"topics": [], "similarity_matrix": [], "topic_words": {}}
        return {"topics": [], "proportions": [], "topic_words": {}}
    
    theta = np.load(theta_files[0])
    beta = np.load(beta_files[0])
    
    # Load topic words
    topic_words = {}
    if topic_words_files:
        with open(topic_words_files[0]) as f:
            raw_data = json.load(f)
            if isinstance(raw_data, dict):
                topic_words = raw_data
            elif isinstance(raw_data, list):
                for item in raw_data:
                    if isinstance(item, list) and len(item) >= 2:
                        topic_id = str(item[0])
                        words_data = item[1]
                        if words_data and isinstance(words_data[0], list):
                            topic_words[topic_id] = [w[0] for w in words_data[:10]]
                        else:
                            topic_words[topic_id] = words_data[:10] if isinstance(words_data, list) else []
    
    if data_type == "topic_distribution":
        # Calculate topic proportions across all documents
        topic_proportions = theta.mean(axis=0).tolist()
        return {
            "topics": [f"Topic {i+1}" for i in range(len(topic_proportions))],
            "proportions": topic_proportions,
            "topic_words": topic_words
        }
    
    elif data_type == "doc_topic_distribution":
        # Sample documents for visualization (limit to 100 for performance)
        num_docs = min(100, theta.shape[0])
        indices = np.linspace(0, theta.shape[0] - 1, num_docs, dtype=int)
        sampled_theta = theta[indices]
        
        return {
            "documents": [f"Doc {i+1}" for i in indices],
            "distributions": sampled_theta.tolist(),
            "num_topics": theta.shape[1]
        }
    
    elif data_type == "topic_similarity":
        # Calculate cosine similarity between topics using beta
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(beta)
        
        return {
            "topics": [f"Topic {i+1}" for i in range(beta.shape[0])],
            "similarity_matrix": similarity_matrix.tolist(),
            "topic_words": topic_words
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")


@router.get("/results/{dataset}/{mode}/export", tags=["results"])
async def export_results(
    dataset: str,
    mode: str,
    types: str = Query(default="metrics,topic_words,visualizations", description="Comma-separated: metrics, topic_words, visualizations"),
    current_user: User = Depends(get_current_active_user),
):
    """导出结果为 ZIP：评估指标 JSON、主题词 JSON、所有图表"""
    import zipfile
    import io
    
    requested = {t.strip() for t in types.split(",") if t.strip()}
    result_path = settings.get_user_result_path(current_user.id, dataset, mode)
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if "metrics" in requested:
            eval_dir = result_path / "evaluation"
            if eval_dir.exists():
                for mf in eval_dir.glob("metrics_*.json"):
                    zf.write(mf, f"metrics/{mf.name}")
            else:
                for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=1):
                    if t.get("mode") == mode and t.get("metrics"):
                        import json as _j
                        zf.writestr("metrics/from_task.json", _j.dumps(t["metrics"], indent=2))
                        break
        
        if "topic_words" in requested:
            model_dir = result_path / "model"
            if model_dir.exists():
                for tf in model_dir.glob("topic_words_*.json"):
                    zf.write(tf, f"topic_words/{tf.name}")
            else:
                for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=1):
                    if t.get("mode") == mode and t.get("topic_words"):
                        import json as _j
                        zf.writestr("topic_words/from_task.json", _j.dumps(t["topic_words"], indent=2))
                        break
        
        if "visualizations" in requested:
            viz_dir = result_path / "visualization"
            if viz_dir.exists():
                for vf in viz_dir.rglob("*"):
                    if vf.is_file():
                        arcname = f"visualizations/{vf.relative_to(viz_dir)}"
                        zf.write(vf, arcname)
    
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={dataset}_{mode}_results.zip"}
    )


def _task_owned_by_user(task: Dict[str, Any], user_id: int) -> bool:
    """Check if task belongs to user (legacy tasks without user_id go to user 1)."""
    tid = task.get("user_id")
    return tid == user_id or (tid is None and user_id == 1)


@router.post("/tasks", response_model=TaskResponse, tags=["tasks"])
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
):
    """
    创建并启动新的 ETM 任务 (异步 Fire-and-Forget 模式)
    
    工作流程：
    1. 验证数据集存在
    2. 创建任务记录并持久化
    3. 在后台启动任务处理
    4. 立即返回 task_id 给前端
    5. 前端可通过 GET /api/tasks/{task_id} 轮询进度
    """
    import uuid
    
    # 1. 验证数据集及用户权限
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset).where(
                UserDataset.user_id == current_user.id,
                UserDataset.dataset_name == request.dataset
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found or access denied")
    dataset_dir = settings.DATA_DIR / request.dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found")
    
    # 2. 在非模拟模式下检查 embeddings
    if not settings.SIMULATION_MODE:
        embeddings_dir = settings.get_user_result_path(current_user.id, request.dataset, request.mode) / "embeddings"
        emb_file = embeddings_dir / f"{request.dataset}_{request.mode}_embeddings.npy"
        # 也检查 result/{user_id}/dataset/embedding 目录
        embedding_dir_alt = settings.RESULT_DIR / str(current_user.id) / request.dataset / "embedding"
        emb_files_alt = list(embedding_dir_alt.glob("*_embeddings.npy")) if embedding_dir_alt.exists() else []
        
        if not emb_file.exists() and not emb_files_alt:
            raise HTTPException(
                status_code=400, 
                detail=f"Embeddings not found for {request.dataset}/{request.mode}. Please generate embeddings first."
            )
    
    # 3. 生成任务 ID 并创建持久化记录
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    task_data = {
        "user_id": current_user.id,
        "dataset": request.dataset,
        "mode": request.mode,
        "num_topics": request.num_topics,
        "vocab_size": request.vocab_size,
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "hidden_dim": request.hidden_dim,
        "current_step": "pending",
        "message": "任务已创建，等待执行..."
    }
    
    # 使用 TaskStore 持久化（支持服务器重启后恢复）
    task = task_store.create_task(task_id, task_data)
    
    # 同时更新内存中的 active_tasks（用于实时状态）
    etm_agent.active_tasks[task_id] = task.copy()
    
    # 4. 在后台启动任务（Fire-and-Forget）
    if settings.SIMULATION_MODE:
        background_tasks.add_task(run_simulated_pipeline, task_id, request)
    else:
        background_tasks.add_task(run_real_pipeline, task_id, request)
    
    logger.info(f"Task created and queued: {task_id} (dataset={request.dataset}, mode={request.mode})")
    
    # 5. 立即返回 task_id
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        current_step="pending",
        progress=0,
        created_at=datetime.fromisoformat(task["created_at"]),
        updated_at=datetime.fromisoformat(task["updated_at"])
    )


async def run_simulated_pipeline(task_id: str, request: TaskRequest):
    """模拟训练流水线（后台执行）"""
    await simulate_training_pipeline(task_id, request)


async def run_real_pipeline(task_id: str, request: TaskRequest):
    """真实训练流水线（后台执行）"""
    try:
        # 构造与 LangGraph 兼容的 initial_state，并固定为当前 task_id
        initial = create_initial_state(request)
        initial["task_id"] = task_id
        etm_agent.active_tasks[task_id] = initial

        # 更新状态为运行中
        task_store.set_running(task_id)
        task_store.add_log(task_id, "start", "info", "开始执行 ETM 训练流水线")

        # 运行实际的 LangGraph pipeline（传入 task_id，复用上面写入的 initial_state）
        result = await etm_agent.run_pipeline(request, task_id=task_id)

        # 更新最终状态
        if result.get("status") == "completed":
            task_store.set_completed(task_id, {
                "metrics": result.get("metrics"),
                "topic_words": result.get("topic_words"),
                "visualization_paths": result.get("visualization_paths")
            })
        else:
            task_store.set_failed(task_id, result.get("error_message", "Unknown error"))

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed for {task_id}: {e}\n{traceback.format_exc()}")
        task_store.set_failed(task_id, str(e))


async def simulate_training_pipeline(task_id: str, request: TaskRequest):
    """模拟训练流水线（用于开发和演示）- 使用 TaskStore 持久化"""
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
        task_store.set_running(task_id)
        task_store.add_log(task_id, "start", "info", "开始模拟训练流水线")
        
        # 同步更新内存状态
        if task_id in etm_agent.active_tasks:
            etm_agent.active_tasks[task_id]["status"] = "running"
            etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        for step_name, message, progress_start, progress_end in steps:
            await asyncio.sleep(0.8)  # 每步等待0.8秒（加快演示速度）
            
            # 检查任务是否存在
            task = task_store.get_task(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found, stopping simulation")
                return
            
            # 检查是否被取消
            if task.get("status") == "cancelled":
                logger.info(f"Task {task_id} was cancelled, stopping simulation")
                return
            
            # 更新进度（持久化）
            task_store.update_progress(task_id, progress_end, step_name, message)
            task_store.add_log(task_id, step_name, "completed", message)
            
            # 同步更新内存状态
            if task_id in etm_agent.active_tasks:
                etm_agent.active_tasks[task_id]["current_step"] = step_name
                etm_agent.active_tasks[task_id]["progress"] = progress_end
                etm_agent.active_tasks[task_id]["message"] = message
                etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # 训练完成 - 生成模拟结果
        num_topics = request.num_topics or 20
        metrics = {
            "topic_coherence_avg": round(0.456 + (num_topics / 100), 4),
            "topic_diversity_td": round(0.789 - (num_topics / 200), 4),
            "topic_diversity_irbo": 0.85,
            "perplexity": round(123.45 - num_topics, 2)
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
        
        visualization_paths = [
            f"/api/results/{request.dataset}/{request.mode}/visualizations/topic_words.png",
            f"/api/results/{request.dataset}/{request.mode}/visualizations/topic_similarity.png",
            f"/api/results/{request.dataset}/{request.mode}/visualizations/doc_topics.png",
        ]
        
        # 持久化完成状态
        task_store.set_completed(task_id, {
            "metrics": metrics,
            "topic_words": topic_words,
            "visualization_paths": visualization_paths
        })
        task_store.add_log(task_id, "complete", "success", "训练完成")
        
        # 同步更新内存状态
        if task_id in etm_agent.active_tasks:
            etm_agent.active_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "metrics": metrics,
                "topic_words": topic_words,
                "visualization_paths": visualization_paths,
                "completed_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
        
        logger.info(f"Simulated pipeline completed for task {task_id}")
            
    except Exception as e:
        import traceback
        logger.error(f"Simulated pipeline failed for {task_id}: {e}\n{traceback.format_exc()}")
        task_store.set_failed(task_id, str(e))
        if task_id in etm_agent.active_tasks:
            etm_agent.active_tasks[task_id]["status"] = "failed"
            etm_agent.active_tasks[task_id]["error_message"] = str(e)
            etm_agent.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()


@router.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    dataset: Optional[str] = Query(None, description="Filter by dataset"),
    limit: int = Query(100, ge=1, le=500, description="Max number of tasks"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取任务列表（支持过滤和分页），仅返回当前用户的任务
    """
    tasks = []
    stored_tasks = task_store.get_tasks_list(
        user_id=current_user.id,
        status=status,
        dataset=dataset,
        limit=limit,
        offset=offset
    )
    
    for state in stored_tasks:
        tasks.append(_build_task_response(state))
    
    return tasks


@router.get("/tasks/stats", tags=["tasks"])
async def get_task_stats(current_user: User = Depends(get_current_active_user)):
    """获取任务统计信息（仅当前用户）"""
    tasks = task_store.get_tasks_list(user_id=current_user.id, limit=10000)
    stats = {"total": 0, "pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
    for t in tasks:
        stats["total"] += 1
        s = t.get("status", "pending")
        if s in stats:
            stats[s] += 1
    return stats


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """获取单个任务详情，仅当前用户可访问"""
    state = etm_agent.get_task_status(task_id)
    if not state:
        state = task_store.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    if not _task_owned_by_user(state, current_user.id):
        raise HTTPException(status_code=404, detail="Task not found")

    # 若任务关联 DLC，查询阿里云实时状态并回写 state
    dlc_job_id = state.get("dlc_job_id")
    if dlc_job_id and state.get("status") not in ("completed", "failed", "error"):
        try:
            from ..api.data_api import _get_dlc_job_status, _list_oss_result_files, DLC_SDK_AVAILABLE
            if DLC_SDK_AVAILABLE and settings.PAI_ENABLED:
                dlc_status = _get_dlc_job_status(dlc_job_id)
                if dlc_status:
                    state = dict(state)  # 避免修改原始对象
                    state["dlc_status"] = dlc_status
                    # DLC 状态 → 进度映射
                    _dlc_progress = {
                        "Creating": 5, "Created": 8, "Queuing": 10, "Waiting": 12,
                        "Scheduling": 15, "Preparing": 20, "Running": 50,
                        "Stopping": 95, "Succeeded": 100,
                    }
                    _dlc_msg = {
                        "Creating": "任务创建中", "Created": "任务已创建，等待调度",
                        "Queuing": "排队等待资源", "Waiting": "等待资源分配",
                        "Scheduling": "正在调度资源", "Preparing": "环境准备中",
                        "Running": "训练运行中", "Stopping": "任务停止中",
                        "Succeeded": "训练完成", "Failed": "训练失败",
                        "Stopped": "训练已停止",
                    }
                    if dlc_status in _dlc_progress:
                        state["progress"] = _dlc_progress[dlc_status]
                    if dlc_status in _dlc_msg:
                        # 计算已运行时长
                        created_at_str = state.get("created_at", "")
                        elapsed = ""
                        if created_at_str:
                            try:
                                from datetime import datetime as _dt
                                if isinstance(created_at_str, str):
                                    ca = _dt.fromisoformat(created_at_str)
                                else:
                                    ca = created_at_str
                                secs = int((_dt.now() - ca).total_seconds())
                                if secs >= 60:
                                    elapsed = f"（已运行 {secs // 60} 分 {secs % 60} 秒）"
                                else:
                                    elapsed = f"（已运行 {secs} 秒）"
                            except Exception:
                                pass
                        state["message"] = f"{_dlc_msg[dlc_status]}{elapsed}"
                    state["current_step"] = "dlc_training"
                    if dlc_status == "Succeeded":
                        state["status"] = "completed"
                        state["progress"] = 100
                        # 列出 OSS 结果文件并下载 metrics/topic_words
                        dataset_name = state.get("dataset", "")
                        task_mode = state.get("mode", "zero_shot")
                        oss_results = _list_oss_result_files(task_id, dataset_name)
                        completion_data: Dict[str, Any] = {"dlc_status": "Succeeded"}
                        if oss_results:
                            state["message"] = f"训练完成，共 {len(oss_results)} 个结果文件"
                            completion_data["oss_result_urls"] = oss_results
                        # 下载 metrics 和 topic_words
                        from ..api.data_api import _fetch_oss_result_data
                        oss_data = _fetch_oss_result_data(dataset_name, task_mode)
                        if oss_data.get("metrics"):
                            completion_data["metrics"] = oss_data["metrics"]
                            state["metrics"] = oss_data["metrics"]
                        if oss_data.get("topic_words"):
                            completion_data["topic_words"] = oss_data["topic_words"]
                            state["topic_words"] = oss_data["topic_words"]
                        task_store.set_completed(task_id, completion_data)
                    elif dlc_status in ("Failed", "Stopped"):
                        state["status"] = "failed"
                        task_store.set_failed(task_id, f"DLC: {dlc_status}")
                else:
                    # DLC 查询返回 None（网络错误等），保持当前 state 不变
                    logger.debug(f"DLC status query returned None for {dlc_job_id}, keeping current state")
        except Exception as e:
            logger.warning(f"DLC status query failed for {dlc_job_id}: {e}")

    # 对已完成但缺少 metrics/topic_words 的 DLC 任务，尝试从 OSS 回填
    if (state.get("status") == "completed"
            and state.get("dlc_job_id")
            and not state.get("metrics")):
        try:
            from ..api.data_api import _fetch_oss_result_data
            dataset_name = state.get("dataset", "")
            task_mode = state.get("mode", "zero_shot")
            if dataset_name:
                oss_data = _fetch_oss_result_data(dataset_name, task_mode)
                if oss_data.get("metrics") or oss_data.get("topic_words"):
                    state = dict(state)
                    updates: Dict[str, Any] = {}
                    if oss_data.get("metrics"):
                        state["metrics"] = oss_data["metrics"]
                        updates["metrics"] = oss_data["metrics"]
                    if oss_data.get("topic_words"):
                        state["topic_words"] = oss_data["topic_words"]
                        updates["topic_words"] = oss_data["topic_words"]
                    task_store.update_task(task_id, updates)
        except Exception as e:
            logger.debug(f"OSS backfill failed for {task_id}: {e}")

    return _build_task_response(state)


@router.get("/tasks/{task_id}/logs", tags=["tasks"])
async def get_task_logs(
    task_id: str,
    tail: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    """获取任务执行日志"""
    state = task_store.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    if not _task_owned_by_user(state, current_user.id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    logs = state.get("logs", [])
    return {
        "task_id": task_id,
        "status": state.get("status"),
        "logs": logs[-tail:],
        "total_count": len(logs)
    }


def _build_task_response(state: Dict[str, Any]) -> TaskResponse:
    """构建 TaskResponse 对象"""
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
    
    # 计算持续时间
    duration_seconds = None
    if completed_at and created_at:
        duration_seconds = (completed_at - created_at).total_seconds()
    
    return TaskResponse(
        task_id=state.get("task_id", "unknown"),
        status=TaskStatus(state.get("status", "pending")),
        current_step=state.get("current_step"),
        progress=_calculate_progress(state),
        message=state.get("message"),
        dlc_job_id=state.get("dlc_job_id"),
        dlc_status=state.get("dlc_status"),
        dataset=state.get("dataset"),
        mode=state.get("mode"),
        num_topics=state.get("num_topics"),
        metrics=state.get("metrics"),
        topic_words=state.get("topic_words"),
        visualization_paths=state.get("visualization_paths"),
        created_at=created_at,
        updated_at=updated_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        error_message=state.get("error_message")
    )


@router.delete("/tasks/{task_id}", tags=["tasks"])
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Cancel a running task"""
    state = task_store.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    if not _task_owned_by_user(state, current_user.id):
        raise HTTPException(status_code=404, detail="Task not found")
    success = await etm_agent.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")
    return {"message": "Task cancelled", "task_id": task_id}


# ==========================================
# Chat API Endpoint
# ==========================================

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(request: ChatRequest):
    """Chat with AI assistant using Qwen API"""
    try:
        response = chat_service.process_message(request)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")


# ==========================================
# Conversation History API Endpoints
# ==========================================

# In-memory storage for conversation history (in production, use database)
_conversation_history: Dict[str, List[Dict[str, Any]]] = {}


class ConversationHistoryRequest(BaseModel):
    """Request to save conversation history"""
    session_id: str
    messages: List[Dict[str, Any]]


@router.post("/chat/history", tags=["chat"])
async def save_conversation_history(request: ConversationHistoryRequest):
    """Save conversation history for a session"""
    try:
        # Store last 100 messages per session
        _conversation_history[request.session_id] = request.messages[-100:]
        return {"message": "History saved", "session_id": request.session_id, "message_count": len(_conversation_history[request.session_id])}
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save history: {str(e)}")


@router.get("/chat/history/{session_id}", tags=["chat"])
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = _conversation_history.get(session_id, [])
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.delete("/chat/history/{session_id}", tags=["chat"])
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        if session_id in _conversation_history:
            del _conversation_history[session_id]
        return {"message": "History cleared", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.post("/chat/suggestions", response_model=SuggestionsResponse, tags=["chat"])
async def get_suggestions(request: SuggestionsRequest = SuggestionsRequest()):
    """Get intelligent suggestions based on current context"""
    try:
        suggestions = chat_service.get_suggestions(request.context or {})
        suggestion_items = [
            SuggestionItem(
                text=s.get("text", ""),
                action=s.get("action", ""),
                description=s.get("description", ""),
                data=s.get("data")
            )
            for s in suggestions
        ]
        return SuggestionsResponse(suggestions=suggestion_items)
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


# ==========================================
# theta_1-main Legacy API 兼容接口
# ==========================================

class ChatV2Request(BaseModel):
    job_id: str
    message: str
    session_id: Optional[str] = None


@router.post("/chat/v2", tags=["chat"])
async def chat_v2_legacy(request: ChatV2Request = Body(...)):
    """Multi-turn conversation - theta_1-main 兼容"""
    try:
        req = ChatRequest(message=request.message, context={"job_id": request.job_id})
        resp = chat_service.process_message(req)
        return {"job_id": request.job_id, "message": resp.message, "status": "ok"}
    except Exception as e:
        logger.error(f"Chat v2 error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_job_to_dataset_mode(job_id: str) -> tuple:
    """job_id -> (dataset, mode)"""
    if "_" in job_id:
        for mode in ("zero_shot", "unsupervised", "supervised"):
            if job_id.endswith("_" + mode):
                return job_id[: -len(mode) - 1], mode
    return job_id, "zero_shot"


@router.post("/interpret/metrics", tags=["interpret"])
async def interpret_metrics_legacy(
    job_id: str = Body(...),
    language: str = Body("zh")
):
    """Interpret evaluation metrics - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    try:
        result_path = settings.get_result_path(dataset, mode)
        eval_dir = result_path / "evaluation"
        metrics = None
        if eval_dir.exists():
            mf = next(sorted(eval_dir.glob("metrics_*.json"), reverse=True), None)
            if mf:
                with open(mf) as f:
                    metrics = json.load(f)
        if not metrics:
            for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=5):
                if t.get("mode") == mode and t.get("metrics"):
                    metrics = t["metrics"]
                    break
        if not metrics:
            return {"job_id": job_id, "interpretations": [], "metrics_count": 0, "summary": "暂无评估指标数据"}
        interpretations = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                interpretations.append({
                    "metric": k,
                    "value": v,
                    "interpretation": f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                })
        return {"job_id": job_id, "interpretations": interpretations, "metrics_count": len(interpretations)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpret metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret/topics", tags=["interpret"])
async def interpret_topics_legacy(
    job_id: str = Body(...),
    language: str = Body("zh"),
    use_llm: bool = Body(True)
):
    """Interpret topic semantics - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    try:
        result_path = settings.get_result_path(dataset, mode)
        model_dir = result_path / "model"
        topic_words = {}
        if model_dir.exists():
            tf = next(sorted(model_dir.glob("topic_words_*.json"), reverse=True), None)
            if tf:
                with open(tf) as f:
                    topic_words = _parse_topic_words(json.load(f), 10)
        if not topic_words:
            for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=5):
                if t.get("mode") == mode and t.get("topic_words"):
                    topic_words = t["topic_words"]
                    break
        if not topic_words:
            return {"job_id": job_id, "topics": [], "topics_count": 0, "summary": "暂无主题词数据"}
        topics = []
        for tid, words in topic_words.items():
            topics.append({
                "topic_id": int(tid.replace("topic_", "")) if "topic_" in tid else int(tid or 0),
                "words": words[:10] if isinstance(words, list) else list(words)[:10],
                "interpretation": f"主题关键词: {', '.join(words[:5]) if isinstance(words, list) else str(words)[:100]}"
            })
        return {"job_id": job_id, "topics": topics, "topics_count": len(topics)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpret topics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret/summary", tags=["interpret"])
async def interpret_summary_legacy(
    job_id: str = Body(...),
    language: str = Body("zh")
):
    """Generate analysis summary - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    try:
        metrics_data = None
        for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=5):
            if t.get("mode") == mode:
                metrics_data = t.get("metrics", {})
                break
        if not metrics_data:
            result_path = settings.get_result_path(dataset, mode)
            eval_dir = result_path / "evaluation"
            if eval_dir.exists():
                mf = next(sorted(eval_dir.glob("metrics_*.json"), reverse=True), None)
                if mf:
                    with open(mf) as f:
                        metrics_data = json.load(f)
        summary = f"数据集 {dataset}，模式 {mode} 的分析结果摘要。"
        if metrics_data:
            summary += " 评估指标: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(metrics_data.items())[:7] if isinstance(v, (int, float)))
        return {"job_id": job_id, "summary": summary, "language": language}
    except Exception as e:
        logger.error(f"Interpret summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision/analyze", tags=["vision"])
async def vision_analyze_legacy(
    image_url: str = Body(...),
    question: str = Body(...),
    language: str = Body("zh")
):
    """Analyze image with Qwen3 Vision - theta_1-main 兼容"""
    if not settings.QWEN_API_KEY:
        raise HTTPException(status_code=501, detail="Configure QWEN_API_KEY for vision analysis")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.QWEN_API_KEY, base_url=settings.QWEN_API_BASE)
        resp = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": question}]}],
            max_tokens=1024
        )
        return {"success": True, "image_url": image_url, "question": question, "answer": resp.choices[0].message.content}
    except ImportError:
        raise HTTPException(status_code=501, detail="Install openai: pip install openai")
    except Exception as e:
        logger.error(f"Vision analyze error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/topics", tags=["jobs"])
async def jobs_topics_legacy(job_id: str):
    """Get topics for job - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    try:
        result_path = settings.get_result_path(dataset, mode)
        model_dir = result_path / "model"
        topic_words = {}
        if model_dir.exists():
            tf = next(sorted(model_dir.glob("topic_words_*.json"), reverse=True), None)
            if tf:
                with open(tf) as f:
                    topic_words = _parse_topic_words(json.load(f), 10)
        if not topic_words:
            for t in task_store.get_tasks_list(status="completed", dataset=dataset, limit=5):
                if t.get("mode") == mode and t.get("topic_words"):
                    topic_words = t["topic_words"]
                    break
        if not topic_words:
            raise HTTPException(status_code=404, detail=f"No topics for job {job_id}")
        topics = []
        for tid, words in topic_words.items():
            topics.append({
                "id": int(tid.replace("topic_", "")) if "topic_" in tid else int(tid or 0),
                "keywords": words[:10] if isinstance(words, list) else list(words)[:10],
            })
        return {"job_id": job_id, "topics": topics}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Jobs topics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/charts", tags=["jobs"])
async def jobs_charts_legacy(job_id: str):
    """Get charts list for job - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    result_path = settings.get_result_path(dataset, mode)
    viz_dir = result_path / "visualization"
    charts = {}
    wordclouds = []
    if viz_dir.exists():
        for f in viz_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                name = f.name
                charts[name] = f"/api/results/{dataset}/{mode}/visualizations/{name}"
                if "wordcloud" in name.lower():
                    wordclouds.append(charts[name])
    return {"job_id": job_id, "charts": charts, "wordclouds": wordclouds, "downloads": {}}


@router.get("/download/{job_id}/{filename}", tags=["jobs"])
async def download_job_file_legacy(job_id: str, filename: str):
    """Download file from job - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    result_path = settings.get_result_path(dataset, mode)
    viz_path = result_path / "visualization" / filename
    if viz_path.exists() and viz_path.is_file():
        return FileResponse(viz_path, filename=filename)
    eval_path = result_path / "evaluation" / filename
    if eval_path.exists() and eval_path.is_file():
        return FileResponse(eval_path, filename=filename)
    raise HTTPException(status_code=404, detail=f"File {filename} not found")


@router.post("/vision/analyze-chart", tags=["vision"])
async def vision_analyze_chart_legacy(
    job_id: str = Body(...),
    chart_name: str = Body(...),
    analysis_type: str = Body("general"),
    language: str = Body("zh")
):
    """Analyze chart from job results - theta_1-main 兼容"""
    dataset, mode = _resolve_job_to_dataset_mode(job_id)
    result_path = settings.get_result_path(dataset, mode)
    viz_path = result_path / "visualization" / chart_name
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail=f"Chart {chart_name} not found for job {job_id}")
    if not settings.QWEN_API_KEY:
        raise HTTPException(status_code=501, detail="Configure QWEN_API_KEY for vision analysis")
    try:
        import base64
        with open(viz_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()
        data_url = f"data:image/png;base64,{b64}"
        from openai import OpenAI
        client = OpenAI(api_key=settings.QWEN_API_KEY, base_url=settings.QWEN_API_BASE)
        prompt = f"分析此图表，类型 {analysis_type}，用{language}回答"
        resp = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": prompt}]}],
            max_tokens=1024
        )
        return {"success": True, "chart": chart_name, "answer": resp.choices[0].message.content}
    except ImportError:
        raise HTTPException(status_code=501, detail="Install openai: pip install openai")
    except Exception as e:
        logger.error(f"Vision analyze chart error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    text_column: Optional[str] = None  # If None, auto-detect
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
async def start_preprocessing(
    request: PreprocessingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
):
    """Start a preprocessing job (BOW + Embedding generation)"""
    import uuid

    # DLC 模式：预处理由云端 DLC 任务完成，本地无需运行，直接返回"已完成"
    if settings.DLC_ENABLED:
        job_id = f"preproc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        mock_job = {
            "job_id": job_id,
            "user_id": current_user.id,
            "dataset": request.dataset,
            "csv_path": "",
            "text_column": request.text_column,
            "config": request.config.model_dump() if request.config else {},
            "status": "completed",
            "progress": 100.0,
            "current_stage": "complete",
            "message": "DLC 模式：预处理将在阿里云 DLC 任务中完成，本地跳过",
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
            "updated_at": datetime.now().isoformat(),
        }
        _preprocessing_jobs[job_id] = mock_job
        return PreprocessingStatus(**{k: v for k, v in mock_job.items() if k in PreprocessingStatus.model_fields})

    # Validate dataset exists and user has access
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset).where(
                UserDataset.user_id == current_user.id,
                UserDataset.dataset_name == request.dataset
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found or access denied")
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
        "user_id": current_user.id,
        "dataset": request.dataset,
        "csv_path": str(csv_file),
        "text_column": request.text_column,
        "config": request.config.model_dump() if request.config else {},
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
    
    return PreprocessingStatus(**{k: v for k, v in job_status.items() if k in PreprocessingStatus.model_fields})


@router.get("/preprocessing/{job_id}", response_model=PreprocessingStatus, tags=["preprocessing"])
async def get_preprocessing_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get status of a preprocessing job"""
    if job_id not in _preprocessing_jobs:
        raise HTTPException(status_code=404, detail=f"Preprocessing job '{job_id}' not found")
    job_status = _preprocessing_jobs[job_id]
    if job_status.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail=f"Preprocessing job '{job_id}' not found")
    return PreprocessingStatus(**{k: v for k, v in job_status.items() if k in PreprocessingStatus.model_fields})


@router.get("/preprocessing", response_model=List[PreprocessingStatus], tags=["preprocessing"])
async def list_preprocessing_jobs(
    current_user: User = Depends(get_current_active_user),
):
    """List preprocessing jobs for current user"""
    return [
        PreprocessingStatus(**{k: v for k, v in job.items() if k in PreprocessingStatus.model_fields})
        for job in _preprocessing_jobs.values()
        if job.get("user_id") == current_user.id
    ]


@router.delete("/preprocessing/{job_id}", tags=["preprocessing"])
async def cancel_preprocessing(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Cancel a preprocessing job"""
    if job_id not in _preprocessing_jobs:
        raise HTTPException(status_code=404, detail=f"Preprocessing job '{job_id}' not found")
    job_status = _preprocessing_jobs[job_id]
    if job_status.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail=f"Preprocessing job '{job_id}' not found")
    if job_status["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    job_status["status"] = "failed"
    job_status["error_message"] = "Cancelled by user"
    job_status["updated_at"] = datetime.now().isoformat()
    
    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/preprocessing/check/{dataset}", tags=["preprocessing"])
async def check_preprocessing_status(
    dataset: str,
    current_user: User = Depends(get_current_active_user),
):
    """Check if a dataset has been preprocessed (has BOW and embeddings)"""
    # DLC 模式：预处理在云端完成，本地无需检查，直接告知前端可以训练
    if settings.DLC_ENABLED:
        return {
            "dataset": dataset,
            "has_bow": True,
            "has_embeddings": True,
            "ready_for_training": True,
            "bow_path": None,
            "embedding_path": None,
            "vocab_path": None,
        }

    async with async_session_maker() as session:
        result = await session.execute(
            select(UserDataset).where(
                UserDataset.user_id == current_user.id,
                UserDataset.dataset_name == dataset
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found or access denied")
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
        from ETM.preprocessing import EmbeddingProcessor, ProcessingConfig, read_csv_auto_encoding
        
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
        
        # Auto-detect text column if not specified or doesn't exist
        import pandas as pd
        text_column = job.get("text_column")
        csv_path = job["csv_path"]

        # 若上传的是 Excel 文件（.xlsx/.xls），先转换为临时 CSV
        if csv_path.lower().endswith(('.xlsx', '.xls')):
            import tempfile
            df_excel = pd.read_excel(csv_path)
            tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=os.path.dirname(csv_path))
            df_excel.to_csv(tmp_csv.name, index=False, encoding='utf-8')
            tmp_csv.close()
            csv_path = tmp_csv.name
            job["csv_path"] = csv_path
            logger.info(f"Converted Excel to CSV: {csv_path}")

        df_sample = read_csv_auto_encoding(csv_path, nrows=5)
        
        # If text_column is None, empty, or not in CSV columns, auto-detect
        if not text_column or text_column not in df_sample.columns:
            # Try common text column names (case-insensitive)
            possible_text_columns = [
                'text', 'content', 'cleaned_content', 'enhanced_text', 
                'cleaned_text', 'processed_text', 'body', 'message',
                'Text', 'Content', 'Body', 'Message',
                'narrative', 'complaint', 'description', 'comment',
                'review', 'article', 'post', 'tweet', 'summary'
            ]
            
            detected_column = None
            
            # First, try exact match (case-sensitive)
            for col in possible_text_columns:
                if col in df_sample.columns:
                    detected_column = col
                    break
            
            # If not found, try case-insensitive match
            if not detected_column:
                df_columns_lower = {col.lower(): col for col in df_sample.columns}
                for col_pattern in possible_text_columns:
                    if col_pattern.lower() in df_columns_lower:
                        detected_column = df_columns_lower[col_pattern.lower()]
                        break
            
            # If still not found, try to find columns containing text-related keywords
            if not detected_column:
                text_keywords = ['text', 'content', 'narrative', 'complaint', 'description', 
                                'comment', 'review', 'article', 'body', 'message', 'summary']
                for col in df_sample.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in text_keywords):
                        detected_column = col
                        break
            
            # If still not found, use the longest column name (likely to be text content)
            if not detected_column and len(df_sample.columns) > 0:
                detected_column = max(df_sample.columns, key=len)
                logger.warning(f"Using longest column name as text column: '{detected_column}'")
            
            if detected_column:
                text_column = detected_column
                original_col = job.get("text_column", "auto")
                logger.info(f"Auto-detected text column: '{text_column}' (requested: '{original_col}')")
            else:
                original_col = job.get("text_column", "auto")
                raise ValueError(
                    f"Column '{original_col}' not found in CSV. "
                    f"Available columns: {df_sample.columns.tolist()}. "
                    f"Please specify a valid text column."
                )
        
        # Run processing
        result = processor.process(
            csv_path=csv_path,
            text_column=text_column,
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

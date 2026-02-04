"""
DLC API Routes
PAI-DLC 训练任务管理 API
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.pai_service import pai_service, TrainingJob
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/dlc", tags=["DLC Training"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TrainingRequest(BaseModel):
    """训练任务请求"""
    dataset: str = Field(..., description="数据集名称")
    job_name: Optional[str] = Field(None, description="任务名称")
    num_topics: int = Field(20, ge=5, le=100, description="主题数量")
    epochs: int = Field(50, ge=10, le=200, description="训练轮数")
    mode: str = Field("zero_shot", description="训练模式: zero_shot, supervised, unsupervised")
    model_size: str = Field("0.6B", description="模型大小: 0.6B, 4B, 8B")
    text_column: Optional[str] = Field(None, description="文本列名")
    time_column: Optional[str] = Field(None, description="时间列名")


class TrainingResponse(BaseModel):
    """训练任务响应"""
    job_id: str
    job_name: str
    dataset: str
    status: str
    created_at: str
    updated_at: str
    dlc_job_id: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    result_path: Optional[str] = None


class JobListResponse(BaseModel):
    """任务列表响应"""
    total: int
    jobs: List[TrainingResponse]


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/jobs", response_model=TrainingResponse)
async def submit_training_job(request: TrainingRequest):
    """
    提交 DLC 训练任务
    
    将训练任务提交到阿里云 PAI-DLC 平台执行
    """
    logger.info(f"Submitting training job for dataset: {request.dataset}")
    
    try:
        job = pai_service.submit_training_job(
            dataset=request.dataset,
            job_name=request.job_name,
            num_topics=request.num_topics,
            epochs=request.epochs,
            mode=request.mode,
            model_size=request.model_size,
            text_column=request.text_column,
            time_column=request.time_column
        )
        
        logger.info(f"Job submitted: {job.job_id}, status: {job.status}")
        
        return TrainingResponse(
            job_id=job.job_id,
            job_name=job.job_name,
            dataset=job.dataset,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            dlc_job_id=job.dlc_job_id,
            progress=job.progress,
            message=job.message,
            result_path=job.result_path
        )
        
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=JobListResponse)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, succeeded, failed, stopped")
):
    """
    列出所有训练任务
    """
    jobs = pai_service.list_jobs(status=status)
    
    return JobListResponse(
        total=len(jobs),
        jobs=[
            TrainingResponse(
                job_id=j.job_id,
                job_name=j.job_name,
                dataset=j.dataset,
                status=j.status,
                created_at=j.created_at,
                updated_at=j.updated_at,
                dlc_job_id=j.dlc_job_id,
                progress=j.progress,
                message=j.message,
                result_path=j.result_path
            )
            for j in jobs
        ]
    )


@router.get("/jobs/{job_id}", response_model=TrainingResponse)
async def get_training_job(job_id: str):
    """
    获取训练任务状态
    """
    job = pai_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return TrainingResponse(
        job_id=job.job_id,
        job_name=job.job_name,
        dataset=job.dataset,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        dlc_job_id=job.dlc_job_id,
        progress=job.progress,
        message=job.message,
        result_path=job.result_path
    )


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """
    取消训练任务
    """
    success = pai_service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to cancel job {job_id}")
    
    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/logs")
async def get_training_logs(
    job_id: str,
    tail: int = Query(100, ge=10, le=1000, description="Number of log lines to return")
):
    """
    获取训练任务日志
    """
    logs = pai_service.get_job_logs(job_id, tail=tail)
    
    if logs is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return {"job_id": job_id, "logs": logs}


@router.get("/health")
async def dlc_health_check():
    """
    DLC 服务健康检查
    """
    from ..services.pai_service import DLC_AVAILABLE, OSS_AVAILABLE
    
    return {
        "status": "healthy",
        "dlc_sdk_available": DLC_AVAILABLE,
        "oss_sdk_available": OSS_AVAILABLE,
        "oss_bucket": pai_service.config.oss_bucket,
        "region": pai_service.config.region_id,
        "has_credentials": bool(pai_service.config.access_key_id)
    }

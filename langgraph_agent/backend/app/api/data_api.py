"""
theta_1 风格数据 API — 兼容 OSS + DLC 训练 + 本地模式

模式判断：
  - OSS + DLC 齐备 → presigned-url 返回 OSS 签名 URL，upload-complete 提交 DLC 任务
  - 仅 OSS → presigned-url 返回 OSS 签名 URL，upload-complete 启动本地流水线
  - 无 OSS → presigned-url 返回后端直传 URL，upload-complete 启动本地流水线
"""

import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import get_logger
from ..services.task_store import task_store
from ..models.user import User
from ..models.user_dataset import UserDataset
from ..services.auth_service import get_current_active_user
from ..core.database import async_session_maker
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

# OSS SDK（可选）
try:
    import oss2
    OSS_SDK_AVAILABLE = True
except ImportError:
    OSS_SDK_AVAILABLE = False

# DLC SDK（可选）
try:
    from alibabacloud_pai_dlc20201203.client import Client as DLCClient
    from alibabacloud_pai_dlc20201203 import models as dlc_models
    from alibabacloud_tea_openapi import models as open_api_models
    DLC_SDK_AVAILABLE = True
except ImportError:
    DLC_SDK_AVAILABLE = False

logger = get_logger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])

# 内存存储（与 theta_1 兼容；本地模式 job_id 即 task_id）
jobs_store: Dict[str, Dict[str, Any]] = {}


# ============ OSS / DLC 工具函数 ============

def _get_oss_bucket():
    """获取 OSS Bucket 实例（仅当 OSS 已配置且 SDK 可用时）"""
    if not OSS_SDK_AVAILABLE:
        return None
    if not settings.OSS_ENABLED:
        return None
    auth = oss2.Auth(settings.CLOUD_ACCESS_KEY_ID, settings.CLOUD_ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.RESOLVED_OSS_BUCKET)


def _generate_oss_signed_url(oss_path: str, method: str = 'PUT', expires: int = 3600,
                              content_type: str = 'text/csv') -> str:
    """生成 OSS 签名 URL"""
    bucket = _get_oss_bucket()
    if not bucket:
        raise RuntimeError("OSS 未配置或 SDK 不可用")
    headers = {'Content-Type': content_type} if method == 'PUT' else {}
    return bucket.sign_url(method, oss_path, expires, headers=headers)


def _create_dlc_client():
    """创建 DLC 客户端"""
    if not DLC_SDK_AVAILABLE:
        raise RuntimeError("DLC SDK 未安装: pip install alibabacloud_pai_dlc20201203")
    if not settings.CLOUD_ACCESS_KEY_ID or not settings.CLOUD_ACCESS_KEY_SECRET:
        raise RuntimeError("阿里云凭证未配置 (ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET)")
    region = settings.RESOLVED_REGION
    config = open_api_models.Config(
        access_key_id=settings.CLOUD_ACCESS_KEY_ID,
        access_key_secret=settings.CLOUD_ACCESS_KEY_SECRET,
        region_id=region,
        endpoint=f"pai-dlc.{region}.aliyuncs.com"
    )
    return DLCClient(config)


def _submit_dlc_job(job_id: str, user_id: int, dataset: str, num_topics: int, epochs: int,
                    mode: str, model_size: str, models: str) -> str:
    """提交 DLC 训练任务 — 逻辑移植自 theta_1-main/api/data_api.py"""
    client = _create_dlc_client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"THETA-{job_id[:8]}-{dataset}_{timestamp}"

    command = f"""
# ============================================
# THETA 完整训练流程
# ============================================
export ETM_BASE_DIR="/mnt"
export ETM_DATA_DIR="/mnt/data"
export ETM_RESULT_DIR="/mnt/result"
export ETM_CODE_DIR="/mnt/code/ETM"
export ETM_EMBEDDING_MODELS_DIR="/mnt/embedding_models"
export THETA_JOB_ID="{job_id}"
export THETA_USER_ID="{user_id}"
export ETM_USER_DATA_DIR="/mnt/data/{user_id}"
export ETM_USER_RESULT_DIR="/mnt/result/{user_id}"
export PYTHONUNBUFFERED=1

echo "=== Installing dependencies ==="
pip install transformers torch numpy scipy scikit-learn tqdm jieba pandas sentence-transformers gensim python-docx PyPDF2 pdfminer.six pdf2docx click

echo "=== Checking uploaded data files ==="
ls -la /mnt/data/{user_id}/{job_id}/

cd /mnt/code/ETM

# Step 0: 多格式文件转 CSV
echo "=== Step 0: Converting multi-format files to CSV ==="
NON_CSV_COUNT=$(find /mnt/data/{user_id}/{job_id}/ -type f ! -name '*.csv' | wc -l)
if [ "$NON_CSV_COUNT" -gt 0 ]; then
    echo "Found $NON_CSV_COUNT non-CSV files, converting..."
    mkdir -p /mnt/data/{user_id}/{dataset}
    python -m dataclean.main convert /mnt/data/{user_id}/{job_id}/ /mnt/data/{user_id}/{dataset}/{dataset}_cleaned.csv --language chinese --recursive
else
    echo "All CSV, copying..."
    mkdir -p /mnt/data/{user_id}/{dataset}
    cp /mnt/data/{user_id}/{job_id}/*.csv /mnt/data/{user_id}/{dataset}/
fi

# Step 1: 数据预处理
echo "=== Step 1: Data Preprocessing ==="
python -c "
import sys; sys.path.insert(0, '/mnt/code/ETM')
from prepare_data import prepare_baseline_data
prepare_baseline_data(dataset='{dataset}', vocab_size=5000, language='zh',
                      data_dir='/mnt/data/{user_id}', result_dir='/mnt/result/{user_id}', job_id='{job_id}')
print('Data preprocessing completed!')
"

# Step 2: 生成嵌入
echo "=== Step 2: Generating Embeddings ==="
if [ -d "/mnt/sbert" ]; then export SBERT_MODEL_PATH="/mnt/sbert"
elif [ -d "/mnt/embedding_models/sbert/all-MiniLM-L6-v2" ]; then export SBERT_MODEL_PATH="/mnt/embedding_models/sbert/all-MiniLM-L6-v2"
fi
python -c "
import sys; sys.path.insert(0, '/mnt/code/ETM')
from prepare_data import generate_embeddings_for_baseline
generate_embeddings_for_baseline(dataset='{dataset}', result_dir='/mnt/result/{user_id}', job_id='{job_id}')
print('Embedding generation completed!')
"

# Step 3: 模型训练
echo "=== Step 3: Model Training ==="
if echo "{models}" | grep -q "theta"; then
    python run_pipeline.py --dataset {dataset} --models theta \\
        --model_size {model_size} --mode {mode} \\
        --num_topics {num_topics} --epochs {epochs} --gpu 0 --job_id {job_id}
fi
BASELINES=$(echo "{models}" | sed 's/theta,//g' | sed 's/,theta//g' | sed 's/theta//g')
if [ -n "$BASELINES" ]; then
    python run_pipeline.py --dataset {dataset} --models $BASELINES \\
        --num_topics {num_topics} --epochs {epochs} --gpu 0 --job_id {job_id}
fi

echo "=== Training completed ==="
ls -la /mnt/result/{user_id}/{dataset}/{mode}/ 2>/dev/null || echo "Results in standard location"
"""

    workspace_id = settings.RESOLVED_WORKSPACE_ID
    image = settings.PAI_TRAINING_IMAGE
    instance_type = settings.DLC_INSTANCE_TYPE
    dataset_id = settings.OSS_DATASET_ID

    create_job_request = dlc_models.CreateJobRequest(
        workspace_id=workspace_id,
        display_name=job_name,
        job_type="PyTorchJob",
        job_specs=[
            dlc_models.JobSpec(
                type="Worker",
                image=image,
                pod_count=1,
                ecs_spec=instance_type
            )
        ],
        user_command=command,
        data_sources=[
            dlc_models.DataSourceItem(
                data_source_type="Dataset",
                data_source_id=dataset_id,
                mount_path="/mnt"
            )
        ],
        envs={
            "THETA_JOB_ID": job_id,
            "THETA_DATASET": dataset,
            "THETA_NUM_TOPICS": str(num_topics),
            "THETA_EPOCHS": str(epochs),
            "THETA_MODE": mode,
            "THETA_MODEL_SIZE": model_size
        }
    )

    response = client.create_job(create_job_request)
    logger.info(f"DLC job submitted: {response.body.job_id} for task {job_id}")
    return response.body.job_id


def _get_dlc_job_status(dlc_job_id: str) -> str:
    """查询 DLC 任务状态"""
    client = _create_dlc_client()
    request = dlc_models.GetJobRequest()
    response = client.get_job(dlc_job_id, request)
    return response.body.status


# ============ Pydantic 模型 ============

class PresignedUrlRequest(BaseModel):
    filename: str
    content_type: Optional[str] = "text/csv"


class PresignedUrlResponse(BaseModel):
    job_id: str
    upload_url: str
    oss_path: str
    content_type: str = "text/csv"
    expires_in: int = 3600


class UploadCompleteRequest(BaseModel):
    job_id: str
    dataset_name: Optional[str] = None
    num_topics: int = 20
    epochs: int = 100
    mode: str = "zero_shot"
    model_size: str = "0.6B"
    models: str = "theta"


class UploadCompleteResponse(BaseModel):
    job_id: str
    status: str
    message: str
    dlc_job_id: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    dlc_job_id: Optional[str] = None
    dlc_status: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ============ 辅助函数 ============

def _get_upload_dir() -> Path:
    """获取 job 上传临时目录"""
    d = settings.DATA_DIR / ".jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_to_task_response(job: Dict[str, Any]) -> Dict[str, Any]:
    """将 job/task 转为前端 TaskResponse 兼容格式"""
    job_id = job.get("job_id") or job.get("task_id")
    status = job.get("status", "unknown")
    if status in ("training", "submitting_dlc"):
        status = "running"
    elif status == "pending_upload" or status == "uploaded":
        status = "pending"
    return {
        "job_id": job_id,
        "task_id": job_id,
        "status": status,
        "dataset": job.get("dataset_name") or job.get("dataset"),
        "mode": job.get("mode"),
        "num_topics": job.get("num_topics"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error") or job.get("error_message"),
        "dlc_job_id": job.get("dlc_job_id"),
        "dlc_status": job.get("dlc_status"),
    }


# ============ API 端点 ============

@router.post("/presigned-url", response_model=PresignedUrlResponse)
async def get_presigned_url(
    request_body: PresignedUrlRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    """
    生成上传 URL
    - OSS 已配置 → 返回 OSS 签名 URL（前端直传 OSS）
    - OSS 未配置 → 返回后端直传 URL（前端 PUT 到本后端）
    """
    job_id = str(uuid.uuid4())
    oss_path = f"data/{current_user.id}/{job_id}/{request_body.filename}"
    content_type = request_body.content_type or "text/csv"

    jobs_store[job_id] = {
        "job_id": job_id,
        "status": "pending_upload",
        "oss_path": oss_path,
        "filename": request_body.filename,
        "user_id": current_user.id,
        "created_at": datetime.now().isoformat(),
        "dlc_job_id": None,
    }

    if settings.OSS_ENABLED and OSS_SDK_AVAILABLE:
        # OSS 模式：生成签名 URL，前端直传 OSS
        try:
            upload_url = _generate_oss_signed_url(oss_path, method='PUT', content_type=content_type)
            logger.info(f"OSS presigned URL generated for job {job_id}")
            return PresignedUrlResponse(
                job_id=job_id,
                upload_url=upload_url,
                oss_path=oss_path,
                content_type=content_type,
                expires_in=3600,
            )
        except Exception as e:
            logger.warning(f"OSS presigned URL failed, falling back to local: {e}")

    # 本地模式：返回后端直传 URL
    base_url = str(request.base_url).rstrip("/")
    upload_url = f"{base_url}/api/data/upload/{job_id}"
    return PresignedUrlResponse(
        job_id=job_id,
        upload_url=upload_url,
        oss_path=oss_path,
        content_type=content_type,
        expires_in=3600,
    )


@router.put("/upload/{job_id}")
async def upload_file_to_backend(
    job_id: str,
    request: Request,
):
    """
    本地模式：接收前端 PUT 上传的文件（raw body）
    仅当 job 存在且状态为 pending_upload 时接受
    """
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs_store[job_id]
    if job.get("status") != "pending_upload":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not in pending_upload status")

    upload_dir = _get_upload_dir() / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = job.get("filename") or "data.csv"
    file_path = upload_dir / filename

    try:
        content = await request.body()
        file_path.write_bytes(content)
    except Exception as e:
        logger.error(f"Failed to save upload for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    job["status"] = "uploaded"
    job["local_path"] = str(file_path)
    return {"job_id": job_id, "status": "uploaded"}


class PrepareDatasetRequest(BaseModel):
    """将 .jobs 中的文件落到 data/{dataset}/，供预处理使用"""
    job_id: str
    dataset_name: str


@router.post("/prepare-dataset")
async def prepare_dataset(
    request_body: PrepareDatasetRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    在开始分析前，将 job 中的上传文件复制到 dataset 目录（命名为 {dataset}_text_only.csv）。
    预处理和训练都依赖该路径，需在 startPreprocessing 之前调用。
    支持两种上传路径：
    - 本地直传：status=uploaded，local_path 已设置
    - OSS 直传：status=pending_upload，从 OSS 下载文件到本地
    """
    job_id = request_body.job_id
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs_store[job_id]

    # OSS 直传场景：文件已在 OSS，但后端 job 状态仍是 pending_upload
    # 从 OSS 下载文件到本地 .jobs 目录，更新 job 状态供后续流程使用
    if job.get("status") == "pending_upload" and settings.OSS_ENABLED and OSS_SDK_AVAILABLE and job.get("oss_path"):
        try:
            bucket = _get_oss_bucket()
            if bucket:
                upload_dir = _get_upload_dir() / job_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                filename = job.get("filename") or "data.csv"
                file_path = upload_dir / filename
                bucket.get_object_to_file(job["oss_path"], str(file_path))
                job["status"] = "uploaded"
                job["local_path"] = str(file_path)
                logger.info(f"Downloaded OSS file for job {job_id}: {job['oss_path']} -> {file_path}")
        except Exception as e:
            logger.error(f"Failed to download OSS file for job {job_id}: {e}")
            raise HTTPException(status_code=500, detail=f"从 OSS 下载文件失败: {e}")

    if job.get("status") != "uploaded" or not job.get("local_path"):
        raise HTTPException(status_code=400, detail=f"Job {job_id} has no uploaded file to prepare")

    safe_name = "".join(c for c in request_body.dataset_name if c.isalnum() or c in ("_", "-", " ")).strip().replace(" ", "_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset_name")

    local_path = Path(job["local_path"])
    dataset_dir = settings.DATA_DIR / str(current_user.id) / safe_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dest = dataset_dir / f"{safe_name}_text_only.csv"
    shutil.copy2(local_path, dest)

    async with async_session_maker() as session:
        try:
            ud = UserDataset(user_id=current_user.id, dataset_name=safe_name)
            session.add(ud)
            await session.commit()
        except IntegrityError:
            await session.rollback()

    return {"status": "ok", "dataset": safe_name, "path": str(dest)}


@router.post("/upload-complete", response_model=UploadCompleteResponse)
async def upload_complete(
    request_body: UploadCompleteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
):
    """
    上传完成通知 — 触发训练
    - DLC 已配置（OSS + DLC）→ 提交阿里云 DLC 训练任务
    - 仅本地 → 复制数据到 dataset 目录，启动 LangGraph 流水线
    """
    job_id = request_body.job_id
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs_store[job_id]

    dataset_name = request_body.dataset_name or job.get("filename", "data").replace(".csv", "").replace(".xlsx", "")
    safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ("_", "-", " ")).strip().replace(" ", "_")
    if not safe_name:
        safe_name = f"dataset_{job_id[:8]}"

    # ========== DLC 模式：OSS + DLC 齐备，且非模拟模式 ========== 
    if settings.DLC_ENABLED and DLC_SDK_AVAILABLE and not settings.SIMULATION_MODE:
        job["status"] = "submitting_dlc"
        job["dataset_name"] = safe_name
        job["mode"] = request_body.mode
        job["num_topics"] = request_body.num_topics
        job["model_size"] = request_body.model_size
        job["models"] = request_body.models
        job["task_id"] = job_id

        try:
            dlc_job_id = _submit_dlc_job(
                job_id=job_id,
                user_id=current_user.id,
                dataset=safe_name,
                num_topics=request_body.num_topics,
                epochs=request_body.epochs,
                mode=request_body.mode,
                model_size=request_body.model_size,
                models=request_body.models,
            )
            job["dlc_job_id"] = dlc_job_id
            job["status"] = "training"

            # 同时在 task_store 中创建记录，便于前端通过 /api/tasks 查询
            task_data = {
                "user_id": current_user.id,
                "dataset": safe_name,
                "mode": request_body.mode,
                "num_topics": request_body.num_topics,
                "epochs": request_body.epochs,
                "current_step": "dlc_training",
                "message": f"DLC 训练任务已提交 (dlc_job_id={dlc_job_id})",
                "dlc_job_id": dlc_job_id,
            }
            task_store.create_task(job_id, task_data)

            return UploadCompleteResponse(
                job_id=job_id,
                status="training",
                message="DLC 训练任务已提交",
                dlc_job_id=dlc_job_id,
            )
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            logger.error(f"DLC job submission failed for {job_id}: {e}")
            raise HTTPException(status_code=500, detail=f"提交 DLC 任务失败: {str(e)}")

    # ========== 本地模式 ==========
    # 从 .jobs/{job_id} 复制到 DATA_DIR/{dataset}
    if job.get("status") == "uploaded" and job.get("local_path"):
        local_path = Path(job["local_path"])
        dataset_dir = settings.DATA_DIR / str(current_user.id) / safe_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dest = dataset_dir / f"{safe_name}_text_only.csv"
        shutil.copy2(local_path, dest)
        try:
            shutil.rmtree(local_path.parent, ignore_errors=True)
        except Exception:
            pass

        async with async_session_maker() as session:
            try:
                ud = UserDataset(user_id=current_user.id, dataset_name=safe_name)
                session.add(ud)
                await session.commit()
            except IntegrityError:
                await session.rollback()

    elif job.get("status") == "pending_upload":
        dataset_dir = settings.DATA_DIR / str(current_user.id) / safe_name
        if not dataset_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset '{safe_name}' not found. Please upload file first via OSS or FormData.",
            )

    # 启动 LangGraph 流水线
    from ..schemas.agent import TaskRequest
    from ..api.routes import run_simulated_pipeline, run_real_pipeline
    from ..agents.etm_agent import etm_agent, create_initial_state

    task_request = TaskRequest(
        dataset=safe_name,
        mode=request_body.mode,
        num_topics=request_body.num_topics,
        epochs=request_body.epochs,
    )

    task_id = job_id
    task_data = {
        "user_id": current_user.id,
        "dataset": safe_name,
        "mode": request_body.mode,
        "num_topics": request_body.num_topics,
        "epochs": request_body.epochs,
        "current_step": "pending",
        "message": "任务已创建，等待执行...",
    }
    task = task_store.create_task(task_id, task_data)
    etm_agent.active_tasks[task_id] = task.copy()

    job["status"] = "training"
    job["dataset_name"] = safe_name
    job["mode"] = request_body.mode
    job["num_topics"] = request_body.num_topics
    job["task_id"] = task_id

    if settings.SIMULATION_MODE:
        background_tasks.add_task(run_simulated_pipeline, task_id, task_request)
    else:
        background_tasks.add_task(run_real_pipeline, task_id, task_request)

    return UploadCompleteResponse(
        job_id=job_id,
        status="training",
        message="训练任务已提交（本地模式）",
        dlc_job_id=None,
    )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """获取任务状态（优先 task_store，兼容 jobs_store；有 DLC 时远程查询）"""
    task = task_store.get_task(job_id)
    job = jobs_store.get(job_id)

    # 如果有 DLC job_id，查询远程状态
    dlc_job_id = None
    dlc_status = None
    if job and job.get("dlc_job_id"):
        dlc_job_id = job["dlc_job_id"]
    elif task and task.get("dlc_job_id"):
        dlc_job_id = task["dlc_job_id"]

    if dlc_job_id and DLC_SDK_AVAILABLE and settings.PAI_ENABLED:
        try:
            dlc_status = _get_dlc_job_status(dlc_job_id)
            # 同步更新本地状态
            if dlc_status in ("Succeeded", "Failed", "Stopped"):
                new_status = "completed" if dlc_status == "Succeeded" else "error"
                now = datetime.now().isoformat()
                if job:
                    job["status"] = new_status
                    job["completed_at"] = now
                    job["dlc_status"] = dlc_status
                if task:
                    if dlc_status == "Succeeded":
                        task_store.set_completed(job_id)
                    else:
                        task_store.set_failed(job_id, f"DLC job {dlc_status}")
            elif dlc_status in ("Running", "Creating", "Queuing"):
                if job:
                    job["dlc_status"] = dlc_status
        except Exception as e:
            dlc_status = f"查询失败: {str(e)}"

    if task:
        status = task.get("status", "unknown")
        return JobStatusResponse(
            job_id=job_id,
            status=status,
            dlc_job_id=dlc_job_id,
            dlc_status=dlc_status,
            created_at=task.get("created_at"),
            completed_at=task.get("completed_at"),
            error=task.get("error_message"),
        )
    if job:
        return JobStatusResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            dlc_job_id=dlc_job_id,
            dlc_status=dlc_status,
            created_at=job.get("created_at"),
            completed_at=job.get("completed_at"),
            error=job.get("error"),
        )
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.get("/jobs/{job_id}/results")
async def get_job_results(
    job_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    """
    获取训练结果
    - DLC 模式：返回 OSS 签名下载 URL
    - 本地模式：返回静态文件 URL
    """
    task = task_store.get_task(job_id)
    if not task:
        if job_id in jobs_store:
            job = jobs_store[job_id]
            task = {"dataset": job.get("dataset_name"), "mode": job.get("mode"), "status": job.get("status")}
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")

    dataset = task.get("dataset")
    mode = task.get("mode")
    if not dataset or not mode:
        raise HTTPException(status_code=500, detail="Missing dataset or mode in task")

    # DLC 模式：从 OSS 动态列出结果文件
    job = jobs_store.get(job_id)
    if job and job.get("dlc_job_id") and settings.OSS_ENABLED and OSS_SDK_AVAILABLE:
        result_base = f"result/{job_id}"
        download_urls = {}
        bucket = _get_oss_bucket()
        if bucket:
            try:
                import oss2 as _oss2
                for obj in _oss2.ObjectIterator(bucket, prefix=f"{result_base}/"):
                    # 跳过目录占位符（0 字节且以 / 结尾）
                    if obj.size == 0 and obj.key.endswith("/"):
                        continue
                    # 只返回有用的文件类型
                    if obj.key.endswith((".npy", ".npz", ".json", ".csv", ".png", ".jpg", ".txt", ".pt", ".md")):
                        rel = obj.key[len(result_base) + 1:]  # 相对路径
                        download_urls[rel] = _generate_oss_signed_url(obj.key, method='GET')
            except Exception as e:
                logger.error(f"Failed to list OSS results for {job_id}: {e}")
        return {"job_id": job_id, "result_base": result_base, "files": download_urls}

    # 本地模式
    result_path = settings.get_result_path(dataset, mode)
    files: Dict[str, str] = {}
    base_url = str(request.base_url).rstrip("/")

    for ext in ["*.png", "*.json", "*.npy"]:
        for f in result_path.rglob(ext):
            if f.is_file():
                rel = str(f.relative_to(result_path)).replace("\\", "/")
                files[rel] = f"{base_url}/static/results/{dataset}/{mode}/{rel}"

    return {
        "job_id": job_id,
        "result_base": str(result_path),
        "files": files,
    }


@router.get("/jobs")
async def list_jobs(
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
):
    """列出任务（合并 task_store 与 jobs_store）"""
    tasks = task_store.get_tasks_list(user_id=current_user.id, limit=limit * 2)
    seen = {t.get("task_id") for t in tasks}
    for jid, job in list(jobs_store.items()):
        if jid not in seen and job.get("user_id") == current_user.id:
            tasks.append({"task_id": jid, "job_id": jid, **job})
    tasks.sort(key=lambda x: x.get("created_at", "") or "", reverse=True)
    return {"jobs": [_job_to_task_response(t) for t in tasks[:limit]]}


# 图表文件扩展名白名单（等同 PowerShell: \.png|\.jpg|\.pdf|\.html）
_CHART_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".html"}


@router.get("/oss-datasets")
async def list_oss_datasets(
    current_user: User = Depends(get_current_active_user),
):
    """
    列出 OSS 上所有拥有可视化图表的数据集名称。
    扫描 result/0.6B/*/ 和 result/baseline/*/ 两级目录。
    返回：[{name, chart_count}] 按 chart_count 降序。
    """
    if not settings.OSS_ENABLED or not OSS_SDK_AVAILABLE:
        return {"datasets": [], "note": "OSS 未配置"}
    bucket = _get_oss_bucket()
    if not bucket:
        return {"datasets": [], "note": "OSS bucket 获取失败"}

    # 先枚举二级目录 result/0.6B/{ds}/ 和 result/baseline/{ds}/
    top_prefixes = ["result/0.6B/", "result/baseline/"]
    dataset_names: set = set()
    try:
        for tp in top_prefixes:
            for obj in oss2.ObjectIterator(bucket, prefix=tp, delimiter="/"):
                if hasattr(obj, "common_prefixes"):
                    # delimiter 模式下拿 prefix 字段
                    pass
            # 用 list_objects 取子前缀
            result = bucket.list_objects(prefix=tp, delimiter="/")
            for cp in result.prefix_list:
                # cp 形如 "result/0.6B/edu_data/"
                parts = cp.rstrip("/").split("/")
                if len(parts) >= 3:
                    dataset_names.add(parts[2])  # edu_data, test1, ...
    except Exception as e:
        logger.error(f"list_oss_datasets error: {e}")
        return {"datasets": [], "note": str(e)}

    # 对每个数据集统计图表数量（非阻塞，只扫 0.6B 前缀）
    results = []
    try:
        for ds in dataset_names:
            count = 0
            for prefix in [f"result/0.6B/{ds}/", f"result/baseline/{ds}/"]:
                for obj in oss2.ObjectIterator(bucket, prefix=prefix):
                    if obj.size == 0:
                        continue
                    ext = "." + obj.key.rsplit(".", 1)[-1].lower() if "." in obj.key else ""
                    if ext in _CHART_EXTENSIONS:
                        count += 1
            results.append({"name": ds, "chart_count": count})
    except Exception as e:
        logger.error(f"list_oss_datasets count error: {e}")

    results.sort(key=lambda x: x["chart_count"], reverse=True)
    return {"datasets": results}


@router.get("/oss-charts/{dataset}")
async def list_oss_chart_files(
    dataset: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    列出 OSS 上指定数据集的所有图表文件（png/jpg/pdf/html）。

    等同于：
        ossutil ls "oss://theta-prod-20260123/result/baseline/{dataset}/" -r
        | Select-String "\\.png|\\.jpg|\\.pdf|\\.html"

    同时扫描三条路径：
        result/baseline/{dataset}/   — Baseline 模型结果
        result/0.6B/{dataset}/       — THETA ETM 结果
        result/{dataset}/            — BOW / 共享文件

    仅当后端配置了 OSS 时返回有效数据；未配置则返回空列表并附说明。
    """
    if not settings.OSS_ENABLED or not OSS_SDK_AVAILABLE:
        return {
            "dataset": dataset,
            "charts": [],
            "total": 0,
            "note": "OSS 未配置，无法列举远程图表文件",
        }

    bucket = _get_oss_bucket()
    if not bucket:
        return {"dataset": dataset, "charts": [], "total": 0, "note": "OSS bucket 获取失败"}

    # 与 _list_oss_result_files 保持一致，同时扫三条路径
    prefixes = [
        f"result/baseline/{dataset}/",   # Baseline models
        f"result/0.6B/{dataset}/",        # THETA ETM (model_size=0.6B)
        f"result/{dataset}/",             # BOW/共享文件
    ]
    charts: List[Dict[str, str]] = []
    seen_keys: set = set()
    try:
        for prefix in prefixes:
            for obj in oss2.ObjectIterator(bucket, prefix=prefix):
                if obj.size == 0 and obj.key.endswith("/"):
                    continue
                if obj.key in seen_keys:
                    continue
                ext = "." + obj.key.rsplit(".", 1)[-1].lower() if "." in obj.key else ""
                if ext not in _CHART_EXTENSIONS:
                    continue
                seen_keys.add(obj.key)
                rel_path = obj.key[len(prefix):]
                charts.append({
                    "key": obj.key,
                    "path": rel_path,
                    "ext": ext.lstrip("."),
                    "size": obj.size,
                    "url": _generate_oss_signed_url(obj.key, method="GET"),
                })
    except Exception as e:
        logger.error(f"list_oss_chart_files error for dataset={dataset}: {e}")
        raise HTTPException(status_code=500, detail=f"列举 OSS 图表文件失败: {e}")

    return {"dataset": dataset, "charts": charts, "total": len(charts)}

"""
THETA 数据上传 API
提供前端直传 OSS 的签名 URL 生成和上传完成通知

功能：
1. 生成 OSS 签名 URL，前端直接上传到 OSS
2. 上传完成后通知后端，触发 DLC 训练任务
3. job_id 隔离用户数据
"""

import os
import uuid
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# 阿里云 OSS SDK
try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False
    print("[WARNING] oss2 未安装，请运行: pip install oss2")

# 阿里云 DLC SDK
try:
    from alibabacloud_pai_dlc20201203.client import Client as DLCClient
    from alibabacloud_pai_dlc20201203 import models as dlc_models
    from alibabacloud_tea_openapi import models as open_api_models
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False
    print("[WARNING] 阿里云 DLC SDK 未安装，请运行: pip install alibabacloud_pai_dlc20201203")


# ============ 配置 ============
ACCESS_KEY_ID = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
ACCESS_KEY_SECRET = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "oss-cn-shanghai.aliyuncs.com")
OSS_BUCKET_NAME = os.environ.get("OSS_BUCKET", "theta-prod-20260123")
REGION_ID = "cn-shanghai"
WORKSPACE_ID = os.environ.get("DLC_WORKSPACE_ID", "464377")

# DLC 配置
DEFAULT_IMAGE = "registry.cn-shanghai.aliyuncs.com/pai-dlc/pytorch-training:2.1-gpu-py310-cu121-ubuntu22.04"
DEFAULT_INSTANCE_TYPE = "ecs.gn7i-c8g1.2xlarge"
OSS_DATASET_ID = os.environ.get("OSS_DATASET_ID", "d-cvx2t6q7t8w3bnrvgl")
# ==============================

# 内存存储（生产环境应使用数据库）
jobs_store = {}

router = APIRouter(prefix="/api/data", tags=["data"])


# ============ 请求/响应模型 ============

class PresignedUrlRequest(BaseModel):
    """签名 URL 请求"""
    filename: str
    content_type: Optional[str] = "text/csv"


class PresignedUrlResponse(BaseModel):
    """签名 URL 响应"""
    job_id: str
    upload_url: str
    oss_path: str
    expires_in: int = 3600


class UploadCompleteRequest(BaseModel):
    """上传完成通知请求"""
    job_id: str
    dataset_name: Optional[str] = None
    num_topics: int = 20
    epochs: int = 100
    mode: str = "zero_shot"
    model_size: str = "0.6B"
    models: str = "theta"  # 支持多模型: "theta,lda,etm"


class UploadCompleteResponse(BaseModel):
    """上传完成响应"""
    job_id: str
    status: str
    message: str
    dlc_job_id: Optional[str] = None


class JobStatusResponse(BaseModel):
    """任务状态响应"""
    job_id: str
    status: str
    dlc_job_id: Optional[str] = None
    dlc_status: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ============ OSS 工具函数 ============

def get_oss_bucket():
    """获取 OSS Bucket 对象"""
    if not OSS_AVAILABLE:
        raise HTTPException(status_code=500, detail="OSS SDK 未安装")
    if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET:
        raise HTTPException(status_code=500, detail="OSS 凭证未配置")
    
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
    return bucket


def generate_signed_url(oss_path: str, expires: int = 3600) -> str:
    """生成 OSS 签名上传 URL"""
    bucket = get_oss_bucket()
    url = bucket.sign_url('PUT', oss_path, expires, headers={'Content-Type': 'text/csv'})
    return url


def generate_download_url(oss_path: str, expires: int = 3600) -> str:
    """生成 OSS 签名下载 URL"""
    bucket = get_oss_bucket()
    url = bucket.sign_url('GET', oss_path, expires)
    return url


# ============ DLC 工具函数 ============

def create_dlc_client():
    """创建 DLC 客户端"""
    if not DLC_AVAILABLE:
        raise HTTPException(status_code=500, detail="DLC SDK 未安装")
    if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET:
        raise HTTPException(status_code=500, detail="DLC 凭证未配置")
    
    config = open_api_models.Config(
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        region_id=REGION_ID,
        endpoint=f"pai-dlc.{REGION_ID}.aliyuncs.com"
    )
    return DLCClient(config)


def submit_dlc_job(job_id: str, dataset: str, num_topics: int, epochs: int, 
                   mode: str, model_size: str, models: str) -> str:
    """提交 DLC 训练任务"""
    client = create_dlc_client()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"THETA-{job_id[:8]}-{dataset}_{timestamp}"
    
    # 构建训练命令 - 使用 job_id 隔离结果，包含完整的数据预处理流程
    command = f"""
# ============================================
# THETA 完整训练流程
# 包含：数据预处理 -> 嵌入生成 -> 模型训练 -> 评估 -> 可视化
# ============================================

# 设置环境变量
export ETM_BASE_DIR="/mnt"
export ETM_DATA_DIR="/mnt/data"
export ETM_RESULT_DIR="/mnt/result"
export ETM_CODE_DIR="/mnt/code/ETM"
export ETM_EMBEDDING_MODELS_DIR="/mnt/embedding_models"
export THETA_JOB_ID="{job_id}"
export PYTHONUNBUFFERED=1

# 安装依赖
echo "=== Installing dependencies ==="
pip install transformers torch numpy scipy scikit-learn tqdm jieba pandas sentence-transformers gensim

# 检查数据文件
echo "=== Checking uploaded data files ==="
ls -la /mnt/data/{job_id}/

# 进入代码目录
cd /mnt/code/ETM

# ============================================
# Step 1: 数据预处理（清洗、分词、构建词汇表）
# ============================================
echo "=== Step 1: Data Preprocessing ==="

# 复制用户上传的数据到标准位置
mkdir -p /mnt/data/{dataset}
cp /mnt/data/{job_id}/*.csv /mnt/data/{dataset}/

# 运行数据预处理脚本
python -c "
import sys
sys.path.insert(0, '/mnt/code/ETM')
from prepare_data import prepare_baseline_data

# 预处理数据：清洗、分词、构建词汇表、生成 BOW 矩阵
prepare_baseline_data(
    dataset='{dataset}',
    vocab_size=5000,
    language='zh',
    data_dir='/mnt/data',
    result_dir='/mnt/result',
    job_id='{job_id}'
)
print('Data preprocessing completed!')
"

# ============================================
# Step 2: 生成嵌入（SBERT + Word2Vec）
# ============================================
echo "=== Step 2: Generating Embeddings ==="

# 检查 SBERT 模型路径
if [ -d "/mnt/sbert" ]; then
    export SBERT_MODEL_PATH="/mnt/sbert"
elif [ -d "/mnt/embedding_models/sbert/all-MiniLM-L6-v2" ]; then
    export SBERT_MODEL_PATH="/mnt/embedding_models/sbert/all-MiniLM-L6-v2"
fi

# 生成嵌入
python -c "
import sys
sys.path.insert(0, '/mnt/code/ETM')
from prepare_data import generate_embeddings_for_baseline

# 生成 SBERT 和 Word2Vec 嵌入
generate_embeddings_for_baseline(
    dataset='{dataset}',
    result_dir='/mnt/result',
    job_id='{job_id}'
)
print('Embedding generation completed!')
"

# ============================================
# Step 3: 模型训练
# ============================================
echo "=== Step 3: Model Training ==="

# 根据模型类型选择训练流程
if echo "{models}" | grep -q "theta"; then
    echo "=== Running THETA training ==="
    python run_pipeline.py --dataset {dataset} --models theta \\
        --model_size {model_size} --mode {mode} \\
        --num_topics {num_topics} --epochs {epochs} --gpu 0 \\
        --job_id {job_id}
fi

# 运行基线模型（如果指定）
BASELINES=$(echo "{models}" | sed 's/theta,//g' | sed 's/,theta//g' | sed 's/theta//g')
if [ -n "$BASELINES" ]; then
    echo "=== Running baseline models: $BASELINES ==="
    python run_pipeline.py --dataset {dataset} --models $BASELINES \\
        --num_topics {num_topics} --epochs {epochs} --gpu 0 \\
        --job_id {job_id}
fi

# ============================================
# 完成
# ============================================
echo "=== Training completed ==="
echo "Results saved to: /mnt/result/{job_id}/"
ls -la /mnt/result/{job_id}/ 2>/dev/null || echo "Results in standard location"
"""
    
    # 创建任务请求
    create_job_request = dlc_models.CreateJobRequest(
        workspace_id=WORKSPACE_ID,
        display_name=job_name,
        job_type="PyTorchJob",
        job_specs=[
            dlc_models.JobSpec(
                type="Worker",
                image=DEFAULT_IMAGE,
                pod_count=1,
                ecs_spec=DEFAULT_INSTANCE_TYPE
            )
        ],
        user_command=command,
        data_sources=[
            dlc_models.DataSourceItem(
                data_source_type="Dataset",
                data_source_id=OSS_DATASET_ID,
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
    return response.body.job_id


def get_dlc_job_status(dlc_job_id: str) -> str:
    """获取 DLC 任务状态"""
    client = create_dlc_client()
    request = dlc_models.GetJobRequest()
    response = client.get_job(dlc_job_id, request)
    return response.body.status


# ============ API 端点 ============

@router.post("/presigned-url", response_model=PresignedUrlResponse)
async def get_presigned_url(request: PresignedUrlRequest):
    """
    生成 OSS 签名上传 URL
    
    前端使用此 URL 直接上传文件到 OSS，无需经过后端
    """
    # 生成唯一 job_id
    job_id = str(uuid.uuid4())
    
    # 构建 OSS 路径：/data/{job_id}/{filename}
    oss_path = f"data/{job_id}/{request.filename}"
    
    # 生成签名 URL
    upload_url = generate_signed_url(oss_path)
    
    # 记录任务信息
    jobs_store[job_id] = {
        "job_id": job_id,
        "status": "pending_upload",
        "oss_path": oss_path,
        "filename": request.filename,
        "created_at": datetime.now().isoformat(),
        "dlc_job_id": None
    }
    
    return PresignedUrlResponse(
        job_id=job_id,
        upload_url=upload_url,
        oss_path=oss_path,
        expires_in=3600
    )


@router.post("/upload-complete", response_model=UploadCompleteResponse)
async def upload_complete(request: UploadCompleteRequest, background_tasks: BackgroundTasks):
    """
    上传完成通知 - 自动触发 DLC 训练任务
    
    前端上传完成后调用此接口，后端自动提交 DLC 任务
    """
    job_id = request.job_id
    
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_store[job_id]
    
    if job["status"] != "pending_upload":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not in pending_upload status")
    
    # 更新任务状态
    dataset_name = request.dataset_name or job["filename"].replace(".csv", "")
    job["status"] = "submitting_dlc"
    job["dataset_name"] = dataset_name
    job["num_topics"] = request.num_topics
    job["epochs"] = request.epochs
    job["mode"] = request.mode
    job["model_size"] = request.model_size
    job["models"] = request.models
    
    try:
        # 提交 DLC 任务
        dlc_job_id = submit_dlc_job(
            job_id=job_id,
            dataset=dataset_name,
            num_topics=request.num_topics,
            epochs=request.epochs,
            mode=request.mode,
            model_size=request.model_size,
            models=request.models
        )
        
        job["dlc_job_id"] = dlc_job_id
        job["status"] = "training"
        
        return UploadCompleteResponse(
            job_id=job_id,
            status="training",
            message="DLC 训练任务已提交",
            dlc_job_id=dlc_job_id
        )
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"提交 DLC 任务失败: {str(e)}")


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    获取任务状态
    """
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_store[job_id]
    
    # 如果有 DLC 任务，查询其状态
    dlc_status = None
    if job.get("dlc_job_id"):
        try:
            dlc_status = get_dlc_job_status(job["dlc_job_id"])
            
            # 更新本地状态
            if dlc_status in ["Succeeded", "Failed", "Stopped"]:
                job["status"] = "completed" if dlc_status == "Succeeded" else "error"
                job["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            dlc_status = f"查询失败: {str(e)}"
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        dlc_job_id=job.get("dlc_job_id"),
        dlc_status=dlc_status,
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """
    获取训练结果下载链接
    
    返回结果文件的签名下载 URL
    """
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_store[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")
    
    # 结果文件路径
    result_base = f"result/{job_id}"
    
    # 常见结果文件
    result_files = [
        "theta_k20.npy",
        "beta_k20.npy",
        "metrics_k20.json",
        "topic_words_k20.json",
        "training_history_k20.json",
        "visualization/topic_wordcloud.png",
        "visualization/topic_distribution.png",
        "visualization/doc_topic_heatmap.png"
    ]
    
    download_urls = {}
    bucket = get_oss_bucket()
    
    for file in result_files:
        oss_path = f"{result_base}/{file}"
        try:
            # 检查文件是否存在
            if bucket.object_exists(oss_path):
                download_urls[file] = generate_download_url(oss_path)
        except:
            pass
    
    return {
        "job_id": job_id,
        "result_base": result_base,
        "files": download_urls
    }


@router.get("/jobs")
async def list_jobs(limit: int = 20):
    """
    列出所有任务
    """
    jobs = list(jobs_store.values())
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": jobs[:limit]}

"""
Scripts API Routes
用于执行和管理服务器上的 bash 脚本
"""

from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.script_service import (
    script_service,
    ScriptInfo,
    AVAILABLE_SCRIPTS,
)
from ..core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/scripts", tags=["scripts"])


class ExecuteScriptRequest(BaseModel):
    """执行脚本请求"""
    script_id: str
    parameters: Dict[str, str] = {}


class ExecuteScriptResponse(BaseModel):
    """执行脚本响应"""
    job_id: str
    script_id: str
    script_name: str
    status: str
    message: str


class ScriptJobResponse(BaseModel):
    """脚本任务响应"""
    job_id: str
    script_id: str
    script_name: str
    parameters: Dict[str, str]
    status: str
    progress: float
    message: str
    logs: List[str]
    exit_code: Optional[int]
    created_at: str
    updated_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


@router.get("", response_model=List[ScriptInfo])
async def list_scripts():
    """获取所有可用脚本列表"""
    return script_service.get_available_scripts()


@router.get("/categories")
async def list_script_categories():
    """获取脚本分类"""
    categories = {}
    for script in AVAILABLE_SCRIPTS.values():
        if script.category not in categories:
            categories[script.category] = []
        categories[script.category].append({
            "id": script.id,
            "name": script.name,
            "description": script.description
        })
    return categories


@router.get("/{script_id}", response_model=ScriptInfo)
async def get_script(script_id: str):
    """获取指定脚本信息"""
    script_info = script_service.get_script_info(script_id)
    if not script_info:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return script_info


@router.post("/execute", response_model=ExecuteScriptResponse)
async def execute_script(request: ExecuteScriptRequest):
    """
    执行指定脚本

    参数说明：
    - script_id: 脚本 ID (如 "04_train_theta")
    - parameters: 脚本参数字典（值统一为字符串）

    示例：
    ```json
    {
        "script_id": "04_train_theta",
        "parameters": {
            "dataset": "edu_data",
            "model_size": "0.6B",
            "mode": "zero_shot",
            "num_topics": "20",
            "epochs": "100",
            "language": "zh"
        }
    }
    ```
    """
    script_info = script_service.get_script_info(request.script_id)
    if not script_info:
        raise HTTPException(status_code=404, detail=f"Script '{request.script_id}' not found")
    
    # 验证必需参数
    for param in script_info.parameters:
        if param.get("required", False) and param["name"] not in request.parameters:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required parameter: {param['name']}"
            )
    
    try:
        job_id = await script_service.execute_script(
            script_id=request.script_id,
            parameters=request.parameters
        )
        
        return ExecuteScriptResponse(
            job_id=job_id,
            script_id=request.script_id,
            script_name=script_info.name,
            status="pending",
            message=f"脚本 {script_info.name} 已提交执行"
        )
    
    except Exception as e:
        logger.error(f"Failed to execute script: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {str(e)}")


@router.get("/jobs", response_model=List[ScriptJobResponse])
async def list_jobs():
    """获取所有脚本任务列表"""
    jobs = script_service.get_all_jobs()
    return [ScriptJobResponse(**job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=ScriptJobResponse)
async def get_job(job_id: str):
    """获取指定任务状态"""
    job = script_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ScriptJobResponse(**job)


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 100):
    """
    获取任务日志
    
    参数：
    - tail: 返回最后N行日志，默认100
    """
    job = script_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    logs = job.get("logs", [])
    if tail > 0:
        logs = logs[-tail:]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "logs": logs,
        "total_lines": len(job.get("logs", []))
    }


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """取消正在执行的任务"""
    success = await script_service.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (not found or already finished)"
        )
    return {"message": "Job cancelled", "job_id": job_id}


# ==========================================
# 便捷端点：直接执行特定脚本
# ==========================================

class CleanDataRequest(BaseModel):
    input: str
    language: str
    text_column: Optional[str] = None
    label_columns: Optional[str] = None
    keep_all: bool = False
    min_words: int = 3


class PrepareDataRequest(BaseModel):
    dataset: str
    model: str
    vocab_size: int = 5000
    language: str = "english"
    model_size: str = "0.6B"
    mode: str = "zero_shot"
    label_column: Optional[str] = None
    time_column: Optional[str] = None
    bow_only: bool = False


class TrainThetaRequest(BaseModel):
    dataset: str
    model_size: str = "0.6B"
    mode: str = "zero_shot"
    num_topics: int = 20
    epochs: int = 100
    batch_size: int = 64
    hidden_dim: int = 512
    learning_rate: str = "0.002"
    language: str = "en"
    skip_viz: bool = False


class TrainBaselineRequest(BaseModel):
    dataset: str
    models: str
    num_topics: int = 20
    epochs: int = 100
    batch_size: int = 64
    language: str = "en"
    with_viz: bool = False


class VisualizeRequest(BaseModel):
    dataset: str
    baseline: bool = False
    model: Optional[str] = None
    num_topics: int = 20
    model_size: str = "0.6B"
    mode: str = "zero_shot"
    language: str = "en"
    dpi: int = 300


class EvaluateRequest(BaseModel):
    dataset: str
    model: str
    num_topics: int = 20
    model_size: str = "0.6B"
    mode: str = "zero_shot"


class CompareRequest(BaseModel):
    dataset: str
    models: str
    num_topics: int = 20


class QuickStartRequest(BaseModel):
    dataset: str
    language: str = "english"


@router.post("/clean", response_model=ExecuteScriptResponse)
async def clean_data(request: CleanDataRequest):
    """便捷端点：数据清洗 (02_clean_data.sh)"""
    params: Dict[str, str] = {
        "input": request.input,
        "language": request.language,
        "min_words": str(request.min_words),
    }
    if request.text_column:
        params["text_column"] = request.text_column
    if request.label_columns:
        params["label_columns"] = request.label_columns
    if request.keep_all:
        params["keep_all"] = "true"
    return await execute_script(ExecuteScriptRequest(
        script_id="02_clean_data", parameters=params,
    ))


@router.post("/prepare", response_model=ExecuteScriptResponse)
async def prepare_data(request: PrepareDataRequest):
    """便捷端点：数据准备 (03_prepare_data.sh)"""
    params: Dict[str, str] = {
        "dataset": request.dataset,
        "model": request.model,
        "vocab_size": str(request.vocab_size),
        "language": request.language,
    }
    if request.model == "theta":
        params["model_size"] = request.model_size
        params["mode"] = request.mode
    if request.label_column:
        params["label_column"] = request.label_column
    if request.time_column and request.model == "dtm":
        params["time_column"] = request.time_column
    if request.bow_only:
        params["bow-only"] = "true"
    return await execute_script(ExecuteScriptRequest(
        script_id="03_prepare_data", parameters=params,
    ))


@router.post("/train-theta", response_model=ExecuteScriptResponse)
async def train_theta(request: TrainThetaRequest):
    """便捷端点：THETA 训练 (04_train_theta.sh)"""
    params: Dict[str, str] = {
        "dataset": request.dataset,
        "model_size": request.model_size,
        "mode": request.mode,
        "num_topics": str(request.num_topics),
        "epochs": str(request.epochs),
        "batch_size": str(request.batch_size),
        "hidden_dim": str(request.hidden_dim),
        "learning_rate": request.learning_rate,
        "language": request.language,
    }
    if request.skip_viz:
        params["skip-viz"] = "true"
    return await execute_script(ExecuteScriptRequest(
        script_id="04_train_theta", parameters=params,
    ))


@router.post("/train-baseline", response_model=ExecuteScriptResponse)
async def train_baseline(request: TrainBaselineRequest):
    """便捷端点：基线模型训练 (05_train_baseline.sh)"""
    params: Dict[str, str] = {
        "dataset": request.dataset,
        "models": request.models,
        "num_topics": str(request.num_topics),
        "epochs": str(request.epochs),
        "batch_size": str(request.batch_size),
        "language": request.language,
    }
    if request.with_viz:
        params["with-viz"] = "true"
    return await execute_script(ExecuteScriptRequest(
        script_id="05_train_baseline", parameters=params,
    ))


@router.post("/visualize", response_model=ExecuteScriptResponse)
async def visualize_results(request: VisualizeRequest):
    """便捷端点：可视化 (06_visualize.sh)"""
    params: Dict[str, str] = {
        "dataset": request.dataset,
        "language": request.language,
        "dpi": str(request.dpi),
    }
    if request.baseline:
        params["baseline"] = "true"
        if request.model:
            params["model"] = request.model
        params["num_topics"] = str(request.num_topics)
    else:
        params["model_size"] = request.model_size
        params["mode"] = request.mode
    return await execute_script(ExecuteScriptRequest(
        script_id="06_visualize", parameters=params,
    ))


@router.post("/evaluate", response_model=ExecuteScriptResponse)
async def evaluate_model(request: EvaluateRequest):
    """便捷端点：模型评估 (07_evaluate.sh)"""
    params: Dict[str, str] = {
        "dataset": request.dataset,
        "model": request.model,
        "num_topics": str(request.num_topics),
    }
    if request.model == "theta":
        params["model_size"] = request.model_size
        params["mode"] = request.mode
    return await execute_script(ExecuteScriptRequest(
        script_id="07_evaluate", parameters=params,
    ))


@router.post("/compare", response_model=ExecuteScriptResponse)
async def compare_models(request: CompareRequest):
    """便捷端点：模型对比 (08_compare_models.sh)"""
    return await execute_script(ExecuteScriptRequest(
        script_id="08_compare_models",
        parameters={
            "dataset": request.dataset,
            "models": request.models,
            "num_topics": str(request.num_topics),
        },
    ))


@router.post("/quick-start", response_model=ExecuteScriptResponse)
async def quick_start(request: QuickStartRequest):
    """便捷端点：快速开始 (10/11_quick_start)"""
    script_id = (
        "11_quick_start_chinese" if request.language == "chinese"
        else "10_quick_start_english"
    )
    return await execute_script(ExecuteScriptRequest(
        script_id=script_id,
        parameters={"dataset": request.dataset},
    ))

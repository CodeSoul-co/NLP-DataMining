"""
Script Execution Service
用于执行服务器上的 bash 脚本并追踪执行状态
"""

import asyncio
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel

from ..core.logging import get_logger
from ..core.config import settings

logger = get_logger(__name__)


class ScriptStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScriptInfo(BaseModel):
    """脚本信息"""
    id: str
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    category: str


class ScriptJob(BaseModel):
    """脚本执行任务"""
    job_id: str
    script_id: str
    script_name: str
    parameters: Dict[str, str]
    status: ScriptStatus
    progress: float = 0.0
    message: str = ""
    logs: List[str] = []
    exit_code: Optional[int] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


SCRIPTS_DIR = Path(os.environ.get(
    "THETA_SCRIPTS_DIR",
    str(settings.BASE_DIR / "theta_1-main" / "scripts"),
))

AVAILABLE_SCRIPTS = {
    # ==================== preprocessing ====================
    "02_clean_data": ScriptInfo(
        id="02_clean_data",
        name="02_clean_data.sh",
        description="数据清洗 - 清洗原始文本 CSV/目录为主题建模标准格式，支持中英德西语",
        parameters=[
            {"name": "input", "type": "string", "required": True, "description": "输入 CSV 文件或目录路径（docx/txt）"},
            {"name": "language", "type": "string", "required": True, "description": "数据语言: english, chinese, german, spanish"},
            {"name": "text_column", "type": "string", "required": False, "description": "CSV 中文本列名称（CSV 模式必填）"},
            {"name": "label_columns", "type": "string", "required": False, "description": "需保留的标签/元数据列，逗号分隔"},
            {"name": "keep_all", "type": "boolean", "required": False, "description": "保留所有原始列，仅清洗文本列"},
            {"name": "preview", "type": "boolean", "required": False, "description": "仅预览 CSV 列和示例行后退出"},
            {"name": "output", "type": "string", "required": False, "description": "输出 CSV 路径（默认自动生成）"},
            {"name": "min_words", "type": "integer", "required": False, "default": "3", "description": "清洗后每文档最少词数"},
        ],
        category="preprocessing",
    ),
    "02_generate_embeddings": ScriptInfo(
        id="02_generate_embeddings",
        name="02_generate_embeddings.sh",
        description="Embedding 生成 - 使用 Qwen3-Embedding 生成文档向量（03 的子脚本，可独立用于恢复/重跑）",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式: zero_shot / unsupervised / supervised"},
            {"name": "model_size", "type": "string", "required": False, "default": "0.6B", "description": "Qwen 模型大小: 0.6B / 4B / 8B"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "16", "description": "批次大小（unsupervised 建议 <=8）"},
            {"name": "epochs", "type": "integer", "required": False, "default": "10", "description": "微调轮数（supervised/unsupervised）"},
            {"name": "learning_rate", "type": "string", "required": False, "default": "2e-5", "description": "学习率"},
            {"name": "max_length", "type": "integer", "required": False, "default": "512", "description": "最大序列长度"},
            {"name": "label_column", "type": "string", "required": False, "description": "标签列名（supervised 模式必填）"},
            {"name": "exp_dir", "type": "string", "required": False, "description": "保存到指定实验目录"},
            {"name": "gpu", "type": "integer", "required": False, "default": "0", "description": "GPU 设备 ID"},
        ],
        category="preprocessing",
    ),
    "03_prepare_data": ScriptInfo(
        id="03_prepare_data",
        name="03_prepare_data.sh",
        description="数据准备 (All-in-one) - 根据目标模型自动生成 BOW 矩阵和 Embedding",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "model", "type": "string", "required": True, "description": "目标模型: lda/hdp/stm/btm/nvdm/gsm/prodlda/ctm/etm/dtm/bertopic/theta"},
            {"name": "vocab_size", "type": "integer", "required": False, "default": "5000", "description": "词汇表大小"},
            {"name": "language", "type": "string", "required": False, "default": "english", "description": "数据语言，控制 BOW 分词: english / chinese"},
            {"name": "model_size", "type": "string", "required": False, "default": "0.6B", "description": "Qwen 模型大小（仅 theta）: 0.6B / 4B / 8B"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "Embedding 模式（仅 theta）: zero_shot / unsupervised / supervised"},
            {"name": "label_column", "type": "string", "required": False, "description": "标签列名（theta supervised 模式必填）"},
            {"name": "time_column", "type": "string", "required": False, "default": "year", "description": "时间列名（仅 dtm）"},
            {"name": "emb_epochs", "type": "integer", "required": False, "default": "10", "description": "Embedding 微调轮数（theta unsupervised/supervised）"},
            {"name": "emb_batch_size", "type": "integer", "required": False, "default": "8", "description": "Embedding 微调批次大小"},
            {"name": "emb_lr", "type": "string", "required": False, "default": "2e-5", "description": "Embedding 微调学习率"},
            {"name": "emb_max_length", "type": "integer", "required": False, "default": "512", "description": "Embedding 最大序列长度"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "32", "description": "Embedding 生成批次大小"},
            {"name": "gpu", "type": "integer", "required": False, "default": "0", "description": "GPU 设备 ID"},
            {"name": "bow-only", "type": "boolean", "required": False, "description": "仅生成 BOW，跳过 Embedding"},
            {"name": "check-only", "type": "boolean", "required": False, "description": "仅检查文件是否存在"},
            {"name": "exp_name", "type": "string", "required": False, "description": "实验名称标签"},
        ],
        category="preprocessing",
    ),

    # ==================== training ====================
    "04_train_theta": ScriptInfo(
        id="04_train_theta",
        name="04_train_theta.sh",
        description="THETA 训练 - 集成训练 + 评估 + 可视化的 THETA 主题模型",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称（需先通过 03_prepare_data 准备数据）"},
            {"name": "model_size", "type": "string", "required": False, "default": "0.6B", "description": "Qwen 模型大小: 0.6B / 4B / 8B"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "Embedding 模式: zero_shot / unsupervised / supervised"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量 K"},
            {"name": "epochs", "type": "integer", "required": False, "default": "100", "description": "训练轮数"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "64", "description": "训练批次大小"},
            {"name": "hidden_dim", "type": "integer", "required": False, "default": "512", "description": "编码器隐藏层维度"},
            {"name": "learning_rate", "type": "string", "required": False, "default": "0.002", "description": "学习率"},
            {"name": "kl_start", "type": "string", "required": False, "default": "0.0", "description": "KL 退火起始权重"},
            {"name": "kl_end", "type": "string", "required": False, "default": "1.0", "description": "KL 退火终止权重"},
            {"name": "kl_warmup", "type": "integer", "required": False, "default": "50", "description": "KL 退火预热轮数"},
            {"name": "patience", "type": "integer", "required": False, "default": "10", "description": "早停耐心值"},
            {"name": "gpu", "type": "integer", "required": False, "default": "0", "description": "GPU 设备 ID"},
            {"name": "language", "type": "string", "required": False, "default": "en", "description": "可视化语言: en / zh"},
            {"name": "data_exp", "type": "string", "required": False, "description": "数据实验 ID（默认自动选择最新）"},
            {"name": "exp_name", "type": "string", "required": False, "description": "训练实验名称标签"},
            {"name": "skip-train", "type": "boolean", "required": False, "description": "跳过训练，仅评估/可视化已有模型"},
            {"name": "skip-viz", "type": "boolean", "required": False, "description": "跳过可视化"},
        ],
        category="training",
    ),
    "05_train_baseline": ScriptInfo(
        id="05_train_baseline",
        name="05_train_baseline.sh",
        description="基线模型训练 - 支持 11 种主题模型 (lda/hdp/stm/btm/nvdm/gsm/prodlda/ctm/etm/dtm/bertopic)",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "models", "type": "string", "required": True, "description": "模型列表，逗号分隔: lda,hdp,stm,btm,nvdm,gsm,prodlda,ctm,etm,dtm,bertopic"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量（hdp/bertopic 忽略此参数）"},
            {"name": "epochs", "type": "integer", "required": False, "default": "100", "description": "训练轮数（神经网络模型）"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "64", "description": "批次大小"},
            {"name": "hidden_dim", "type": "integer", "required": False, "default": "512", "description": "隐藏层维度"},
            {"name": "learning_rate", "type": "string", "required": False, "default": "0.002", "description": "学习率"},
            {"name": "gpu", "type": "integer", "required": False, "default": "0", "description": "GPU 设备 ID"},
            {"name": "language", "type": "string", "required": False, "default": "en", "description": "可视化语言: en / zh"},
            {"name": "with-viz", "type": "boolean", "required": False, "description": "启用可视化（默认关闭）"},
            {"name": "skip-train", "type": "boolean", "required": False, "description": "跳过训练，仅评估/可视化已有模型"},
            {"name": "data_exp", "type": "string", "required": False, "description": "数据实验 ID"},
            {"name": "exp_name", "type": "string", "required": False, "description": "实验名称标签"},
            {"name": "max_iter", "type": "integer", "required": False, "default": "100", "description": "最大迭代次数（lda/stm）"},
            {"name": "max_topics", "type": "integer", "required": False, "default": "150", "description": "最大主题数（hdp）"},
            {"name": "n_iter", "type": "integer", "required": False, "default": "100", "description": "Gibbs 采样迭代次数（btm）"},
            {"name": "alpha", "type": "string", "required": False, "default": "1.0", "description": "Alpha 先验（hdp/btm）"},
            {"name": "beta", "type": "string", "required": False, "default": "0.01", "description": "Beta 先验（btm）"},
            {"name": "inference_type", "type": "string", "required": False, "default": "zeroshot", "description": "推断类型（ctm）: zeroshot / combined"},
            {"name": "dropout", "type": "string", "required": False, "default": "0.2", "description": "Dropout 比率（神经网络模型）"},
        ],
        category="training",
    ),
    "12_train_multi_gpu": ScriptInfo(
        id="12_train_multi_gpu",
        name="12_train_multi_gpu.sh",
        description="多 GPU 分布式训练 - 使用 PyTorch DDP 进行 THETA 多卡训练",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "num_gpus", "type": "integer", "required": False, "default": "2", "description": "使用的 GPU 数量"},
            {"name": "model_size", "type": "string", "required": False, "default": "0.6B", "description": "Qwen 模型大小: 0.6B / 4B / 8B"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "Embedding 模式: zero_shot / unsupervised / supervised"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量"},
            {"name": "epochs", "type": "integer", "required": False, "default": "100", "description": "训练轮数"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "64", "description": "每 GPU 批次大小"},
        ],
        category="training",
    ),

    # ==================== visualization ====================
    "06_visualize": ScriptInfo(
        id="06_visualize",
        name="06_visualize.sh",
        description="可视化 - 为已训练模型生成词云、热力图、t-SNE/UMAP、pyLDAvis 等可视化",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "baseline", "type": "boolean", "required": False, "description": "基线模型模式（不传则为 THETA 模式）"},
            {"name": "model", "type": "string", "required": False, "description": "基线模型名称（baseline 模式必填）"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量（baseline 模式使用）"},
            {"name": "model_size", "type": "string", "required": False, "default": "0.6B", "description": "THETA 模型大小: 0.6B / 4B / 8B"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "THETA Embedding 模式"},
            {"name": "model_exp", "type": "string", "required": False, "description": "模型实验 ID（默认自动选择最新）"},
            {"name": "language", "type": "string", "required": False, "default": "en", "description": "图表语言: en / zh"},
            {"name": "dpi", "type": "integer", "required": False, "default": "300", "description": "图片 DPI"},
        ],
        category="visualization",
    ),

    # ==================== evaluation ====================
    "07_evaluate": ScriptInfo(
        id="07_evaluate",
        name="07_evaluate.sh",
        description="模型评估 - 计算 TD/iRBO/NPMI/C_V/UMass/Exclusivity/PPL 七项指标",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "model", "type": "string", "required": True, "description": "模型名称: lda/hdp/stm/btm/nvdm/gsm/prodlda/ctm/etm/dtm/bertopic/theta"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量"},
            {"name": "vocab_size", "type": "integer", "required": False, "default": "5000", "description": "词汇表大小"},
            {"name": "model_size", "type": "string", "required": False, "default": "0.6B", "description": "THETA 模型大小（仅 model=theta 时使用）"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "THETA 模式（仅 model=theta 时使用）"},
        ],
        category="evaluation",
    ),
    "08_compare_models": ScriptInfo(
        id="08_compare_models",
        name="08_compare_models.sh",
        description="模型对比 - 读取各模型评估指标，生成跨模型对比表",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "models", "type": "string", "required": True, "description": "模型列表，逗号分隔"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量"},
            {"name": "output", "type": "string", "required": False, "description": "输出 CSV 文件路径（默认仅终端输出）"},
        ],
        category="evaluation",
    ),

    # ==================== quickstart ====================
    "10_quick_start_english": ScriptInfo(
        id="10_quick_start_english",
        name="10_quick_start_english.sh",
        description="英文快速开始 - 一键完成英文数据集的数据准备 + THETA 训练",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
        ],
        category="quickstart",
    ),
    "11_quick_start_chinese": ScriptInfo(
        id="11_quick_start_chinese",
        name="11_quick_start_chinese.sh",
        description="中文快速开始 - 一键完成中文数据集的数据准备 + THETA 训练 + 中文可视化",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
        ],
        category="quickstart",
    ),
}


class ScriptService:
    """脚本执行服务"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.scripts_dir = SCRIPTS_DIR
    
    def get_available_scripts(self) -> List[ScriptInfo]:
        """获取所有可用脚本"""
        return list(AVAILABLE_SCRIPTS.values())
    
    def get_script_info(self, script_id: str) -> Optional[ScriptInfo]:
        """获取指定脚本信息"""
        return AVAILABLE_SCRIPTS.get(script_id)
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Dict]:
        """获取所有任务"""
        return list(self.jobs.values())
    
    async def execute_script(
        self,
        script_id: str,
        parameters: Dict[str, str]
    ) -> str:
        """
        执行脚本
        返回 job_id
        """
        script_info = AVAILABLE_SCRIPTS.get(script_id)
        if not script_info:
            raise ValueError(f"Unknown script: {script_id}")
        
        # 生成任务ID
        job_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 创建任务记录
        job = {
            "job_id": job_id,
            "script_id": script_id,
            "script_name": script_info.name,
            "parameters": parameters,
            "status": ScriptStatus.PENDING.value,
            "progress": 0.0,
            "message": "任务已创建，等待执行",
            "logs": [],
            "exit_code": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None
        }
        
        self.jobs[job_id] = job
        
        # 异步执行脚本
        asyncio.create_task(self._run_script(job_id, script_info, parameters))
        
        return job_id
    
    async def _run_script(
        self,
        job_id: str,
        script_info: ScriptInfo,
        parameters: Dict[str, str]
    ):
        """实际执行脚本"""
        job = self.jobs[job_id]
        
        try:
            # 更新状态为运行中
            job["status"] = ScriptStatus.RUNNING.value
            job["message"] = f"正在执行 {script_info.name}"
            job["updated_at"] = datetime.now().isoformat()
            
            script_path = self.scripts_dir / script_info.name
            cmd_args = ["bash", str(script_path)]

            param_lookup = {p["name"]: p for p in script_info.parameters}
            for param_name, param_value in parameters.items():
                if param_name not in param_lookup:
                    continue
                pdef = param_lookup[param_name]
                if pdef.get("type") == "boolean":
                    if str(param_value).lower() in ("true", "1", "yes"):
                        cmd_args.append(f"--{param_name}")
                else:
                    cmd_args.extend([f"--{param_name}", str(param_value)])
            
            logger.info(f"Executing script: {' '.join(cmd_args)}")
            job["logs"].append(f"[{datetime.now().isoformat()}] 执行命令: {' '.join(cmd_args)}")
            
            # 执行脚本
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.scripts_dir.parent),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            # 实时读取输出
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_text = line.decode('utf-8', errors='replace').rstrip()
                job["logs"].append(f"[{datetime.now().isoformat()}] {line_text}")
                job["updated_at"] = datetime.now().isoformat()
                
                # 解析进度（如果脚本输出包含进度信息）
                if "%" in line_text:
                    try:
                        match = re.search(r'(\d+(?:\.\d+)?)\s*%', line_text)
                        if match:
                            job["progress"] = float(match.group(1))
                    except Exception:
                        pass
                
                # 更新消息
                job["message"] = line_text[:200] if len(line_text) > 200 else line_text
            
            # 等待进程完成
            await process.wait()
            exit_code = process.returncode
            
            job["exit_code"] = exit_code
            job["completed_at"] = datetime.now().isoformat()
            job["updated_at"] = datetime.now().isoformat()
            
            if exit_code == 0:
                job["status"] = ScriptStatus.COMPLETED.value
                job["progress"] = 100.0
                job["message"] = f"{script_info.name} 执行完成"
                logger.info(f"Script {script_info.name} completed successfully")
            else:
                job["status"] = ScriptStatus.FAILED.value
                job["error_message"] = f"脚本退出码: {exit_code}"
                job["message"] = f"执行失败 (退出码: {exit_code})"
                logger.error(f"Script {script_info.name} failed with exit code {exit_code}")
        
        except asyncio.CancelledError:
            job["status"] = ScriptStatus.CANCELLED.value
            job["message"] = "任务已取消"
            job["updated_at"] = datetime.now().isoformat()
            logger.info(f"Script job {job_id} cancelled")
        
        except Exception as e:
            job["status"] = ScriptStatus.FAILED.value
            job["error_message"] = str(e)
            job["message"] = f"执行出错: {str(e)}"
            job["updated_at"] = datetime.now().isoformat()
            logger.error(f"Script execution error: {e}", exc_info=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job["status"] not in [ScriptStatus.PENDING.value, ScriptStatus.RUNNING.value]:
            return False
        
        job["status"] = ScriptStatus.CANCELLED.value
        job["message"] = "任务已取消"
        job["updated_at"] = datetime.now().isoformat()
        
        return True


# 全局服务实例
script_service = ScriptService()

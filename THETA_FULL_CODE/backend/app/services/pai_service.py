"""
PAI-DLC Service
阿里云 PAI-DLC 训练任务管理服务

功能：
1. 提交 DLC 训练任务
2. 监控任务状态
3. 获取训练结果
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# Aliyun SDK
try:
    from alibabacloud_pai_dlc20201203.client import Client as DLCClient
    from alibabacloud_pai_dlc20201203 import models as dlc_models
    from alibabacloud_tea_openapi import models as open_api_models
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False
    print("[WARNING] alibabacloud_pai_dlc20201203 not installed. DLC features disabled.")

try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False
    print("[WARNING] oss2 not installed. OSS features disabled.")


def _get_dlc_config():
    """从 settings 获取 DLC 配置"""
    from ..core.config import settings
    return DLCConfig(
        access_key_id=settings.ALIBABA_CLOUD_ACCESS_KEY_ID,
        access_key_secret=settings.ALIBABA_CLOUD_ACCESS_KEY_SECRET,
        region_id=settings.PAI_REGION_ID,
        workspace_id=settings.PAI_WORKSPACE_ID,
        oss_bucket=settings.OSS_BUCKET,
        oss_endpoint=settings.OSS_ENDPOINT,
    )


@dataclass
class DLCConfig:
    """DLC 配置"""
    # 阿里云认证
    access_key_id: str = ""
    access_key_secret: str = ""
    region_id: str = "cn-shanghai"
    
    # PAI-DLC 配置
    workspace_id: str = ""
    
    # OSS 配置
    oss_bucket: str = "theta-prod-20260123"
    oss_endpoint: str = "oss-cn-shanghai-internal.aliyuncs.com"
    oss_code_path: str = "code/"
    oss_data_path: str = "data/"
    oss_result_path: str = "result/"
    oss_model_path: str = "embedding_models/"
    
    # 训练配置
    default_image: str = "registry.cn-shanghai.aliyuncs.com/pai-dlc/pytorch-training:2.1-gpu-py310-cu121-ubuntu22.04"
    default_instance_type: str = "ecs.gn6i-c4g1.xlarge"  # V100 16GB
    default_instance_count: int = 1


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    job_name: str
    dataset: str
    status: str  # pending, running, succeeded, failed, stopped
    created_at: str
    updated_at: str
    dlc_job_id: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    result_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class PAIService:
    """PAI-DLC 服务"""
    
    def __init__(self, config: Optional[DLCConfig] = None):
        self.config = config or _get_dlc_config()
        self._dlc_client = None
        self._oss_bucket = None
        self._jobs: Dict[str, TrainingJob] = {}
    
    @property
    def dlc_client(self):
        """懒加载 DLC 客户端"""
        if self._dlc_client is None and DLC_AVAILABLE:
            config = open_api_models.Config(
                access_key_id=self.config.access_key_id,
                access_key_secret=self.config.access_key_secret,
                region_id=self.config.region_id
            )
            config.endpoint = f"pai-dlc.{self.config.region_id}.aliyuncs.com"
            self._dlc_client = DLCClient(config)
        return self._dlc_client
    
    @property
    def oss_bucket(self):
        """懒加载 OSS Bucket"""
        if self._oss_bucket is None and OSS_AVAILABLE:
            auth = oss2.Auth(
                self.config.access_key_id,
                self.config.access_key_secret
            )
            self._oss_bucket = oss2.Bucket(
                auth,
                self.config.oss_endpoint,
                self.config.oss_bucket
            )
        return self._oss_bucket
    
    def submit_training_job(
        self,
        dataset: str,
        job_name: Optional[str] = None,
        num_topics: int = 20,
        epochs: int = 50,
        mode: str = "zero_shot",
        model_size: str = "0.6B",
        text_column: Optional[str] = None,
        time_column: Optional[str] = None,
        **kwargs
    ) -> TrainingJob:
        """
        提交 DLC 训练任务
        
        Args:
            dataset: 数据集名称
            job_name: 任务名称
            num_topics: 主题数量
            epochs: 训练轮数
            mode: 训练模式 (zero_shot, supervised, unsupervised)
            model_size: 模型大小 (0.6B, 4B, 8B)
            text_column: 文本列名
            time_column: 时间列名
        
        Returns:
            TrainingJob: 训练任务对象
        """
        # 生成任务 ID
        job_id = f"theta_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_name = job_name or f"THETA-{dataset}-{mode}"
        
        # 创建任务对象
        job = TrainingJob(
            job_id=job_id,
            job_name=job_name,
            dataset=dataset,
            status="pending",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # 构建训练命令
        oss_base = f"oss://{self.config.oss_bucket}"
        
        command = f"""
# 设置环境变量
export THETA_BASE="{oss_base}"
export THETA_MODEL_SIZE="{model_size}"
export THETA_DATA_DIR="{oss_base}/{self.config.oss_data_path}"
export THETA_RESULT_DIR="{oss_base}/{self.config.oss_result_path}"
export THETA_MODEL_DIR="{oss_base}/{self.config.oss_model_path}"

# 安装依赖
pip install -r /workspace/code/ETM/requirements.txt

# 运行完整流水线
cd /workspace/code
bash scripts/run_full_pipeline.sh "{job_id}" "{dataset}" "{text_column or 'text'}" "{time_column or ''}" "{num_topics}"
"""
        
        if not DLC_AVAILABLE:
            # 模拟模式（用于测试）
            job.status = "pending"
            job.message = "DLC SDK not available, running in simulation mode"
            self._jobs[job_id] = job
            return job
        
        try:
            # 创建 DLC 任务
            create_job_request = dlc_models.CreateJobRequest(
                workspace_id=self.config.workspace_id,
                display_name=job_name,
                job_type="TFJob",
                job_specs=[
                    dlc_models.JobSpec(
                        type="Worker",
                        image=self.config.default_image,
                        pod_count=self.config.default_instance_count,
                        ecs_spec=self.config.default_instance_type,
                        resource_config=dlc_models.ResourceConfig(
                            cpu="4",
                            memory="16Gi",
                            gpu="1"
                        )
                    )
                ],
                user_command=command,
                data_sources=[
                    dlc_models.DataSourceItem(
                        data_source_type="OSS",
                        data_source_id=f"oss://{self.config.oss_bucket}/",
                        mount_path="/workspace"
                    )
                ],
                code_source=dlc_models.CodeSource(
                    code_source_type="OSS",
                    mount_path="/workspace/code",
                    code_source_id=f"oss://{self.config.oss_bucket}/{self.config.oss_code_path}"
                ),
                envs={
                    "THETA_JOB_ID": job_id,
                    "THETA_DATASET": dataset,
                    "THETA_MODE": mode,
                    "THETA_NUM_TOPICS": str(num_topics),
                    "THETA_EPOCHS": str(epochs)
                }
            )
            
            response = self.dlc_client.create_job(create_job_request)
            
            job.dlc_job_id = response.body.job_id
            job.status = "running"
            job.message = "Job submitted to DLC"
            
        except Exception as e:
            job.status = "failed"
            job.message = f"Failed to submit job: {str(e)}"
        
        self._jobs[job_id] = job
        return job
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """获取任务状态"""
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        # 如果有 DLC job ID，从 DLC 获取最新状态
        if job.dlc_job_id and DLC_AVAILABLE and self.dlc_client:
            try:
                request = dlc_models.GetJobRequest(job_id=job.dlc_job_id)
                response = self.dlc_client.get_job(request)
                
                dlc_status = response.body.status
                status_map = {
                    "Creating": "pending",
                    "Queuing": "pending",
                    "Running": "running",
                    "Succeeded": "succeeded",
                    "Failed": "failed",
                    "Stopped": "stopped"
                }
                
                job.status = status_map.get(dlc_status, "unknown")
                job.updated_at = datetime.now().isoformat()
                
                # 如果成功，获取结果路径
                if job.status == "succeeded":
                    job.result_path = f"oss://{self.config.oss_bucket}/{self.config.oss_result_path}job_{job_id}/"
                    job.progress = 100.0
                    
            except Exception as e:
                job.message = f"Failed to get job status: {str(e)}"
        
        return job
    
    def list_jobs(self, status: Optional[str] = None) -> List[TrainingJob]:
        """列出所有任务"""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.dlc_job_id and DLC_AVAILABLE and self.dlc_client:
            try:
                request = dlc_models.StopJobRequest(job_id=job.dlc_job_id)
                self.dlc_client.stop_job(request)
                job.status = "stopped"
                job.updated_at = datetime.now().isoformat()
                return True
            except Exception as e:
                job.message = f"Failed to cancel job: {str(e)}"
                return False
        
        job.status = "stopped"
        return True
    
    def get_job_logs(self, job_id: str, tail: int = 100) -> Optional[str]:
        """获取任务日志"""
        job = self._jobs.get(job_id)
        if not job or not job.dlc_job_id:
            return None
        
        if not DLC_AVAILABLE or not self.dlc_client:
            return "DLC SDK not available"
        
        try:
            request = dlc_models.GetJobEventsRequest(
                job_id=job.dlc_job_id,
                max_events_num=tail
            )
            response = self.dlc_client.get_job_events(request)
            
            logs = []
            for event in response.body.events or []:
                logs.append(f"[{event.time}] {event.message}")
            
            return "\n".join(logs)
            
        except Exception as e:
            return f"Failed to get logs: {str(e)}"
    
    def upload_dataset(self, local_path: str, dataset_name: str) -> str:
        """上传数据集到 OSS"""
        if not OSS_AVAILABLE or not self.oss_bucket:
            raise RuntimeError("OSS SDK not available")
        
        oss_path = f"{self.config.oss_data_path}{dataset_name}/"
        
        # 上传文件
        if os.path.isfile(local_path):
            filename = os.path.basename(local_path)
            self.oss_bucket.put_object_from_file(
                f"{oss_path}{filename}",
                local_path
            )
        elif os.path.isdir(local_path):
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, local_path)
                    self.oss_bucket.put_object_from_file(
                        f"{oss_path}{rel_path}",
                        file_path
                    )
        
        return f"oss://{self.config.oss_bucket}/{oss_path}"
    
    def download_results(self, job_id: str, local_path: str) -> bool:
        """下载训练结果"""
        job = self._jobs.get(job_id)
        if not job or job.status != "succeeded":
            return False
        
        if not OSS_AVAILABLE or not self.oss_bucket:
            return False
        
        oss_prefix = f"{self.config.oss_result_path}job_{job_id}/"
        
        os.makedirs(local_path, exist_ok=True)
        
        for obj in oss2.ObjectIterator(self.oss_bucket, prefix=oss_prefix):
            rel_path = obj.key[len(oss_prefix):]
            if rel_path:
                local_file = os.path.join(local_path, rel_path)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                self.oss_bucket.get_object_to_file(obj.key, local_file)
        
        return True


# 全局服务实例
pai_service = PAIService()

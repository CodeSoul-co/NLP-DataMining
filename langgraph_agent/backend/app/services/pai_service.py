"""
PAI Service
Alibaba Cloud PAI-DLC (Training) and PAI-EAS (Inference) integration
"""

import json
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class PAIDLCService:
    """PAI-DLC service for training jobs"""
    
    def __init__(self):
        self.enabled = settings.PAI_ENABLED
        if self.enabled:
            from alibabacloud_pai_dlc20201203.client import Client as DLCClient
            from alibabacloud_tea_openapi import models as open_api_models
            from alibabacloud_credentials import providers
            
            config = open_api_models.Config(
                access_key_id=settings.PAI_ACCESS_KEY_ID,
                access_key_secret=settings.PAI_ACCESS_KEY_SECRET,
                region_id=settings.PAI_REGION
            )
            config.endpoint = f"pai-dlc.{settings.PAI_REGION}.aliyuncs.com"
            self.client = DLCClient(config)
            logger.info("PAI-DLC client initialized")
        else:
            self.client = None
            logger.warning("PAI-DLC is not configured, running in simulation mode")

    async def submit_training_job(
        self,
        task_id: str,
        dataset_oss_path: str,
        output_oss_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit a training job to PAI-DLC
        
        Args:
            task_id: Unique task identifier
            dataset_oss_path: OSS path to input dataset
            output_oss_path: OSS path for output results
            config: Training configuration
        
        Returns:
            Job info including job_id
        """
        if not self.enabled:
            # Simulation mode
            logger.info(f"[SIMULATION] Would submit training job: {task_id}")
            return {
                "job_id": f"simulated-{task_id}",
                "status": "Running",
                "message": "Simulation mode - no actual job submitted"
            }
        
        from alibabacloud_pai_dlc20201203 import models as dlc_models
        
        # Prepare training command
        num_topics = config.get("num_topics", 20)
        epochs = config.get("epochs", 50)
        batch_size = config.get("batch_size", 64)
        
        command = f"""
python /app/ETM/train_etm.py \\
    --data_path {dataset_oss_path} \\
    --output_path {output_oss_path} \\
    --num_topics {num_topics} \\
    --epochs {epochs} \\
    --batch_size {batch_size} \\
    --task_id {task_id}
"""
        
        # Create job request
        request = dlc_models.CreateJobRequest(
            display_name=f"theta-etm-{task_id[:8]}",
            job_type="TFJob",
            resource_id=settings.PAI_RESOURCE_GROUP_ID,
            job_specs=[
                dlc_models.JobSpec(
                    type="Worker",
                    image=settings.PAI_TRAINING_IMAGE,
                    pod_count=1,
                    resource_config=dlc_models.ResourceConfig(
                        cpu="4",
                        memory="16Gi",
                        gpu="1",
                        gpu_type="V100"
                    ),
                    envs=[
                        dlc_models.EnvVar(name="TASK_ID", value=task_id),
                        dlc_models.EnvVar(name="OSS_ENDPOINT", value=settings.OSS_ENDPOINT),
                    ]
                )
            ],
            user_command=command,
            data_sources=[
                dlc_models.DataSourceItem(
                    data_source_type="OSS",
                    mount_path="/data",
                    uri=f"oss://{settings.OSS_BUCKET_NAME}/"
                )
            ]
        )
        
        try:
            response = self.client.create_job(settings.PAI_WORKSPACE_ID, request)
            job_id = response.body.job_id
            logger.info(f"Training job submitted: {job_id}")
            return {
                "job_id": job_id,
                "status": "Submitted",
                "message": "Job submitted successfully"
            }
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status"""
        if not self.enabled or job_id.startswith("simulated-"):
            return {
                "job_id": job_id,
                "status": "Succeeded",
                "progress": 100,
                "message": "Simulation mode"
            }
        
        from alibabacloud_pai_dlc20201203 import models as dlc_models
        
        try:
            request = dlc_models.GetJobRequest()
            response = self.client.get_job(settings.PAI_WORKSPACE_ID, job_id, request)
            job = response.body
            
            # Map PAI status to our status
            status_map = {
                "Creating": "pending",
                "Queuing": "pending",
                "Running": "training",
                "Succeeded": "completed",
                "Failed": "failed",
                "Stopped": "cancelled"
            }
            
            return {
                "job_id": job_id,
                "status": status_map.get(job.status, job.status),
                "pai_status": job.status,
                "progress": self._estimate_progress(job.status),
                "message": job.reason_message or "",
                "created_at": job.gmt_create_time,
                "finished_at": job.gmt_finish_time
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if not self.enabled or job_id.startswith("simulated-"):
            return True
        
        from alibabacloud_pai_dlc20201203 import models as dlc_models
        
        try:
            request = dlc_models.StopJobRequest()
            self.client.stop_job(settings.PAI_WORKSPACE_ID, job_id, request)
            logger.info(f"Job cancelled: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    def _estimate_progress(self, status: str) -> float:
        """Estimate progress based on status"""
        progress_map = {
            "Creating": 5,
            "Queuing": 10,
            "Running": 50,
            "Succeeded": 100,
            "Failed": 100,
            "Stopped": 100
        }
        return progress_map.get(status, 0)


class PAIEASService:
    """PAI-EAS service for model inference"""
    
    def __init__(self):
        self.enabled = settings.EAS_ENABLED
        if self.enabled:
            self.endpoint = settings.EAS_ENDPOINT
            self.token = settings.EAS_TOKEN
            logger.info("PAI-EAS client initialized")
        else:
            self.endpoint = None
            self.token = None
            logger.warning("PAI-EAS is not configured, running in simulation mode")

    async def infer_topics(
        self,
        texts: List[str],
        model_name: str = "etm-default"
    ) -> Dict[str, Any]:
        """
        Get topic distribution for texts
        
        Args:
            texts: List of text documents
            model_name: Name of the deployed model service
        
        Returns:
            Topic distributions and top topics
        """
        if not self.enabled:
            # Simulation mode - return mock data
            import random
            num_topics = 20
            return {
                "topic_distributions": [
                    [random.random() for _ in range(num_topics)]
                    for _ in texts
                ],
                "top_topics": [
                    [(i, random.random()) for i in random.sample(range(num_topics), 3)]
                    for _ in texts
                ]
            }
        
        url = f"{self.endpoint}/api/predict/{model_name}"
        headers = {
            "Authorization": self.token,
            "Content-Type": "application/json"
        }
        payload = {
            "texts": texts
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"EAS inference failed: {e}")
                raise

    async def get_embeddings(
        self,
        texts: List[str],
        model_name: str = "qwen-embedding"
    ) -> List[List[float]]:
        """
        Get text embeddings
        
        Args:
            texts: List of text documents
            model_name: Name of the embedding service
        
        Returns:
            List of embedding vectors
        """
        if not self.enabled:
            # Simulation mode - return mock embeddings
            import random
            embedding_dim = 768
            return [
                [random.random() for _ in range(embedding_dim)]
                for _ in texts
            ]
        
        url = f"{self.endpoint}/api/predict/{model_name}"
        headers = {
            "Authorization": self.token,
            "Content-Type": "application/json"
        }
        payload = {
            "texts": texts,
            "return_embeddings": True
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result.get("embeddings", [])
            except Exception as e:
                logger.error(f"EAS embedding failed: {e}")
                raise

    async def health_check(self, model_name: str = "etm-default") -> bool:
        """Check if EAS service is healthy"""
        if not self.enabled:
            return True
        
        url = f"{self.endpoint}/api/predict/{model_name}/health"
        headers = {
            "Authorization": self.token
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=10)
                return response.status_code == 200
            except Exception:
                return False


# Create global instances
pai_dlc = PAIDLCService()
pai_eas = PAIEASService()

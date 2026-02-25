"""
Integration of LangChain Multi-Agent System with FastAPI
This module provides the bridge between the existing FastAPI app and the new agent system
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import the orchestrator agent
from agent.core.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger(__name__)

class AgentIntegration:
    """Integration layer for the multi-agent system"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp", llm_config: Optional[Dict] = None,
                 embedding_model_path: str = None):
        """Initialize the agent system"""
        self.base_dir = Path(base_dir)
        self.llm_config = llm_config or self._get_default_llm_config()
        self.embedding_model_path = embedding_model_path or os.environ.get("QWEN_EMBEDDING_MODEL_PATH")
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent(base_dir, self.llm_config, self.embedding_model_path)
        
        logger.info(f"Agent integration initialized with base_dir: {base_dir}")
    
    def _get_default_llm_config(self) -> Dict[str, Any]:
        """Get default LLM configuration with environment variable support"""
        return {
            "provider": "qwen",
            "api_key": os.environ.get("DASHSCOPE_API_KEY", "sk-ca1e46556f584e50aa74a2f6ff5659f0"),
            "base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
            "model": os.environ.get("QWEN_MODEL", "qwen-flash"),
            "temperature": float(os.environ.get("QWEN_TEMPERATURE", 0.3)),
            "top_p": float(os.environ.get("QWEN_TOP_P", 0.9)),
            "vision_enabled": False
        }
    
    def run_full_analysis(self, job_id: str) -> Dict[str, Any]:
        """
        Run complete topic model analysis pipeline
        
        Args:
            job_id: Unique identifier for the analysis job
            
        Returns:
            Dict with analysis result and status
        """
        try:
            logger.info(f"Starting full analysis for job_id: {job_id}")
            
            # Check if input data exists
            data_path = self.base_dir / "data" / job_id / "data.csv"
            if not data_path.exists():
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": f"Input data not found: {data_path}"
                }
            
            # Run full pipeline
            result = self.orchestrator.run_full(job_id)
            
            logger.info(f"Full analysis completed for job_id: {job_id}, status: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in full analysis for job_id {job_id}: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    def handle_query(self, job_id: str, question: str) -> Dict[str, Any]:
        """
        Handle user query with intelligent routing
        
        Args:
            job_id: Unique identifier for the analysis job
            question: User question
            
        Returns:
            Dict with answer and status
        """
        try:
            logger.info(f"Handling query for job_id: {job_id}, question: {question[:50]}...")
            
            # Use orchestrator to handle query
            result = self.orchestrator.handle_query(job_id, question)
            
            logger.info(f"Query handled for job_id: {job_id}, status: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling query for job_id {job_id}: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "question": question,
                "error": str(e)
            }
    
    def get_analysis_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis result for a job
        
        Args:
            job_id: Unique identifier for the analysis job
            
        Returns:
            Analysis result or None if not found
        """
        try:
            result_path = self.base_dir / "result" / job_id / "analysis_result.json"
            if result_path.exists():
                with open(result_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            logger.error(f"Error getting analysis result for job_id {job_id}: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get current status of a job
        
        Args:
            job_id: Unique identifier for the analysis job
            
        Returns:
            Dict with job status information
        """
        try:
            # Check if analysis result exists
            result = self.get_analysis_result(job_id)
            if result:
                return {
                    "job_id": job_id,
                    "status": result.get("status", "unknown"),
                    "completed_at": result.get("completed_at"),
                    "duration_seconds": result.get("duration_seconds"),
                    "error": result.get("error")
                }
            
            # Check if log exists (processing)
            log_path = self.base_dir / "result" / job_id / "log.txt"
            if log_path.exists():
                return {
                    "job_id": job_id,
                    "status": "processing",
                    "completed_at": None,
                    "duration_seconds": None,
                    "error": None
                }
            
            # Check if data exists (not started)
            data_path = self.base_dir / "data" / job_id / "data.csv"
            if data_path.exists():
                return {
                    "job_id": job_id,
                    "status": "not_started",
                    "completed_at": None,
                    "duration_seconds": None,
                    "error": None
                }
            
            return {
                "job_id": job_id,
                "status": "not_found",
                "completed_at": None,
                "duration_seconds": None,
                "error": "Job not found"
            }
            
        except Exception as e:
            logger.error(f"Error getting job status for job_id {job_id}: {str(e)}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
    
    def list_jobs(self) -> Dict[str, Any]:
        """
        List all jobs in the system
        
        Returns:
            Dict with job list
        """
        try:
            jobs = []
            
            # Check result directory for completed jobs
            result_dir = self.base_dir / "result"
            if result_dir.exists():
                for job_dir in result_dir.iterdir():
                    if job_dir.is_dir():
                        job_id = job_dir.name
                        status_info = self.get_job_status(job_id)
                        jobs.append(status_info)
            
            # Check data directory for jobs not yet processed
            data_dir = self.base_dir / "data"
            if data_dir.exists():
                for job_dir in data_dir.iterdir():
                    if job_dir.is_dir():
                        job_id = job_dir.name
                        if not any(job["job_id"] == job_id for job in jobs):
                            jobs.append(self.get_job_status(job_id))
            
            return {
                "status": "success",
                "jobs": sorted(jobs, key=lambda x: x["job_id"])
            }
            
        except Exception as e:
            logger.error(f"Error listing jobs: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_download_path(self, job_id: str, filename: str) -> Optional[str]:
        """
        Get actual file path for download endpoint
        
        Args:
            job_id: Unique identifier for the analysis job
            filename: Filename to download
            
        Returns:
            Actual file path or None if not found
        """
        try:
            # Try result directory first
            result_path = self.base_dir / "result" / job_id / filename
            if result_path.exists():
                return str(result_path)
            
            # Try visualization directory
            viz_path = self.base_dir / "visualization" / "outputs" / job_id / filename
            if viz_path.exists():
                return str(viz_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting download path for job_id {job_id}, filename {filename}: {str(e)}")
            return None

# Global instance for integration
_agent_integration = None

def get_agent_integration(base_dir: str = None, llm_config: Dict = None) -> AgentIntegration:
    """Get or create global agent integration instance"""
    global _agent_integration
    
    if _agent_integration is None:
        base_dir = base_dir or os.environ.get("THETA_ROOT", "/root/autodl-tmp")
        _agent_integration = AgentIntegration(base_dir, llm_config)
    
    return _agent_integration

def reset_agent_integration():
    """Reset the global agent integration instance (for testing)"""
    global _agent_integration
    _agent_integration = None

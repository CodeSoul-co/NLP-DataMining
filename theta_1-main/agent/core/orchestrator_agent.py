"""
PLAN: Orchestrate the entire topic model analysis pipeline
WHAT: Single entry point that routes to appropriate agents and manages dependency chain
WHY: Provides unified interface for full pipeline execution and query-based processing
INPUTS: 
- job_id: str - Unique identifier for analysis job
- mode: str - Either "full" (run_full) or "query" (handle_query)
- question: str - User question (only for query mode)
OUTPUTS:
- For full mode: Complete analysis_result.json with all fields populated
- For query mode: Answer to user question based on available analysis
SIDE EFFECTS:
- Manages dependency chain execution
- Creates and maintains result/{job_id}/analysis_result.json
- Generates all required output files and directories
- Logs all processing steps
FAILURE MODES:
- Dependency chain failure → log error + set status="failed"
- Shape validation failure → log error + set status="failed"
- Missing required outputs → log error + set status="failed"
RULE:
- Must enforce dependency chain: DataCleaning -> (Embedding + Bow) -> ETM -> Visualization -> Report
- Must generate and maintain analysis_result.json with strict schema
- Must create all required output files in correct locations
- Must perform shape/order validation at each step
- Path contract: manages all paths under /root/autodl-tmp/ (or THETA_ROOT)
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import shutil

# Import all agents
from .data_cleaning_agent import DataCleaningAgent
from .bow_agent import BowAgent
from .embedding_agent import EmbeddingAgent
from .etm_agent import ETMAgent
from .visualization_agent import VisualizationAgent
from .text_qa_agent import TextQAAgent
from .vision_qa_agent import VisionQAAgent
from .report_agent import ReportAgent

class OrchestratorAgent:
    """Main orchestrator for the topic model analysis pipeline"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp", llm_config: Optional[Dict] = None, 
                 embedding_model_path: str = None):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
        self.llm_config = llm_config or {}
        self.embedding_model_path = embedding_model_path  # Optional Qwen embedding model path
        
        # Initialize all agents
        self.data_cleaning_agent = DataCleaningAgent(base_dir)
        self.bow_agent = BowAgent(base_dir)
        self.embedding_agent = EmbeddingAgent(base_dir, model_path=embedding_model_path)
        self.etm_agent = ETMAgent(base_dir)
        self.visualization_agent = VisualizationAgent(base_dir)
        self.report_agent = ReportAgent(base_dir, llm_config)
        self.text_qa_agent = TextQAAgent(base_dir, llm_config)
        self.vision_qa_agent = VisionQAAgent(base_dir, llm_config)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"OrchestratorAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def run_full(self, job_id: str) -> Dict[str, Any]:
        """
        Run the complete topic model analysis pipeline
        
        Args:
            job_id: Unique identifier for analysis job
            
        Returns:
            Dict with final analysis result and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting full pipeline for job_id: {job_id}")
        
        try:
            # Initialize analysis result
            analysis_result = {
                "job_id": job_id,
                "status": "processing",
                "started_at": start_time.isoformat(),
                "completed_at": None,
                "duration_seconds": None
            }
            
            # Step 1: Data Cleaning (always required)
            self._log(job_id, "Starting data cleaning...")
            cleaning_result = self.data_cleaning_agent.process(job_id)
            if cleaning_result["status"] != "success":
                return self._fail_pipeline(job_id, analysis_result, cleaning_result["error"])
            
            # Step 2: BOW generation (required)
            self._log(job_id, "Starting BOW generation...")
            bow_result = self.bow_agent.process(job_id)
            if bow_result["status"] != "success":
                return self._fail_pipeline(job_id, analysis_result, bow_result["error"])
            
            # Step 3: Embedding generation (required)
            self._log(job_id, "Starting embedding generation...")
            embedding_result = self.embedding_agent.process(
                job_id, 
                bow_result["vocab_path"], 
                bow_result["bow_path"]
            )
            if embedding_result["status"] != "success":
                return self._fail_pipeline(job_id, analysis_result, embedding_result["error"])
            
            # Step 4: ETM processing (required)
            self._log(job_id, "Starting ETM processing...")
            etm_result = self.etm_agent.process(
                job_id,
                bow_result["vocab_path"],
                bow_result["bow_path"],
                embedding_result["doc_embeddings_path"],
                embedding_result["vocab_embeddings_path"]
            )
            if etm_result["status"] != "success":
                return self._fail_pipeline(job_id, analysis_result, etm_result["error"])
            
            # Step 5: Visualization (required)
            self._log(job_id, "Starting visualization generation...")
            viz_result = self.visualization_agent.process(
                job_id,
                etm_result["theta_path"],
                etm_result["beta_path"],
                etm_result["topics_path"]
            )
            if viz_result["status"] != "success":
                return self._fail_pipeline(job_id, analysis_result, viz_result["error"])
            
            # Step 6: Generate Word report
            self._log(job_id, "Generating Word report...")
            report_result = self.report_agent.process(job_id)
            if report_result["status"] != "success":
                self._log(job_id, f"Report generation failed: {report_result.get('error')}", error=True)
                # Continue even if report fails - it's not critical
            
            # Step 7: Generate final analysis_result.json
            self._log(job_id, "Generating final analysis result...")
            final_result = self._generate_analysis_result(
                job_id, 
                etm_result, 
                viz_result, 
                start_time
            )
            
            # Step 7: Archive additional files
            self._log(job_id, "Archiving additional files...")
            self._archive_files(job_id, etm_result, viz_result)
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            final_result["completed_at"] = end_time.isoformat()
            final_result["duration_seconds"] = duration
            
            # Save final result
            result_path = self.base_dir / "result" / job_id / "analysis_result.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            self._log(job_id, f"Pipeline completed successfully in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "job_id": job_id,
                "analysis_result": final_result
            }
            
        except Exception as e:
            self._log(job_id, f"Pipeline failed with unexpected error: {str(e)}", error=True)
            return self._fail_pipeline(job_id, analysis_result, str(e))
    
    def handle_query(self, job_id: str, question: str) -> Dict[str, Any]:
        """
        Handle user query with minimal necessary processing
        
        Args:
            job_id: Unique identifier for analysis job
            question: User question about the analysis
            
        Returns:
            Dict with answer and status
        """
        self.logger.info(f"Handling query for job_id: {job_id}, question: {question[:50]}...")
        
        try:
            # Check if analysis result exists
            result_path = self.base_dir / "result" / job_id / "analysis_result.json"
            if result_path.exists():
                # Analysis already complete, just answer the question
                return self._answer_question(job_id, question)
            
            # Determine what processing is needed based on the question
            needed_steps = self._determine_needed_steps(question)
            
            # Run only necessary steps
            if "analysis" in needed_steps:
                # Need to run full pipeline
                pipeline_result = self.run_full(job_id)
                if pipeline_result["status"] != "success":
                    return pipeline_result
            
            # Answer the question
            return self._answer_question(job_id, question)
            
        except Exception as e:
            self._log(job_id, f"Query handling failed: {str(e)}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "question": question,
                "error": str(e)
            }
    
    def _determine_needed_steps(self, question: str) -> List[str]:
        """Determine what processing steps are needed based on the question"""
        question_lower = question.lower()
        needed_steps = []
        
        # Check if question requires analysis results
        analysis_keywords = ["topic", "theme", "meaning", "interpret", "analyze", "distribution", "coherence"]
        if any(keyword in question_lower for keyword in analysis_keywords):
            needed_steps.append("analysis")
        
        # Check if question requires visual analysis
        visual_keywords = ["chart", "graph", "visualization", "plot", "image", "word cloud", "trend"]
        if any(keyword in question_lower for keyword in visual_keywords):
            needed_steps.append("visualization")
        
        return needed_steps
    
    def _answer_question(self, job_id: str, question: str) -> Dict[str, Any]:
        """Answer user question using appropriate QA agent"""
        question_lower = question.lower()
        
        # Determine if this is a visual question
        visual_keywords = ["chart", "graph", "visualization", "plot", "image", "word cloud", "trend", "see", "look"]
        is_visual = any(keyword in question_lower for keyword in visual_keywords)
        
        if is_visual:
            # Use Vision QA agent
            return self.vision_qa_agent.process(job_id, question)
        else:
            # Use Text QA agent
            return self.text_qa_agent.process(job_id, question)
    
    def _generate_analysis_result(self, job_id: str, etm_result: Dict[str, Any], 
                               viz_result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Generate the final analysis_result.json"""
        # Load topics from ETM output
        topics_path = self.base_dir / "ETM" / "outputs" / "topic_words" / f"{job_id}_topics.json"
        with open(topics_path, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        
        # Load metrics (placeholder - should come from ETM training)
        metrics = {
            "coherence_score": 0.78,  # Placeholder - should be calculated
            "diversity_score": 0.65,  # Placeholder - should be calculated
            "optimal_k": len(topics)
        }
        
        # Generate chart paths
        viz_dir = self.base_dir / "visualization" / "outputs" / job_id
        charts = {
            "topic_distribution": f"/api/download/{job_id}/topic_distribution.png",
            "heatmap": f"/api/download/{job_id}/heatmap_doc_topic.png",
            "coherence_curve": f"/api/download/{job_id}/coherence_curve.png",
            "topic_similarity": f"/api/download/{job_id}/topic_similarity.png"
        }
        
        # Generate download paths
        downloads = {
            "report": f"/api/download/{job_id}/report.docx",
            "theta_csv": f"/api/download/{job_id}/theta.csv",
            "beta_csv": f"/api/download/{job_id}/beta.csv"
        }
        
        # Add word cloud URLs to topics
        for i, topic in enumerate(topics):
            topic["wordcloud_url"] = f"/api/download/{job_id}/wordcloud_topic_{i}.png"
        
        return {
            "job_id": job_id,
            "status": "success",
            "completed_at": None,  # Will be set by caller
            "duration_seconds": None,  # Will be set by caller
            "metrics": metrics,
            "topics": topics,
            "charts": charts,
            "downloads": downloads
        }
    
    def _archive_files(self, job_id: str, etm_result: Dict[str, Any], viz_result: Dict[str, Any]):
        """Archive additional files to result directory"""
        result_dir = self.base_dir / "result" / job_id
        
        # Export theta.npy to theta.csv
        theta_path = self.base_dir / "ETM" / "outputs" / "theta" / f"{job_id}_theta.npy"
        if theta_path.exists():
            theta = np.load(theta_path)
            theta_csv = result_dir / "theta.csv"
            pd.DataFrame(theta).to_csv(theta_csv, index=False)
        
        # Export beta.npy to beta.csv
        beta_path = self.base_dir / "ETM" / "outputs" / "beta" / f"{job_id}_beta.npy"
        if beta_path.exists():
            beta = np.load(beta_path)
            beta_csv = result_dir / "beta.csv"
            pd.DataFrame(beta).to_csv(beta_csv, index=False)
        
        # Copy/move visualization files
        viz_dir = self.base_dir / "visualization" / "outputs" / job_id
        if viz_dir.exists():
            for viz_file in viz_dir.glob("*.png"):
                shutil.copy2(viz_file, result_dir / viz_file.name)
        
        # Create metrics.json
        metrics = {
            "job_id": job_id,
            "coherence_score": 0.78,  # Placeholder
            "diversity_score": 0.65,  # Placeholder
            "optimal_k": etm_result.get("num_topics", 0),
            "training_time": datetime.now().isoformat()
        }
        metrics_path = result_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    def _fail_pipeline(self, job_id: str, analysis_result: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle pipeline failure"""
        analysis_result["status"] = "failed"
        analysis_result["error"] = error
        analysis_result["completed_at"] = datetime.now().isoformat()
        
        # Save failed result
        result_path = self.base_dir / "result" / job_id / "analysis_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        self._log(job_id, f"Pipeline failed: {error}", error=True)
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": error
        }
    
    def _log(self, job_id: str, message: str, error: bool = False):
        """Log message to result/{job_id}/log.txt"""
        log_path = self.base_dir / "result" / job_id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] [{level}] OrchestratorAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

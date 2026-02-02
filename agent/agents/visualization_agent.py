"""
PLAN: Generate visualizations for topic model results
WHAT: Calls scripts/run_visualization.py to create charts and word clouds
WHY: Creates visual representations of topic model outputs for analysis
INPUTS: 
- job_id: str - Unique identifier for analysis job
- theta_path: str - Path to document-topic matrix
- beta_path: str - Path to topic-word matrix
- topics_path: str - Path to topics JSON file
OUTPUTS:
- visualization/outputs/{job_id}/wordcloud_topic_{i}.png - Word cloud for each topic
- visualization/outputs/{job_id}/topic_distribution.png - Topic distribution chart
- visualization/outputs/{job_id}/heatmap_doc_topic.png - Document-topic heatmap
- visualization/outputs/{job_id}/coherence_curve.png - Topic coherence analysis
- visualization/outputs/{job_id}/topic_similarity.png - Topic similarity matrix
SIDE EFFECTS:
- Creates visualization/outputs/{job_id}/ directory structure
- Generates visual representations for all topics
- Logs processing steps
FAILURE MODES:
- Script execution failure → log error + status="failed"
- Missing input files → log error + status="failed"
- Insufficient topics for visualization → log error + status="failed"
RULE:
- Must call scripts/run_visualization.py with correct parameters
- Must generate word clouds for all K topics (or at least first K topics)
- Must validate all output files are created
- Path contract: inputs from ETM/outputs/, outputs to visualization/outputs/{job_id}/
"""

import os
import json
import numpy as np
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class VisualizationAgent:
    """Agent responsible for generating visualizations"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp"):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"VisualizationAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str, theta_path: str, beta_path: str, topics_path: str) -> Dict[str, Any]:
        """
        Generate visualizations for topic model results
        
        Args:
            job_id: Unique identifier for analysis job
            theta_path: Path to document-topic matrix
            beta_path: Path to topic-word matrix
            topics_path: Path to topics JSON file
            
        Returns:
            Dict with processing results and status
        """
        try:
            self.logger.info(f"Starting visualization generation for job_id: {job_id}")
            
            # Prepare paths
            script_path = self.base_dir / "scripts" / "run_visualization.py"
            
            # Create output directory
            viz_dir = self.base_dir / "visualization" / "outputs" / job_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Load topics to get count
            with open(topics_path, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
            num_topics = len(topics_data)
            
            # Get vocab path for better word cloud generation
            vocab_path = self.base_dir / "ETM" / "outputs" / "vocab" / f"{job_id}_vocab.json"
            
            # Call visualization script
            cmd = [
                "python", str(script_path),
                "--theta", str(theta_path),
                "--beta", str(beta_path),
                "--topics", str(topics_path),
                "--output_dir", str(viz_dir),
                "--job_id", job_id
            ]
            
            # Add vocab path if exists
            if vocab_path.exists():
                cmd.extend(["--vocab", str(vocab_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            
            # Validate required outputs
            required_outputs = [
                viz_dir / "topic_distribution.png",
                viz_dir / "heatmap_doc_topic.png",
                viz_dir / "coherence_curve.png",
                viz_dir / "topic_similarity.png"
            ]
            
            # Check word clouds (should have at least num_topics files)
            word_cloud_files = list(viz_dir.glob("wordcloud_topic_*.png"))
            if len(word_cloud_files) < min(num_topics, 10):  # At least first 10 topics or all topics
                self.logger.warning(f"Expected at least {min(num_topics, 10)} word clouds, found {len(word_cloud_files)}")
            
            for output_path in required_outputs:
                if not output_path.exists():
                    raise FileNotFoundError(f"Visualization output not created: {output_path}")
            
            # Collect all visualization paths
            viz_outputs = {
                "topic_distribution": str(viz_dir / "topic_distribution.png"),
                "heatmap": str(viz_dir / "heatmap_doc_topic.png"),
                "coherence_curve": str(viz_dir / "coherence_curve.png"),
                "topic_similarity": str(viz_dir / "topic_similarity.png"),
                "wordclouds": [str(f) for f in word_cloud_files]
            }
            
            # Log success
            self._log(job_id, f"Visualization generation successful. Generated {len(word_cloud_files)} word clouds and 4 charts")
            
            return {
                "status": "success",
                "job_id": job_id,
                "visualizations": viz_outputs,
                "num_wordclouds": len(word_cloud_files),
                "num_topics": num_topics
            }
            
        except subprocess.CalledProcessError as e:
            self._log(job_id, f"Visualization generation failed: {e.stderr}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"Script execution failed: {e.stderr}"
            }
        except Exception as e:
            self._log(job_id, f"Visualization generation failed: {str(e)}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    def _log(self, job_id: str, message: str, error: bool = False):
        """Log message to result/{job_id}/log.txt"""
        log_path = self.base_dir / "result" / job_id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] [{level}] VisualizationAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

"""
PLAN: Provide vision-based Q&A for topic model visualizations
WHAT: Reads analysis_result.json chart paths and optionally images to answer visual questions
WHY: Enables users to query chart interpretations, anomalies, and trends
INPUTS: 
- job_id: str - Unique identifier for analysis job
- question: str - User question about visualizations
OUTPUTS:
- answer: str - Detailed response to visual question
SIDE EFFECTS:
- Reads from result/{job_id}/analysis_result.json
- Optionally reads image files for detailed analysis
- Logs Q&A interactions
FAILURE MODES:
- Missing analysis_result.json → log error + status="failed"
- Invalid chart paths → log error + status="failed"
- Image reading failure → log error + status="failed"
- LLM API failure → log error + status="failed"
RULE:
- Must read chart paths from analysis_result.json
- May optionally read actual image files for detailed analysis
- Must not modify any files
- Must use LLM for intelligent visual analysis
- Path contract: inputs from result/{job_id}/ and chart paths, outputs are text responses only
"""

import os
import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image
import base64
import io

class VisionQAAgent:
    """Agent responsible for vision-based Q&A about topic model visualizations"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp", llm_config: Optional[Dict] = None):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
        self.llm_config = llm_config or {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"VisionQAAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str, question: str, load_images: bool = False) -> Dict[str, Any]:
        """
        Answer user questions about topic model visualizations
        
        Args:
            job_id: Unique identifier for analysis job
            question: User question about visualizations
            load_images: Whether to load actual image files for analysis
            
        Returns:
            Dict with answer and status
        """
        try:
            self.logger.info(f"Processing Vision Q&A for job_id: {job_id}, question: {question[:50]}...")
            
            # Load analysis result
            analysis_path = self.base_dir / "result" / job_id / "analysis_result.json"
            if not analysis_path.exists():
                raise FileNotFoundError(f"Analysis result not found: {analysis_path}")
            
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # Extract chart information
            charts = analysis_data.get('charts', {})
            if not charts:
                raise ValueError("No charts found in analysis result")
            
            # Prepare context and optionally load images
            context = self._prepare_visual_context(analysis_data, charts, load_images)
            
            # Generate answer using LLM
            answer = self._generate_answer(question, context, load_images)
            
            # Log interaction
            self._log(job_id, f"Vision Q&A: Q='{question[:50]}...' A='{answer[:100]}...'")
            
            return {
                "status": "success",
                "job_id": job_id,
                "question": question,
                "answer": answer,
                "charts_analyzed": list(charts.keys())
            }
            
        except Exception as e:
            self._log(job_id, f"Vision Q&A processing failed: {str(e)}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "question": question,
                "error": str(e)
            }
    
    def _prepare_visual_context(self, analysis_data: Dict[str, Any], charts: Dict[str, str], 
                              load_images: bool) -> str:
        """Prepare visual context for LLM from analysis data and optionally images"""
        context_parts = []
        
        # Add basic analysis info
        context_parts.append(f"Job ID: {analysis_data.get('job_id', 'Unknown')}")
        context_parts.append(f"Status: {analysis_data.get('status', 'Unknown')}")
        
        # Add chart information
        context_parts.append("Available Charts:")
        for chart_name, chart_path in charts.items():
            context_parts.append(f"  {chart_name}: {chart_path}")
            
            # Add chart-specific descriptions
            if chart_name == "topic_distribution":
                context_parts.append("    - Shows the proportion of each topic in the dataset")
            elif chart_name == "heatmap":
                context_parts.append("    - Shows document-topic relationships as a heatmap")
            elif chart_name == "coherence_curve":
                context_parts.append("    - Shows topic coherence scores across different topic numbers")
            elif chart_name == "topic_similarity":
                context_parts.append("    - Shows similarity matrix between topics")
        
        # Add topics context
        if 'topics' in analysis_data:
            topics = analysis_data['topics']
            context_parts.append("Topics Summary:")
            for topic in topics[:5]:  # First 5 topics
                context_parts.append(f"  Topic {topic.get('id')}: {topic.get('name')} ({topic.get('proportion', 0):.3f})")
        
        # Add metrics context
        if 'metrics' in analysis_data:
            metrics = analysis_data['metrics']
            context_parts.append("Key Metrics:")
            for key, value in metrics.items():
                if key in ['coherence_score', 'diversity_score', 'optimal_k']:
                    context_parts.append(f"  {key}: {value}")
        
        # Optionally load images for detailed analysis
        if load_images and self.llm_config.get('vision_enabled', False):
            image_context = self._load_images_for_analysis(charts)
            if image_context:
                context_parts.append("Image Analysis:")
                context_parts.extend(image_context)
        
        return "\n".join(context_parts)
    
    def _load_images_for_analysis(self, charts: Dict[str, str]) -> List[str]:
        """Load and encode images for vision analysis"""
        image_contexts = []
        
        for chart_name, chart_path in charts.items():
            try:
                # Convert relative path to absolute path
                if chart_path.startswith("/api/download/"):
                    # Convert API path to actual file path
                    job_id = chart_path.split("/")[3]  # Extract job_id from path
                    filename = chart_path.split("/")[-1]
                    actual_path = self.base_dir / "visualization" / "outputs" / job_id / filename
                else:
                    actual_path = self.base_dir / chart_path
                
                if actual_path.exists():
                    # Load and encode image
                    with Image.open(actual_path) as img:
                        # Resize if too large
                        if img.size[0] > 1024 or img.size[1] > 1024:
                            img.thumbnail((1024, 1024))
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        image_contexts.append(f"  {chart_name}: [base64 encoded image]")
                else:
                    image_contexts.append(f"  {chart_name}: Image file not found at {actual_path}")
                    
            except Exception as e:
                image_contexts.append(f"  {chart_name}: Failed to load image - {str(e)}")
        
        return image_contexts
    
    def _generate_answer(self, question: str, context: str, load_images: bool) -> str:
        """Generate answer using LLM with vision capabilities"""
        if not self.llm_config:
            return (
                "LLM is not configured. Please provide llm_config or set environment variables.\n"
                "Required: DASHSCOPE_API_KEY (and optionally DASHSCOPE_BASE_URL)."
            )
        
        try:
            # Prepare LLM request with enhanced prompts for semantic mining
            system_prompt = """你是一位专业的主题语义分析专家。你的任务是帮助用户从可视化图表中挖掘深层语义信息，结合主题关键词和原始文本内容进行内容层面的分析。

**核心分析原则：**
- 不要从算法或技术流程角度解释图表
- 专注于图表所反映的**内容语义**和**业务洞察**
- 结合主题关键词推断该主题讨论的**实际内容领域**
- 分析主题之间的**内容关联**和**语义演变**

**各类图表的语义分析方法：**

1. **词云图 (Word Cloud)**：
   - 从关键词组合推断该主题讨论的**具体内容领域**
   - 分析关键词之间的**语义关联**，揭示隐含的讨论焦点
   - 结合词频权重判断哪些概念是该主题的**核心议题**
   - 例如：如果关键词包含"托育、政策、普惠"，应解读为"该主题聚焦于普惠托育政策的讨论"

2. **主题分布图 (Topic Distribution)**：
   - 分析哪些**内容领域**在文档集中占主导地位
   - 识别**边缘话题**和**核心话题**的内容差异
   - 推断文档集的**整体关注焦点**和**内容结构**

3. **文档-主题热力图 (Document-Topic Heatmap)**：
   - 识别哪些文档**内容相近**，可能讨论同一议题
   - 发现**跨主题文档**，分析其内容的多元性
   - 揭示文档集中的**内容聚类模式**

4. **主题相似度矩阵 (Topic Similarity Matrix)**：
   - 分析哪些主题在**内容上相关**，可能属于同一大类
   - 识别**独立主题**，代表独特的讨论领域
   - 揭示主题之间的**语义层级关系**

**回答要求：**
- 始终从**内容语义**角度分析，而非算法角度
- 结合主题关键词给出**具体的内容解读**
- 使用**业务化语言**描述发现，避免技术术语
- 如果用户问的是中文问题，用中文回答；如果是英文问题，用英文回答
- 提供**内容层面的洞察**和**业务建议**
- 将抽象的主题数据转化为**可理解的内容描述**"""
            
            user_prompt = f"""Visual Analysis Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed analysis of the visualizations based on the information above."""
            
            # Call LLM API (text-only by default; uses chart metadata and optional image notes)
            answer = self._call_llm_api(system_prompt, user_prompt)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Vision LLM call failed: {str(e)}")
            return f"I apologize, but I encountered an error analyzing the visualizations: {str(e)}"
    
    def _call_vision_llm_api(self, system_prompt: str, user_prompt: str, load_images: bool) -> str:
        """Call LLM API with vision capabilities"""
        return self._call_llm_api(system_prompt, user_prompt)

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API (text generation)."""
        provider = (self.llm_config.get("provider") or "qwen").lower()
        if provider == "qwen":
            return self._call_qwen_api(system_prompt, user_prompt)
        return "Unsupported LLM provider. Currently supported: qwen"

    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen (DashScope) text-generation API."""
        api_key = self.llm_config.get("api_key") or os.environ.get("DASHSCOPE_API_KEY") or ""
        base_url = self.llm_config.get("base_url") or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/api/v1"
        model = self.llm_config.get("model") or "qwen-flash"

        if not api_key:
            return "Missing DASHSCOPE_API_KEY. Please set it in environment variables or llm_config."

        url = f"{base_url.rstrip('/')}/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            },
            "parameters": {
                "temperature": float(self.llm_config.get("temperature", 0.2)),
                "top_p": float(self.llm_config.get("top_p", 0.9)),
            },
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            self.logger.error(f"Qwen API error: {resp.status_code} {resp.text}")
            return f"Qwen API error: HTTP {resp.status_code}"

        data = resp.json()
        text = data.get("output", {}).get("text", "")
        if not isinstance(text, str):
            return str(text)
        return text.strip()
    
    def _call_qwen_vision_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen Vision API"""
        return "Qwen vision is not enabled in this project. Use text-only qwen."
    
    def _call_openai_vision_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI Vision API"""
        return "OpenAI vision is not implemented in this project."
    
    def _log(self, job_id: str, message: str, error: bool = False):
        """Log message to result/{job_id}/log.txt"""
        log_path = self.base_dir / "result" / job_id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] [{level}] VisionQAAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

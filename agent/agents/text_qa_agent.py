"""
PLAN: Provide text-based Q&A for topic model analysis
WHAT: Reads analysis_result.json and topics/metrics to answer questions
WHY: Enables users to query topic meanings, metrics interpretation, and report content
INPUTS: 
- job_id: str - Unique identifier for analysis job
- question: str - User question about the analysis
OUTPUTS:
- answer: str - Detailed response to user question
SIDE EFFECTS:
- Reads from result/{job_id}/analysis_result.json
- Logs Q&A interactions
FAILURE MODES:
- Missing analysis_result.json → log error + status="failed"
- Invalid question format → log error + status="failed"
- LLM API failure → log error + status="failed"
RULE:
- Must only read from result/{job_id}/analysis_result.json
- Must not modify any files
- Must use LLM for intelligent question answering
- Path contract: inputs from result/{job_id}/, outputs are text responses only
"""

import os
import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class TextQAAgent:
    """Agent responsible for text-based Q&A about topic model analysis"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp", llm_config: Optional[Dict] = None):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
        self.llm_config = llm_config or {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"TextQAAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str, question: str) -> Dict[str, Any]:
        """
        Answer user questions about topic model analysis
        
        Args:
            job_id: Unique identifier for analysis job
            question: User question about the analysis
            
        Returns:
            Dict with answer and status
        """
        try:
            self.logger.info(f"Processing Q&A for job_id: {job_id}, question: {question[:50]}...")
            
            # Load analysis result
            analysis_path = self.base_dir / "result" / job_id / "analysis_result.json"
            if not analysis_path.exists():
                raise FileNotFoundError(f"Analysis result not found: {analysis_path}")
            
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # Prepare context for LLM
            context = self._prepare_context(analysis_data)
            
            # Generate answer using LLM
            answer = self._generate_answer(question, context)
            
            # Log interaction
            self._log(job_id, f"Q&A: Q='{question[:50]}...' A='{answer[:100]}...'")
            
            return {
                "status": "success",
                "job_id": job_id,
                "question": question,
                "answer": answer
            }
            
        except Exception as e:
            self._log(job_id, f"Q&A processing failed: {str(e)}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "question": question,
                "error": str(e)
            }
    
    def _prepare_context(self, analysis_data: Dict[str, Any]) -> str:
        """Prepare context for LLM from analysis data"""
        context_parts = []
        
        # Add job info
        if 'job_id' in analysis_data:
            context_parts.append(f"Job ID: {analysis_data['job_id']}")
        if 'status' in analysis_data:
            context_parts.append(f"Status: {analysis_data['status']}")
        if 'completed_at' in analysis_data:
            context_parts.append(f"Completed: {analysis_data['completed_at']}")
        
        # Add metrics
        if 'metrics' in analysis_data:
            metrics = analysis_data['metrics']
            context_parts.append("Metrics:")
            for key, value in metrics.items():
                context_parts.append(f"  {key}: {value}")
        
        # Add topics
        if 'topics' in analysis_data:
            topics = analysis_data['topics']
            context_parts.append("Topics:")
            for topic in topics:
                context_parts.append(f"  Topic {topic.get('id')}: {topic.get('name')} (Proportion: {topic.get('proportion', 0):.3f})")
                keywords = topic.get('keywords', [])
                if keywords:
                    context_parts.append(f"    Keywords: {', '.join(keywords[:10])}")
        
        # Add charts info
        if 'charts' in analysis_data:
            charts = analysis_data['charts']
            context_parts.append("Available Charts:")
            for key, value in charts.items():
                context_parts.append(f"  {key}: {value}")
        
        # Add downloads info
        if 'downloads' in analysis_data:
            downloads = analysis_data['downloads']
            context_parts.append("Available Downloads:")
            for key, value in downloads.items():
                context_parts.append(f"  {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        if not self.llm_config:
            return (
                "LLM is not configured. Please provide llm_config or set environment variables.\n"
                "Required: DASHSCOPE_API_KEY (and optionally DASHSCOPE_BASE_URL)."
            )
        
        try:
            # Prepare LLM request with enhanced prompts for semantic mining
            system_prompt = """你是一位专业的主题语义分析专家。你的任务是帮助用户从主题模型分析结果中挖掘深层语义信息，结合主题关键词和文本内容进行内容层面的分析。

**核心分析原则：**
- 不要从算法或技术流程角度解释结果
- 专注于分析结果所反映的**内容语义**和**业务洞察**
- 结合主题关键词推断该主题讨论的**实际内容领域**
- 分析主题之间的**内容关联**和**语义演变**

**分析能力：**

1. **主题语义解读**：
   - 根据关键词组合推断该主题讨论的**具体内容领域**
   - 分析关键词之间的**语义关联**，揭示隐含的讨论焦点
   - 例如：关键词"LDA、VAE、language"应解读为"该主题聚焦于语言建模技术的讨论"

2. **内容结构分析**：
   - 分析哪些**内容领域**在文档集中占主导地位
   - 识别**边缘话题**和**核心话题**的内容差异
   - 推断文档集的**整体关注焦点**

3. **主题关联分析**：
   - 分析哪些主题在**内容上相关**，可能属于同一大类
   - 识别**独立主题**，代表独特的讨论领域
   - 揭示主题之间的**语义层级关系**

4. **业务洞察**：
   - 将主题分析结果转化为**可操作的业务建议**
   - 识别文档集中的**关键议题**和**发展趋势**
   - 提供**内容层面的决策支持**

**回答要求：**
- 始终从**内容语义**角度分析，而非算法角度
- 结合主题关键词给出**具体的内容解读**
- 使用**业务化语言**描述发现，避免技术术语
- 如果用户问的是中文问题，用中文回答；如果是英文问题，用英文回答
- 提供**内容层面的洞察**和**业务建议**
- 将抽象的主题数据转化为**可理解的内容描述**"""
            
            user_prompt = f"""Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed answer based on the analysis data above."""
            
            answer = self._call_llm_api(system_prompt, user_prompt)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            return f"I apologize, but I encountered an error generating the answer: {str(e)}"
    
    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API - implementation depends on your LLM service"""
        provider = (self.llm_config.get("provider") or "qwen").lower()
        if provider == "qwen":
            return self._call_qwen_api(system_prompt, user_prompt)
        return "Unsupported LLM provider. Currently supported: qwen"
    
    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen API with improved error handling and OpenAI compatibility"""
        api_key = self.llm_config.get("api_key") or os.environ.get("DASHSCOPE_API_KEY") or ""
        base_url = self.llm_config.get("base_url") or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/api/v1"
        model = self.llm_config.get("model") or os.environ.get("QWEN_MODEL") or "qwen-flash"

        if not api_key:
            return "Missing DASHSCOPE_API_KEY. Please set it in environment variables or llm_config."

        # Try OpenAI-compatible endpoint first
        if "dashscope" in base_url.lower():
            return self._call_qwen_dashscope(system_prompt, user_prompt, api_key, base_url, model)
        else:
            return self._call_qwen_openai_compatible(system_prompt, user_prompt, api_key, base_url, model)
    
    def _call_qwen_dashscope(self, system_prompt: str, user_prompt: str, api_key: str, base_url: str, model: str) -> str:
        """Call Qwen via DashScope API"""
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
                "temperature": float(self.llm_config.get("temperature", os.environ.get("QWEN_TEMPERATURE", 0.3))),
                "top_p": float(self.llm_config.get("top_p", os.environ.get("QWEN_TOP_P", 0.9))),
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
    
    def _call_qwen_openai_compatible(self, system_prompt: str, user_prompt: str, api_key: str, base_url: str, model: str) -> str:
        """Call Qwen via OpenAI-compatible endpoint"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=float(self.llm_config.get("temperature", os.environ.get("QWEN_TEMPERATURE", 0.3))),
                top_p=float(self.llm_config.get("top_p", os.environ.get("QWEN_TOP_P", 0.9))),
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            return "OpenAI library not installed. Please install it with: pip install openai"
        except Exception as e:
            self.logger.error(f"OpenAI-compatible API error: {str(e)}")
            return f"Qwen API error: {str(e)}"
    
    def _call_openai_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API"""
        return "OpenAI provider is not implemented in this project. Use provider=qwen."
    
    def _log(self, job_id: str, message: str, error: bool = False):
        """Log message to result/{job_id}/log.txt"""
        log_path = self.base_dir / "result" / job_id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] [{level}] TextQAAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

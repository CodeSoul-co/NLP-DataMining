"""
QA Tools for LangChain Agent
"""

import json
import os
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import Field
import requests


class TextQATool(BaseTool):
    """文本问答工具 - 回答关于主题分析的问题"""
    
    name: str = "text_qa"
    description: str = """用于回答关于主题分析结果的问题。
    输入: job_id,question (任务ID和问题，用逗号分隔)
    前置条件: 分析结果必须存在
    功能: 基于分析结果，使用LLM回答用户问题
    输出: 问题的答案
    使用场景: 用户想了解主题含义、关键词解释、分布分析等"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, input_str: str) -> str:
        """执行问答"""
        try:
            parts = input_str.split(",", 1)
            if len(parts) < 2:
                return "错误: 需要提供 job_id 和 question，格式: job_id,question"
            
            job_id = parts[0].strip()
            question = parts[1].strip()
            
            # 加载分析结果
            result_path = self.base_dir / "result" / job_id / "analysis_result.json"
            topics_path = self.base_dir / "ETM" / "outputs" / "topic_words" / f"{job_id}_topics.json"
            
            context = ""
            if result_path.exists():
                with open(result_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                context = f"分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}"
            elif topics_path.exists():
                with open(topics_path, 'r', encoding='utf-8') as f:
                    topics = json.load(f)
                context = f"主题数据: {json.dumps(topics, ensure_ascii=False, indent=2)}"
            else:
                return f"错误: 找不到任务 {job_id} 的分析结果，请先完成分析流程"
            
            # 调用LLM
            answer = self._call_llm(question, context)
            return answer
            
        except Exception as e:
            return f"问答失败: {str(e)}"
    
    def _call_llm(self, question: str, context: str) -> str:
        """调用Qwen LLM"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-ca1e46556f584e50aa74a2f6ff5659f0")
        base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.environ.get("QWEN_MODEL", "qwen-plus")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """你是一位专业的主题语义分析专家。根据提供的主题分析结果，从内容语义角度回答用户问题。
不要从算法角度解释，而是分析主题的实际内容含义和业务洞察。"""
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"上下文:\n{context}\n\n问题: {question}"}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "无法生成回答")
        except Exception as e:
            return f"LLM调用失败: {str(e)}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


class VisionQATool(BaseTool):
    """视觉问答工具 - 回答关于可视化图表的问题"""
    
    name: str = "vision_qa"
    description: str = """用于回答关于可视化图表的问题。
    输入: job_id,question (任务ID和问题，用逗号分隔)
    前置条件: 可视化图表必须已生成
    功能: 基于图表信息，使用LLM回答用户关于词云、热力图等的问题
    输出: 问题的答案
    使用场景: 用户想了解词云中某个词的含义、热力图的模式等"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, input_str: str) -> str:
        """执行视觉问答"""
        try:
            parts = input_str.split(",", 1)
            if len(parts) < 2:
                return "错误: 需要提供 job_id 和 question，格式: job_id,question"
            
            job_id = parts[0].strip()
            question = parts[1].strip()
            
            # 加载主题数据和图表信息
            topics_path = self.base_dir / "ETM" / "outputs" / "topic_words" / f"{job_id}_topics.json"
            viz_dir = self.base_dir / "visualization" / "outputs" / job_id
            
            if not topics_path.exists():
                return f"错误: 找不到任务 {job_id} 的主题数据"
            
            with open(topics_path, 'r', encoding='utf-8') as f:
                topics = json.load(f)
            
            # 构建上下文
            charts = list(viz_dir.glob("*.png")) if viz_dir.exists() else []
            context = f"""主题数据: {json.dumps(topics, ensure_ascii=False, indent=2)}
可用图表: {[c.name for c in charts]}"""
            
            # 调用LLM
            answer = self._call_llm(question, context)
            return answer
            
        except Exception as e:
            return f"视觉问答失败: {str(e)}"
    
    def _call_llm(self, question: str, context: str) -> str:
        """调用Qwen LLM"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-ca1e46556f584e50aa74a2f6ff5659f0")
        base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.environ.get("QWEN_MODEL", "qwen-plus")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """你是一位专业的主题语义分析专家。根据主题数据和图表信息，从内容语义角度回答用户关于可视化的问题。
分析词云中关键词的实际含义、热力图的内容模式、相似度矩阵的语义关联等。"""
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"上下文:\n{context}\n\n问题: {question}"}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "无法生成回答")
        except Exception as e:
            return f"LLM调用失败: {str(e)}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

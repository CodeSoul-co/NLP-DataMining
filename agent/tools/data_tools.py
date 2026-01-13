"""
Data Processing Tools for LangChain Agent
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional
from langchain.tools import BaseTool
from pydantic import Field
from docx import Document


class DataCleaningTool(BaseTool):
    """数据清洗工具 - 验证和准备输入数据"""
    
    name: str = "data_cleaning"
    description: str = """用于验证和清洗输入数据。
    输入: job_id (任务ID)
    功能: 验证data/{job_id}/data.csv是否存在且格式正确，生成配置文件
    输出: 配置文件路径和数据统计信息
    使用场景: 在开始任何分析之前，需要先验证数据"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, job_id: str) -> str:
        """执行数据清洗"""
        try:
            data_path = self.base_dir / "data" / job_id / "data.csv"
            
            if not data_path.exists():
                return f"错误: 数据文件不存在 {data_path}"
            
            df = pd.read_csv(data_path)
            
            if 'text' not in df.columns:
                return "错误: CSV文件必须包含'text'列"
            
            if len(df) == 0:
                return "错误: 数据文件为空"
            
            # 创建配置
            result_dir = self.base_dir / "result" / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            config = {
                "job_id": job_id,
                "data_path": str(data_path),
                "document_count": len(df),
                "columns": list(df.columns)
            }
            
            config_path = result_dir / "config.yaml"
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
            
            return f"数据验证成功: {len(df)}个文档, 配置已保存到 {config_path}"
            
        except Exception as e:
            return f"数据清洗失败: {str(e)}"
    
    async def _arun(self, job_id: str) -> str:
        return self._run(job_id)


class DocxConverterTool(BaseTool):
    """Word文档转换工具 - 将docx转换为CSV格式"""
    
    name: str = "docx_converter"
    description: str = """用于将Word文档(.docx)转换为CSV格式。
    输入: docx_path (Word文档路径), job_id (任务ID)
    功能: 提取Word文档中的文本段落，保存为data/{job_id}/data.csv
    输出: 转换后的CSV文件路径和文档数量
    使用场景: 当用户提供Word文档作为输入时使用"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, input_str: str) -> str:
        """执行文档转换"""
        try:
            # 解析输入参数
            parts = input_str.split(",")
            if len(parts) < 2:
                return "错误: 需要提供 docx_path 和 job_id，格式: docx_path,job_id"
            
            docx_path = parts[0].strip()
            job_id = parts[1].strip()
            
            if not Path(docx_path).exists():
                return f"错误: Word文档不存在 {docx_path}"
            
            # 提取文本
            doc = Document(docx_path)
            paragraphs = []
            current_section = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    if len(text) < 50 and (text[0].isdigit() or text.endswith('模型')):
                        if current_section:
                            paragraphs.append(' '.join(current_section))
                        current_section = [text]
                    else:
                        current_section.append(text)
            
            if current_section:
                paragraphs.append(' '.join(current_section))
            
            paragraphs = [p for p in paragraphs if len(p) > 50]
            
            # 保存CSV
            data_dir = self.base_dir / "data" / job_id
            data_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = data_dir / "data.csv"
            df = pd.DataFrame({'text': paragraphs})
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            return f"转换成功: 从Word文档提取了{len(paragraphs)}个文档段落，保存到 {csv_path}"
            
        except Exception as e:
            return f"文档转换失败: {str(e)}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

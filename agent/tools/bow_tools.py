"""
BOW Generation Tool for LangChain Agent
"""

import subprocess
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import Field


class BowGeneratorTool(BaseTool):
    """BOW生成工具 - 生成词袋表示和词汇表"""
    
    name: str = "bow_generator"
    description: str = """用于生成词袋(Bag-of-Words)表示和词汇表。
    输入: job_id (任务ID)
    前置条件: data/{job_id}/data.csv 必须存在
    功能: 调用BOW生成模块，生成词汇表和词袋矩阵
    输出: vocab.json和bow.npz文件路径
    使用场景: 在数据清洗完成后，进行文本向量化"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, job_id: str) -> str:
        """执行BOW生成"""
        try:
            data_path = self.base_dir / "data" / job_id / "data.csv"
            if not data_path.exists():
                return f"错误: 数据文件不存在 {data_path}，请先运行data_cleaning或docx_converter"
            
            # 创建输出目录
            vocab_dir = self.base_dir / "ETM" / "outputs" / "vocab"
            bow_dir = self.base_dir / "ETM" / "outputs" / "bow"
            vocab_dir.mkdir(parents=True, exist_ok=True)
            bow_dir.mkdir(parents=True, exist_ok=True)
            
            vocab_output = vocab_dir / f"{job_id}_vocab.json"
            bow_output = bow_dir / f"{job_id}_bow.npz"
            
            # 调用脚本
            script_path = self.base_dir / "scripts" / "run_engine_a.py"
            cmd = [
                "python", str(script_path),
                "--input", str(data_path),
                "--vocab_output", str(vocab_output),
                "--bow_output", str(bow_output),
                "--job_id", job_id
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if vocab_output.exists() and bow_output.exists():
                import json
                with open(vocab_output, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                vocab_size = len(vocab_data.get('vocab', []))
                return f"BOW生成成功: 词汇表大小={vocab_size}, 文件保存到 {vocab_output} 和 {bow_output}"
            else:
                return f"BOW生成失败: {result.stderr}"
                
        except Exception as e:
            return f"BOW生成失败: {str(e)}"
    
    async def _arun(self, job_id: str) -> str:
        return self._run(job_id)

"""
Embedding Generation Tool for LangChain Agent
"""

import subprocess
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import Field


class EmbeddingGeneratorTool(BaseTool):
    """Embedding生成工具 - 生成文档和词汇嵌入"""
    
    name: str = "embedding_generator"
    description: str = """用于生成文档嵌入和词汇嵌入向量。
    输入: job_id (任务ID)
    前置条件: BOW生成必须已完成 (vocab.json和bow.npz存在)
    功能: 调用Embedding模块，生成文档和词汇的向量表示
    输出: embeddings.npy和vocab_emb.npy文件路径
    使用场景: 在BOW生成完成后，为ETM训练准备嵌入向量"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, job_id: str) -> str:
        """执行Embedding生成"""
        try:
            vocab_path = self.base_dir / "ETM" / "outputs" / "vocab" / f"{job_id}_vocab.json"
            bow_path = self.base_dir / "ETM" / "outputs" / "bow" / f"{job_id}_bow.npz"
            
            if not vocab_path.exists():
                return f"错误: 词汇表不存在 {vocab_path}，请先运行bow_generator"
            if not bow_path.exists():
                return f"错误: BOW矩阵不存在 {bow_path}，请先运行bow_generator"
            
            # 创建输出目录
            embed_dir = self.base_dir / "embedding" / "outputs" / "zero_shot"
            embed_dir.mkdir(parents=True, exist_ok=True)
            
            doc_emb_output = embed_dir / f"{job_id}_embeddings.npy"
            vocab_emb_output = embed_dir / f"{job_id}_vocab_emb.npy"
            
            # 调用脚本
            script_path = self.base_dir / "scripts" / "run_engine_b.py"
            cmd = [
                "python", str(script_path),
                "--vocab", str(vocab_path),
                "--bow", str(bow_path),
                "--doc_emb_output", str(doc_emb_output),
                "--vocab_emb_output", str(vocab_emb_output),
                "--job_id", job_id
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if doc_emb_output.exists() and vocab_emb_output.exists():
                import numpy as np
                doc_emb = np.load(doc_emb_output)
                vocab_emb = np.load(vocab_emb_output)
                return f"Embedding生成成功: 文档嵌入{doc_emb.shape}, 词汇嵌入{vocab_emb.shape}"
            else:
                return f"Embedding生成失败: {result.stderr}"
                
        except Exception as e:
            return f"Embedding生成失败: {str(e)}"
    
    async def _arun(self, job_id: str) -> str:
        return self._run(job_id)

"""
ETM Training Tool for LangChain Agent
"""

import subprocess
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import Field


class ETMTrainerTool(BaseTool):
    """ETM训练工具 - 训练嵌入式主题模型"""
    
    name: str = "etm_trainer"
    description: str = """用于训练嵌入式主题模型(ETM)。
    输入: job_id (任务ID)
    前置条件: BOW和Embedding必须已生成
    功能: 调用ETM训练模块，生成主题分布和主题-词分布
    输出: theta.npy(文档-主题), beta.npy(主题-词), topics.json(主题关键词)
    使用场景: 在Embedding生成完成后，进行主题建模"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, job_id: str) -> str:
        """执行ETM训练"""
        try:
            vocab_path = self.base_dir / "ETM" / "outputs" / "vocab" / f"{job_id}_vocab.json"
            bow_path = self.base_dir / "ETM" / "outputs" / "bow" / f"{job_id}_bow.npz"
            doc_emb_path = self.base_dir / "embedding" / "outputs" / "zero_shot" / f"{job_id}_embeddings.npy"
            vocab_emb_path = self.base_dir / "embedding" / "outputs" / "zero_shot" / f"{job_id}_vocab_emb.npy"
            
            # 检查前置条件
            missing = []
            if not vocab_path.exists(): missing.append("vocab.json")
            if not bow_path.exists(): missing.append("bow.npz")
            if not doc_emb_path.exists(): missing.append("doc_embeddings.npy")
            if not vocab_emb_path.exists(): missing.append("vocab_embeddings.npy")
            
            if missing:
                return f"错误: 缺少前置文件 {missing}，请先运行bow_generator和embedding_generator"
            
            # 创建输出目录
            theta_dir = self.base_dir / "ETM" / "outputs" / "theta"
            beta_dir = self.base_dir / "ETM" / "outputs" / "beta"
            alpha_dir = self.base_dir / "ETM" / "outputs" / "alpha"
            topics_dir = self.base_dir / "ETM" / "outputs" / "topic_words"
            
            for d in [theta_dir, beta_dir, alpha_dir, topics_dir]:
                d.mkdir(parents=True, exist_ok=True)
            
            theta_output = theta_dir / f"{job_id}_theta.npy"
            beta_output = beta_dir / f"{job_id}_beta.npy"
            alpha_output = alpha_dir / f"{job_id}_alpha.npy"
            topics_output = topics_dir / f"{job_id}_topics.json"
            
            # 调用脚本
            script_path = self.base_dir / "scripts" / "run_engine_c.py"
            cmd = [
                "python", str(script_path),
                "--vocab", str(vocab_path),
                "--bow", str(bow_path),
                "--doc_emb", str(doc_emb_path),
                "--vocab_emb", str(vocab_emb_path),
                "--theta_output", str(theta_output),
                "--beta_output", str(beta_output),
                "--alpha_output", str(alpha_output),
                "--topics_output", str(topics_output),
                "--job_id", job_id
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if theta_output.exists() and topics_output.exists():
                import json
                with open(topics_output, 'r', encoding='utf-8') as f:
                    topics = json.load(f)
                return f"ETM训练成功: 生成了{len(topics)}个主题，文件保存到ETM/outputs/"
            else:
                return f"ETM训练失败: {result.stderr}"
                
        except Exception as e:
            return f"ETM训练失败: {str(e)}"
    
    async def _arun(self, job_id: str) -> str:
        return self._run(job_id)

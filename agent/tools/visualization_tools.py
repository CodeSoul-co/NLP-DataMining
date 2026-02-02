"""
Visualization Tool for LangChain Agent
"""

import subprocess
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import Field


class VisualizationTool(BaseTool):
    """可视化工具 - 生成主题分析可视化图表"""
    
    name: str = "visualization_generator"
    description: str = """用于生成主题分析的可视化图表。
    输入: job_id (任务ID)
    前置条件: ETM训练必须已完成
    功能: 生成词云图、主题分布图、热力图、相似度矩阵等
    输出: 多个PNG图片文件
    使用场景: 在ETM训练完成后，生成可视化结果供用户查看"""
    
    base_dir: Path = Field(default_factory=lambda: Path("."))
    
    def _run(self, job_id: str) -> str:
        """执行可视化生成"""
        try:
            theta_path = self.base_dir / "ETM" / "outputs" / "theta" / f"{job_id}_theta.npy"
            beta_path = self.base_dir / "ETM" / "outputs" / "beta" / f"{job_id}_beta.npy"
            topics_path = self.base_dir / "ETM" / "outputs" / "topic_words" / f"{job_id}_topics.json"
            
            # 检查前置条件
            if not theta_path.exists() or not beta_path.exists() or not topics_path.exists():
                return "错误: ETM输出文件不存在，请先运行etm_trainer"
            
            # 创建输出目录
            viz_dir = self.base_dir / "visualization" / "outputs" / job_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 调用脚本
            script_path = self.base_dir / "scripts" / "run_visualization.py"
            vocab_path = self.base_dir / "ETM" / "outputs" / "vocab" / f"{job_id}_vocab.json"
            
            cmd = [
                "python", str(script_path),
                "--theta", str(theta_path),
                "--beta", str(beta_path),
                "--topics", str(topics_path),
                "--output_dir", str(viz_dir),
                "--job_id", job_id
            ]
            
            if vocab_path.exists():
                cmd.extend(["--vocab", str(vocab_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            # 检查生成的文件
            generated = list(viz_dir.glob("*.png"))
            if generated:
                # 复制到result目录
                result_dir = self.base_dir / "result" / job_id
                result_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                for f in generated:
                    shutil.copy(f, result_dir / f.name)
                
                return f"可视化生成成功: 生成了{len(generated)}个图表，保存到 {viz_dir}"
            else:
                return f"可视化生成失败: {result.stderr}"
                
        except Exception as e:
            return f"可视化生成失败: {str(e)}"
    
    async def _arun(self, job_id: str) -> str:
        return self._run(job_id)

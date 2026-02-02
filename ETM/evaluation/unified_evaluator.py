"""
Unified Topic Model Evaluator

统一的主题模型评估器，支持所有模型类型（THETA, LDA, ETM, CTM, DTM）
使用相同的7个评估指标：
1. PPL (Perplexity) - 困惑度
2. TD (Topic Diversity) - 主题多样性
3. iRBO (Inverse Rank-Biased Overlap) - 主题多样性
4. NPMI (Normalized PMI) - 主题一致性
5. C_V - 主题一致性
6. UMass - 主题一致性
7. Exclusivity - 主题排他性
"""

import os
import json
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from .topic_metrics import (
    compute_topic_diversity,
    compute_topic_coherence_npmi,
    compute_topic_coherence_cv,
    compute_topic_coherence_umass,
    compute_topic_exclusivity,
    compute_topic_significance,
    compute_perplexity
)


class UnifiedEvaluator:
    """
    统一的主题模型评估器
    
    支持评估所有模型类型，生成统一格式的评估结果和可视化
    """
    
    def __init__(
        self,
        beta: np.ndarray,
        theta: np.ndarray,
        bow_matrix: Union[np.ndarray, sp.csr_matrix],
        vocab: List[str],
        training_history: Optional[Dict] = None,
        model_name: str = "unknown",
        dataset: str = "unknown",
        output_dir: Optional[str] = None,
        num_topics: int = 20,
        dev_mode: bool = False
    ):
        """
        初始化评估器
        
        Args:
            beta: 主题-词分布 (K, V)
            theta: 文档-主题分布 (N, K)
            bow_matrix: BOW矩阵 (N, V)
            vocab: 词汇表
            training_history: 训练历史（含loss等）
            model_name: 模型名称
            dataset: 数据集名称
            output_dir: 输出目录
            num_topics: 主题数
            dev_mode: 调试模式
        """
        self.beta = beta
        self.theta = theta
        self.bow_matrix = bow_matrix
        self.vocab = vocab
        self.training_history = training_history
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.num_topics = num_topics
        self.dev_mode = dev_mode
        
        self.metrics = {}
    
    def compute_all_metrics(self, top_k: int = 10) -> Dict[str, Any]:
        """
        计算所有7个评估指标
        
        Args:
            top_k: 每个主题取top-k词计算
            
        Returns:
            包含所有指标的字典
        """
        print(f"\n{'='*60}")
        print(f"Computing metrics for {self.model_name} on {self.dataset}")
        print(f"{'='*60}")
        
        # 1. Topic Diversity (TD)
        print("  Computing Topic Diversity (TD)...")
        td = compute_topic_diversity(self.beta, top_k=25)
        self.metrics['topic_diversity_td'] = td
        
        # 2. Topic Diversity (iRBO)
        print("  Computing Topic Diversity (iRBO)...")
        irbo = self._compute_irbo(top_k=25)
        self.metrics['topic_diversity_irbo'] = irbo
        
        # 3. Topic Coherence (NPMI)
        print("  Computing Topic Coherence (NPMI)...")
        npmi_avg, npmi_per_topic = compute_topic_coherence_npmi(
            self.beta, self.bow_matrix, top_k=top_k
        )
        self.metrics['topic_coherence_npmi_avg'] = npmi_avg
        self.metrics['topic_coherence_npmi_per_topic'] = npmi_per_topic
        
        # 4. Topic Coherence (C_V)
        print("  Computing Topic Coherence (C_V)...")
        try:
            cv_avg, cv_per_topic = compute_topic_coherence_cv(
                self.beta, self.bow_matrix, top_k=top_k
            )
            self.metrics['topic_coherence_cv_avg'] = cv_avg
            self.metrics['topic_coherence_cv_per_topic'] = cv_per_topic
        except Exception as e:
            print(f"    Warning: C_V computation failed: {e}")
            self.metrics['topic_coherence_cv_avg'] = None
            self.metrics['topic_coherence_cv_per_topic'] = None
        
        # 5. Topic Coherence (UMass)
        print("  Computing Topic Coherence (UMass)...")
        umass_avg, umass_per_topic = compute_topic_coherence_umass(
            self.beta, self.bow_matrix, top_k=top_k
        )
        self.metrics['topic_coherence_umass_avg'] = umass_avg
        self.metrics['topic_coherence_umass_per_topic'] = umass_per_topic
        
        # 6. Topic Exclusivity
        print("  Computing Topic Exclusivity...")
        excl_avg, excl_per_topic = compute_topic_exclusivity(self.beta, top_k=top_k)
        self.metrics['topic_exclusivity_avg'] = excl_avg
        self.metrics['topic_exclusivity_per_topic'] = excl_per_topic
        
        # 7. Topic Significance
        print("  Computing Topic Significance...")
        sig_per_topic = compute_topic_significance(self.theta)
        self.metrics['topic_significance_per_topic'] = sig_per_topic
        self.metrics['topic_significance_avg'] = float(np.mean(sig_per_topic))
        
        # 8. Perplexity (if possible)
        print("  Computing Perplexity...")
        try:
            ppl = compute_perplexity(self.beta, self.theta, self.bow_matrix)
            self.metrics['perplexity'] = ppl
        except Exception as e:
            print(f"    Warning: Perplexity computation failed: {e}")
            self.metrics['perplexity'] = None
        
        # 打印结果摘要
        print(f"\n  Results:")
        print(f"    - Topic Diversity (TD): {td:.4f}")
        print(f"    - Topic Diversity (iRBO): {irbo:.4f}")
        print(f"    - Coherence (NPMI): {npmi_avg:.4f}")
        if self.metrics['topic_coherence_cv_avg'] is not None:
            print(f"    - Coherence (C_V): {self.metrics['topic_coherence_cv_avg']:.4f}")
        print(f"    - Coherence (UMass): {umass_avg:.4f}")
        print(f"    - Exclusivity: {excl_avg:.4f}")
        print(f"    - Significance: {self.metrics['topic_significance_avg']:.4f}")
        if self.metrics['perplexity'] is not None:
            print(f"    - Perplexity: {self.metrics['perplexity']:.2f}")
        
        return self.metrics
    
    def _compute_irbo(self, top_k: int = 25, p: float = 0.9) -> float:
        """
        计算 Inverse Rank-Biased Overlap (iRBO)
        
        Args:
            top_k: 每个主题取top-k词
            p: RBO参数
            
        Returns:
            iRBO分数
        """
        num_topics = self.beta.shape[0]
        top_words = np.argsort(-self.beta, axis=1)[:, :top_k]
        
        rbo_sum = 0
        count = 0
        
        for i in range(num_topics):
            for j in range(i + 1, num_topics):
                # 计算两个主题的RBO
                list1 = set(top_words[i])
                list2 = set(top_words[j])
                
                # 简化的RBO计算
                overlap = len(list1 & list2) / top_k
                rbo_sum += overlap
                count += 1
        
        avg_overlap = rbo_sum / count if count > 0 else 0
        irbo = 1 - avg_overlap  # Inverse
        return irbo
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        保存评估指标到JSON文件
        
        Args:
            filename: 文件名，默认为 metrics_k{K}.json
            
        Returns:
            保存的文件路径
        """
        if self.output_dir is None:
            raise ValueError("output_dir not set")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f'metrics_k{self.num_topics}.json'
        
        filepath = self.output_dir / filename
        
        # 转换numpy类型为Python原生类型
        metrics_json = {}
        for k, v in self.metrics.items():
            if isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                metrics_json[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                metrics_json[k] = int(v)
            elif isinstance(v, list):
                metrics_json[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
            else:
                metrics_json[k] = v
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Metrics saved to {filepath}")
        return str(filepath)
    
    def generate_training_plots(self) -> List[str]:
        """
        生成训练过程可视化图表
        
        Returns:
            生成的图片路径列表
        """
        if self.training_history is None:
            print("  [SKIP] No training history available")
            return []
        
        if self.output_dir is None:
            raise ValueError("output_dir not set")
        
        viz_dir = self.output_dir / 'visualization' / 'global'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Training Loss Curve
        if 'train_loss' in self.training_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            train_loss = self.training_history['train_loss']
            epochs = range(1, len(train_loss) + 1)
            
            ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
            
            if 'val_loss' in self.training_history:
                val_loss = self.training_history['val_loss']
                ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{self.model_name.upper()} Training Loss - {self.dataset}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filepath = viz_dir / 'training_loss.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            generated_files.append(str(filepath))
            print(f"  ✓ training_loss.png")
        
        # 2. Reconstruction Loss + KL Loss (if available)
        if 'recon_loss' in self.training_history and 'kl_loss' in self.training_history:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            recon_loss = self.training_history['recon_loss']
            kl_loss = self.training_history['kl_loss']
            epochs = range(1, len(recon_loss) + 1)
            
            axes[0].plot(epochs, recon_loss, 'g-', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Reconstruction Loss', fontsize=12)
            axes[0].set_title('Reconstruction Loss', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs, kl_loss, 'm-', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('KL Divergence Loss', fontsize=12)
            axes[1].set_title('KL Divergence Loss', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{self.model_name.upper()} Loss Components - {self.dataset}', fontsize=14)
            plt.tight_layout()
            
            filepath = viz_dir / 'loss_components.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            generated_files.append(str(filepath))
            print(f"  ✓ loss_components.png")
        
        # 3. Perplexity Curve (if available)
        if 'perplexity' in self.training_history:
            ppl_history = self.training_history['perplexity']
            if isinstance(ppl_history, list) and len(ppl_history) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                epochs = range(1, len(ppl_history) + 1)
                ax.plot(epochs, ppl_history, 'c-', linewidth=2)
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Perplexity', fontsize=12)
                ax.set_title(f'{self.model_name.upper()} Perplexity - {self.dataset}', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                filepath = viz_dir / 'perplexity_curve.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                generated_files.append(str(filepath))
                print(f"  ✓ perplexity_curve.png")
        
        return generated_files
    
    def generate_metrics_plots(self) -> List[str]:
        """
        生成评估指标可视化图表
        
        Returns:
            生成的图片路径列表
        """
        if not self.metrics:
            print("  [SKIP] No metrics computed yet")
            return []
        
        if self.output_dir is None:
            raise ValueError("output_dir not set")
        
        viz_dir = self.output_dir / 'visualization' / 'global'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Metrics Summary Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metric_names = []
        metric_values = []
        
        for name, key in [
            ('TD', 'topic_diversity_td'),
            ('iRBO', 'topic_diversity_irbo'),
            ('NPMI', 'topic_coherence_npmi_avg'),
            ('C_V', 'topic_coherence_cv_avg'),
            ('Exclusivity', 'topic_exclusivity_avg'),
            ('Significance', 'topic_significance_avg')
        ]:
            if key in self.metrics and self.metrics[key] is not None:
                metric_names.append(name)
                metric_values.append(self.metrics[key])
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.2)
        
        # 添加数值标签
        for bar, val in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.model_name.upper()} Evaluation Metrics - {self.dataset}', fontsize=14)
        ax.set_ylim(0, max(metric_values) * 1.15)
        
        filepath = viz_dir / 'metrics_summary.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_files.append(str(filepath))
        print(f"  ✓ metrics_summary.png")
        
        # 2. Per-topic metrics (Coherence, Exclusivity, Significance)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        topics = range(1, self.num_topics + 1)
        
        # Coherence per topic
        if 'topic_coherence_npmi_per_topic' in self.metrics:
            coherence = self.metrics['topic_coherence_npmi_per_topic']
            axes[0].bar(topics, coherence, color='steelblue', edgecolor='black')
            axes[0].axhline(y=np.mean(coherence), color='red', linestyle='--', label=f'Mean: {np.mean(coherence):.3f}')
            axes[0].set_xlabel('Topic')
            axes[0].set_ylabel('NPMI Coherence')
            axes[0].set_title('Topic Coherence (NPMI)')
            axes[0].legend()
        
        # Exclusivity per topic
        if 'topic_exclusivity_per_topic' in self.metrics:
            exclusivity = self.metrics['topic_exclusivity_per_topic']
            axes[1].bar(topics, exclusivity, color='seagreen', edgecolor='black')
            axes[1].axhline(y=np.mean(exclusivity), color='red', linestyle='--', label=f'Mean: {np.mean(exclusivity):.3f}')
            axes[1].set_xlabel('Topic')
            axes[1].set_ylabel('Exclusivity')
            axes[1].set_title('Topic Exclusivity')
            axes[1].legend()
        
        # Significance per topic
        if 'topic_significance_per_topic' in self.metrics:
            significance = self.metrics['topic_significance_per_topic']
            axes[2].bar(topics, significance, color='coral', edgecolor='black')
            axes[2].axhline(y=np.mean(significance), color='red', linestyle='--', label=f'Mean: {np.mean(significance):.3f}')
            axes[2].set_xlabel('Topic')
            axes[2].set_ylabel('Significance')
            axes[2].set_title('Topic Significance')
            axes[2].legend()
        
        plt.suptitle(f'{self.model_name.upper()} Per-Topic Metrics - {self.dataset}', fontsize=14)
        plt.tight_layout()
        
        filepath = viz_dir / 'per_topic_metrics.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_files.append(str(filepath))
        print(f"  ✓ per_topic_metrics.png")
        
        return generated_files


def evaluate_model(
    model_dir: str,
    bow_matrix: Union[np.ndarray, sp.csr_matrix],
    vocab: List[str],
    model_name: str,
    dataset: str,
    num_topics: int = 20
) -> Dict[str, Any]:
    """
    便捷函数：评估已训练的模型
    
    Args:
        model_dir: 模型目录（包含theta, beta等文件）
        bow_matrix: BOW矩阵
        vocab: 词汇表
        model_name: 模型名称
        dataset: 数据集名称
        num_topics: 主题数
        
    Returns:
        评估结果
    """
    model_dir = Path(model_dir)
    
    # 加载theta和beta
    theta_path = model_dir / f'theta_k{num_topics}.npy'
    beta_path = model_dir / f'beta_k{num_topics}.npy'
    
    if not theta_path.exists() or not beta_path.exists():
        # 尝试其他命名格式
        theta_files = list(model_dir.glob('theta_*.npy'))
        beta_files = list(model_dir.glob('beta_*.npy'))
        if theta_files and beta_files:
            theta_path = sorted(theta_files)[-1]
            beta_path = sorted(beta_files)[-1]
        else:
            raise FileNotFoundError(f"theta/beta not found in {model_dir}")
    
    theta = np.load(theta_path)
    beta = np.load(beta_path)
    
    # 加载训练历史（如果存在）
    training_history = None
    history_path = model_dir / f'training_history_k{num_topics}.json'
    if not history_path.exists():
        history_files = list(model_dir.glob('training_history_*.json'))
        if history_files:
            history_path = sorted(history_files)[-1]
    
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # 创建评估器
    evaluator = UnifiedEvaluator(
        beta=beta,
        theta=theta,
        bow_matrix=bow_matrix,
        vocab=vocab,
        training_history=training_history,
        model_name=model_name,
        dataset=dataset,
        output_dir=str(model_dir),
        num_topics=num_topics
    )
    
    # 计算指标
    metrics = evaluator.compute_all_metrics()
    
    # 保存指标
    evaluator.save_metrics()
    
    # 生成可视化
    evaluator.generate_training_plots()
    evaluator.generate_metrics_plots()
    
    return metrics

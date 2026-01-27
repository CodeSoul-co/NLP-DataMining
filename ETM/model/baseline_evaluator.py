"""
Baseline Evaluator - Baseline模型统一评估器

确保Baseline模型（LDA, ETM, CTM）的输出与THETA方法的评估和可视化系统兼容。

评估指标:
- Perplexity (困惑度)
- Topic Diversity TD (主题多样性)
- Topic Diversity iRBO (基于排名的多样性)
- Topic Coherence NPMI (归一化点互信息)
- Topic Coherence C_V (C_V一致性)
- Topic Coherence UMass (UMass一致性)
- Topic Exclusivity (主题独占性)
"""

import os
import json
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd

# 导入THETA的评估模块
import sys
sys.path.insert(0, '/root/autodl-tmp/ETM')

from evaluation.topic_metrics import (
    compute_all_metrics,
    compute_topic_diversity_td,
    compute_topic_diversity_inverted_rbo,
    compute_topic_coherence_npmi,
    compute_topic_coherence_cv,
    compute_topic_coherence_umass,
    compute_topic_exclusivity,
    compute_perplexity
)

from visualization.topic_visualizer import TopicVisualizer


class BaselineEvaluator:
    """
    Baseline模型评估器
    
    使用与THETA相同的评估指标和可视化方法，确保公平对比。
    """
    
    def __init__(
        self,
        result_dir: str,
        dataset: str,
        bow_matrix: np.ndarray = None,
        vocab: List[str] = None
    ):
        """
        初始化评估器
        
        Args:
            result_dir: Baseline结果目录
            dataset: 数据集名称
            bow_matrix: BOW矩阵 (用于计算coherence)
            vocab: 词汇表
        """
        self.result_dir = result_dir
        self.dataset = dataset
        self.bow_matrix = bow_matrix
        self.vocab = vocab
        
        # 加载BOW矩阵（如果未提供）
        if self.bow_matrix is None:
            bow_path = os.path.join(result_dir, 'bow_matrix.npz')
            if os.path.exists(bow_path):
                self.bow_matrix = sp.load_npz(bow_path)
        
        # 加载词汇表（如果未提供）
        if self.vocab is None:
            vocab_path = os.path.join(result_dir, 'vocab.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.vocab = json.load(f)
    
    def evaluate_model(
        self,
        model_name: str,
        theta: np.ndarray,
        beta: np.ndarray,
        num_topics: int = None
    ) -> Dict[str, Any]:
        """
        评估单个模型
        
        Args:
            model_name: 模型名称 ('lda', 'etm', 'ctm')
            theta: 文档-主题分布 (D x K)
            beta: 主题-词分布 (K x V)
            num_topics: 主题数量
            
        Returns:
            评估指标字典
        """
        if num_topics is None:
            num_topics = beta.shape[0]
        
        print(f"\n评估 {model_name.upper()} (K={num_topics})...")
        
        # 使用THETA的评估函数
        metrics = compute_all_metrics(
            beta=beta,
            theta=theta,
            doc_term_matrix=self.bow_matrix,
            top_k_coherence=10,
            top_k_diversity=25,
            compute_extended=True
        )
        
        # 添加模型信息
        metrics['model'] = model_name
        metrics['num_topics'] = num_topics
        metrics['num_docs'] = theta.shape[0]
        metrics['vocab_size'] = beta.shape[1]
        
        return metrics
    
    def evaluate_from_files(
        self,
        model_name: str,
        num_topics: int = 20
    ) -> Dict[str, Any]:
        """
        从文件加载结果并评估
        
        Args:
            model_name: 模型名称
            num_topics: 主题数量
            
        Returns:
            评估指标字典
        """
        # 确定模型目录
        if model_name == 'ctm':
            model_dir = os.path.join(self.result_dir, 'ctm_zeroshot')
            if not os.path.exists(model_dir):
                model_dir = os.path.join(self.result_dir, 'ctm_combined')
        else:
            model_dir = os.path.join(self.result_dir, model_name)
        
        # 加载theta和beta
        theta_path = os.path.join(model_dir, f'theta_k{num_topics}.npy')
        beta_path = os.path.join(model_dir, f'beta_k{num_topics}.npy')
        
        if not os.path.exists(theta_path) or not os.path.exists(beta_path):
            raise FileNotFoundError(f"找不到 {model_name} 的结果文件: {theta_path}, {beta_path}")
        
        theta = np.load(theta_path)
        beta = np.load(beta_path)
        
        return self.evaluate_model(model_name, theta, beta, num_topics)
    
    def evaluate_all_baselines(
        self,
        models: List[str] = None,
        num_topics: int = 20
    ) -> pd.DataFrame:
        """
        评估所有Baseline模型
        
        Args:
            models: 模型列表
            num_topics: 主题数量
            
        Returns:
            评估结果DataFrame
        """
        if models is None:
            models = ['lda', 'etm', 'ctm']
        
        results = []
        
        for model_name in models:
            try:
                metrics = self.evaluate_from_files(model_name, num_topics)
                results.append(metrics)
                print(f"\n{model_name.upper()} 评估完成:")
                print(f"  - Perplexity: {metrics.get('perplexity', 'N/A')}")
                print(f"  - Diversity (TD): {metrics['topic_diversity_td']:.4f}")
                print(f"  - Coherence (NPMI): {metrics['topic_coherence_npmi_avg']:.4f}")
                print(f"  - Exclusivity: {metrics.get('topic_exclusivity_avg', 'N/A')}")
            except Exception as e:
                print(f"评估 {model_name} 失败: {e}")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 选择关键列
        key_columns = [
            'model', 'num_topics', 'perplexity',
            'topic_diversity_td', 'topic_diversity_irbo',
            'topic_coherence_npmi_avg', 'topic_coherence_cv_avg',
            'topic_coherence_umass_avg', 'topic_exclusivity_avg'
        ]
        df = df[[c for c in key_columns if c in df.columns]]
        
        return df
    
    def save_evaluation_results(
        self,
        results: pd.DataFrame,
        output_path: str = None
    ):
        """
        保存评估结果（与THETA格式兼容）
        
        Args:
            results: 评估结果DataFrame
            output_path: 输出路径
        """
        if output_path is None:
            output_path = os.path.join(self.result_dir, 'baseline_evaluation_metrics.csv')
        
        results.to_csv(output_path, index=False)
        print(f"\n评估结果已保存到: {output_path}")
        
        # 同时保存JSON格式
        json_path = output_path.replace('.csv', '.json')
        results.to_json(json_path, orient='records', indent=2)
    
    def visualize_model(
        self,
        model_name: str,
        theta: np.ndarray,
        beta: np.ndarray,
        topic_words: Dict[str, List[str]],
        output_dir: str = None
    ):
        """
        使用THETA的可视化工具可视化模型结果
        
        Args:
            model_name: 模型名称
            theta: 文档-主题分布
            beta: 主题-词分布
            topic_words: 主题词字典
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = os.path.join(self.result_dir, model_name, 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建可视化器
        visualizer = TopicVisualizer(output_dir=output_dir)
        
        # 转换topic_words格式为THETA格式
        # THETA格式: List[Tuple[int, List[Tuple[str, float]]]]
        topic_words_theta_format = []
        for topic_key, words in topic_words.items():
            topic_idx = int(topic_key.split('_')[1])
            # 获取对应的概率
            word_probs = []
            for word in words:
                if self.vocab and word in self.vocab:
                    word_idx = self.vocab.index(word)
                    prob = beta[topic_idx, word_idx]
                else:
                    prob = 1.0 / len(words)  # 均匀分布作为fallback
                word_probs.append((word, float(prob)))
            topic_words_theta_format.append((topic_idx, word_probs))
        
        # 排序
        topic_words_theta_format.sort(key=lambda x: x[0])
        
        print(f"\n生成 {model_name.upper()} 可视化...")
        
        # 1. 主题词条形图
        visualizer.visualize_topic_words(
            topic_words_theta_format,
            num_words=10,
            filename=f'{model_name}_topic_words.png'
        )
        
        # 2. 主题词云
        visualizer.visualize_all_wordclouds(
            topic_words_theta_format,
            num_words=30,
            filename=f'{model_name}_wordclouds.png'
        )
        
        # 3. 主题相似度热力图
        visualizer.visualize_topic_similarity(
            beta,
            topic_words_theta_format,
            filename=f'{model_name}_topic_similarity.png'
        )
        
        # 4. 文档-主题分布
        visualizer.visualize_document_topics(
            theta,
            method='umap',
            topic_words=topic_words_theta_format,
            filename=f'{model_name}_document_topics.png'
        )
        
        # 5. 主题比例
        visualizer.visualize_topic_proportions(
            theta,
            topic_words=topic_words_theta_format,
            filename=f'{model_name}_topic_proportions.png'
        )
        
        print(f"可视化已保存到: {output_dir}")
    
    def visualize_from_files(
        self,
        model_name: str,
        num_topics: int = 20
    ):
        """
        从文件加载结果并可视化
        
        Args:
            model_name: 模型名称
            num_topics: 主题数量
        """
        # 确定模型目录
        if model_name == 'ctm':
            model_dir = os.path.join(self.result_dir, 'ctm_zeroshot')
            if not os.path.exists(model_dir):
                model_dir = os.path.join(self.result_dir, 'ctm_combined')
        else:
            model_dir = os.path.join(self.result_dir, model_name)
        
        # 加载数据
        theta = np.load(os.path.join(model_dir, f'theta_k{num_topics}.npy'))
        beta = np.load(os.path.join(model_dir, f'beta_k{num_topics}.npy'))
        
        with open(os.path.join(model_dir, f'topic_words_k{num_topics}.json'), 'r', encoding='utf-8') as f:
            topic_words = json.load(f)
        
        self.visualize_model(model_name, theta, beta, topic_words)
    
    def compare_with_theta(
        self,
        theta_result_dir: str,
        mode: str = 'zero_shot',
        num_topics: int = 20
    ) -> pd.DataFrame:
        """
        与THETA方法进行对比
        
        Args:
            theta_result_dir: THETA结果目录
            mode: THETA模式
            num_topics: 主题数量
            
        Returns:
            对比结果DataFrame
        """
        # 加载THETA结果
        theta_model_dir = os.path.join(theta_result_dir, self.dataset, mode, 'model')
        
        # 查找最新的timestamp
        import glob
        theta_files = sorted(glob.glob(os.path.join(theta_model_dir, 'theta_*.npy')), reverse=True)
        
        if not theta_files:
            print(f"找不到THETA结果: {theta_model_dir}")
            return None
        
        # 提取timestamp
        timestamp = os.path.basename(theta_files[0]).replace('theta_', '').replace('.npy', '')
        
        theta_theta = np.load(os.path.join(theta_model_dir, f'theta_{timestamp}.npy'))
        theta_beta = np.load(os.path.join(theta_model_dir, f'beta_{timestamp}.npy'))
        
        # 评估THETA
        theta_metrics = self.evaluate_model('THETA', theta_theta, theta_beta)
        
        # 评估Baseline
        baseline_results = self.evaluate_all_baselines(num_topics=num_topics)
        
        # 合并结果
        theta_df = pd.DataFrame([theta_metrics])
        all_results = pd.concat([baseline_results, theta_df], ignore_index=True)
        
        return all_results


def compare_all_models(
    dataset: str,
    baseline_dir: str = '/root/autodl-tmp/result/baseline',
    theta_dir: str = '/root/autodl-tmp/result/0.6B',
    mode: str = 'zero_shot',
    num_topics: int = 20,
    models: List[str] = None
) -> pd.DataFrame:
    """
    对比所有模型的便捷函数
    
    Args:
        dataset: 数据集名称
        baseline_dir: Baseline结果目录
        theta_dir: THETA结果目录
        mode: THETA模式
        num_topics: 主题数量
        models: Baseline模型列表
        
    Returns:
        对比结果DataFrame
    """
    if models is None:
        models = ['lda', 'etm', 'ctm']
    
    # 创建评估器
    evaluator = BaselineEvaluator(
        result_dir=os.path.join(baseline_dir, dataset),
        dataset=dataset
    )
    
    # 评估Baseline
    baseline_results = evaluator.evaluate_all_baselines(models=models, num_topics=num_topics)
    
    # 尝试加载THETA结果
    try:
        theta_model_dir = os.path.join(theta_dir, dataset, mode, 'model')
        import glob
        theta_files = sorted(glob.glob(os.path.join(theta_model_dir, 'theta_*.npy')), reverse=True)
        
        if theta_files:
            timestamp = os.path.basename(theta_files[0]).replace('theta_', '').replace('.npy', '')
            theta_theta = np.load(os.path.join(theta_model_dir, f'theta_{timestamp}.npy'))
            theta_beta = np.load(os.path.join(theta_model_dir, f'beta_{timestamp}.npy'))
            
            theta_metrics = evaluator.evaluate_model('THETA', theta_theta, theta_beta)
            theta_df = pd.DataFrame([theta_metrics])
            baseline_results = pd.concat([baseline_results, theta_df], ignore_index=True)
    except Exception as e:
        print(f"加载THETA结果失败: {e}")
    
    # 保存结果
    output_path = os.path.join(baseline_dir, dataset, 'comparison_results.csv')
    baseline_results.to_csv(output_path, index=False)
    print(f"\n对比结果已保存到: {output_path}")
    
    return baseline_results


def print_comparison_table(results: pd.DataFrame):
    """
    打印对比表格
    
    Args:
        results: 对比结果DataFrame
    """
    print("\n" + "="*80)
    print("模型对比结果")
    print("="*80)
    
    # 格式化输出
    display_cols = {
        'model': '模型',
        'perplexity': 'Perplexity↓',
        'topic_diversity_td': 'TD↑',
        'topic_diversity_irbo': 'iRBO↑',
        'topic_coherence_npmi_avg': 'NPMI↑',
        'topic_coherence_cv_avg': 'C_V↑',
        'topic_exclusivity_avg': 'Exclusivity↑'
    }
    
    # 选择存在的列
    cols = [c for c in display_cols.keys() if c in results.columns]
    df_display = results[cols].copy()
    df_display.columns = [display_cols[c] for c in cols]
    
    # 格式化数值
    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(
            lambda x: f'{x:.4f}' if pd.notna(x) and isinstance(x, (int, float)) else str(x)
        )
    
    print(df_display.to_string(index=False))
    print("="*80)
    print("↑ = 越高越好, ↓ = 越低越好")

"""
Short Text Topic Analysis Visualization (BTM-style)

Creates BTM-style visualizations for short text topic modeling:
- Perplexity curve for optimal K selection
- Topic keywords table (publication-ready)
- Biterm aggregation preprocessing guidance

Reference: Biterm Topic Model (BTM) for short texts
Note: Uses ETM + biterm/aggregation preprocessing to approximate BTM benefits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


class ShortTextAnalysisVisualizer:
    """
    Visualize short text topic analysis results (BTM-style).
    
    Uses ETM outputs with biterm/aggregation preprocessing
    to approximate BTM's benefits for short texts.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150
    ):
        """
        Initialize short text analysis visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def plot_perplexity_curve(
        self,
        k_values: List[int],
        perplexity_values: List[float],
        optimal_k: int = None,
        filename: Optional[str] = None,
        title: str = "Confusion degree corresponding to different topics number",
        xlabel: str = "主题数",
        ylabel: str = "困惑度(×10⁴)"
    ) -> plt.Figure:
        """
        Create BTM-style perplexity curve for K selection.
        
        Shows perplexity vs number of topics, with elbow point
        indicating optimal K.
        
        Args:
            k_values: List of topic numbers tested
            perplexity_values: Perplexity at each K
            optimal_k: Optimal K value (will be marked)
            filename: Output filename
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Plot perplexity curve
        ax.plot(k_values, perplexity_values, 'k-', linewidth=1.5, marker='o',
                markersize=4, markerfacecolor='white', markeredgecolor='black')
        
        # Mark optimal K if provided
        if optimal_k and optimal_k in k_values:
            idx = k_values.index(optimal_k)
            ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.scatter([optimal_k], [perplexity_values[idx]], color='red', s=80, zorder=5)
            ax.annotate(f'K={optimal_k}', xy=(optimal_k, perplexity_values[idx]),
                       xytext=(optimal_k + 2, perplexity_values[idx] + 5),
                       fontsize=10, color='red')
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set x-axis ticks
        ax.set_xticks(k_values[::2] if len(k_values) > 15 else k_values)
        ax.tick_params(axis='both', labelsize=9)
        
        # Grid
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def generate_topic_keywords_table(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        n_words: int = 15,
        topic_labels: Dict[int, str] = None,
        filename: Optional[str] = None,
        title: str = "Key words for safety risk of aircraft maintenance safety hazard record"
    ) -> plt.Figure:
        """
        Create BTM-style topic keywords table.
        
        Generates a publication-ready table showing top keywords
        for each topic, similar to BTM paper Table 1.
        
        Args:
            topic_words: Dict mapping topic_idx to list of (word, weight) tuples
            n_words: Number of keywords per topic
            topic_labels: Optional custom labels for topics (e.g., {0: 'A', 1: 'B'})
            filename: Output filename
            title: Table title
            
        Returns:
            Figure
        """
        n_topics = len(topic_words)
        
        # Generate topic labels if not provided
        if topic_labels is None:
            topic_labels = {i: chr(65 + i) if i < 26 else f"T{i}" 
                          for i in range(n_topics)}
        
        # Prepare table data
        table_data = []
        for topic_idx in sorted(topic_words.keys()):
            label = topic_labels.get(topic_idx, f"T{topic_idx}")
            words_list = topic_words[topic_idx]
            if not words_list:
                keywords = ""
            elif isinstance(words_list, dict):
                # Dict format: {word: weight}
                sorted_words = sorted(words_list.items(), key=lambda x: x[1], reverse=True)[:n_words]
                keywords = "、".join([w[0] for w in sorted_words])
            elif isinstance(words_list[0], (tuple, list)):
                # List of (word, weight) tuples
                keywords = "、".join([w[0] for w in words_list[:n_words]])
            else:
                # Plain list of words
                keywords = "、".join([str(w) for w in words_list[:n_words]])
            table_data.append([label, keywords])
        
        # Calculate figure height based on number of topics
        row_height = 0.4
        fig_height = max(4, 1.5 + n_topics * row_height)
        
        fig, ax = plt.subplots(figsize=(12, fig_height), facecolor='white')
        ax.set_facecolor('white')
        ax.axis('off')
        
        # Create table
        col_widths = [0.08, 0.92]
        table = ax.table(
            cellText=table_data,
            colLabels=['序号', '关键词'],
            colWidths=col_widths,
            loc='center',
            cellLoc='left'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style header
        for j in range(2):
            cell = table[(0, j)]
            cell.set_text_props(fontweight='bold', fontsize=10)
            cell.set_facecolor('#E8E8E8')
        
        # Style data cells
        for i in range(1, n_topics + 1):
            # Topic label column
            table[(i, 0)].set_text_props(fontweight='bold', ha='center')
            table[(i, 0)].set_facecolor('#F8F8F8')
            # Keywords column
            table[(i, 1)].set_text_props(ha='left')
        
        # Title
        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_combined_analysis(
        self,
        k_values: List[int],
        perplexity_values: List[float],
        topic_words: Dict[int, List[Tuple[str, float]]],
        optimal_k: int = None,
        n_words: int = 15,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create combined BTM-style analysis figure.
        
        Left: Perplexity curve for K selection
        Right: Topic keywords table
        
        Args:
            k_values: Topic numbers tested
            perplexity_values: Perplexity values
            topic_words: Topic keywords
            optimal_k: Optimal K
            n_words: Keywords per topic
            filename: Output filename
            
        Returns:
            Figure
        """
        n_topics = len(topic_words)
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(16, max(8, 2 + n_topics * 0.5)), facecolor='white')
        
        # Left: Perplexity curve
        ax1 = fig.add_axes([0.05, 0.15, 0.35, 0.75])
        ax1.set_facecolor('white')
        
        ax1.plot(k_values, perplexity_values, 'k-', linewidth=1.5, marker='o',
                markersize=4, markerfacecolor='white', markeredgecolor='black')
        
        if optimal_k and optimal_k in k_values:
            idx = k_values.index(optimal_k)
            ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax1.scatter([optimal_k], [perplexity_values[idx]], color='red', s=80, zorder=5)
        
        ax1.set_xlabel('主题数', fontsize=11)
        ax1.set_ylabel('困惑度(×10⁴)', fontsize=11)
        ax1.set_title('Fig. 5  不同主题数对应的困惑度', fontsize=10)
        ax1.set_xticks(k_values[::2] if len(k_values) > 15 else k_values)
        ax1.tick_params(axis='both', labelsize=9)
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # Right: Keywords table
        ax2 = fig.add_axes([0.45, 0.1, 0.52, 0.8])
        ax2.set_facecolor('white')
        ax2.axis('off')
        
        # Generate topic labels
        topic_labels = {i: chr(65 + i) if i < 26 else f"T{i}" 
                       for i in range(n_topics)}
        
        # Prepare table data
        table_data = []
        for topic_idx in sorted(topic_words.keys()):
            label = topic_labels.get(topic_idx, f"T{topic_idx}")
            words_list = topic_words[topic_idx]
            if not words_list:
                keywords = ""
            elif isinstance(words_list, dict):
                sorted_words = sorted(words_list.items(), key=lambda x: x[1], reverse=True)[:n_words]
                keywords = "、".join([w[0] for w in sorted_words])
            elif isinstance(words_list[0], (tuple, list)):
                keywords = "、".join([w[0] for w in words_list[:n_words]])
            else:
                keywords = "、".join([str(w) for w in words_list[:n_words]])
            table_data.append([label, keywords])
        
        # Create table
        table = ax2.table(
            cellText=table_data,
            colLabels=['序号', '关键词'],
            colWidths=[0.06, 0.94],
            loc='center',
            cellLoc='left'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style header
        for j in range(2):
            cell = table[(0, j)]
            cell.set_text_props(fontweight='bold', fontsize=9)
            cell.set_facecolor('#E8E8E8')
        
        # Style data cells
        for i in range(1, n_topics + 1):
            table[(i, 0)].set_text_props(fontweight='bold', ha='center')
            table[(i, 0)].set_facecolor('#F8F8F8')
        
        ax2.set_title('Table 1  机务维修安全隐患记录的安全隐患关键词', fontsize=10, pad=10)
        
        plt.suptitle('ETM + Biterm聚合预处理：短文本主题分析结果', 
                    fontsize=12, fontweight='bold', y=0.98)
        
        return self._save_or_show(fig, filename)
    
    def simulate_perplexity_data(
        self,
        k_range: Tuple[int, int] = (8, 72),
        step: int = 4,
        optimal_k: int = 12,
        seed: int = 42
    ) -> Dict:
        """
        Simulate realistic perplexity curve data.
        
        Simulates BTM-style perplexity curve with elbow at optimal_k.
        
        Args:
            k_range: Range of K values (min, max)
            step: Step size for K values
            optimal_k: Optimal K (elbow point)
            seed: Random seed
            
        Returns:
            Dict with k_values, perplexity_values, optimal_k
        """
        np.random.seed(seed)
        
        k_values = list(range(k_range[0], k_range[1] + 1, step))
        
        # Simulate perplexity: high at small K, drops sharply, then plateaus
        perplexity = []
        for k in k_values:
            if k <= optimal_k:
                # Sharp decrease before optimal K
                base = 100 - 50 * (k / optimal_k)
            else:
                # Slow decrease after optimal K
                base = 50 - 10 * np.log(k / optimal_k)
            
            # Add noise
            noise = np.random.randn() * 3
            perplexity.append(max(30, base + noise))
        
        return {
            'k_values': k_values,
            'perplexity_values': perplexity,
            'optimal_k': optimal_k
        }
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure or show it."""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Figure saved to {filepath}")
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return None


def plot_short_text_analysis(
    k_values: List[int],
    perplexity_values: List[float],
    topic_words: Dict[int, List[Tuple[str, float]]],
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function for BTM-style short text analysis plot.
    
    Args:
        k_values: Topic numbers tested
        perplexity_values: Perplexity values
        topic_words: Topic keywords
        output_dir: Output directory
        filename: Output filename
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    visualizer = ShortTextAnalysisVisualizer(output_dir=output_dir)
    
    return visualizer.plot_combined_analysis(
        k_values=k_values,
        perplexity_values=perplexity_values,
        topic_words=topic_words,
        filename=filename,
        **kwargs
    )


# Biterm preprocessing guidance
BITERM_PREPROCESSING_GUIDE = """
# ETM + Biterm聚合预处理指南

## 目标
用 ETM + biterm/聚合预处理 最大程度逼近 BTM 的核心收益：
- 短文本更稳定的主题
- 评估曲线选K
- 主题关键词表

## 预处理步骤

### 1. Biterm生成
对于每个短文本文档，生成所有词对（biterms）：
```python
def generate_biterms(doc_tokens, window_size=None):
    biterms = []
    if window_size is None:
        # 全文档词对
        for i, w1 in enumerate(doc_tokens):
            for w2 in doc_tokens[i+1:]:
                biterms.append((w1, w2))
    else:
        # 滑动窗口词对
        for i in range(len(doc_tokens) - window_size + 1):
            window = doc_tokens[i:i+window_size]
            for j, w1 in enumerate(window):
                for w2 in window[j+1:]:
                    biterms.append((w1, w2))
    return biterms
```

### 2. 文档聚合
将相似短文本聚合为伪长文档：
```python
def aggregate_short_texts(texts, method='temporal'):
    if method == 'temporal':
        # 按时间窗口聚合
        return aggregate_by_time_window(texts)
    elif method == 'user':
        # 按用户聚合
        return aggregate_by_user(texts)
    elif method == 'thread':
        # 按对话线程聚合
        return aggregate_by_thread(texts)
```

### 3. ETM训练
使用聚合后的文档训练ETM：
- 词嵌入：使用预训练词向量
- 主题数：通过困惑度曲线选择
- 正则化：适当增加以防止过拟合

## 评估指标
- 困惑度 (Perplexity)：越低越好
- 主题一致性 (Coherence)：越高越好
- 主题多样性 (Diversity)：避免重复主题

## 输出
1. 困惑度曲线图：选择拐点作为最优K
2. 主题关键词表：每个主题的top-N关键词
3. 主题分布：文档-主题分布矩阵
"""


if __name__ == '__main__':
    # Example usage
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = ShortTextAnalysisVisualizer(output_dir=output_dir)
    
    # Simulate perplexity data
    pp_data = visualizer.simulate_perplexity_data()
    
    # Simulate topic words
    topic_words = {
        i: [(f"word_{j}", 0.1 - j * 0.005) for j in range(15)]
        for i in range(12)
    }
    
    # Create combined analysis
    visualizer.plot_combined_analysis(
        k_values=pp_data['k_values'],
        perplexity_values=pp_data['perplexity_values'],
        topic_words=topic_words,
        optimal_k=pp_data['optimal_k'],
        filename='short_text_analysis.png'
    )
    
    print("Short text analysis plot saved!")
    print("\n" + BITERM_PREPROCESSING_GUIDE)

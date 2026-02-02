"""
Topic Visualization Tools for ETM

This module provides visualization tools for ETM results:
- Topic word clouds
- Topic similarity heatmap
- Document-topic distribution visualization
- Topic evolution over time (if timestamps available)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import logging

# Try to import wordcloud, but don't fail if it's not available
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logging.warning("WordCloud package not available. Install with 'pip install wordcloud'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicVisualizer:
    """
    Visualization tools for ETM results.
    
    Provides methods to visualize:
    - Topic word clouds
    - Topic similarity heatmap
    - Document-topic distribution
    - Topic embeddings in 2D space
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        cmap: str = "viridis",
        random_state: int = 42,
        language: str = 'en'
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Default figure DPI
            cmap: Default colormap
            random_state: Random state for reproducibility
            language: Language for labels ('en' or 'zh')
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.random_state = random_state
        self.language = language
        
        # 双语标签字典
        self._labels = {
            'topic': {'en': 'Topic', 'zh': '主题'},
            'word': {'en': 'Word', 'zh': '词'},
            'value': {'en': 'Value', 'zh': '值'},
            'frequency': {'en': 'Frequency', 'zh': '频率'},
            'similarity': {'en': 'Similarity', 'zh': '相似度'},
            'proportion': {'en': 'Proportion', 'zh': '比例'},
            'average_proportion': {'en': 'Average Proportion', 'zh': '平均比例'},
            'epoch': {'en': 'Epoch', 'zh': '轮次'},
            'loss': {'en': 'Loss', 'zh': '损失'},
            'perplexity': {'en': 'Perplexity', 'zh': '困惑度'},
            'train_loss': {'en': 'Train Loss', 'zh': '训练损失'},
            'val_loss': {'en': 'Val Loss', 'zh': '验证损失'},
            'train_perplexity': {'en': 'Train Perplexity', 'zh': '训练困惑度'},
            'val_perplexity': {'en': 'Validation Perplexity', 'zh': '验证困惑度'},
            # 图表标题
            'top_words_per_topic': {'en': 'Top 10 Words per Topic', 'zh': '每个主题的前10个词'},
            'topic_word_clouds': {'en': 'Topic Word Clouds', 'zh': '主题词云'},
            'etm_metrics': {'en': 'Evaluation Metrics', 'zh': '评估指标'},
            'topic_similarity_matrix': {'en': 'Topic Similarity Matrix', 'zh': '主题相似度矩阵'},
            'doc_topic_dist': {'en': 'Document-topic distribution', 'zh': '文档-主题分布'},
            'topic_proportions': {'en': 'Topic Proportions Across All Documents', 'zh': '所有文档的主题比例'},
            'train_val_loss': {'en': 'Training & Validation Loss', 'zh': '训练与验证损失'},
            'perplexity_training': {'en': 'Perplexity During Training', 'zh': '训练过程中的困惑度'},
            'intertopic_distance': {'en': 'Intertopic Distance Map', 'zh': '主题间距离图'},
            'via_mds': {'en': '(via multidimensional scaling)', 'zh': '(通过多维缩放)'},
            'marginal_topic_dist': {'en': 'Marginal topic distribution', 'zh': '边际主题分布'},
            'top_salient_terms': {'en': 'Most Salient Terms', 'zh': '最显著词汇'},
            'overall_term_freq': {'en': 'Overall term frequency', 'zh': '整体词频'},
            'estimated_term_freq': {'en': 'Estimated term frequency within the selected topic', 'zh': '所选主题内的估计词频'},
            # 指标名称
            'diversity_td': {'en': 'Diversity (TD)', 'zh': '多样性 (TD)'},
            'diversity_irbo': {'en': 'Diversity (iRBO)', 'zh': '多样性 (iRBO)'},
            'coherence_npmi': {'en': 'Coherence (NPMI)', 'zh': '一致性 (NPMI)'},
            'coherence_cv': {'en': 'Coherence (C_V)', 'zh': '一致性 (C_V)'},
            'exclusivity': {'en': 'Exclusivity', 'zh': '排他性'},
        }
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 设置中文字体支持
        self._setup_chinese_fonts()
    
    def _setup_chinese_fonts(self):
        """设置中文字体支持"""
        import matplotlib
        import matplotlib.font_manager as fm
        
        # 中文字体列表（按优先级）
        chinese_fonts = [
            'WenQuanYi Micro Hei',      # 文泉驿微米黑 (Linux)
            'WenQuanYi Zen Hei',        # 文泉驿正黑 (Linux)
            'Noto Sans CJK SC',         # Google Noto
            'SimHei',                   # 黑体 (Windows)
            'Microsoft YaHei',          # 微软雅黑
            'PingFang SC',              # 苹方 (macOS)
            'DejaVu Sans'               # 后备
        ]
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 查找可用的中文字体路径（用于WordCloud）
        self.chinese_font_path = None
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                self.chinese_font_path = fp
                break
    
    def _get_label(self, key: str) -> str:
        """获取双语标签"""
        if key in self._labels:
            return self._labels[key].get(self.language, self._labels[key].get('en', key))
        return key
    
    def _get_topic_label(self, topic_idx: int, short: bool = False) -> str:
        """获取主题标签"""
        if self.language == 'zh':
            return f"主题{topic_idx + 1}" if short else f"主题 {topic_idx + 1}"
        else:
            return f"T{topic_idx + 1}" if short else f"Topic {topic_idx + 1}"
        
    def _save_or_show(self, fig, filename=None):
        """Save figure to file or show it"""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {filepath}")
            return filepath
        else:
            plt.show()
            return None
    
    def visualize_topic_words(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_topics: int = None,
        num_words: int = 10,
        as_wordcloud: bool = False,
        filename: str = None
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Visualize top words for each topic.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_topics: Number of topics to visualize (None for all)
            num_words: Number of words per topic
            as_wordcloud: Whether to use word clouds
            filename: Filename to save visualization
            
        Returns:
            Figure or list of figures
        """
        if num_topics is None:
            num_topics = len(topic_words)
        else:
            num_topics = min(num_topics, len(topic_words))
        
        if as_wordcloud and not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available, falling back to bar plots")
            as_wordcloud = False
        
        if as_wordcloud:
            # Create a word cloud for each topic
            figs = []
            for topic_idx, words in topic_words[:num_topics]:
                # Create word frequency dictionary
                word_freq = {word: prob for word, prob in words[:num_words*2]}
                
                # Create word cloud
                fig, ax = plt.subplots(figsize=(10, 6))
                wc = WordCloud(
                    background_color='white',
                    width=800,
                    height=400,
                    max_words=num_words,
                    random_state=self.random_state
                ).generate_from_frequencies(word_freq)
                
                ax.imshow(wc, interpolation='bilinear')
                ax.set_title(self._get_topic_label(topic_idx), fontsize=16)
                ax.axis('off')
                
                figs.append(fig)
                
                # Save or show
                if filename:
                    base, ext = os.path.splitext(filename)
                    topic_filename = f"{base}_topic{topic_idx + 1}{ext}"
                    self._save_or_show(fig, topic_filename)
            
            return figs
        else:
            # Create bar plots for topics with Spectral colormap
            n_cols = min(5, num_topics)
            n_rows = (num_topics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4 * n_cols, 2.5 * n_rows),
                facecolor='white'
            )
            
            # Flatten axes for easier indexing
            if n_rows * n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            # Use Spectral colormap for different topic colors
            colors = plt.cm.Spectral(np.linspace(0, 1, num_topics))
            
            for i, (topic_idx, words) in enumerate(topic_words[:num_topics]):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Extract words and probabilities
                top_words = [word for word, _ in words[:num_words]]
                top_probs = [prob for _, prob in words[:num_words]]
                
                # Create horizontal bar plot with topic-specific color
                y_pos = np.arange(len(top_words))
                ax.barh(y_pos, top_probs, align='center', color=colors[i], edgecolor='white', linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_words, fontsize=8)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_title(self._get_topic_label(topic_idx), fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', labelsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle(self._get_label('top_words_per_topic'), fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save or show
            return self._save_or_show(fig, filename)
    
    def visualize_all_wordclouds(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_words: int = 30,
        filename: str = None
    ) -> plt.Figure:
        """
        Generate a single figure with all topic wordclouds in a grid.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_words: Number of words per wordcloud
            filename: Filename to save visualization
            
        Returns:
            Figure with all wordclouds
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available")
            return None
        
        num_topics = len(topic_words)
        n_cols = min(5, num_topics)
        n_rows = (num_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            facecolor='white'
        )
        
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Use viridis colormap for different topic colors
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_topics))
        
        def make_color_func(color):
            """Create a color function with proper closure"""
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                # Add some variation based on font_size
                factor = 0.7 + 0.3 * (font_size / 100)
                r = int(min(255, color[0] * 255 * factor))
                g = int(min(255, color[1] * 255 * factor))
                b = int(min(255, color[2] * 255 * factor))
                return f"rgb({r}, {g}, {b})"
            return color_func
        
        for i, (topic_idx, words) in enumerate(topic_words):
            if i >= len(axes):
                break
            
            ax = axes[i]
            word_freq = {word: prob for word, prob in words[:num_words]}
            
            try:
                # 使用中文字体（如果可用）
                wc_kwargs = {
                    'background_color': 'white',
                    'width': 400,
                    'height': 300,
                    'max_words': num_words,
                    'random_state': self.random_state,
                    'color_func': make_color_func(colors[i])
                }
                if self.chinese_font_path:
                    wc_kwargs['font_path'] = self.chinese_font_path
                
                wc = WordCloud(**wc_kwargs).generate_from_frequencies(word_freq)
                
                ax.imshow(wc, interpolation='bilinear')
            except Exception as e:
                ax.text(0.5, 0.5, f'Topic {topic_idx + 1}\n(error)', ha='center', va='center')
            
            ax.set_title(self._get_topic_label(topic_idx), fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for j in range(len(topic_words), len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(self._get_label('topic_word_clouds'), fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def visualize_combined_wordcloud(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_words: int = 50,
        filename: str = None
    ) -> plt.Figure:
        """
        Generate a single combined wordcloud with all topic words.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_words: Number of words per topic to include
            filename: Filename to save visualization
            
        Returns:
            Figure with combined wordcloud
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available")
            return None
        
        # Combine all words from all topics
        combined_freq = {}
        num_topics = len(topic_words)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_topics))
        word_colors = {}
        
        for i, (topic_idx, words) in enumerate(topic_words):
            for word, prob in words[:num_words]:
                if word not in combined_freq:
                    combined_freq[word] = prob
                    word_colors[word] = colors[i]
                else:
                    # Keep the higher probability and its color
                    if prob > combined_freq[word]:
                        combined_freq[word] = prob
                        word_colors[word] = colors[i]
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if word in word_colors:
                c = word_colors[word]
                return f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
            return "rgb(100, 100, 100)"
        
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
        
        try:
            wc = WordCloud(
                background_color='white',
                width=1000,
                height=1000,
                max_words=300,
                random_state=self.random_state,
                color_func=color_func
            ).generate_from_frequencies(combined_freq)
            
            ax.imshow(wc, interpolation='bilinear')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        
        ax.axis('off')
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def visualize_metrics(
        self,
        metrics: Dict,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize evaluation metrics as a bar chart.
        
        Args:
            metrics: Dictionary of metric names and values
            filename: Filename to save visualization
            
        Returns:
            Figure with metrics bar chart
        """
        # Filter and select key metrics (exclude perplexity and UMass)
        key_metrics = {}
        metric_mapping = {
            'topic_diversity_td': 'diversity_td',
            'topic_diversity_irbo': 'diversity_irbo',
            'topic_coherence_npmi_avg': 'coherence_npmi',
            'topic_coherence_cv_avg': 'coherence_cv',
            'topic_exclusivity_avg': 'exclusivity',
            # Legacy keys
            'diversity_td': 'diversity_td',
            'diversity_irbo': 'diversity_irbo',
            'coherence_npmi_avg': 'coherence_npmi',
            'coherence_cv_avg': 'coherence_cv',
            'exclusivity_avg': 'exclusivity',
        }
        
        for key, label_key in metric_mapping.items():
            display_name = self._get_label(label_key)
            if key in metrics and display_name not in key_metrics:
                key_metrics[display_name] = metrics[key]
        
        if not key_metrics:
            logger.warning("No metrics to visualize")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        names = list(key_metrics.keys())
        values = list(key_metrics.values())
        
        # Use Spectral colormap
        colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(names)))
        
        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(self._get_label('value'), fontsize=12)
        ax.set_title(self._get_label('etm_metrics'), fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(values) * 1.15)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def visualize_topic_similarity(
        self,
        beta: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        metric: str = 'cosine',
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize topic similarity as a heatmap.
        
        Args:
            beta: Topic-word distribution matrix (K x V)
            topic_words: Optional list of topic words for labels
            metric: Similarity metric ('cosine', 'euclidean', 'correlation')
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        num_topics = beta.shape[0]
        
        # Compute similarity matrix
        if metric == 'cosine':
            sim_matrix = cosine_similarity(beta)
        elif metric == 'euclidean':
            # Convert distances to similarities
            dist_matrix = euclidean_distances(beta)
            sim_matrix = 1 / (1 + dist_matrix)
        elif metric == 'correlation':
            sim_matrix = np.corrcoef(beta)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Create labels if topic_words is provided
        if topic_words:
            labels = []
            for topic_idx, words in topic_words:
                top_words = [word for word, _ in words[:3]]
                label = f"{topic_idx}: {', '.join(top_words)}"
                labels.append(label)
            labels = labels[:num_topics]
        else:
            labels = [self._get_topic_label(i) for i in range(num_topics)]
        
        # Adjust figure size and annotation based on number of topics
        if num_topics > 20:
            # For many topics, use larger figure and no annotations
            base_size = max(14, num_topics * 0.5)
            fig, ax = plt.subplots(figsize=(base_size, base_size - 2))
            show_annot = False
            fontsize = 8
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            show_annot = True
            fontsize = 10
        
        sns.heatmap(
            sim_matrix,
            annot=show_annot,
            fmt='.2f' if show_annot else '',
            annot_kws={'size': 8} if show_annot else {},
            cmap='RdYlBu_r',
            xticklabels=[self._get_topic_label(i, short=True) for i in range(num_topics)],
            yticklabels=[self._get_topic_label(i, short=True) for i in range(num_topics)],
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=0.3 if num_topics > 20 else 0.5,
            square=True,
            cbar_kws={'shrink': 0.8, 'label': self._get_label('similarity')}
        )
        ax.set_title(f"{self._get_label('topic_similarity_matrix')} ({metric.title()})", fontsize=16, fontweight='bold')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(rotation=0, fontsize=fontsize)
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_document_topics(
        self,
        theta: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'umap',
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        max_docs: int = 10000,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize document-topic distributions in 2D space.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            labels: Optional document labels for coloring
            method: Dimensionality reduction method ('umap', 'pca')
            topic_words: Optional list of topic words for annotation
            max_docs: Maximum number of documents to visualize
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Sample documents if too many
        n_docs = theta.shape[0]
        if n_docs > max_docs:
            indices = np.random.choice(n_docs, max_docs, replace=False)
            theta_sample = theta[indices]
        else:
            theta_sample = theta
            max_docs = n_docs
        
        # Get dominant topic for each document
        dominant_topics = np.argmax(theta_sample, axis=1)
        num_topics = theta_sample.shape[1]
        
        # Apply dimensionality reduction
        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    n_neighbors=30,
                    min_dist=0.3,
                    spread=1.0,
                    metric='cosine'
                )
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                reducer = PCA(n_components=2, random_state=self.random_state)
                method = 'pca'
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        theta_2d = reducer.fit_transform(theta_sample)
        
        # Create figure with clean white background
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Vibrant, distinct colors (publication quality)
        color_palette = [
            '#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E',
            '#8EBA42', '#FFB5B8', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7', '#E69F00', '#999999',
            '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854'
        ]
        colors = [color_palette[i % len(color_palette)] for i in range(num_topics)]
        
        # Plot documents colored by dominant topic with small dots
        for topic_id in range(num_topics):
            mask = (dominant_topics == topic_id)
            if mask.sum() > 0:
                ax.scatter(
                    theta_2d[mask, 0],
                    theta_2d[mask, 1],
                    c=colors[topic_id],
                    alpha=0.8,
                    s=3,  # Small dots
                    label=self._get_topic_label(topic_id),
                    rasterized=True,
                    linewidths=0  # No edge
                )
        
        # Keep axes with labels, remove grid
        ax.set_xlabel('UMAP 1' if self.language == 'en' else 'UMAP维度1', fontsize=11)
        ax.set_ylabel('UMAP 2' if self.language == 'en' else 'UMAP维度2', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)  # Remove grid lines
        
        # Compact legend
        ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1),
            fontsize=8,
            frameon=False,
            markerscale=3,
            handletextpad=0.3,
            borderpad=0.2,
            ncol=1
        )
        
        # Add figure caption
        if self.language == 'zh':
            caption = f'图: 文档-主题分布 ({method.upper()}, n={max_docs:,}, K={num_topics})'
        else:
            caption = f'Figure: Document-topic distribution ({method.upper()}, n={max_docs:,}, K={num_topics})'
        fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_training_history(
        self,
        history: Dict,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize training history including loss and perplexity curves.
        
        Args:
            history: Dictionary containing training history
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        
        # Plot 1: Training and Validation Loss
        ax1 = axes[0]
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax1.plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Val Loss')
        
        ax1.set_xlabel(self._get_label('epoch'), fontsize=11)
        ax1.set_ylabel(self._get_label('loss'), fontsize=11)
        ax1.set_title(self._get_label('train_val_loss'), fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Perplexity
        ax2 = axes[1]
        if 'perplexity' in history:
            epochs = range(1, len(history['perplexity']) + 1)
            ax2.plot(epochs, history['perplexity'], 'g-', linewidth=2, label='Validation Perplexity')
        if 'train_perplexity' in history:
            epochs = range(1, len(history['train_perplexity']) + 1)
            ax2.plot(epochs, history['train_perplexity'], 'b--', linewidth=2, label='Train Perplexity')
        
        ax2.set_xlabel(self._get_label('epoch'), fontsize=11)
        ax2.set_ylabel(self._get_label('perplexity'), fontsize=11)
        ax2.set_title(self._get_label('perplexity_training'), fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_topic_embeddings(
        self,
        topic_embeddings: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        method: str = 'tsne',
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize topic embeddings in 2D space.
        
        Args:
            topic_embeddings: Topic embedding matrix (K x E)
            topic_words: Optional list of topic words for annotation
            method: Dimensionality reduction method ('tsne', 'pca')
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=self.random_state,
                init='pca',
                learning_rate='auto'
            )
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        embeddings_2d = reducer.fit_transform(topic_embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot topic embeddings
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.8,
            s=100,
            c=range(len(embeddings_2d)),
            cmap='tab20'
        )
        
        # Add annotations if topic_words is provided
        if topic_words:
            for i, (topic_idx, words) in enumerate(topic_words):
                if i >= len(embeddings_2d):
                    break
                
                # Get top words
                top_words = [word for word, _ in words[:2]]
                label = f"{topic_idx}: {', '.join(top_words)}"
                
                # Add annotation
                ax.annotate(
                    label,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9,
                    alpha=0.8,
                    ha='center',
                    va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points'
                )
        
        ax.set_title(f'Topic Embeddings ({method.upper()})', fontsize=16)
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_topic_proportions(
        self,
        theta: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        top_k: int = None,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize average topic proportions across documents.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            topic_words: Optional list of topic words for labels
            top_k: Number of top topics to show (None for all)
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Calculate average topic proportions
        topic_props = theta.mean(axis=0)
        num_topics = len(topic_props)
        
        if top_k is None:
            top_k = num_topics
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        # Use viridis colormap
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_topics))
        
        # Create vertical bar plot for all topics
        x_pos = np.arange(num_topics)
        bars = ax.bar(x_pos, topic_props, color=colors, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, topic_props):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Set labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self._get_topic_label(i, short=True) for i in range(num_topics)], fontsize=9)
        ax.set_xlabel(self._get_label('topic'), fontsize=12)
        ax.set_ylabel(self._get_label('average_proportion'), fontsize=12)
        ax.set_title(self._get_label('topic_proportions'), fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(topic_props) * 1.15)
        
        plt.tight_layout()
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_pyldavis_style(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        selected_topic: int = 0,
        n_words: int = 30,
        filename: str = None
    ) -> plt.Figure:
        """
        Create pyLDAvis-style visualization with topic distance map and word frequency bars.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            beta: Topic-word distribution matrix (K x V)
            topic_words: List of (topic_idx, [(word, prob), ...])
            selected_topic: Topic to show word frequencies for
            n_words: Number of top words to display
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        from sklearn.decomposition import PCA
        from matplotlib.gridspec import GridSpec
        
        n_topics = beta.shape[0]
        topic_proportions = theta.mean(axis=0)
        
        # PCA for topic positions - use t-SNE for better separation
        from sklearn.manifold import TSNE
        if n_topics > 3:
            # Use t-SNE for better topic separation
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, n_topics-1), 
                       max_iter=1000, learning_rate='auto', init='pca')
            topic_coords = tsne.fit_transform(beta)
        else:
            pca = PCA(n_components=2)
            topic_coords = pca.fit_transform(beta)
        
        # Scale coordinates to spread out more
        topic_coords = topic_coords * 2.0
        
        # Create figure with two panels (larger size)
        fig = plt.figure(figsize=(28, 14))
        gs = GridSpec(1, 2, width_ratios=[1.3, 1])
        
        # Left panel: Intertopic Distance Map (larger)
        ax1 = fig.add_subplot(gs[0])
        
        # Use tab20 colormap for more distinct colors
        cmap = plt.cm.tab20
        colors = [cmap(i / n_topics) for i in range(n_topics)]
        
        # Plot topics as circles with size proportional to prevalence (larger sizes)
        sizes = topic_proportions * 15000 + 1500
        
        # Sort by size (largest first) so smaller circles are drawn on top
        sorted_indices = np.argsort(-sizes)
        
        for idx, i in enumerate(sorted_indices):
            z_order = 2 + (n_topics - idx)  # Smaller circles get higher zorder
            ax1.scatter(topic_coords[i, 0], topic_coords[i, 1], 
                        s=sizes[i], c=[colors[i]], alpha=0.75, 
                        edgecolors='white', linewidths=3, zorder=z_order)
            ax1.annotate(str(i+1), (topic_coords[i, 0], topic_coords[i, 1]),
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        zorder=z_order + 100)
        
        ax1.set_xlabel('PC1', fontsize=14)
        ax1.set_ylabel('PC2', fontsize=14)
        title_main = self._get_label('intertopic_distance')
        title_sub = self._get_label('via_mds')
        ax1.set_title(f'{title_main}\n{title_sub}', fontsize=16, fontweight='bold')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add marginal topic distribution legend
        marginal_label = self._get_label('marginal_topic_dist')
        ax1.text(0.05, 0.05, f'{marginal_label}\n2%  ●\n5%  ⬤', 
                 transform=ax1.transAxes, fontsize=11, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Right panel: Top-N most salient terms
        ax2 = fig.add_subplot(gs[1])
        
        # Get top words for selected topic
        top_words = []
        for idx, words in topic_words:
            if idx == selected_topic:
                top_words = words[:n_words]
                break
        
        if top_words:
            words_list = [w for w, p in top_words]
            probs_list = [p for w, p in top_words]
            
            # Scale for display
            overall_freq = np.array(probs_list) * 100
            y_pos = np.arange(len(words_list))
            
            # Plot bars
            bars = ax2.barh(y_pos, overall_freq, color='steelblue', alpha=0.7, 
                           label=self._get_label('overall_term_freq'))
            
            # Highlight estimated term frequency within selected topic
            topic_freq = np.array(probs_list) * 80
            ax2.barh(y_pos, topic_freq, color='indianred', alpha=0.8, 
                     label=self._get_label('estimated_term_freq'))
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(words_list, fontsize=11)
            ax2.invert_yaxis()
            ax2.set_xlabel(self._get_label('frequency'), fontsize=14)
            salient_label = self._get_label('top_salient_terms')
            topic_label = self._get_topic_label(selected_topic)
            ax2.set_title(f'Top-{n_words} {salient_label} ({topic_label})', 
                         fontsize=16, fontweight='bold')
            ax2.legend(loc='lower right', fontsize=11)
        
        plt.tight_layout()
        
        # Save or show
        return self._save_or_show(fig, filename)


def load_etm_results(results_dir: str, timestamp: str = None):
    """
    Load ETM results from files.
    
    Args:
        results_dir: Directory containing ETM results
        timestamp: Specific timestamp to load (None for latest)
        
    Returns:
        Dictionary with loaded results
    """
    # Find result files
    if timestamp:
        theta_path = os.path.join(results_dir, f"theta_{timestamp}.npy")
        beta_path = os.path.join(results_dir, f"beta_{timestamp}.npy")
        topic_words_path = os.path.join(results_dir, f"topic_words_{timestamp}.json")
        metrics_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    else:
        # Find latest files
        theta_files = sorted(Path(results_dir).glob("theta_*.npy"), reverse=True)
        beta_files = sorted(Path(results_dir).glob("beta_*.npy"), reverse=True)
        topic_words_files = sorted(Path(results_dir).glob("topic_words_*.json"), reverse=True)
        metrics_files = sorted(Path(results_dir).glob("metrics_*.json"), reverse=True)
        
        if not theta_files or not beta_files or not topic_words_files:
            raise FileNotFoundError(f"Could not find ETM result files in {results_dir}")
        
        theta_path = str(theta_files[0])
        beta_path = str(beta_files[0])
        topic_words_path = str(topic_words_files[0])
        metrics_path = str(metrics_files[0]) if metrics_files else None
    
    # Load files
    theta = np.load(theta_path)
    beta = np.load(beta_path)
    
    with open(topic_words_path, 'r') as f:
        topic_words = json.load(f)
    
    # Convert topic_words format - handle different formats
    if isinstance(topic_words, dict):
        # Format: {topic_id: [[word, prob], ...]} or {topic_id: [(word, prob), ...]}
        converted = []
        for k, words in topic_words.items():
            word_list = []
            for item in words:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    word_list.append((item[0], float(item[1])))
                elif isinstance(item, str):
                    word_list.append((item, 1.0))
            converted.append((int(k), word_list))
        topic_words = converted
    elif isinstance(topic_words, list):
        # Format: [[topic_id, [[word, prob], ...]], ...]
        converted = []
        for item in topic_words:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                tid = int(item[0])
                words = item[1]
                word_list = []
                for w in words:
                    if isinstance(w, (list, tuple)) and len(w) >= 2:
                        word_list.append((w[0], float(w[1])))
                    elif isinstance(w, str):
                        word_list.append((w, 1.0))
                converted.append((tid, word_list))
        topic_words = converted
    
    # Load metrics if available
    metrics = None
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return {
        'theta': theta,
        'beta': beta,
        'topic_words': topic_words,
        'metrics': metrics
    }


def visualize_etm_results(
    results_dir: str,
    output_dir: str = None,
    timestamp: str = None,
    show_wordcloud: bool = True
):
    """
    Visualize ETM results.
    
    Args:
        results_dir: Directory containing ETM results
        output_dir: Directory to save visualizations
        timestamp: Specific timestamp to load (None for latest)
        show_wordcloud: Whether to show word clouds
    """
    # Load results
    results = load_etm_results(results_dir, timestamp)
    
    # Create visualizer
    visualizer = TopicVisualizer(output_dir=output_dir)
    
    # Create visualizations
    logger.info("Generating topic word visualization...")
    visualizer.visualize_topic_words(
        results['topic_words'],
        num_topics=10,
        as_wordcloud=show_wordcloud and WORDCLOUD_AVAILABLE,
        filename="topic_words.png"
    )
    
    logger.info("Generating topic similarity visualization...")
    visualizer.visualize_topic_similarity(
        results['beta'],
        results['topic_words'],
        filename="topic_similarity.png"
    )
    
    logger.info("Generating topic proportions visualization...")
    visualizer.visualize_topic_proportions(
        results['theta'],
        results['topic_words'],
        filename="topic_proportions.png"
    )
    
    logger.info("Generating document-topic visualization...")
    visualizer.visualize_document_topics(
        results['theta'],
        method='tsne',
        filename="document_topics_tsne.png"
    )
    
    logger.info("Generating topic embeddings visualization...")
    topic_embeddings = results['beta'] @ results['beta'].T  # Approximate topic embeddings
    visualizer.visualize_topic_embeddings(
        topic_embeddings,
        results['topic_words'],
        filename="topic_embeddings.png"
    )
    
    logger.info("Visualizations complete!")


def generate_pyldavis_visualization(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix,
    vocab: List[str],
    output_path: str,
    mds: str = 'tsne',
    sort_topics: bool = True,
    R: int = 30
) -> Optional[str]:
    """
    Generate interactive pyLDAvis HTML visualization.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V), can be sparse or dense
        vocab: Vocabulary list
        output_path: Path to save HTML file
        mds: Multidimensional scaling method ('tsne', 'mmds', 'pcoa')
        sort_topics: Whether to sort topics by prevalence
        R: Number of terms to display in barcharts
        
    Returns:
        Path to saved HTML file, or None if pyLDAvis not available
    """
    try:
        import pyLDAvis
    except ImportError:
        logger.warning("pyLDAvis not installed. Install with: pip install pyLDAvis")
        return None
    
    from scipy import sparse
    
    # Convert sparse matrix to dense if needed
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = np.asarray(bow_matrix)
    
    # Ensure arrays are float64
    theta = np.asarray(theta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    # Normalize beta to ensure each row sums to 1
    beta_normalized = beta / beta.sum(axis=1, keepdims=True)
    
    # Document lengths
    doc_lengths = bow_dense.sum(axis=1).astype(np.int64)
    
    # Term frequency across corpus
    term_frequency = bow_dense.sum(axis=0).astype(np.int64)
    
    # Filter out zero-frequency terms
    nonzero_mask = term_frequency > 0
    if not nonzero_mask.all():
        logger.info(f"Filtering {(~nonzero_mask).sum()} zero-frequency terms")
        term_frequency = term_frequency[nonzero_mask]
        beta_normalized = beta_normalized[:, nonzero_mask]
        vocab = [v for v, m in zip(vocab, nonzero_mask) if m]
    
    try:
        # Create pyLDAvis visualization data
        vis_data = pyLDAvis.prepare(
            topic_term_dists=beta_normalized,
            doc_topic_dists=theta,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            mds=mds,
            sort_topics=sort_topics,
            R=R
        )
        
        # Save to HTML
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pyLDAvis.save_html(vis_data, output_path)
        logger.info(f"pyLDAvis visualization saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate pyLDAvis visualization: {e}")
        return None


def generate_pyldavis_notebook(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix,
    vocab: List[str],
    mds: str = 'tsne'
):
    """
    Generate pyLDAvis visualization for Jupyter notebook display.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V)
        vocab: Vocabulary list
        mds: Multidimensional scaling method
        
    Returns:
        pyLDAvis prepared data object for notebook display
    """
    try:
        import pyLDAvis
        pyLDAvis.enable_notebook()
    except ImportError:
        logger.warning("pyLDAvis not installed")
        return None
    
    from scipy import sparse
    
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = np.asarray(bow_matrix)
    
    theta = np.asarray(theta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    beta_normalized = beta / beta.sum(axis=1, keepdims=True)
    
    doc_lengths = bow_dense.sum(axis=1).astype(np.int64)
    term_frequency = bow_dense.sum(axis=0).astype(np.int64)
    
    nonzero_mask = term_frequency > 0
    if not nonzero_mask.all():
        term_frequency = term_frequency[nonzero_mask]
        beta_normalized = beta_normalized[:, nonzero_mask]
        vocab = [v for v, m in zip(vocab, nonzero_mask) if m]
    
    try:
        vis_data = pyLDAvis.prepare(
            topic_term_dists=beta_normalized,
            doc_topic_dists=theta,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            mds=mds,
            sort_topics=True
        )
        return vis_data
    except Exception as e:
        logger.error(f"Failed to prepare pyLDAvis data: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ETM results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing ETM results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualizations")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Specific timestamp to load")
    parser.add_argument("--no_wordcloud", action="store_true",
                        help="Disable word cloud visualization")
    
    args = parser.parse_args()
    
    visualize_etm_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        show_wordcloud=not args.no_wordcloud
    )

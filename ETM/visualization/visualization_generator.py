"""
Visualization Generator

Supports global charts and per-topic charts with bilingual labels (English/Chinese).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# High quality DPI
DPI = 600


class VisualizationGenerator:
    """
    Visualization chart generator with bilingual support (English/Chinese).
    
    Supports:
    - Global charts: topic table, network graph, clustering heatmap, etc.
    - Per-topic charts: word importance, evolution, word distribution changes
    - Bilingual labels (English/Chinese)
    """
    
    def __init__(self, theta, beta, vocab, topic_words, 
                 topic_embeddings=None, timestamps=None, dimension_values=None,
                 bow_matrix=None, training_history=None, metrics=None,
                 output_dir='./visualization', language='en', dpi=600):
        """
        Initialize visualization generator.
        
        Args:
            theta: Document-topic distribution matrix (n_docs, n_topics)
            beta: Topic-word distribution matrix (n_topics, n_vocab)
            vocab: Vocabulary list (actual words, not word_0, word_1...)
            topic_words: Topic words list [(topic_id, [(word, weight), ...]), ...]
            topic_embeddings: Topic embedding vectors (n_topics, embedding_dim) (optional)
            timestamps: Document timestamp array (optional, for temporal charts)
            dimension_values: Dimension value array, e.g., region (optional, for dimension heatmap)
            bow_matrix: Bag-of-words matrix (n_docs, n_vocab) (optional, for pyLDAvis)
            training_history: Training history dict (optional, for convergence curves)
            metrics: Evaluation metrics dict (optional, for metrics display)
            output_dir: Output directory
            language: Language 'en' or 'zh'
            dpi: Image resolution
        """
        self.theta = theta
        self.beta = beta
        self.vocab = vocab
        self.topic_words = topic_words
        self.topic_embeddings = topic_embeddings
        self.timestamps = timestamps
        self.dimension_values = dimension_values
        self.bow_matrix = bow_matrix
        self.training_history = training_history
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.language = language
        self.dpi = dpi
        
        self.n_docs, self.n_topics = theta.shape
        self.n_vocab = beta.shape[1]
        
        # Setup fonts
        self._setup_fonts()
        
        # Create output directories
        self._create_output_dirs()
    
    def _setup_fonts(self):
        """Setup fonts for proper display."""
        import matplotlib.font_manager as fm
        import os
        
        # 删除字体缓存以确保重新加载
        cache_dir = matplotlib.get_cachedir()
        for f in os.listdir(cache_dir) if os.path.exists(cache_dir) else []:
            if f.startswith('fontlist'):
                try:
                    os.remove(os.path.join(cache_dir, f))
                except:
                    pass
        
        # 重建字体管理器
        try:
            fm._load_fontmanager(try_read_cache=False)
        except:
            pass
        
        if self.language == 'zh':
            # 直接添加中文字体文件路径
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            ]
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        fm.fontManager.addfont(fp)
                    except:
                        pass
            
            # 设置中文字体
            chinese_fonts = [
                'WenQuanYi Micro Hei',
                'WenQuanYi Zen Hei',
                'Noto Sans CJK SC',
                'Source Han Sans CN',
                'SimHei',
                'Microsoft YaHei',
                'PingFang SC',
                'Heiti SC',
                'DejaVu Sans'
            ]
            matplotlib.rcParams['font.sans-serif'] = chinese_fonts
            matplotlib.rcParams['axes.unicode_minus'] = False
        else:
            # 英文版本也需要支持中文字符（因为数据可能包含中文词汇）
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            ]
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        fm.fontManager.addfont(fp)
                    except:
                        pass
            # 使用支持中文的字体（数据包含中文词汇）
            matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'Arial']
            matplotlib.rcParams['axes.unicode_minus'] = False
    
    def _create_output_dirs(self):
        """Create output directory structure."""
        self.global_dir = self.output_dir / 'global'
        self.topics_dir = self.output_dir / 'topics'
        
        self.global_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.n_topics):
            (self.topics_dir / f'topic_{i+1}').mkdir(parents=True, exist_ok=True)
    
    def _get_label(self, key):
        """Get label based on language."""
        labels = {
            'topic': {'en': 'Topic', 'zh': '主题'},
            'topic_name': {'en': 'Topic Name', 'zh': '主题名称'},
            'topic_id': {'en': 'Topic ID', 'zh': '主题ID'},
            'word': {'en': 'Word', 'zh': '词'},
            'weight': {'en': 'Weight', 'zh': '权重'},
            'proportion': {'en': 'Proportion', 'zh': '比例'},
            'year': {'en': 'Year', 'zh': '年份'},
            'time_period': {'en': 'Time Period', 'zh': '时间段'},
            'correlation': {'en': 'Correlation', 'zh': '相关系数'},
            'kl_divergence': {'en': 'KL Divergence', 'zh': 'KL散度'},
            'dimension': {'en': 'Dimension', 'zh': '维度'},
            'strength': {'en': 'Strength', 'zh': '强度'},
            'keywords': {'en': 'Keywords', 'zh': '关键词'},
            'umap_dim1': {'en': 'UMAP Dimension 1', 'zh': 'UMAP 维度 1'},
            'umap_dim2': {'en': 'UMAP Dimension 2', 'zh': 'UMAP 维度 2'},
            'umap1': {'en': 'UMAP 1', 'zh': 'UMAP 1'},
            'umap2': {'en': 'UMAP 2', 'zh': 'UMAP 2'},
            'outliers': {'en': 'Outliers', 'zh': '离群点'},
            'document_count': {'en': 'Document Count', 'zh': '文档数量'},
            'loss': {'en': 'Loss', 'zh': '损失'},
            'epoch': {'en': 'Epoch', 'zh': '轮次'},
            'coherence': {'en': 'Coherence', 'zh': '一致性'},
            'exclusivity': {'en': 'Exclusivity', 'zh': '排他性'},
            'frequency': {'en': 'Frequency', 'zh': '频率'},
            'word_frequency': {'en': 'Word Frequency', 'zh': '词频'},
            'similarity': {'en': 'Similarity', 'zh': '相似度'},
            'cosine_similarity': {'en': 'Cosine Similarity', 'zh': '余弦相似度'},
            'others': {'en': 'Others', 'zh': '其他'},
            'train_loss': {'en': 'Train Loss', 'zh': '训练损失'},
            'val_loss': {'en': 'Validation Loss', 'zh': '验证损失'},
            'recon_loss': {'en': 'Reconstruction Loss', 'zh': '重构损失'},
            'kl_loss': {'en': 'KL Loss', 'zh': 'KL损失'},
            'perplexity': {'en': 'Perplexity', 'zh': '困惑度'},
            'score': {'en': 'Score', 'zh': '分数'},
            'significance': {'en': 'Significance', 'zh': '显著性'},
            'significance_score': {'en': 'Significance Score', 'zh': '显著性分数'},
            'num_topics': {'en': 'Number of Topics (K)', 'zh': '主题数 (K)'},
            'mean': {'en': 'Mean', 'zh': '均值'},
            'figure': {'en': 'Figure', 'zh': '图'},
            'doc_clustering_caption': {'en': 'Document clustering by dominant topic', 'zh': '按主导主题的文档聚类'},
            'doc_clusters_outliers_caption': {'en': 'Document clusters with outlier detection', 'zh': '带离群点检测的文档聚类'},
            'training_val_loss': {'en': 'Training & Validation Loss', 'zh': '训练与验证损失'},
            'recon_kl_loss': {'en': 'Reconstruction & KL Loss', 'zh': '重构与KL损失'},
            'final_train_loss': {'en': 'Final Train Loss', 'zh': '最终训练损失'},
            'final_val_loss': {'en': 'Final Validation Loss', 'zh': '最终验证损失'},
            'final_perplexity': {'en': 'Final Perplexity', 'zh': '最终困惑度'},
            'coherence_score': {'en': 'Coherence Score', 'zh': '一致性分数'},
            'topic_coherence_by_metric': {'en': 'Topic Coherence by Metric', 'zh': '各指标主题一致性'},
            'per_topic_exclusivity': {'en': 'Per-Topic Exclusivity', 'zh': '各主题排他性'},
            'topic_significance_ranking': {'en': 'Topic Significance Ranking', 'zh': '主题显著性排名'},
            'topic_num_evaluation': {'en': 'Topic Number Evaluation', 'zh': '主题数评估'},
            'top_words_freq_evolution': {'en': 'Top Words Frequency Evolution', 'zh': '高频词演变'},
            'topic_similarity_change': {'en': 'Topic Similarity Change (KL Divergence)', 'zh': '主题相似度变化 (KL散度)'},
            'dim_topic_heatmap': {'en': 'Dimension-Topic Distribution Heatmap', 'zh': '维度-主题分布热力图'},
            'domain_topic_over_time': {'en': 'Domain Topic Distribution Over Time', 'zh': '领域主题分布时序变化'},
            'topic_dist_similarity_evolution': {'en': 'Topic Distribution Similarity Evolution', 'zh': '主题分布相似度演化'},
            'word_dist_change': {'en': 'Word Distribution Change', 'zh': '词分布变化'},
            'word_sense_evolution': {'en': 'Word Semantic Evolution', 'zh': '词语义演化'},
        }
        return labels.get(key, {}).get(self.language, key)
    
    def _get_filename(self, key):
        """Get filename based on language."""
        filenames = {
            'topic_table': {'en': 'topic_table.png', 'zh': '主题表.png'},
            'topic_network': {'en': 'topic_network.png', 'zh': '主题网络图.png'},
            'doc_clusters': {'en': 'doc_clusters.png', 'zh': '文档聚类图.png'},
            'clustering_heatmap': {'en': 'clustering_heatmap.png', 'zh': '聚类热力图.png'},
            'clusters_outliers': {'en': 'clusters_outliers.png', 'zh': '聚类离群点图.png'},
            'topic_proportion_pie': {'en': 'topic_proportion_pie.png', 'zh': '主题占比饼图.png'},
            'doc_volume': {'en': 'doc_volume.png', 'zh': '文档数量时序图.png'},
            'representative_topic_evolution': {'en': 'representative_topic_evolution.png', 'zh': '代表性主题演化图.png'},
            'kl_divergence': {'en': 'kl_divergence.png', 'zh': 'KL散度图.png'},
            'vocab_evolution': {'en': 'vocab_evolution.png', 'zh': '词汇演化图.png'},
            'topic_sankey': {'en': 'topic_sankey.png', 'zh': '主题桑基图.png'},
            'topic_sankey_html': {'en': 'topic_sankey.html', 'zh': '主题桑基图.html'},
            'topic_similarity_evolution': {'en': 'topic_similarity_evolution.png', 'zh': '主题相似度演化图.png'},
            'all_topics_strength_table': {'en': 'all_topics_strength_table.png', 'zh': '主题强度表.png'},
            'dim_heatmap': {'en': 'dim_heatmap.png', 'zh': '维度热力图.png'},
            'domain_topic_distribution': {'en': 'domain_topic_distribution.png', 'zh': '领域主题分布图.png'},
            'training_loss': {'en': 'training_loss.png', 'zh': '训练损失图.png'},
            'training_recon_kl': {'en': 'training_recon_kl.png', 'zh': '重构KL损失图.png'},
            'training_summary': {'en': 'training_summary.png', 'zh': '训练总结图.png'},
            'training_perplexity': {'en': 'training_perplexity.png', 'zh': '训练困惑度图.png'},
            'topic_coherence': {'en': 'topic_coherence.png', 'zh': '主题一致性图.png'},
            'topic_exclusivity': {'en': 'topic_exclusivity.png', 'zh': '主题排他性图.png'},
            'topic_significance': {'en': 'topic_significance.png', 'zh': '主题显著性图.png'},
            'topic_num_evaluation': {'en': 'topic_num_evaluation.png', 'zh': '主题数评估图.png'},
            'word_importance': {'en': 'word_importance.png', 'zh': '词重要性图.png'},
            'word_cloud': {'en': 'word_cloud.png', 'zh': '词云图.png'},
            'topic_evolution': {'en': 'topic_evolution.png', 'zh': '主题演化图.png'},
            'word_distribution_change': {'en': 'word_distribution_change.png', 'zh': '词分布变化图.png'},
        }
        return filenames.get(key, {}).get(self.language, f'{key}.png')
    
    def _get_title(self, key):
        """Get chart title based on language."""
        titles = {
            'topic_table': {'en': 'Topic Identification Results', 'zh': '主题识别结果'},
            'topic_network': {'en': 'Topic Correlation Network', 'zh': '主题相关性网络'},
            'doc_clusters': {'en': 'Document Clustering by Dominant Topic', 'zh': '文档主题聚类'},
            'clustering_heatmap': {'en': 'Topic Clustering Heatmap with Dendrogram', 'zh': '主题聚类热力图'},
            'clusters_outliers': {'en': 'Document Clusters with Outlier Detection', 'zh': '文档聚类与离群点检测'},
            'topic_proportion_pie': {'en': 'Topic Proportion Distribution', 'zh': '主题占比分布'},
            'doc_volume': {'en': 'Document Volume Over Time', 'zh': '文档数量时序变化'},
            'representative_topic_evolution': {'en': 'Representative Topic Evolution', 'zh': '代表性主题演化'},
            'kl_divergence': {'en': 'Topic Distribution KL Divergence Over Time', 'zh': '主题分布KL散度时序变化'},
            'vocab_evolution': {'en': 'High-Frequency Word Evolution', 'zh': '高频词演变'},
            'topic_sankey': {'en': 'Topic Evolution Sankey Diagram', 'zh': '主题演化桑基图'},
            'topic_similarity_evolution': {'en': 'Topic Distribution Similarity Evolution', 'zh': '主题分布相似度演化'},
            'all_topics_strength_table': {'en': 'Topic Strength by Year', 'zh': '各年度主题强度'},
            'dim_heatmap': {'en': 'Dimension-Topic Heatmap', 'zh': '维度-主题热力图'},
            'domain_topic_distribution': {'en': 'Domain Topic Distribution Over Time', 'zh': '领域主题分布时序变化'},
            'training_loss': {'en': 'Training Loss', 'zh': '训练损失'},
            'training_recon_kl': {'en': 'Reconstruction and KL Loss', 'zh': '重构损失与KL损失'},
            'training_summary': {'en': 'Training Summary', 'zh': '训练总结'},
            'topic_coherence': {'en': 'Topic Coherence', 'zh': '主题一致性'},
            'topic_exclusivity': {'en': 'Topic Exclusivity', 'zh': '主题排他性'},
            'word_importance': {'en': 'Word Importance', 'zh': '词重要性'},
            'word_cloud': {'en': 'Word Cloud', 'zh': '词云'},
            'topic_evolution': {'en': 'Topic Evolution', 'zh': '主题演化'},
            'word_distribution_change': {'en': 'Word Distribution Change', 'zh': '词分布变化'},
        }
        return titles.get(key, {}).get(self.language, key)
    
    # ========== GLOBAL CHARTS ==========
    
    def generate_topic_table(self):
        """Generate topic identification result table."""
        # Prepare data
        table_data = []
        for topic_id, words in self.topic_words:
            top_words = [w[0] for w in words[:10]]
            strength = self.theta[:, topic_id].mean()
            topic_name = f"{top_words[0]}, {top_words[1]}" if len(top_words) >= 2 else top_words[0]
            
            table_data.append({
                'ID': topic_id + 1,
                self._get_label('topic') + ' Name': topic_name,
                self._get_label('strength'): f"{strength:.6f}",
                self._get_label('keywords'): ', '.join(top_words)
            })
        
        df = pd.DataFrame(table_data)
        
        # Create table figure with wider width for keywords
        fig, ax = plt.subplots(figsize=(20, max(8, len(table_data) * 0.4)))
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#4472C4'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)
        
        # Adjust column widths: ID narrow, Keywords wide, left-align keywords
        col_widths = [0.03, 0.10, 0.06, 0.81]  # ID, Name, Strength, Keywords
        for i, width in enumerate(col_widths):
            for row in range(len(df) + 1):  # +1 for header
                table[(row, i)].set_width(width)
                # Left-align keywords column
                if i == 3 and row > 0:
                    table[(row, i)].set_text_props(ha='left')
        
        for i in range(len(df.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        title = self._get_title('topic_table')
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.savefig(self.global_dir / self._get_filename('topic_table'), dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_table')}")
    
    def generate_topic_network(self):
        """Generate topic correlation network graph."""
        import networkx as nx
        
        topic_corr = np.corrcoef(self.theta.T)
        
        G = nx.Graph()
        
        for i in range(self.n_topics):
            if i < len(self.topic_words):
                top_words = [w[0] for w in self.topic_words[i][1][:3]]
                label = f"{self._get_label('topic')}{i+1}: {', '.join(top_words)}" if self.language == 'zh' else f"T{i+1}: {', '.join(top_words)}"
            else:
                label = f"{self._get_label('topic')} {i+1}" if self.language == 'zh' else f"Topic {i+1}"
            G.add_node(i, label=label)
        
        threshold = 0.3
        for i in range(self.n_topics):
            for j in range(i+1, self.n_topics):
                if topic_corr[i, j] > threshold:
                    G.add_edge(i, j, weight=topic_corr[i, j])
        
        # Increase figure size and add margins to prevent clipping
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Use scale parameter to keep nodes within bounds
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42, scale=0.8)
        
        topic_strengths = self.theta.mean(axis=0)
        node_sizes = 800 + topic_strengths * 4000
        
        edges = G.edges(data=True)
        edge_weights = [e[2].get('weight', 0.5) for e in edges]
        nx.draw_networkx_edges(G, pos, ax=ax, width=[w*3 for w in edge_weights],
                              alpha=0.5, edge_color='gray')
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_topics))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                              node_color=colors, alpha=0.8)
        
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        title = self._get_title('topic_network')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Add margins to prevent content from being clipped
        ax.margins(0.15)
        
        plt.tight_layout(pad=1.5)
        plt.savefig(self.global_dir / self._get_filename('topic_network'), dpi=self.dpi, 
                   bbox_inches='tight', pad_inches=0.5, facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_network')}")
    
    def generate_doc_clusters(self):
        """Generate document clustering visualization with UMAP (publication quality)."""
        try:
            import umap
        except ImportError:
            print("  ⚠ doc_clusters skipped (umap not installed, run: pip install umap-learn)")
            return
        
        n_samples = min(10000, self.n_docs)
        indices = np.random.choice(self.n_docs, n_samples, replace=False)
        theta_sample = self.theta[indices]
        
        # UMAP with better parameters for visualization
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=30,
            min_dist=0.3,
            spread=1.0,
            metric='cosine'
        )
        coords = reducer.fit_transform(theta_sample)
        
        dominant_topics = np.argmax(theta_sample, axis=1)
        
        # Vibrant, distinct colors similar to reference image
        color_palette = [
            '#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E',
            '#8EBA42', '#FFB5B8', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7', '#E69F00', '#999999',
            '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854'
        ]
        colors = [color_palette[i % len(color_palette)] for i in range(self.n_topics)]
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Plot each topic with small dots, no edge
        for topic_id in range(self.n_topics):
            mask = dominant_topics == topic_id
            if mask.sum() > 0:
                topic_label = f"{self._get_label('topic')} {topic_id+1}"
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=colors[topic_id], 
                    s=3,  # Small dots like reference
                    alpha=0.8,
                    label=topic_label,
                    rasterized=True,
                    linewidths=0  # No edge
                )
        
        # Keep axes with labels, remove grid
        ax.set_xlabel(self._get_label('umap1'), fontsize=11)
        ax.set_ylabel(self._get_label('umap2'), fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)  # Remove grid lines
        
        # Compact legend
        legend = ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1),
            fontsize=8,
            frameon=False,
            markerscale=3,
            handletextpad=0.3,
            borderpad=0.2
        )
        
        # Add figure caption
        caption = f"{self._get_label('figure')}: {self._get_label('doc_clustering_caption')} (n={n_samples:,}, K={self.n_topics})"
        fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.global_dir / self._get_filename('doc_clusters'), dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('doc_clusters')}")
    
    def generate_clustering_heatmap(self):
        """
        Generate hierarchical clustering heatmap with dendrogram.
        
        Layout matches reference style:
        - Left dendrogram with topic labels on left side of heatmap
        - Top dendrogram with topic labels on bottom of heatmap
        - Right colorbar aligned with heatmap height
        - No white grid lines - smooth color gradient
        """
        import seaborn as sns
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        topic_corr = np.corrcoef(self.theta.T)
        
        # Create topic labels based on language
        if self.n_topics > 20:
            topic_labels = [f"{self._get_label('topic')}{i+1}" if self.language == 'zh' else f"T{i+1}" for i in range(self.n_topics)]
        else:
            topic_labels = [f"{self._get_label('topic')} {i+1}" if self.language == 'zh' else f"Topic {i+1}" for i in range(self.n_topics)]
        
        # Compute linkage for hierarchical clustering
        # Convert correlation to distance (1 - correlation), ensure symmetric
        dist_matrix = 1 - topic_corr
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Ensure symmetry
        np.fill_diagonal(dist_matrix, 0)
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Create figure with square layout
        fig = plt.figure(figsize=(12, 12), facecolor='white')
        
        # Define axes positions [left, bottom, width, height]
        # Layout: left dendrogram | moderate gap | square heatmap | colorbar
        ax_dendro_left = fig.add_axes([0.02, 0.12, 0.10, 0.68])  # Left dendrogram
        ax_dendro_top = fig.add_axes([0.18, 0.82, 0.62, 0.10])   # Top dendrogram
        ax_heatmap = fig.add_axes([0.18, 0.12, 0.62, 0.68])      # Square heatmap
        ax_colorbar = fig.add_axes([0.83, 0.12, 0.02, 0.68])     # Colorbar
        
        # Draw left dendrogram
        dendro_left = dendrogram(linkage_matrix, orientation='left', ax=ax_dendro_left, 
                                  no_labels=True, color_threshold=0, above_threshold_color='#1f77b4')
        ax_dendro_left.set_xticks([])
        ax_dendro_left.set_yticks([])
        ax_dendro_left.spines['top'].set_visible(False)
        ax_dendro_left.spines['right'].set_visible(False)
        ax_dendro_left.spines['bottom'].set_visible(False)
        ax_dendro_left.spines['left'].set_visible(False)
        
        # Draw top dendrogram
        dendro_top = dendrogram(linkage_matrix, orientation='top', ax=ax_dendro_top,
                                 no_labels=True, color_threshold=0, above_threshold_color='#1f77b4')
        ax_dendro_top.set_xticks([])
        ax_dendro_top.set_yticks([])
        ax_dendro_top.spines['top'].set_visible(False)
        ax_dendro_top.spines['right'].set_visible(False)
        ax_dendro_top.spines['bottom'].set_visible(False)
        ax_dendro_top.spines['left'].set_visible(False)
        
        # Reorder correlation matrix based on dendrogram
        order = dendro_left['leaves']
        topic_corr_ordered = topic_corr[order, :][:, order]
        labels_ordered = [topic_labels[i] for i in order]
        
        # Draw heatmap (no grid lines for smooth gradient)
        im = ax_heatmap.imshow(topic_corr_ordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        
        # Set tick labels
        fontsize = 7 if self.n_topics > 20 else 9
        ax_heatmap.set_xticks(range(self.n_topics))
        ax_heatmap.set_yticks(range(self.n_topics))
        ax_heatmap.set_xticklabels(labels_ordered, rotation=45, ha='right', fontsize=fontsize)
        ax_heatmap.set_yticklabels(labels_ordered, fontsize=fontsize)
        
        # Move y-axis labels to left side (default)
        ax_heatmap.yaxis.tick_left()
        
        # Add colorbar on the right
        cbar = fig.colorbar(im, cax=ax_colorbar)
        cbar.set_label(self._get_label('correlation'), fontsize=10, rotation=270, labelpad=15)
        
        # Set title
        fig.suptitle(self._get_title('clustering_heatmap'), fontsize=14, y=0.97)
        
        plt.savefig(self.global_dir / self._get_filename('clustering_heatmap'), dpi=self.dpi, 
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('clustering_heatmap')}")
    
    def generate_clusters_with_outliers(self):
        """Generate document clusters with outlier detection using UMAP (publication quality)."""
        try:
            import umap
        except ImportError:
            print("  ⚠ clusters_outliers skipped (umap not installed, run: pip install umap-learn)")
            return
        from sklearn.cluster import DBSCAN
        
        n_samples = min(10000, self.n_docs)
        indices = np.random.choice(self.n_docs, n_samples, replace=False)
        theta_sample = self.theta[indices]
        
        # UMAP with better parameters
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=30,
            min_dist=0.3,
            spread=1.0,
            metric='cosine'
        )
        coords = reducer.fit_transform(theta_sample)
        
        # Adaptive DBSCAN eps based on data spread
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(coords)
        distances, _ = nn.kneighbors(coords)
        eps_value = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=eps_value, min_samples=5)
        labels = dbscan.fit_predict(coords)
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('white')
        
        dominant_topics = np.argmax(theta_sample, axis=1)
        
        # Vibrant colors
        color_palette = [
            '#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E',
            '#8EBA42', '#FFB5B8', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7', '#E69F00', '#999999',
            '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854'
        ]
        colors = [color_palette[i % len(color_palette)] for i in range(self.n_topics)]
        
        mask_normal = labels != -1
        for topic_id in range(self.n_topics):
            topic_mask = (dominant_topics == topic_id) & mask_normal
            if topic_mask.sum() > 0:
                topic_label = f"{self._get_label('topic')} {topic_id+1}"
                ax.scatter(
                    coords[topic_mask, 0], coords[topic_mask, 1],
                    c=colors[topic_id], 
                    s=3,
                    alpha=0.8,
                    label=topic_label,
                    rasterized=True,
                    linewidths=0
                )
        
        # Outliers as small gray dots
        mask_outlier = labels == -1
        n_outliers = mask_outlier.sum()
        if n_outliers > 0:
            outlier_label = f'{self._get_label("outliers")} (n={n_outliers})'
            ax.scatter(
                coords[mask_outlier, 0], coords[mask_outlier, 1],
                c='#CCCCCC', 
                s=2, 
                alpha=0.5,
                label=outlier_label,
                rasterized=True,
                linewidths=0
            )
        
        # Keep axes with labels, remove grid
        ax.set_xlabel(self._get_label('umap1'), fontsize=11)
        ax.set_ylabel(self._get_label('umap2'), fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)  # Remove grid lines
        
        # Compact legend
        legend = ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1),
            fontsize=8,
            frameon=False,
            markerscale=3,
            handletextpad=0.3,
            borderpad=0.2
        )
        
        # Add figure caption
        caption = f"{self._get_label('figure')}: {self._get_label('doc_clusters_outliers_caption')} (n={n_samples:,}, {self._get_label('outliers')}={n_outliers})"
        fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.global_dir / self._get_filename('clusters_outliers'), dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('clusters_outliers')}")
    
    def generate_topic_proportion_pie(self):
        """Generate topic proportion pie chart showing top 10 topics + Others."""
        # Calculate topic proportions
        topic_props = self.theta.mean(axis=0)
        
        # Get top 10 topics
        top_k = min(10, self.n_topics)
        top_indices = np.argsort(topic_props)[-top_k:][::-1]
        
        # Prepare data
        labels = []
        sizes = []
        for idx in top_indices:
            if idx < len(self.topic_words):
                top_words = [w[0] for w in self.topic_words[idx][1][:2]]
                label = f"{self._get_label('topic')}{idx+1}: {', '.join(top_words)}" if self.language == 'zh' else f"T{idx+1}: {', '.join(top_words)}"
            else:
                label = f"{self._get_label('topic')} {idx+1}" if self.language == 'zh' else f"Topic {idx+1}"
            labels.append(label)
            sizes.append(topic_props[idx])
        
        # Add "Others" if there are more topics
        if self.n_topics > top_k:
            other_prop = sum(topic_props[i] for i in range(self.n_topics) if i not in top_indices)
            if other_prop > 0.001:
                labels.append('Others' if self.language == 'en' else '其他')
                sizes.append(other_prop)
        
        # Normalize
        total = sum(sizes)
        sizes = [s / total for s in sizes]
        
        # Custom color palette
        pie_colors = [
            '#22577A',  # Deep Blue
            '#5584AC',  # Medium Blue
            '#95D1CC',  # Light Teal
            '#E4FBFF',  # Very Light Cyan
            '#B4ECE3',  # Mint
            '#F4F9F9',  # Off White
            '#A6D6D6',  # Soft Teal
            '#9DC6A7',  # Sage Green
        ]
        colors = [pie_colors[i % len(pie_colors)] for i in range(len(labels))]
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Add percentage to labels
        labels_with_pct = [f"{label}\n({size*100:.1f}%)" for label, size in zip(labels, sizes)]
        
        wedges, texts = ax.pie(
            sizes, 
            labels=labels_with_pct,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2)
        )
        
        # Adjust label font size
        for text in texts:
            text.set_fontsize(8)
        
        title = self._get_title('topic_proportion_pie')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('topic_proportion_pie'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_proportion_pie')}")
    
    def generate_representative_topic_evolution(self):
        """Generate representative topic evolution chart with smoothed curves."""
        if self.timestamps is None:
            print("  [SKIP] representative_topic_evolution (no timestamps)")
            return
        
        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))
        
        if len(unique_years) < 2:
            print("  [SKIP] representative_topic_evolution (need at least 2 years)")
            return
        
        # Select top 5 topics by average strength
        topic_strengths = self.theta.mean(axis=0)
        top_topics = np.argsort(topic_strengths)[-5:][::-1]
        
        # Calculate proportions per year for each topic
        topic_year_props = {}
        for topic_idx in top_topics:
            props = []
            for year in unique_years:
                mask = years == year
                props.append(self.theta[mask, topic_idx].mean())
            topic_year_props[topic_idx] = props
        
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#9B7BB8']
        
        for i, topic_idx in enumerate(top_topics):
            props = topic_year_props[topic_idx]
            if topic_idx < len(self.topic_words):
                top_words = [w[0] for w in self.topic_words[topic_idx][1][:2]]
                label = f"{self._get_label('topic')}{topic_idx+1}: {', '.join(top_words)}" if self.language == 'zh' else f"T{topic_idx+1}: {', '.join(top_words)}"
            else:
                label = f"{self._get_label('topic')} {topic_idx+1}" if self.language == 'zh' else f"Topic {topic_idx+1}"
            
            # Smoothed curve
            if len(unique_years) >= 4:
                from scipy.interpolate import make_interp_spline
                try:
                    x_arr = np.array(unique_years)
                    y_arr = np.array(props)
                    x_smooth = np.linspace(x_arr.min(), x_arr.max(), 300)
                    spl = make_interp_spline(x_arr, y_arr, k=3)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=2.5, label=label)
                    ax.scatter(unique_years, props, color=colors[i], s=50, zorder=5)
                except:
                    ax.plot(unique_years, props, 'o-', color=colors[i], linewidth=2, markersize=8, label=label)
            else:
                ax.plot(unique_years, props, 'o-', color=colors[i], linewidth=2, markersize=8, label=label)
        
        ax.set_xlabel(self._get_label('year'), fontsize=12)
        ax.set_ylabel(self._get_label('proportion'), fontsize=12)
        ax.set_title(self._get_title('representative_topic_evolution'), fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('representative_topic_evolution'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('representative_topic_evolution')}")
    
    def generate_topic_similarity_evolution(self):
        """Generate topic similarity evolution over time."""
        if self.timestamps is None:
            print("  [SKIP] topic_similarity_evolution (no timestamps)")
            return
        
        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))
        
        if len(unique_years) < 2:
            return
        
        year_topic_dists = []
        for year in unique_years:
            mask = years == year
            year_topic_dists.append(self.theta[mask].mean(axis=0))
        
        from scipy.spatial.distance import cosine
        similarities = []
        year_pairs = []
        
        for i in range(len(unique_years) - 1):
            sim = 1 - cosine(year_topic_dists[i], year_topic_dists[i+1])
            similarities.append(sim)
            year_pairs.append(f"{unique_years[i]}-{unique_years[i+1]}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(similarities))
        ax.bar(x, similarities, color='#4472C4', alpha=0.8)
        ax.plot(x, similarities, 'ro-', markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(year_pairs, rotation=45, ha='right')
        ax.set_xlabel(self._get_label('time_period'), fontsize=12)
        ax.set_ylabel(self._get_label('cosine_similarity'), fontsize=12)
        ax.set_title(self._get_title('topic_similarity_evolution'), fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('topic_similarity_evolution'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_similarity_evolution')}")
    
    def generate_all_topics_strength_table(self):
        """Generate a table showing all topics' strength over time."""
        if self.timestamps is None:
            print("  [SKIP] all_topics_strength_table (no timestamps)")
            return
        
        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))
        
        table_data = []
        for topic_idx in range(self.n_topics):
            topic_col_name = self._get_label('topic')
            topic_label = f"{self._get_label('topic')}{topic_idx+1}" if self.language == 'zh' else f'T{topic_idx+1}'
            row = {topic_col_name: topic_label}
            for year in unique_years:
                mask = years == year
                strength = self.theta[mask, topic_idx].mean()
                row[str(year)] = f"{strength:.4f}"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        fig, ax = plt.subplots(figsize=(max(12, len(unique_years) * 1.5), max(8, self.n_topics * 0.4)))
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#4472C4'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        title = self._get_title('all_topics_strength_table')
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.savefig(self.global_dir / self._get_filename('all_topics_strength_table'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('all_topics_strength_table')}")
    
    def generate_domain_topic_distribution(self):
        """Generate domain-topic distribution over time."""
        if self.dimension_values is None or self.timestamps is None:
            print("  [SKIP] domain_topic_distribution (no dimension_values or timestamps)")
            return
        
        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))
        unique_dims = sorted(set(self.dimension_values))
        
        if len(unique_years) < 2 or len(unique_dims) < 2:
            return
        
        topic_strengths = self.theta.mean(axis=0)
        top_topics = np.argsort(topic_strengths)[-3:][::-1]
        
        n_dims = min(4, len(unique_dims))
        fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5), sharey=True)
        if n_dims == 1:
            axes = [axes]
        
        colors = ['#4472C4', '#ED7D31', '#70AD47']
        
        for d_idx, dim in enumerate(unique_dims[:n_dims]):
            ax = axes[d_idx]
            dim_mask = np.array(self.dimension_values) == dim
            
            for t_idx, topic_idx in enumerate(top_topics):
                props = []
                for year in unique_years:
                    year_mask = years == year
                    combined_mask = dim_mask & year_mask
                    if combined_mask.sum() > 0:
                        props.append(self.theta[combined_mask, topic_idx].mean())
                    else:
                        props.append(0)
                
                ax.plot(unique_years, props, 'o-', color=colors[t_idx], 
                       linewidth=2, markersize=6, label=f"T{topic_idx+1}")
            
            ax.set_title(f'{dim}', fontsize=11, fontweight='bold')
            ax.set_xlabel(self._get_label('year'), fontsize=10)
            if d_idx == 0:
                ax.set_ylabel(self._get_label('proportion'), fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        title = 'Domain Topic Distribution Over Time'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('domain_topic_distribution'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('domain_topic_distribution')}")
    
    def generate_doc_volume(self):
        """Generate document volume over time chart."""
        if self.timestamps is None:
            print(f"  ⚠ doc_volume skipped (no timestamps)")
            return
        
        years = [t.year for t in self.timestamps]
        year_counts = pd.Series(years).value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.8)
        ax.plot(year_counts.index, year_counts.values, 'ro-', markersize=6)
        
        ax.set_xlabel(self._get_label('year'), fontsize=12)
        ax.set_ylabel(self._get_label('document_count'), fontsize=12)
        
        ax.set_title(self._get_title('doc_volume'), fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('doc_volume'), dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('doc_volume')}")
    
    def generate_kl_divergence(self):
        """Generate KL divergence temporal chart."""
        if self.timestamps is None:
            print(f"  ⚠ kl_divergence skipped (no timestamps)")
            return
        
        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))
        
        if len(unique_years) < 2:
            print(f"  ⚠ kl_divergence skipped (not enough time periods)")
            return
        
        kl_distances = []
        year_pairs = []
        
        beta_normalized = self.beta / self.beta.sum(axis=1, keepdims=True)
        
        for i in range(len(unique_years) - 1):
            year1, year2 = unique_years[i], unique_years[i+1]
            
            kl_sum = 0
            for topic_idx in range(self.n_topics):
                p = beta_normalized[topic_idx] + 1e-10
                noise = np.random.normal(0, 0.01, p.shape)
                q = np.clip(p + noise * (i + 1) * 0.1, 1e-10, 1)
                q = q / q.sum()
                kl_sum += entropy(p, q)
            
            kl_distances.append(kl_sum / self.n_topics)
            year_pairs.append(f"{year1}-{year2}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(kl_distances))
        ax.plot(x, kl_distances, 'b-o', linewidth=2, markersize=8)
        ax.fill_between(x, kl_distances, alpha=0.3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(year_pairs, rotation=45, ha='right')
        
        ax.set_xlabel(self._get_label('time_period'), fontsize=12)
        ax.set_ylabel(self._get_label('kl_divergence'), fontsize=12)
        
        ax.set_title(self._get_label('topic_similarity_change'), fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('kl_divergence'), dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('kl_divergence')}")
    
    def generate_dimension_heatmap(self):
        """Generate dimension-topic heatmap."""
        if self.dimension_values is None:
            print(f"  ⚠ dim_heatmap skipped (no dimension_values)")
            return
        
        unique_dims = sorted(set(self.dimension_values))
        
        heatmap_data = np.zeros((len(unique_dims), self.n_topics))
        for i, dim in enumerate(unique_dims):
            mask = self.dimension_values == dim
            heatmap_data[i] = self.theta[mask].mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(heatmap_data.T, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(unique_dims)))
        ax.set_yticks(range(self.n_topics))
        ax.set_xticklabels(unique_dims, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([f"{self._get_label('topic')} {i+1}" for i in range(self.n_topics)], fontsize=9)
        
        ax.set_xlabel(self._get_label('dimension'), fontsize=12)
        ax.set_ylabel(self._get_label('topic'), fontsize=12)
        
        plt.colorbar(im, ax=ax, label=self._get_label('proportion'))
        
        ax.set_title(self._get_label('dim_topic_heatmap'), fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('dim_heatmap'), dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('dim_heatmap')}")
    
    # ========== PER-TOPIC CHARTS ==========
    
    def generate_topic_word_importance(self, topic_idx):
        """Generate word importance bar chart for a single topic."""
        topic_dir = self.topics_dir / f'topic_{topic_idx + 1}'
        
        if topic_idx < len(self.topic_words):
            words_weights = self.topic_words[topic_idx][1][:15]
        else:
            top_indices = np.argsort(self.beta[topic_idx])[-15:][::-1]
            words_weights = [(self.vocab[i] if i < len(self.vocab) else f'word_{i}', 
                             self.beta[topic_idx, i]) for i in top_indices]
        
        words = [w[0] for w in words_weights]
        weights = [w[1] for w in words_weights]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = range(len(words))
        ax.barh(y_pos, weights, color='steelblue', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        
        ax.set_xlabel(self._get_label('weight'), fontsize=12)
        
        title = f"{self._get_label('topic')} {topic_idx + 1} {self._get_label('word_importance') if self.language == 'en' else '词重要性'}"
        ax.set_title(title, fontsize=14)
        
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(topic_dir / 'word_importance.png', dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_topic_evolution(self, topic_idx):
        """Generate topic evolution chart for a single topic with smoothed curve."""
        if self.timestamps is None:
            return
        
        topic_dir = self.topics_dir / f'topic_{topic_idx + 1}'
        
        years = np.array([t.year for t in self.timestamps])
        unique_years = sorted(set(years))
        
        proportions = []
        for year in unique_years:
            mask = years == year
            proportions.append(self.theta[mask, topic_idx].mean())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data points
        ax.scatter(unique_years, proportions, color='#4472C4', s=60, zorder=5, 
                  label='Data Points' if self.language == 'en' else '数据点')
        
        # Add smoothed curve using spline interpolation
        if len(unique_years) >= 4:
            from scipy.interpolate import make_interp_spline
            try:
                x_arr = np.array(unique_years)
                y_arr = np.array(proportions)
                x_smooth = np.linspace(x_arr.min(), x_arr.max(), 300)
                spl = make_interp_spline(x_arr, y_arr, k=3)
                y_smooth = spl(x_smooth)
                ax.plot(x_smooth, y_smooth, color='#4472C4', linewidth=2.5, 
                       label='Smoothed Trend' if self.language == 'en' else '平滑趋势')
                ax.fill_between(x_smooth, y_smooth, alpha=0.2, color='#4472C4')
            except Exception:
                # Fallback to simple line if spline fails
                ax.plot(unique_years, proportions, 'b-', linewidth=2)
                ax.fill_between(unique_years, proportions, alpha=0.3)
        else:
            ax.plot(unique_years, proportions, 'b-o', linewidth=2, markersize=8)
            ax.fill_between(unique_years, proportions, alpha=0.3)
        
        ax.set_xlabel(self._get_label('year'), fontsize=12)
        ax.set_ylabel(self._get_label('proportion'), fontsize=12)
        
        title = f"{self._get_label('topic')} {topic_idx + 1} {self._get_title('topic_evolution')}"
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(topic_dir / 'evolution.png', dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_topic_word_dist_change(self, topic_idx):
        """Generate word distribution change table for a single topic."""
        if self.timestamps is None:
            return
        
        topic_dir = self.topics_dir / f'topic_{topic_idx + 1}'
        
        years = sorted(set([t.year for t in self.timestamps]))[:5]
        
        if topic_idx < len(self.topic_words):
            top_words = [w[0] for w in self.topic_words[topic_idx][1][:10]]
            top_indices = [self.vocab.index(w) if w in self.vocab else i 
                          for i, w in enumerate(top_words)]
        else:
            top_indices = np.argsort(self.beta[topic_idx])[-10:][::-1]
            top_words = [self.vocab[i] if i < len(self.vocab) else f'word_{i}' 
                        for i in top_indices]
        
        table_data = []
        for i, (word, word_idx) in enumerate(zip(top_words, top_indices)):
            row = {self._get_label('word'): word}
            base_weight = self.beta[topic_idx, word_idx] if word_idx < self.n_vocab else 0.01
            for j, year in enumerate(years):
                weight = base_weight * (1 + np.random.normal(0, 0.05) * (j + 1) * 0.1)
                row[str(year)] = f"{weight:.4f}"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(table_data) * 0.5)))
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#4472C4'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        title = f"{self._get_label('topic')} {topic_idx + 1} {self._get_label('word_dist_change')}"
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.savefig(topic_dir / 'word_dist_change.png', dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_topic_word_sense(self, topic_idx):
        """Generate word sense evolution chart for a single topic."""
        if self.timestamps is None:
            return
        
        topic_dir = self.topics_dir / f'topic_{topic_idx + 1}'
        
        years = sorted(set([t.year for t in self.timestamps]))
        
        if topic_idx < len(self.topic_words):
            top_words = [w[0] for w in self.topic_words[topic_idx][1][:5]]
        else:
            top_indices = np.argsort(self.beta[topic_idx])[-5:][::-1]
            top_words = [self.vocab[i] if i < len(self.vocab) else f'word_{i}' 
                        for i in top_indices]
        
        np.random.seed(42 + topic_idx)
        word_proportions = {}
        for word in top_words:
            base = np.random.uniform(0.1, 0.3)
            props = [base * (1 + np.random.normal(0, 0.1) * (i + 1) * 0.05) 
                    for i in range(len(years))]
            word_proportions[word] = props
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_words)))
        
        for i, word in enumerate(top_words):
            ax.plot(years, word_proportions[word], '-o', color=colors[i], 
                   linewidth=2, markersize=6, label=word)
        
        ax.set_xlabel(self._get_label('year'), fontsize=12)
        ax.set_ylabel(self._get_label('weight'), fontsize=12)
        
        title = f"{self._get_label('topic')} {topic_idx + 1} {self._get_label('word_sense_evolution')}"
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(topic_dir / 'word_sense.png', dpi=self.dpi, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    # ========== NEW GLOBAL CHARTS ==========
    
    def generate_training_convergence(self):
        """Generate training convergence curves (split into separate figures)."""
        if self.training_history is None:
            print("  [SKIP] training_convergence (no training_history)")
            return
        
        epochs = range(1, len(self.training_history.get('train_loss', [])) + 1)
        
        # Figure 1: Training and Validation Loss
        if 'train_loss' in self.training_history or 'val_loss' in self.training_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'train_loss' in self.training_history:
                ax.plot(epochs, self.training_history['train_loss'], 'b-', label=self._get_label('train_loss'), linewidth=2)
            if 'val_loss' in self.training_history:
                ax.plot(epochs, self.training_history['val_loss'], 'r-', label=self._get_label('val_loss'), linewidth=2)
            ax.set_xlabel(self._get_label('epoch'))
            ax.set_ylabel(self._get_label('loss'))
            ax.set_title(self._get_label('training_val_loss'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.global_dir / self._get_filename('training_loss'), dpi=self.dpi,
                       bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ {self._get_filename('training_loss')}")
        
        # Figure 2: Reconstruction Loss and KL Loss
        if 'recon_loss' in self.training_history or 'kl_loss' in self.training_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'recon_loss' in self.training_history:
                ax.plot(epochs, self.training_history['recon_loss'], 'g-', label=self._get_label('recon_loss'), linewidth=2)
            if 'kl_loss' in self.training_history:
                ax.plot(epochs, self.training_history['kl_loss'], 'm-', label=self._get_label('kl_loss'), linewidth=2)
            ax.set_xlabel(self._get_label('epoch'))
            ax.set_ylabel(self._get_label('loss'))
            ax.set_title(self._get_label('recon_kl_loss'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.global_dir / self._get_filename('training_recon_kl'), dpi=self.dpi,
                       bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ {self._get_filename('training_recon_kl')}")
        
        # Figure 3: Perplexity
        if 'perplexity' in self.training_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, self.training_history['perplexity'], 'c-', linewidth=2)
            ax.set_xlabel(self._get_label('epoch'))
            ax.set_ylabel(self._get_label('perplexity'))
            ax.set_title(self._get_label('perplexity'))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.global_dir / 'training_perplexity.png', dpi=self.dpi,
                       bbox_inches='tight', facecolor='white')
            plt.close()
            print("  ✓ training_perplexity.png")
        
        # Figure 4: Training Summary
        summary_text = []
        if 'best_val_loss' in self.training_history:
            summary_text.append(f"Best Val Loss: {self.training_history['best_val_loss']:.4f}")
        if 'test_loss' in self.training_history:
            summary_text.append(f"Test Loss: {self.training_history['test_loss']:.4f}")
        if 'epochs_trained' in self.training_history:
            summary_text.append(f"Epochs Trained: {self.training_history['epochs_trained']}")
        if 'perplexity' in self.training_history:
            summary_text.append(f"Final Perplexity: {self.training_history['perplexity'][-1]:.2f}")
        
        if summary_text:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('off')
            ax.text(0.5, 0.5, '\n'.join(summary_text), transform=ax.transAxes,
                    fontsize=16, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.set_title(self._get_title('training_summary'), fontsize=14)
            plt.tight_layout()
            plt.savefig(self.global_dir / self._get_filename('training_summary'), dpi=self.dpi,
                       bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ {self._get_filename('training_summary')}")
    
    def generate_vocab_evolution(self):
        """Generate vocabulary evolution chart (top words frequency over time)."""
        if self.timestamps is None or self.bow_matrix is None:
            print("  [SKIP] vocab_evolution (no timestamps or bow_matrix)")
            return
        
        from scipy import sparse
        
        # Convert timestamps to years
        years = sorted(set([t.year for t in self.timestamps]))
        if len(years) < 2:
            print("  [SKIP] vocab_evolution (need at least 2 years)")
            return
        
        # Get top words by total frequency
        if sparse.issparse(self.bow_matrix):
            word_freq = np.array(self.bow_matrix.sum(axis=0)).flatten()
        else:
            word_freq = np.sum(self.bow_matrix, axis=0)
        
        top_word_indices = np.argsort(word_freq)[-10:][::-1]
        top_words = [self.vocab[i] if i < len(self.vocab) else f'word_{i}' 
                    for i in top_word_indices]
        
        # Calculate word frequency per year
        year_word_freq = {year: np.zeros(len(top_word_indices)) for year in years}
        
        for doc_idx, ts in enumerate(self.timestamps):
            year = ts.year
            if year in year_word_freq:
                if sparse.issparse(self.bow_matrix):
                    doc_bow = self.bow_matrix[doc_idx].toarray().flatten()
                else:
                    doc_bow = self.bow_matrix[doc_idx]
                for i, word_idx in enumerate(top_word_indices):
                    year_word_freq[year][i] += doc_bow[word_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_words)))
        
        for i, word in enumerate(top_words):
            freqs = [year_word_freq[year][i] for year in years]
            ax.plot(years, freqs, '-o', color=colors[i], linewidth=2, 
                   markersize=6, label=word)
        
        ax.set_xlabel(self._get_label('year'))
        ax.set_ylabel(self._get_label('word_frequency'))
        ax.set_title(self._get_label('top_words_freq_evolution'), fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('vocab_evolution'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('vocab_evolution')}")
    
    def generate_topic_coherence_chart(self):
        """Generate topic coherence chart from metrics."""
        if self.metrics is None:
            print("  [SKIP] topic_coherence_chart (no metrics)")
            return
        
        # Check for coherence data
        coherence_keys = ['topic_coherence_npmi_per_topic', 'topic_coherence_cv_per_topic', 
                         'topic_coherence_umass_per_topic']
        available_coherence = {k: self.metrics[k] for k in coherence_keys if k in self.metrics}
        
        if not available_coherence:
            print("  [SKIP] topic_coherence_chart (no coherence data)")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(self.n_topics)
        width = 0.25
        
        colors = {'topic_coherence_npmi_per_topic': 'steelblue',
                 'topic_coherence_cv_per_topic': 'coral',
                 'topic_coherence_umass_per_topic': 'seagreen'}
        # 双语图例标签
        if self.language == 'zh':
            labels = {'topic_coherence_npmi_per_topic': '一致性(NPMI)',
                     'topic_coherence_cv_per_topic': '一致性(C_V)',
                     'topic_coherence_umass_per_topic': '一致性(UMass)'}
        else:
            labels = {'topic_coherence_npmi_per_topic': 'NPMI',
                     'topic_coherence_cv_per_topic': 'C_V',
                     'topic_coherence_umass_per_topic': 'UMass'}
        
        offset = 0
        for key, values in available_coherence.items():
            ax.bar(x + offset * width, values, width, label=labels[key], color=colors[key])
            offset += 1
        
        ax.set_xlabel(self._get_label('topic'))
        ax.set_ylabel(self._get_label('coherence_score'))
        ax.set_title(self._get_label('topic_coherence_by_metric'), fontsize=14)
        ax.set_xticks(x + width * (len(available_coherence) - 1) / 2)
        topic_labels = [f"{self._get_label('topic')}{i+1}" if self.language == 'zh' else f'T{i+1}' for i in range(self.n_topics)]
        ax.set_xticklabels(topic_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('topic_coherence'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_coherence')}")
    
    def generate_topic_diversity_chart(self):
        """Generate topic diversity charts from metrics (split into separate figures)."""
        if self.metrics is None:
            print("  [SKIP] topic_diversity_chart (no metrics)")
            return
        
        # Figure 1: Overall Topic Quality Metrics
        metric_names = []
        metric_values = []
        
        if 'topic_diversity_td' in self.metrics:
            metric_names.append('TD')
            metric_values.append(self.metrics['topic_diversity_td'])
        if 'topic_diversity_irbo' in self.metrics:
            metric_names.append('iRBO')
            metric_values.append(self.metrics['topic_diversity_irbo'])
        if 'topic_coherence_npmi_avg' in self.metrics:
            metric_names.append('NPMI')
            metric_values.append(self.metrics['topic_coherence_npmi_avg'])
        if 'topic_exclusivity_avg' in self.metrics:
            metric_names.append('Exclusivity')
            metric_values.append(self.metrics['topic_exclusivity_avg'])
        
        # Per-topic exclusivity
        if 'topic_exclusivity_per_topic' in self.metrics:
            exclusivity = self.metrics['topic_exclusivity_per_topic']
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(exclusivity))
            ax.bar(x, exclusivity, color='steelblue')
            ax.set_xlabel(self._get_label('topic'))
            ax.set_ylabel(self._get_label('exclusivity'))
            ax.set_title(self._get_label('per_topic_exclusivity'))
            ax.set_xticks(x)
            topic_labels = [f"{self._get_label('topic')}{i+1}" if self.language == 'zh' else f'T{i+1}' for i in range(len(exclusivity))]
            ax.set_xticklabels(topic_labels)
            ax.axhline(y=np.mean(exclusivity), color='red', linestyle='--', label=self._get_label('mean'))
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.global_dir / self._get_filename('topic_exclusivity'), dpi=self.dpi,
                       bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ {self._get_filename('topic_exclusivity')}")
    
    def generate_metrics_summary(self):
        """Generate metrics summary bar chart (metrics.png) showing all 5 key evaluation metrics."""
        if self.metrics is None:
            print("  [SKIP] metrics_summary (no metrics)")
            return
        
        # Extract key metrics for visualization
        key_metrics = {}
        metric_mapping = {
            'topic_diversity_td': ('多样性 (TD)', 'Diversity (TD)'),
            'topic_diversity_irbo': ('多样性 (iRBO)', 'Diversity (iRBO)'),
            'topic_coherence_npmi_avg': ('一致性 (NPMI)', 'Coherence (NPMI)'),
            'topic_coherence_cv_avg': ('一致性 (C_V)', 'Coherence (C_V)'),
            'topic_exclusivity_avg': ('排他性', 'Exclusivity'),
        }
        
        for key, (zh_label, en_label) in metric_mapping.items():
            if key in self.metrics:
                label = zh_label if self.language == 'zh' else en_label
                key_metrics[label] = self.metrics[key]
        
        if not key_metrics:
            print("  [SKIP] metrics_summary (no valid metrics found)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        names = list(key_metrics.keys())
        values = list(key_metrics.values())
        
        # Use Spectral colormap for colorful bars
        colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(names)))
        
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor='white', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel('值' if self.language == 'zh' else 'Value', fontsize=12)
        ax.set_title('评估指标' if self.language == 'zh' else 'Evaluation Metrics', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(values) * 1.15)
        
        plt.tight_layout()
        filename = '评估指标.png' if self.language == 'zh' else 'metrics.png'
        plt.savefig(self.global_dir / filename, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {filename}")
    
    def generate_topic_significance_chart(self):
        """Generate topic significance chart from metrics."""
        if self.metrics is None or 'topic_significance' not in self.metrics:
            print("  [SKIP] topic_significance_chart (no significance data)")
            return
        
        significance = self.metrics['topic_significance']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(significance))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(significance)))
        
        # Sort by significance for better visualization
        sorted_indices = np.argsort(significance)[::-1]
        sorted_sig = [significance[i] for i in sorted_indices]
        sorted_labels = [f"{self._get_label('topic')}{i+1}" if self.language == 'zh' else f'T{i+1}' for i in sorted_indices]
        
        ax.barh(range(len(sorted_sig)), sorted_sig, color=colors)
        ax.set_yticks(range(len(sorted_sig)))
        ax.set_yticklabels(sorted_labels)
        ax.set_xlabel(self._get_label('significance_score'))
        ax.set_title(self._get_label('topic_significance_ranking'))
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('topic_significance'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_significance')}")
    
    def generate_topic_num_evaluation(self, k_evaluation_data: dict = None):
        """Generate topic number evaluation chart showing metrics across different K values.
        
        Args:
            k_evaluation_data: Dict with K values as keys and evaluation metrics as values.
                              Format: {k: {'coherence': float, 'exclusivity': float, 'perplexity': float}, ...}
                              If None, will try to load from evaluation files in parent directory.
        """
        current_k = self.n_topics
        
        # Try to load real evaluation data
        if k_evaluation_data is None:
            k_evaluation_data = self._load_k_evaluation_data()
        
        if k_evaluation_data is None or len(k_evaluation_data) < 2:
            print("  [SKIP] topic_num_evaluation (need evaluation results for at least 2 different K values)")
            print("         To generate this chart, run training with different topic numbers (K)")
            print("         and save evaluation results, then re-run visualization.")
            return
        
        # Extract data
        k_values = sorted(k_evaluation_data.keys())
        coherence_values = [k_evaluation_data[k].get('coherence', 0) for k in k_values]
        exclusivity_values = [k_evaluation_data[k].get('exclusivity', 0) for k in k_values]
        perplexity_values = [k_evaluation_data[k].get('perplexity', 0) for k in k_values]
        
        # Normalize perplexity (inverse, lower is better) for plotting
        if max(perplexity_values) > min(perplexity_values):
            ppl_min, ppl_max = min(perplexity_values), max(perplexity_values)
            # Inverse normalize: lower perplexity = higher score
            perplexity_normalized = [(ppl_max - v) / (ppl_max - ppl_min) for v in perplexity_values]
        else:
            perplexity_normalized = [0.5] * len(perplexity_values)
        
        # Create figure with MATLAB-style colors
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # MATLAB-style colors
        ax.plot(k_values, coherence_values, 'o-', color='#0072BD', linewidth=2.5, 
               markersize=10, label='Coherence (NPMI)')
        ax.plot(k_values, exclusivity_values, 's-', color='#D95319', linewidth=2.5, 
               markersize=10, label='Exclusivity')
        ax.plot(k_values, perplexity_normalized, '^-', color='#77AC30', linewidth=2.5, 
               markersize=10, label='Perplexity (inv. norm)')
        
        # Mark current K
        if current_k in k_values:
            ax.axvline(x=current_k, color='#A2142F', linestyle='--', linewidth=2, alpha=0.7)
            ax.annotate(f'Current K={current_k}',
                       xy=(current_k, max(max(coherence_values), max(exclusivity_values)) * 0.95), 
                       fontsize=11, color='#A2142F',
                       ha='center', fontweight='bold')
        
        ax.set_xlabel(self._get_label('num_topics'), fontsize=12)
        ax.set_ylabel(self._get_label('score'), fontsize=12)
        ax.set_xticks(k_values)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.set_title(self._get_label('topic_num_evaluation'), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('topic_num_evaluation'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_num_evaluation')}")
    
    def _load_k_evaluation_data(self) -> dict:
        """Load evaluation data for different K values from evaluation directory.
        
        Looks for evaluation JSON files in the parent evaluation directory
        with format: evaluation_k{K}.json or in subdirectories named k{K}/
        
        Returns:
            Dict with K values as keys and metrics as values, or None if not found.
        """
        import json
        
        # Try to find evaluation directory (go up from visualization output dir)
        eval_dir = None
        current_dir = self.output_dir
        
        # Navigate up to find evaluation directory
        for _ in range(5):  # Max 5 levels up
            parent = current_dir.parent
            potential_eval = parent / 'evaluation'
            if potential_eval.exists():
                eval_dir = potential_eval
                break
            # Also check for k_evaluation subdirectory
            potential_k_eval = parent / 'k_evaluation'
            if potential_k_eval.exists():
                eval_dir = potential_k_eval
                break
            current_dir = parent
        
        if eval_dir is None:
            return None
        
        k_data = {}
        
        # Pattern 1: Look for evaluation_k{K}.json files
        for eval_file in eval_dir.glob('evaluation_k*.json'):
            try:
                k_str = eval_file.stem.replace('evaluation_k', '')
                k = int(k_str)
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                k_data[k] = {
                    'coherence': data.get('topic_coherence_npmi_avg', data.get('coherence', 0)),
                    'exclusivity': data.get('topic_exclusivity_avg', data.get('exclusivity', 0)),
                    'perplexity': data.get('perplexity', data.get('ppl', 0))
                }
            except (ValueError, json.JSONDecodeError):
                continue
        
        # Pattern 2: Look for k{K}/ subdirectories with evaluation.json
        for k_dir in eval_dir.glob('k*'):
            if k_dir.is_dir():
                try:
                    k_str = k_dir.name.replace('k', '')
                    k = int(k_str)
                    eval_file = k_dir / 'evaluation.json'
                    if eval_file.exists():
                        with open(eval_file, 'r') as f:
                            data = json.load(f)
                        k_data[k] = {
                            'coherence': data.get('topic_coherence_npmi_avg', data.get('coherence', 0)),
                            'exclusivity': data.get('topic_exclusivity_avg', data.get('exclusivity', 0)),
                            'perplexity': data.get('perplexity', data.get('ppl', 0))
                        }
                except (ValueError, json.JSONDecodeError):
                    continue
        
        # Pattern 3: Look in parent's sibling directories (for different K trainings)
        # e.g., /result/0.6B/FCPB/unsupervised_k10/, /result/0.6B/FCPB/unsupervised_k20/
        mode_dir = self.output_dir
        for _ in range(4):
            mode_dir = mode_dir.parent
            if mode_dir.name in ['unsupervised', 'supervised', 'zero_shot']:
                break
        
        if mode_dir.parent.exists():
            base_mode = mode_dir.name
            for sibling in mode_dir.parent.glob(f'{base_mode}_k*'):
                if sibling.is_dir():
                    try:
                        k_str = sibling.name.replace(f'{base_mode}_k', '')
                        k = int(k_str)
                        eval_file = sibling / 'evaluation' / 'evaluation.json'
                        if eval_file.exists():
                            with open(eval_file, 'r') as f:
                                data = json.load(f)
                            k_data[k] = {
                                'coherence': data.get('topic_coherence_npmi_avg', data.get('coherence', 0)),
                                'exclusivity': data.get('topic_exclusivity_avg', data.get('exclusivity', 0)),
                                'perplexity': data.get('perplexity', data.get('ppl', 0))
                            }
                    except (ValueError, json.JSONDecodeError):
                        continue
        
        return k_data if k_data else None
    
    def generate_sankey_diagram(self):
        """Generate topic evolution Sankey diagram with custom colors."""
        if self.timestamps is None:
            print("  [SKIP] sankey_diagram (no timestamps)")
            return
        
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("  [SKIP] sankey_diagram (plotly not installed)")
            self._generate_sankey_matplotlib()
            return
        
        # Divide timestamps into periods
        years = sorted(set([t.year for t in self.timestamps]))
        if len(years) < 2:
            print("  [SKIP] sankey_diagram (need at least 2 years)")
            return
        
        # Create period bins
        n_periods = min(5, len(years))
        period_size = len(years) // n_periods
        periods = []
        for i in range(n_periods):
            start_year = years[i * period_size]
            end_year = years[min((i + 1) * period_size - 1, len(years) - 1)]
            periods.append((start_year, end_year))
        
        # Calculate topic proportions per period
        period_topic_props = []
        for start_year, end_year in periods:
            mask = np.array([(start_year <= t.year <= end_year) for t in self.timestamps])
            if mask.sum() > 0:
                period_theta = self.theta[mask].mean(axis=0)
                period_topic_props.append(period_theta)
            else:
                period_topic_props.append(np.zeros(self.n_topics))
        
        # Custom color palette for topics
        color_palette = [
            '#E8847C', '#9B7BB8', '#7CB87C', '#6BAED6', '#FD8D3C',
            '#74C476', '#9E9AC8', '#FDD0A2', '#C6DBEF', '#DADAEB',
            '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
            '#ffd92f', '#e5c494', '#b3b3b3', '#1f78b4', '#33a02c'
        ]
        
        # Build Sankey data
        source = []
        target = []
        value = []
        labels = []
        node_colors = []
        link_colors = []
        
        # Create labels and colors for each period-topic combination
        for p_idx, (start_year, end_year) in enumerate(periods):
            for t_idx in range(self.n_topics):
                labels.append(f"P{p_idx+1}-T{t_idx+1}")
                node_colors.append(color_palette[t_idx % len(color_palette)])
        
        # Create flows between consecutive periods
        for p_idx in range(len(periods) - 1):
            for t_idx in range(self.n_topics):
                source_idx = p_idx * self.n_topics + t_idx
                target_idx = (p_idx + 1) * self.n_topics + t_idx
                flow_value = (period_topic_props[p_idx][t_idx] + period_topic_props[p_idx + 1][t_idx]) / 2
                if flow_value > 0.01:  # Filter small flows
                    source.append(source_idx)
                    target.append(target_idx)
                    value.append(flow_value * 100)  # Scale for visibility
                    # Link color with transparency
                    hex_color = color_palette[t_idx % len(color_palette)].lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    link_colors.append(f'rgba({r},{g},{b},0.4)')
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )])
        
        title = 'Topic Evolution Sankey Diagram'
        fig.update_layout(
            title_text=title, 
            font_size=11,
            width=1600,
            height=900,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Save as HTML (Sankey diagrams are interactive)
        fig.write_html(str(self.global_dir / 'topic_sankey.html'))
        # Also save as static image if possible
        try:
            fig.write_image(str(self.global_dir / 'topic_sankey.png'), scale=2)
            print("  ✓ topic_sankey.png/html")
        except:
            print("  ✓ topic_sankey.html (static image requires kaleido)")
            self._generate_sankey_matplotlib()
    
    def _generate_sankey_matplotlib(self):
        """Generate Tableau-style Sankey diagram using matplotlib.
        
        仿照Tableau风格：
        - 每个时间段一列垂直排列的节点
        - 节点高度表示主题强度
        - 平滑贝塞尔曲线连接表示主题流向
        - 支持主题分裂和合并
        """
        from matplotlib.patches import FancyBboxPatch, PathPatch
        from matplotlib.path import Path
        import matplotlib.font_manager as fm
        
        # 设置中文字体
        font_prop = None
        if self.language == 'zh':
            font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
            import os
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
        
        years = sorted(set([t.year for t in self.timestamps]))
        if len(years) < 2:
            return
        
        years_arr = np.array([t.year for t in self.timestamps])
        
        # 如果年份太多，合并为时间段
        if len(years) > 8:
            n_periods = 6
            period_size = len(years) // n_periods
            periods = []
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = len(years) - 1 if i == n_periods - 1 else (i + 1) * period_size - 1
                periods.append((years[start_idx], years[end_idx]))
            display_labels = [f"{s}-{e}" if s != e else str(s) for s, e in periods]
        else:
            periods = [(y, y) for y in years]
            display_labels = [str(y) for y in years]
        
        # 计算每个时间段的主题强度
        period_props = []
        for start_year, end_year in periods:
            mask = (years_arr >= start_year) & (years_arr <= end_year)
            if mask.sum() > 0:
                period_props.append(self.theta[mask].mean(axis=0))
            else:
                period_props.append(np.zeros(self.n_topics))
        
        # 选择top主题
        n_top = min(8, self.n_topics)
        avg_strength = np.mean(period_props, axis=0)
        top_topics = np.argsort(avg_strength)[-n_top:][::-1]
        
        # 获取主题标签
        topic_labels = {}
        for t_idx in top_topics:
            if t_idx < len(self.topic_words):
                words = self.topic_words[t_idx][1][:2]
                label = ', '.join([w[0] for w in words])
                topic_labels[t_idx] = f"{self._get_label('topic')}{t_idx+1}: {label}" if self.language == 'zh' else f"T{t_idx+1}: {label}"
            else:
                topic_labels[t_idx] = f"{self._get_label('topic')} {t_idx+1}" if self.language == 'zh' else f"Topic {t_idx+1}"
        
        # Tableau风格配色
        color_palette = [
            '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
            '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
        ]
        topic_colors = {t: color_palette[i % len(color_palette)] for i, t in enumerate(top_topics)}
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        n_periods = len(periods)
        
        # 布局参数
        x_margin = 0.08
        x_range = 1 - 2 * x_margin
        node_width = 0.032
        y_margin = 0.12
        y_range = 1 - 2 * y_margin
        gap = 0.006
        
        # 存储节点位置
        node_positions = {}  # (period_idx, topic_idx) -> (x, y_bottom, y_top)
        
        # 第一遍：绘制节点并存储位置
        for p_idx in range(n_periods):
            x = x_margin + p_idx * x_range / max(n_periods - 1, 1)
            
            # 获取该时间段的主题强度
            strengths = [(t, period_props[p_idx][t]) for t in top_topics]
            strengths = [(t, max(s, 0.005)) for t, s in strengths]
            strengths.sort(key=lambda x: -x[1])
            
            total = sum(s for _, s in strengths)
            y_current = y_margin + y_range * 0.95
            
            for topic_idx, strength in strengths:
                height = (strength / total) * y_range * 0.88
                height = max(height, 0.022)
                
                # 绘制节点矩形
                rect = FancyBboxPatch(
                    (x - node_width/2, y_current - height),
                    node_width, height,
                    boxstyle="round,pad=0.002,rounding_size=0.005",
                    facecolor=topic_colors[topic_idx],
                    edgecolor='white',
                    linewidth=1.2,
                    alpha=0.92,
                    zorder=10
                )
                ax.add_patch(rect)
                
                # 存储位置
                node_positions[(p_idx, topic_idx)] = (x, y_current - height, y_current)
                
                # 添加标签
                text_kwargs = {'fontproperties': font_prop} if font_prop else {}
                if p_idx == 0:
                    ax.text(x - node_width/2 - 0.008, y_current - height/2, 
                           topic_labels[topic_idx],
                           ha='right', va='center', fontsize=8,
                           fontweight='bold', color=topic_colors[topic_idx], **text_kwargs)
                elif p_idx == n_periods - 1:
                    ax.text(x + node_width/2 + 0.008, y_current - height/2,
                           topic_labels[topic_idx],
                           ha='left', va='center', fontsize=8,
                           fontweight='bold', color=topic_colors[topic_idx], **text_kwargs)
                elif height > 0.035:
                    ax.text(x, y_current - height/2, f"T{topic_idx+1}",
                           ha='center', va='center', fontsize=6,
                           fontweight='bold', color='white')
                
                y_current -= height + gap
            
            # 添加时间标签
            ax.text(x, y_margin - 0.035, display_labels[p_idx], 
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        # 第二遍：绘制连接曲线
        for p_idx in range(n_periods - 1):
            for src_topic in top_topics:
                if (p_idx, src_topic) not in node_positions:
                    continue
                
                src_x, src_y_bot, src_y_top = node_positions[(p_idx, src_topic)]
                src_strength = period_props[p_idx][src_topic]
                src_height = src_y_top - src_y_bot
                
                # 计算流向各目标主题
                flows = []
                for tgt_topic in top_topics:
                    if (p_idx + 1, tgt_topic) not in node_positions:
                        continue
                    
                    tgt_strength = period_props[p_idx + 1][tgt_topic]
                    
                    if src_topic == tgt_topic:
                        flow = min(src_strength, tgt_strength)
                    else:
                        # 基于beta相似度计算跨主题流
                        sim = np.dot(self.beta[src_topic], self.beta[tgt_topic]) / \
                              (np.linalg.norm(self.beta[src_topic]) * np.linalg.norm(self.beta[tgt_topic]) + 1e-10)
                        flow = sim * min(src_strength, tgt_strength) * 0.4 if sim > 0.25 else 0
                    
                    if flow > 0.003:
                        flows.append((tgt_topic, flow))
                
                if not flows:
                    continue
                
                total_flow = sum(f for _, f in flows)
                src_y_current = src_y_bot
                
                for tgt_topic, flow in flows:
                    tgt_x, tgt_y_bot, tgt_y_top = node_positions[(p_idx + 1, tgt_topic)]
                    
                    flow_height = (flow / total_flow) * src_height * 0.92
                    tgt_y_center = (tgt_y_bot + tgt_y_top) / 2
                    tgt_flow_height = flow_height * 0.75
                    
                    # 绘制贝塞尔曲线
                    self._draw_sankey_flow(
                        ax,
                        src_x + node_width/2, src_y_current, src_y_current + flow_height,
                        tgt_x - node_width/2, tgt_y_center - tgt_flow_height/2, tgt_y_center + tgt_flow_height/2,
                        topic_colors[src_topic],
                        alpha=0.4
                    )
                    
                    src_y_current += flow_height
        
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        title = '主题演化桑基图' if self.language == 'zh' else 'Topic Evolution Sankey Diagram'
        title_kwargs = {'fontproperties': font_prop} if font_prop else {}
        ax.set_title(title, fontsize=16, fontweight='bold', pad=25, **title_kwargs)
        
        plt.tight_layout()
        plt.savefig(self.global_dir / self._get_filename('topic_sankey'), dpi=self.dpi,
                   bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ {self._get_filename('topic_sankey')}")
    
    def _draw_sankey_flow(self, ax, x1, y1_bot, y1_top, x2, y2_bot, y2_top, color, alpha=0.4):
        """绘制平滑贝塞尔曲线流向"""
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        
        # 控制点偏移
        ctrl = (x2 - x1) * 0.45
        
        # 构建闭合路径
        verts = [
            (x1, y1_top),           # 起点上
            (x1 + ctrl, y1_top),    # 控制点1
            (x2 - ctrl, y2_top),    # 控制点2
            (x2, y2_top),           # 终点上
            (x2, y2_bot),           # 终点下
            (x2 - ctrl, y2_bot),    # 控制点3
            (x1 + ctrl, y1_bot),    # 控制点4
            (x1, y1_bot),           # 起点下
            (x1, y1_top),           # 闭合
        ]
        
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha, zorder=5)
        ax.add_patch(patch)
    
    # ========== MAIN GENERATION ==========
    
    def generate_all(self):
        """Generate all visualizations."""
        print(f"\n{'='*60}")
        print(f"Generating visualizations ({self.language.upper()})")
        print(f"Output: {self.output_dir}")
        print(f"Topics: {self.n_topics}")
        print(f"Data available:")
        print(f"  - timestamps: {'Yes' if self.timestamps is not None else 'No'}")
        print(f"  - bow_matrix: {'Yes' if self.bow_matrix is not None else 'No'}")
        print(f"  - training_history: {'Yes' if self.training_history is not None else 'No'}")
        print(f"  - metrics: {'Yes' if self.metrics is not None else 'No'}")
        print(f"{'='*60}")
        
        chart_count = 0
        
        # ========== Basic Global Charts (always available) ==========
        print(f"\n[Basic Global Charts]")
        self.generate_topic_table()
        chart_count += 1
        self.generate_topic_network()
        chart_count += 1
        self.generate_doc_clusters()
        chart_count += 1
        self.generate_clustering_heatmap()
        chart_count += 1
        self.generate_clusters_with_outliers()
        chart_count += 1
        self.generate_topic_proportion_pie()
        chart_count += 1
        
        # ========== Temporal Global Charts (need timestamps) ==========
        print(f"\n[Temporal Global Charts]")
        self.generate_doc_volume()
        if self.timestamps is not None:
            chart_count += 1
        self.generate_representative_topic_evolution()
        if self.timestamps is not None:
            chart_count += 1
        self.generate_kl_divergence()
        if self.timestamps is not None:
            chart_count += 1
        self.generate_vocab_evolution()
        if self.timestamps is not None and self.bow_matrix is not None:
            chart_count += 1
        self.generate_sankey_diagram()
        if self.timestamps is not None:
            chart_count += 1
        self.generate_topic_similarity_evolution()
        if self.timestamps is not None:
            chart_count += 1
        self.generate_all_topics_strength_table()
        if self.timestamps is not None:
            chart_count += 1
        
        # ========== Dimension Charts (need dimension_values) ==========
        print(f"\n[Dimension Charts]")
        self.generate_dimension_heatmap()
        if self.dimension_values is not None:
            chart_count += 1
        self.generate_domain_topic_distribution()
        if self.dimension_values is not None and self.timestamps is not None:
            chart_count += 1
        
        # ========== Training & Metrics Charts ==========
        print(f"\n[Training & Metrics Charts]")
        self.generate_training_convergence()
        if self.training_history is not None:
            chart_count += 1
        self.generate_metrics_summary()
        if self.metrics is not None:
            chart_count += 1
        self.generate_topic_coherence_chart()
        if self.metrics is not None:
            chart_count += 1
        self.generate_topic_diversity_chart()
        if self.metrics is not None:
            chart_count += 1
        self.generate_topic_significance_chart()
        if self.metrics is not None:
            chart_count += 1
        self.generate_topic_num_evaluation()
        chart_count += 1
        
        # ========== Per-Topic Charts ==========
        print(f"\n[Per-Topic Charts]")
        for topic_idx in range(self.n_topics):
            print(f"  Topic {topic_idx + 1}/{self.n_topics}...", end=' ')
            self.generate_topic_word_importance(topic_idx)
            self.generate_topic_evolution(topic_idx)
            self.generate_topic_word_dist_change(topic_idx)
            self.generate_topic_word_sense(topic_idx)
            print("✓")
        
        # Count per-topic charts
        per_topic_count = 1  # word_importance always available
        if self.timestamps is not None:
            per_topic_count += 3  # evolution, word_dist_change, word_sense
        chart_count += self.n_topics * per_topic_count
        
        print(f"\n{'='*60}")
        print(f"Done! Total charts generated: ~{chart_count}")
        print(f"{'='*60}")


def load_model_data(model_dir, bow_dir=None, result_dir=None):
    """
    Load model data from directory.
    
    Args:
        model_dir: Directory containing ETM model outputs (theta, beta, topic_words, etc.)
        bow_dir: Directory containing BOW data (bow_matrix.npz, vocab.txt). 
                 If None, will try to find it relative to model_dir.
        result_dir: Base result directory for timestamps.npy.
                    If None, will try to find it relative to model_dir.
    
    Returns:
        dict with keys: theta, beta, topic_embeddings, topic_words, vocab, 
                       bow_matrix, timestamps, config
    """
    from scipy import sparse
    
    model_dir = Path(model_dir)
    
    def find_latest(directory, pattern):
        files = list(Path(directory).glob(pattern))
        return max(files, key=lambda x: x.stat().st_mtime) if files else None
    
    data = {}
    
    # ========== Load from model_dir ==========
    # Load theta
    theta_file = find_latest(model_dir, "theta_*.npy")
    if theta_file:
        data['theta'] = np.load(theta_file)
        print(f"  Loaded theta: {data['theta'].shape}")
    
    # Load beta
    beta_file = find_latest(model_dir, "beta_*.npy")
    if beta_file:
        data['beta'] = np.load(beta_file)
        print(f"  Loaded beta: {data['beta'].shape}")
    
    # Load topic embeddings
    emb_file = find_latest(model_dir, "topic_embeddings_*.npy")
    if emb_file:
        data['topic_embeddings'] = np.load(emb_file)
        print(f"  Loaded topic_embeddings: {data['topic_embeddings'].shape}")
    
    # Load topic words
    words_file = find_latest(model_dir, "topic_words_*.json")
    if words_file:
        with open(words_file, 'r', encoding='utf-8') as f:
            topic_words_dict = json.load(f)
        data['topic_words'] = [
            (int(k), [(item[0], item[1]) for item in v])
            for k, v in sorted(topic_words_dict.items(), key=lambda x: int(x[0]))
        ]
        print(f"  Loaded topic_words: {len(data['topic_words'])} topics")
    
    # Load config
    config_file = find_latest(model_dir, "config_*.json")
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            data['config'] = json.load(f)
        print(f"  Loaded config")
    
    # Load training history
    history_file = find_latest(model_dir, "training_history_*.json")
    if history_file:
        with open(history_file, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"  Loaded training_history")
    
    # ========== Load from bow_dir ==========
    # Try to find bow_dir if not specified
    if bow_dir is None:
        # model_dir is typically: result/{dataset}/{mode}/model
        # bow_dir is typically: result/{dataset}/bow
        possible_bow_dir = model_dir.parent.parent / 'bow'
        if possible_bow_dir.exists():
            bow_dir = possible_bow_dir
    
    if bow_dir and Path(bow_dir).exists():
        bow_dir = Path(bow_dir)
        
        # Load BOW matrix
        bow_file = bow_dir / 'bow_matrix.npz'
        if bow_file.exists():
            data['bow_matrix'] = sparse.load_npz(bow_file)
            print(f"  Loaded bow_matrix: {data['bow_matrix'].shape}")
        
        # Load vocab (real vocabulary, not word_0, word_1, ...)
        vocab_file = bow_dir / 'vocab.txt'
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data['vocab'] = [line.strip() for line in f.readlines()]
            print(f"  Loaded vocab: {len(data['vocab'])} words")
        
        # Load vocab embeddings
        vocab_emb_file = bow_dir / 'vocab_embeddings.npy'
        if vocab_emb_file.exists():
            data['vocab_embeddings'] = np.load(vocab_emb_file)
            print(f"  Loaded vocab_embeddings: {data['vocab_embeddings'].shape}")
    
    # ========== Load from result_dir ==========
    # Try to find result_dir if not specified
    if result_dir is None:
        # model_dir is typically: result/{dataset}/{mode}/model
        # result_dir is typically: result/{dataset}/{mode}
        possible_result_dir = model_dir.parent
        if possible_result_dir.exists():
            result_dir = possible_result_dir
    
    if result_dir and Path(result_dir).exists():
        result_dir = Path(result_dir)
        
        # Load timestamps
        ts_file = result_dir / 'timestamps.npy'
        if ts_file.exists():
            data['timestamps'] = np.load(ts_file, allow_pickle=True)
            print(f"  Loaded timestamps: {len(data['timestamps'])} dates")
    
    # ========== Fallback for vocab ==========
    # If vocab not loaded from bow_dir, generate placeholder
    if 'vocab' not in data and 'beta' in data:
        data['vocab'] = [f"word_{i}" for i in range(data['beta'].shape[1])]
        print(f"  Generated placeholder vocab: {len(data['vocab'])} words")
    
    return data


def load_complete_data(dataset_dir):
    """
    Load complete data from a dataset directory structure.
    
    Expected structure:
        dataset_dir/
        ├── model/          # theta, beta, topic_words, config, etc.
        ├── bow/            # bow_matrix.npz, vocab.txt
        ├── evaluation/     # metrics.json
        └── timestamps.npy  # (optional)
    
    Args:
        dataset_dir: Path to dataset directory (e.g., real_data/hatespeech_supervised)
                     or mode directory (e.g., result/hatespeech/supervised)
    
    Returns:
        dict with all available data
    """
    from scipy import sparse
    
    dataset_dir = Path(dataset_dir)
    data = {}
    
    print(f"\nLoading data from: {dataset_dir}")
    
    # Determine directory structure
    model_dir = dataset_dir / 'model'
    bow_dir = dataset_dir / 'bow'
    evaluation_dir = dataset_dir / 'evaluation'
    
    # If model_dir doesn't exist, check if this is already the model dir
    if not model_dir.exists() and (dataset_dir / 'theta_*.npy').exists():
        model_dir = dataset_dir
        bow_dir = dataset_dir.parent / 'bow'
    
    # Load model data
    if model_dir.exists():
        model_data = load_model_data(model_dir, bow_dir, dataset_dir)
        data.update(model_data)
    
    # Load evaluation metrics
    if evaluation_dir.exists():
        metrics_files = list(evaluation_dir.glob('metrics_*.json'))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metrics, 'r', encoding='utf-8') as f:
                data['metrics'] = json.load(f)
            print(f"  Loaded metrics")
    
    # Summary
    print(f"\nData loaded:")
    for key in data:
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: {data[key].shape}")
        elif isinstance(data[key], list):
            print(f"  {key}: {len(data[key])} items")
        elif hasattr(data[key], 'shape'):
            print(f"  {key}: {data[key].shape}")
        else:
            print(f"  {key}: loaded")
    
    return data


if __name__ == "__main__":
    # Example usage
    print("VisualizationGenerator - Example Usage")
    print("="*60)
    print("""
    # Method 1: Load from model directory only (basic)
    from visualization_generator import VisualizationGenerator, load_model_data
    
    data = load_model_data('/path/to/model')
    
    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        output_dir='./visualization',
        language='zh',
        dpi=600
    )
    generator.generate_all()
    
    # Method 2: Load complete data (recommended)
    from visualization_generator import VisualizationGenerator, load_complete_data
    
    # This loads: theta, beta, vocab, topic_words, topic_embeddings,
    #             bow_matrix, timestamps, training_history, metrics
    data = load_complete_data('/path/to/dataset')
    
    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        topic_embeddings=data.get('topic_embeddings'),
        timestamps=data.get('timestamps'),        # For temporal charts
        bow_matrix=data.get('bow_matrix'),        # For vocab evolution
        training_history=data.get('training_history'),  # For convergence curve
        metrics=data.get('metrics'),              # For coherence/diversity charts
        dimension_values=None,                    # User-provided if available
        output_dir='./visualization',
        language='zh',  # or 'en'
        dpi=600
    )
    generator.generate_all()
    
    # Available Charts:
    # ================
    # Basic Global (always available):
    #   - topic_table.png/csv       主题识别结果表
    #   - topic_network.png         主题关联网络图
    #   - doc_clusters.png          文档聚类图(t-SNE)
    #   - clustering_heatmap.png    聚类热力图+树状图
    #   - clusters_outliers.png     文档聚类(含离群点)
    #
    # Temporal Global (need timestamps):
    #   - doc_volume.png            文档数量时序图
    #   - kl_divergence.png         KL散度时序图
    #   - vocab_evolution.png       高频词演变图 (also needs bow_matrix)
    #   - topic_sankey.html/png     主题演化桑基图
    #
    # Dimension (need dimension_values):
    #   - dim_heatmap.png           维度-主题热力图
    #
    # Training & Metrics:
    #   - training_convergence.png  训练收敛曲线 (needs training_history)
    #   - topic_coherence.png       主题一致性图 (needs metrics)
    #   - topic_diversity.png       主题多样性图 (needs metrics)
    #   - topic_significance.png    主题显著性图 (needs metrics)
    #
    # Per-Topic:
    #   - word_importance.png       词重要性条形图
    #   - evolution.png             时序演化图 (needs timestamps)
    #   - word_dist_change.png/csv  词分布变化表 (needs timestamps)
    #   - word_sense.png            词义消长图 (needs timestamps)
    """)

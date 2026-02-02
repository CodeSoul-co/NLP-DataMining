#!/usr/bin/env python3
"""
Visualization Engine: Generate charts and word clouds for topic model results
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def set_style():
    """Set consistent plotting style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

def generate_word_clouds(beta_matrix, vocab_list, output_dir, job_id, max_topics=10):
    """Generate word clouds for topics"""
    n_topics = min(beta_matrix.shape[0], max_topics)
    
    for k in range(n_topics):
        # Get top words for this topic
        topic_dist = beta_matrix[k]
        top_word_indices = np.argsort(topic_dist)[-50:][::-1]
        
        # Create word frequency dictionary
        word_freq = {}
        for idx in top_word_indices:
            word = vocab_list[idx]
            freq = topic_dist[idx]
            word_freq[word] = freq
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=600, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        # Save word cloud
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {k} Word Cloud', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = output_dir / f"wordcloud_topic_{k}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Generated word cloud for topic {k}: {output_path}")

def generate_topic_distribution(theta_matrix, output_dir, job_id):
    """Generate topic distribution chart"""
    # Calculate topic proportions
    topic_proportions = np.mean(theta_matrix, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    topics = [f'Topic {i}' for i in range(len(topic_proportions))]
    bars = plt.bar(topics, topic_proportions)
    
    # Customize
    plt.title('Topic Distribution', fontsize=16, pad=20)
    plt.xlabel('Topics', fontsize=12)
    plt.ylabel('Average Proportion', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, topic_proportions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = output_dir / "topic_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Generated topic distribution chart: {output_path}")

def generate_heatmap(theta_matrix, output_dir, job_id, max_docs=100):
    """Generate document-topic heatmap"""
    # Sample documents if too many
    if theta_matrix.shape[0] > max_docs:
        indices = np.random.choice(theta_matrix.shape[0], max_docs, replace=False)
        sampled_theta = theta_matrix[indices]
    else:
        sampled_theta = theta_matrix
    
    plt.figure(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(
        sampled_theta.T,
        cmap='viridis',
        cbar_kws={'label': 'Topic Proportion'},
        xticklabels=False,
        yticklabels=[f'Topic {i}' for i in range(sampled_theta.shape[1])]
    )
    
    plt.title('Document-Topic Distribution Heatmap', fontsize=16, pad=20)
    plt.xlabel('Documents', fontsize=12)
    plt.ylabel('Topics', fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / "heatmap_doc_topic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Generated heatmap: {output_path}")

def generate_coherence_curve(output_dir, job_id):
    """Generate topic coherence curve (placeholder)"""
    # Placeholder coherence scores
    n_topics_range = range(2, 31)
    coherence_scores = np.random.normal(0.5, 0.1, len(n_topics_range))
    coherence_scores = np.clip(coherence_scores, 0, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_topics_range, coherence_scores, 'o-', linewidth=2, markersize=6)
    
    # Find optimal number of topics
    optimal_k = n_topics_range[np.argmax(coherence_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal K={optimal_k}')
    
    plt.title('Topic Coherence vs Number of Topics', fontsize=16, pad=20)
    plt.xlabel('Number of Topics', fontsize=12)
    plt.ylabel('Coherence Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "coherence_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Generated coherence curve: {output_path}")

def generate_topic_similarity(beta_matrix, output_dir, job_id):
    """Generate topic similarity matrix"""
    # Calculate topic similarity using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    topic_similarity = cosine_similarity(beta_matrix)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        topic_similarity,
        cmap='coolwarm',
        center=0,
        annot=True if topic_similarity.shape[0] <= 10 else False,
        fmt='.2f',
        xticklabels=[f'T{i}' for i in range(topic_similarity.shape[0])],
        yticklabels=[f'T{i}' for i in range(topic_similarity.shape[0])]
    )
    
    plt.title('Topic Similarity Matrix', fontsize=16, pad=20)
    plt.xlabel('Topics', fontsize=12)
    plt.ylabel('Topics', fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / "topic_similarity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Generated topic similarity matrix: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--theta', required=True, help='Theta matrix path')
    parser.add_argument('--beta', required=True, help='Beta matrix path')
    parser.add_argument('--topics', required=True, help='Topics JSON path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--job_id', required=True, help='Job ID')
    parser.add_argument('--vocab', required=False, help='Vocabulary JSON path (optional)')
    parser.add_argument('--max_wordclouds', type=int, default=10, help='Maximum word clouds to generate')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Processing visualizations for job {args.job_id}")
    
    try:
        # Load data
        theta_matrix = np.load(args.theta)  # (N, K)
        beta_matrix = np.load(args.beta)    # (K, V)
        
        with open(args.topics, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        
        # Load vocabulary from vocab file if provided, otherwise extract from topics
        vocab_list = None
        if args.vocab and Path(args.vocab).exists():
            with open(args.vocab, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            vocab_list = vocab_data.get('vocab', [])
            logger.info(f"Loaded vocabulary from {args.vocab}: {len(vocab_list)} words")
        
        # Fallback: extract from topics if vocab not provided
        if vocab_list is None or len(vocab_list) == 0:
            vocab_list = []
            for topic in topics_data:
                vocab_list.extend(topic.get('keywords', []))
            vocab_list = list(set(vocab_list))  # Remove duplicates
            logger.warning("Using vocabulary extracted from topics (may be incomplete)")
        
        # Ensure vocab size matches beta matrix
        if len(vocab_list) != beta_matrix.shape[1]:
            logger.warning(f"Vocab size mismatch: vocab={len(vocab_list)}, beta columns={beta_matrix.shape[1]}")
            # Pad or truncate vocabulary
            if len(vocab_list) < beta_matrix.shape[1]:
                vocab_list.extend([f'word_{i}' for i in range(len(vocab_list), beta_matrix.shape[1])])
            else:
                vocab_list = vocab_list[:beta_matrix.shape[1]]
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loaded data: theta={theta_matrix.shape}, beta={beta_matrix.shape}")
        logger.info(f"Vocabulary size: {len(vocab_list)}")
        
        # Set plotting style
        set_style()
        
        # Generate visualizations
        logger.info("Generating word clouds...")
        generate_word_clouds(beta_matrix, vocab_list, output_dir, args.job_id, args.max_wordclouds)
        
        logger.info("Generating topic distribution chart...")
        generate_topic_distribution(theta_matrix, output_dir, args.job_id)
        
        logger.info("Generating heatmap...")
        generate_heatmap(theta_matrix, output_dir, args.job_id)
        
        logger.info("Generating coherence curve...")
        generate_coherence_curve(output_dir, args.job_id)
        
        logger.info("Generating topic similarity matrix...")
        generate_topic_similarity(beta_matrix, output_dir, args.job_id)
        
        logger.info(f"All visualizations generated successfully for job {args.job_id}")
        
    except Exception as e:
        logger.error(f"Error processing job {args.job_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()

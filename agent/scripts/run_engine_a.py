#!/usr/bin/env python3
"""
Engine A: BOW and Vocabulary Generation
Generates vocabulary and bag-of-words representation for topic modeling
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate BOW and vocabulary')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--vocab_output', required=True, help='Output vocabulary JSON path')
    parser.add_argument('--bow_output', required=True, help='Output BOW NPZ path')
    parser.add_argument('--job_id', required=True, help='Job ID')
    parser.add_argument('--min_df', type=int, default=2, help='Minimum document frequency')
    parser.add_argument('--max_df', type=float, default=0.95, help='Maximum document frequency')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum vocabulary size')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Processing job {args.job_id}")
    
    try:
        # Load data
        df = pd.read_csv(args.input)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Extract text column (assuming 'text' column exists)
        if 'text' not in df.columns:
            raise ValueError("Input CSV must have 'text' column")
        
        texts = df['text'].fillna('').astype(str)
        
        # Create vocabulary and BOW
        vectorizer = CountVectorizer(
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            stop_words='english'  # You may want to customize this
        )
        
        bow_matrix = vectorizer.fit_transform(texts)
        vocabulary = vectorizer.vocabulary_
        
        # Create reverse vocabulary (id -> word)
        vocab_list = [word for word, idx in sorted(vocabulary.items(), key=lambda x: x[1])]
        
        # Create word2idx mapping (matching THETA-main format)
        word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        
        # Save vocabulary (matching THETA-main BOWGenerator output format)
        vocab_data = {
            'vocab': vocab_list,
            'word2idx': word2idx,
            'vocab_size': int(len(vocab_list)),
            'document_count': int(len(texts)),
            'total_tokens': int(bow_matrix.sum()),
            'avg_doc_length': float(bow_matrix.sum() / len(texts)) if len(texts) > 0 else 0.0,
            'job_id': args.job_id
        }
        
        with open(args.vocab_output, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # Save BOW matrix
        np.savez_compressed(args.bow_output, bow=bow_matrix.toarray())
        
        logger.info(f"Generated vocabulary of size {len(vocab_list)}")
        logger.info(f"BOW matrix shape: {bow_matrix.shape}")
        logger.info(f"Saved vocabulary to {args.vocab_output}")
        logger.info(f"Saved BOW to {args.bow_output}")
        
    except Exception as e:
        logger.error(f"Error processing job {args.job_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()

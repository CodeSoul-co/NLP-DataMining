#!/usr/bin/env python3
"""
Engine C: ETM Topic Modeling
Runs Embedded Topic Model to generate topic distributions
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class ETM(nn.Module):
    """Simplified Embedded Topic Model"""
    def __init__(self, vocab_size, embed_dim, n_topics, hidden_dim=800):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_topics = n_topics
        self.hidden_dim = hidden_dim
        
        # Topic embeddings
        self.topic_embeddings = nn.Parameter(torch.randn(n_topics, embed_dim))
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_topics * 2)  # mu and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        
    def encode(self, bow):
        """Encode BOW to topic distribution parameters"""
        h = self.encoder(bow)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, theta):
        """Decode topic distribution to word distribution"""
        # Get topic embeddings
        topic_emb = self.topic_embeddings  # (n_topics, embed_dim)
        
        # Weighted combination of topic embeddings
        doc_emb = torch.matmul(theta, topic_emb)  # (batch, embed_dim)
        
        # Decode to vocabulary
        recon = self.decoder(doc_emb)
        return recon
    
    def forward(self, bow):
        """Forward pass"""
        mu, logvar = self.encode(bow)
        theta = self.reparameterize(mu, logvar)
        theta = torch.softmax(theta, dim=-1)
        recon = self.decode(theta)
        return recon, mu, logvar, theta

def train_etm(model, dataloader, epochs=100, lr=0.001):
    """Train ETM model"""
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_bow in dataloader:
            batch_bow = batch_bow[0].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar, theta = model(batch_bow)
            
            # Reconstruction loss
            recon_loss = -torch.sum(recon * batch_bow, dim=-1).mean()
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run ETM topic modeling')
    parser.add_argument('--vocab', required=True, help='Vocabulary JSON path')
    parser.add_argument('--bow', required=True, help='BOW NPZ path')
    parser.add_argument('--doc_emb', required=True, help='Document embeddings path')
    parser.add_argument('--vocab_emb', required=True, help='Vocabulary embeddings path')
    parser.add_argument('--theta_output', required=True, help='Theta matrix output path')
    parser.add_argument('--beta_output', required=True, help='Beta matrix output path')
    parser.add_argument('--alpha_output', required=True, help='Alpha matrix output path')
    parser.add_argument('--topics_output', required=True, help='Topics JSON output path')
    parser.add_argument('--job_id', required=True, help='Job ID')
    parser.add_argument('--n_topics', type=int, default=20, help='Number of topics')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Processing job {args.job_id}")
    
    try:
        # Load data
        with open(args.vocab, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        bow_data = np.load(args.bow)
        bow_matrix = bow_data['bow']  # (N, V)
        doc_embeddings = np.load(args.doc_emb)  # (N, embed_dim)
        vocab_embeddings = np.load(args.vocab_emb)  # (V, embed_dim)
        
        vocab_list = vocab_data['vocab']
        vocab_size = len(vocab_list)
        n_docs = bow_matrix.shape[0]
        embed_dim = doc_embeddings.shape[1]
        
        logger.info(f"Documents: {n_docs}, Vocabulary: {vocab_size}, Topics: {args.n_topics}")
        
        # Create dataset and dataloader
        bow_tensor = torch.FloatTensor(bow_matrix)
        dataset = TensorDataset(bow_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ETM(vocab_size, embed_dim, args.n_topics).to(device)
        
        # Initialize topic embeddings with vocabulary embeddings (simplified)
        with torch.no_grad():
            # Use average of top vocabulary words for each topic (placeholder)
            model.topic_embeddings.data = torch.randn(args.n_topics, embed_dim)
        
        # Train model
        logger.info(f"Training ETM for {args.epochs} epochs...")
        train_etm(model, dataloader, epochs=args.epochs)
        
        # Generate outputs
        model.eval()
        with torch.no_grad():
            # Get theta (document-topic distributions)
            all_theta = []
            for batch_bow in dataloader:
                batch_bow = batch_bow[0].to(device)
                mu, logvar = model.encode(batch_bow)
                theta = model.reparameterize(mu, logvar)
                theta = torch.softmax(theta, dim=-1)
                all_theta.append(theta.cpu().numpy())
            
            theta_matrix = np.vstack(all_theta)  # (N, K)
            
            # Get beta (topic-word distributions)
            topic_embeddings = model.topic_embeddings.cpu().numpy()  # (K, embed_dim)
            
            # Simple way to get beta: use topic embeddings to generate word distributions
            beta_matrix = np.zeros((args.n_topics, vocab_size))
            for k in range(args.n_topics):
                # Compute similarity between topic embedding and word embeddings
                similarities = np.dot(topic_embeddings[k], vocab_embeddings.T)
                beta_matrix[k] = np.exp(similarities) / np.sum(np.exp(similarities))
            
            # Alpha is topic embeddings
            alpha_matrix = topic_embeddings  # (K, embed_dim)
        
        # Generate topics JSON
        topics = []
        for k in range(args.n_topics):
            # Get top words for this topic
            top_word_indices = np.argsort(beta_matrix[k])[-10:][::-1]
            top_words = [vocab_list[i] for i in top_word_indices]
            
            topic = {
                "id": k,
                "name": f"Topic_{k}",
                "keywords": top_words,
                "proportion": float(np.mean(theta_matrix[:, k]))
            }
            topics.append(topic)
        
        # Save outputs
        np.save(args.theta_output, theta_matrix.astype(np.float32))
        np.save(args.beta_output, beta_matrix.astype(np.float32))
        np.save(args.alpha_output, alpha_matrix.astype(np.float32))
        
        with open(args.topics_output, 'w', encoding='utf-8') as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated theta matrix: {theta_matrix.shape}")
        logger.info(f"Generated beta matrix: {beta_matrix.shape}")
        logger.info(f"Generated alpha matrix: {alpha_matrix.shape}")
        logger.info(f"Generated {len(topics)} topics")
        
    except Exception as e:
        logger.error(f"Error processing job {args.job_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()

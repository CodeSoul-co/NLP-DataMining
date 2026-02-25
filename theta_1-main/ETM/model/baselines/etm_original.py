"""
Original ETM (Embedded Topic Model) - 原始ETM实现

这是原始ETM论文的实现，作为Baseline使用。
与你的ETM不同，原始ETM:
1. 使用BOW作为编码器输入（不是Qwen doc embedding）
2. 使用Word2Vec/GloVe词向量（不是Qwen word embedding）

参考: Dieng et al., "Topic Modeling in Embedding Spaces", TACL 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..base import NeuralTopicModel


class OriginalETM(NeuralTopicModel):
    """
    原始ETM实现 - 作为Baseline
    
    架构:
        BOW -> Encoder -> theta (主题分布)
        theta * beta -> word_dist
        beta = softmax(topic_embeddings @ word_embeddings.T)
    
    与你的ETM的区别:
    - 编码器输入: BOW (不是Qwen doc embedding)
    - 词向量: Word2Vec/GloVe或可训练 (不是Qwen word embedding)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        embedding_dim: int = 300,  # Word2Vec/GloVe通常是300维
        hidden_dim: int = 800,
        dropout: float = 0.5,
        activation: str = 'softplus',
        word_embeddings: Optional[np.ndarray] = None,
        train_embeddings: bool = True,
        kl_weight: float = 1.0,
        dev_mode: bool = False,
        # 以下参数为了接口兼容
        doc_embedding_dim: int = None,
        word_embedding_dim: int = None,
        **kwargs
    ):
        """
        初始化原始ETM
        
        Args:
            vocab_size: 词表大小
            num_topics: 主题数量
            embedding_dim: 词向量维度 (Word2Vec=300, GloVe=300)
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            activation: 激活函数
            word_embeddings: 预训练词向量 (vocab_size, embedding_dim)
            train_embeddings: 是否训练词向量
            kl_weight: KL散度权重
            dev_mode: 调试模式
        """
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self._vocab_size = vocab_size
        self._num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.kl_weight = kl_weight
        self.dev_mode = dev_mode
        self.train_embeddings = train_embeddings
        
        # 激活函数
        self.activation = self._get_activation(activation)
        
        # 词向量矩阵 rho (V x E)
        if word_embeddings is not None:
            # 使用预训练词向量
            self.rho = nn.Parameter(
                torch.tensor(word_embeddings, dtype=torch.float32),
                requires_grad=train_embeddings
            )
            self.embedding_dim = word_embeddings.shape[1]
        else:
            # 可训练词向量 - 使用 Xavier 初始化
            self.rho = nn.Parameter(
                torch.empty(vocab_size, embedding_dim),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.rho)
        
        # 主题向量矩阵 alpha (K x E)
        # 使用 Xavier 初始化以获得更好的主题分化
        self.alphas = nn.Parameter(
            torch.empty(num_topics, self.embedding_dim)
        )
        nn.init.xavier_uniform_(self.alphas)
        
        # 编码器: BOW -> theta
        # 原始ETM使用BOW作为输入
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # VAE参数
        self.mu_layer = nn.Linear(hidden_dim, num_topics)
        self.logvar_layer = nn.Linear(hidden_dim, num_topics)
        
        if self.dev_mode:
            print(f"[DEV] OriginalETM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   embedding_dim={self.embedding_dim}")
            print(f"[DEV]   hidden_dim={hidden_dim}")
            print(f"[DEV]   train_embeddings={train_embeddings}")
    
    def _get_activation(self, act: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'softplus': nn.Softplus(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }
        return activations.get(act, nn.Softplus())
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码BOW到主题分布
        
        Args:
            bow: BOW矩阵 (batch, vocab_size)
            
        Returns:
            theta: 主题分布 (batch, num_topics)
            mu: 均值
            logvar: 对数方差
        """
        # 归一化BOW
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # 编码
        hidden = self.encoder(bow_norm)
        
        # VAE参数
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        # 重参数化
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # Softmax得到主题分布
        theta = F.softmax(z, dim=-1)
        
        return theta, mu, logvar
    
    def get_beta(self) -> torch.Tensor:
        """
        计算主题-词分布 beta
        
        beta = softmax(alpha @ rho.T)
        
        Returns:
            beta: (num_topics, vocab_size)
        """
        # alpha: (K, E), rho: (V, E)
        # logits: (K, V)
        logits = torch.mm(self.alphas, self.rho.t())
        beta = F.softmax(logits, dim=-1)
        return beta
    
    def decode(self, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        解码主题分布到词分布
        
        Args:
            theta: 主题分布 (batch, num_topics)
            beta: 主题-词分布 (num_topics, vocab_size)
            
        Returns:
            log_word_dist: 对数词分布 (batch, vocab_size)
        """
        word_dist = torch.mm(theta, beta)
        log_word_dist = torch.log(word_dist + 1e-10)
        return log_word_dist
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,  # 为了接口兼容，但不使用
        bow: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            doc_embeddings: 文档embedding (不使用，仅为接口兼容)
            bow: BOW矩阵 (batch, vocab_size)
            
        Returns:
            Dict containing:
                - loss / total_loss: 总损失
                - recon_loss: 重建损失
                - kl_loss: KL散度损失
                - theta: 主题分布
        """
        # 编码
        theta, mu, logvar = self.encode(bow)
        
        # 获取beta
        beta = self.get_beta()
        
        # 解码
        log_word_dist = self.decode(theta, beta)
        
        # 重建损失
        recon_loss = -(bow * log_word_dist).sum(dim=1).mean()
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # 总损失
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'theta': theta,
            'beta': beta
        }
    
    def get_theta(self, bow: torch.Tensor = None, **kwargs) -> np.ndarray:
        """获取文档-主题分布"""
        if bow is None:
            raise ValueError("bow is required for OriginalETM.get_theta()")
        
        self.eval()
        with torch.no_grad():
            theta, _, _ = self.encode(bow)
        return theta.cpu().numpy()
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        获取主题词
        
        Args:
            vocab: 词汇表
            top_k: 每个主题返回的词数
            
        Returns:
            主题词字典 {topic_id: [word1, word2, ...]}
        """
        with torch.no_grad():
            beta = self.get_beta().cpu().numpy()
        
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        
        return topic_words
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'kl_weight': self.kl_weight,
            'train_embeddings': self.train_embeddings
        }


def train_word2vec_embeddings(
    texts: List[str],
    vocab: List[str],
    embedding_dim: int = 300,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4
) -> np.ndarray:
    """
    使用gensim训练Word2Vec词向量
    
    Args:
        texts: 文本列表
        vocab: 词汇表
        embedding_dim: 词向量维度
        window: 窗口大小
        min_count: 最小词频
        workers: 工作线程数
        
    Returns:
        embeddings: (vocab_size, embedding_dim)
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("gensim not installed. Install with: pip install gensim")
    
    print(f"Training Word2Vec embeddings (dim={embedding_dim})...")
    
    # 分词
    tokenized_texts = [text.split() for text in texts]
    
    # 训练Word2Vec
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10
    )
    
    # 提取词向量
    embeddings = np.zeros((len(vocab), embedding_dim))
    oov_count = 0
    
    for i, word in enumerate(vocab):
        if word in model.wv:
            embeddings[i] = model.wv[word]
        else:
            # OOV词使用随机初始化
            embeddings[i] = np.random.randn(embedding_dim) * 0.01
            oov_count += 1
    
    print(f"Word2Vec training complete. OOV words: {oov_count}/{len(vocab)}")
    return embeddings


def create_original_etm(
    vocab_size: int,
    num_topics: int = 20,
    embedding_dim: int = 300,
    word_embeddings: Optional[np.ndarray] = None,
    **kwargs
) -> OriginalETM:
    """
    创建原始ETM模型的工厂函数
    
    Args:
        vocab_size: 词表大小
        num_topics: 主题数
        embedding_dim: 词向量维度
        word_embeddings: 预训练词向量
        **kwargs: 其他参数
        
    Returns:
        OriginalETM模型实例
    """
    return OriginalETM(
        vocab_size=vocab_size,
        num_topics=num_topics,
        embedding_dim=embedding_dim,
        word_embeddings=word_embeddings,
        **kwargs
    )

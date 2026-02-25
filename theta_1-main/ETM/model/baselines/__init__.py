"""
Baseline Topic Models

Traditional and neural baseline models for comparison with THETA:

Traditional Models:
- LDA: Latent Dirichlet Allocation (sklearn implementation)
- HDP: Hierarchical Dirichlet Process (auto topic number)
- STM: Structural Topic Model (with covariates)
- BTM: Biterm Topic Model (for short texts)

Neural Models:
- CTM: Contextualized Topic Model (SBERT + VAE)
- ETM: Embedded Topic Model (Word2Vec + VAE)
- DTM: Dynamic Topic Model (time-aware)
- NTM: Neural Topic Model variants:
  - NVDM: Neural Variational Document Model
  - GSM: Gaussian Softmax Model
  - ProdLDA: Product of Experts LDA
- BERTopic: BERT-based topic modeling
"""

# Traditional models
from .lda import LDA, SklearnLDA, NeuralLDA, create_lda
from .hdp import HDP, create_hdp
from .stm import STM, create_stm
from .btm import BTM, create_btm

# Neural models
from .ctm import CTM, ZeroShotTM, CombinedTM, create_ctm
from .dtm import DTM, create_dtm
from .etm_original import OriginalETM, create_original_etm, train_word2vec_embeddings
from .ntm import NVDM, GSM, ProdLDA, create_nvdm, create_gsm, create_prodlda
from .bertopic_model import BERTopicModel, create_bertopic

__all__ = [
    # LDA
    'LDA', 'SklearnLDA', 'NeuralLDA', 'create_lda',
    # HDP
    'HDP', 'create_hdp',
    # STM
    'STM', 'create_stm',
    # BTM
    'BTM', 'create_btm',
    # CTM
    'CTM', 'ZeroShotTM', 'CombinedTM', 'create_ctm',
    # DTM
    'DTM', 'create_dtm',
    # Original ETM
    'OriginalETM', 'create_original_etm', 'train_word2vec_embeddings',
    # NTM variants
    'NVDM', 'GSM', 'ProdLDA', 'create_nvdm', 'create_gsm', 'create_prodlda',
    # BERTopic
    'BERTopicModel', 'create_bertopic',
]

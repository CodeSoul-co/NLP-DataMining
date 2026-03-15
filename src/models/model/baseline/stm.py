"""
STM (Structural Topic Model)

A topic model that incorporates document-level metadata (covariates) to model
how topic prevalence and content vary across documents.

Key Features:
- Incorporates document metadata (time, author, category, etc.)
- Models topic prevalence as function of covariates
- Models topic content as function of covariates
- Useful for analyzing how topics vary across groups

Reference:
- Roberts et al., "A Model of Text for Experimentation in the Social Sciences", JASA 2016
- Roberts et al., "stm: R Package for Structural Topic Models", JSS 2019

IMPORTANT: STM *requires* document-level covariates (metadata). Without covariates,
STM degenerates into a Correlated Topic Model (CTM) with logistic-normal prior.
If your dataset has no covariates, use CTM or LDA instead.

Implementation Strategy:
- If R + rpy2 + stm package are available: uses the original R implementation
- Otherwise: uses a Python implementation with logistic-normal prior + covariate regression
- In both cases, covariates are REQUIRED. The model will refuse to train without them.
"""

import numpy as np
from typing import Dict, Optional, List, Any, Tuple, Union
from sklearn.preprocessing import StandardScaler
import warnings
import json

from ..base import TraditionalTopicModel


class CovariatesRequiredError(ValueError):
    """Raised when STM is called without required covariates metadata."""
    
    def __init__(self, dataset: str = None):
        msg = (
            "STM (Structural Topic Model) requires document-level covariates (metadata) "
            "such as time, author, source, category, etc. "
            "Without covariates, STM has no advantage over CTM/LDA.\n\n"
        )
        if dataset:
            msg += f"Dataset '{dataset}' does not provide covariates metadata.\n\n"
        msg += (
            "Options:\n"
            "  1. Add covariates to your dataset (e.g., timestamp, source, category columns)\n"
            "  2. Use CTM (Correlated Topic Model) instead — same logistic-normal prior, no covariates needed\n"
            "  3. Use LDA as the traditional baseline\n\n"
            "To add covariates, include a 'covariate_columns' field in DATASET_CONFIGS in config.py:\n"
            "  'my_dataset': {\n"
            "      ...\n"
            "      'covariate_columns': ['year', 'source', 'category'],\n"
            "  }"
        )
        super().__init__(msg)


def _check_r_stm_available() -> bool:
    """Check if R + rpy2 + stm package are available."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        importr('stm')
        return True
    except (ImportError, Exception):
        return False


class STM(TraditionalTopicModel):
    """
    Structural Topic Model
    
    STM models how document-level covariates affect topic prevalence and content.
    It uses a logistic-normal prior (instead of Dirichlet) where the mean is a
    linear function of covariates.
    
    REQUIRES covariates. Will raise CovariatesRequiredError if none provided.
    
    If R's stm package is available (via rpy2), uses the original implementation.
    Otherwise, uses a Python approximation with logistic-normal variational inference.
    
    Attributes:
        num_topics: Number of topics
        max_iter: Maximum iterations
    """
    
    # Class-level flag
    REQUIRES_COVARIATES = True
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        max_iter: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize STM model.
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics
            max_iter: Maximum iterations
            random_state: Random seed
        """
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self.max_iter = max_iter
        self.random_state = random_state
        self._use_r = _check_r_stm_available()
        
        # Covariate information
        self._covariates = None
        self._covariate_names = None
        self._covariate_effects = None
        
        # Results
        self._theta = None
        self._beta = None
        self._vocab = None
    
    @staticmethod
    def check_requirements(dataset: str = None, covariates: np.ndarray = None,
                           dataset_config: dict = None) -> Tuple[bool, str]:
        """
        Check if STM requirements are met before attempting training.
        
        Args:
            dataset: Dataset name (for error messages)
            covariates: Covariates array, if already loaded
            dataset_config: Dataset config dict from DATASET_CONFIGS
            
        Returns:
            (can_run: bool, reason: str)
        """
        # Check if covariates are provided directly
        if covariates is not None and covariates.shape[1] > 0:
            return True, "Covariates available"
        
        # Check if dataset config specifies covariate columns
        if dataset_config and dataset_config.get('covariate_columns'):
            return True, f"Covariates configured: {dataset_config['covariate_columns']}"
        
        reason = (
            f"STM requires covariates (document-level metadata). "
            f"Dataset '{dataset or 'unknown'}' does not provide covariate_columns in its config. "
            f"Use CTM or LDA instead, or add covariate_columns to DATASET_CONFIGS."
        )
        return False, reason
    
    def fit(
        self,
        bow_matrix: np.ndarray,
        covariates: np.ndarray = None,
        covariate_names: Optional[List[str]] = None,
        vocab: Optional[List[str]] = None,
        dataset: str = None,
        **kwargs
    ) -> 'STM':
        """
        Fit STM model. Covariates are REQUIRED.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
            covariates: Document covariates, shape (num_docs, num_covariates). REQUIRED.
            covariate_names: Names of covariates
            vocab: Vocabulary list
            dataset: Dataset name (for error messages)
        
        Returns:
            self
            
        Raises:
            CovariatesRequiredError: If covariates are not provided
        """
        if covariates is None or (hasattr(covariates, 'shape') and 
                                   (covariates.ndim < 2 or covariates.shape[1] == 0)):
            raise CovariatesRequiredError(dataset=dataset)
        
        self._covariates = covariates
        self._covariate_names = covariate_names
        self._vocab = vocab
        
        n_docs = bow_matrix.shape[0]
        if covariates.shape[0] != n_docs:
            raise ValueError(
                f"Covariates rows ({covariates.shape[0]}) must match "
                f"documents ({n_docs})"
            )
        
        print(f"  STM: {n_docs} documents, {covariates.shape[1]} covariates")
        if covariate_names:
            print(f"  Covariates: {', '.join(covariate_names)}")
        
        if self._use_r:
            self._fit_r(bow_matrix, covariates, covariate_names, vocab)
        else:
            self._fit_python(bow_matrix, covariates, covariate_names, vocab)
        
        return self
    
    def _fit_r(self, bow_matrix, covariates, covariate_names, vocab):
        """Fit using R's stm package via rpy2."""
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        
        numpy2ri.activate()
        stm_pkg = importr('stm')
        
        # Convert BOW to R format (list of documents)
        # R's stm expects a specific format; we use the slam sparse matrix approach
        r_docs = self._bow_to_r_docs(bow_matrix)
        r_vocab = ro.StrVector(vocab) if vocab else ro.StrVector([str(i) for i in range(bow_matrix.shape[1])])
        
        # Build prevalence formula from covariates
        cov_df = self._covariates_to_r_df(covariates, covariate_names)
        
        # Run STM
        formula_str = "~ " + " + ".join(covariate_names or [f"V{i+1}" for i in range(covariates.shape[1])])
        
        result = stm_pkg.stm(
            documents=r_docs,
            vocab=r_vocab,
            K=self.num_topics,
            prevalence=ro.Formula(formula_str),
            data=cov_df,
            max_em_its=self.max_iter,
            seed=self.random_state,
            verbose=True
        )
        
        # Extract theta and beta
        self._theta = np.array(result.rx2('theta'))
        beta_log = np.array(result.rx2('beta')[0])
        self._beta = np.exp(beta_log)
        self._beta = self._beta / self._beta.sum(axis=1, keepdims=True)
        
        # Analyze covariate effects
        self._analyze_covariate_effects_r(result, stm_pkg, covariates, covariate_names)
        
        numpy2ri.deactivate()
    
    def _fit_python(self, bow_matrix, covariates, covariate_names, vocab):
        """
        Python STM approximation using logistic-normal prior with covariate regression.
        
        Unlike plain LDA (Dirichlet prior), this uses:
        1. Logistic-normal prior for topic proportions (like CTM)
        2. Covariate-dependent mean: mu_d = X_d @ Gamma (prevalence covariates)
        3. Variational EM with Laplace approximation for the E-step
        """
        np.random.seed(self.random_state)
        
        n_docs, V = bow_matrix.shape
        K = self.num_topics
        n_cov = covariates.shape[1]
        
        # Standardize covariates (add intercept)
        scaler = StandardScaler()
        X = np.hstack([np.ones((n_docs, 1)), scaler.fit_transform(covariates)])  # (n_docs, n_cov+1)
        self._scaler = scaler
        n_cov_with_intercept = X.shape[1]
        
        # Initialize parameters
        # Gamma: covariate coefficients for topic prevalence, shape (n_cov+1, K-1)
        Gamma = np.zeros((n_cov_with_intercept, K - 1))
        # Sigma: topic covariance matrix, shape (K-1, K-1)
        Sigma = np.eye(K - 1) * 0.5
        # Beta: topic-word distributions, shape (K, V) — initialize with LDA
        from sklearn.decomposition import LatentDirichletAllocation
        lda_init = LatentDirichletAllocation(
            n_components=K, max_iter=10, random_state=self.random_state
        )
        lda_init.fit(bow_matrix)
        Beta = lda_init.components_ / lda_init.components_.sum(axis=1, keepdims=True)
        
        # Variational parameters for each document
        # eta_d: logistic-normal parameter, shape (n_docs, K-1)
        eta = X @ Gamma  # Initialize at prior mean
        
        # EM iterations
        prev_elbo = -np.inf
        for iteration in range(self.max_iter):
            # === E-step: update variational parameters eta_d via Newton's method ===
            theta = np.zeros((n_docs, K))
            for d in range(n_docs):
                doc_bow = bow_matrix[d]
                if hasattr(doc_bow, 'toarray'):
                    doc_bow = doc_bow.toarray().flatten()
                doc_total = doc_bow.sum()
                if doc_total == 0:
                    theta[d] = 1.0 / K
                    continue
                
                mu_d = X[d] @ Gamma  # (K-1,)
                eta_d = mu_d.copy()
                
                # Newton-Raphson (3 iterations for speed)
                for _ in range(3):
                    # Softmax to get theta
                    eta_full = np.append(eta_d, 0.0)  # K-dim, last is reference
                    eta_full -= eta_full.max()
                    exp_eta = np.exp(eta_full)
                    theta_d = exp_eta / exp_eta.sum()
                    
                    # Gradient: doc_counts * (I_{-K} - theta_{-K}) - Sigma_inv @ (eta_d - mu_d)
                    Sigma_inv = np.linalg.solve(Sigma, np.eye(K - 1))
                    
                    # Expected word counts contribution
                    grad = doc_total * (doc_bow @ Beta[:, :].T)  # not used directly
                    
                    # Simplified gradient for logistic-normal
                    theta_mk = theta_d[:-1]  # first K-1
                    diff = eta_d - mu_d
                    gradient = doc_total * (theta_mk - theta_d[:-1]) - Sigma_inv @ diff
                    
                    # Approximate Hessian (diagonal)
                    hess_diag = -doc_total * theta_mk * (1 - theta_mk) - np.diag(Sigma_inv)
                    
                    # Update
                    step = gradient / (hess_diag - 1e-8)
                    eta_d -= 0.5 * step  # Damped update
                
                # Final theta
                eta_full = np.append(eta_d, 0.0)
                eta_full -= eta_full.max()
                exp_eta = np.exp(eta_full)
                theta[d] = exp_eta / exp_eta.sum()
                eta[d] = eta_d
            
            # === M-step ===
            # Update Gamma (covariate coefficients)
            # Gamma = (X^T X)^{-1} X^T eta
            XtX = X.T @ X + 1e-6 * np.eye(n_cov_with_intercept)
            Gamma = np.linalg.solve(XtX, X.T @ eta)
            
            # Update Sigma (topic covariance)
            residuals = eta - X @ Gamma
            Sigma = (residuals.T @ residuals) / n_docs + 1e-6 * np.eye(K - 1)
            
            # Update Beta (topic-word distributions)
            # Weighted word counts
            Beta_new = theta.T @ bow_matrix if not hasattr(bow_matrix, 'toarray') else theta.T @ bow_matrix
            if hasattr(Beta_new, 'toarray'):
                Beta_new = np.asarray(Beta_new)
            Beta_new = np.maximum(Beta_new, 1e-10)
            Beta = Beta_new / Beta_new.sum(axis=1, keepdims=True)
            
            # Check convergence (simplified ELBO proxy)
            elbo = np.sum(theta * np.log(Beta @ bow_matrix.T + 1e-10).T) if n_docs < 50000 else 0
            if iteration > 0 and iteration % 10 == 0:
                print(f"  STM iteration {iteration}/{self.max_iter}")
            
            if abs(elbo - prev_elbo) < 1e-4 and elbo != 0:
                print(f"  STM converged at iteration {iteration}")
                break
            prev_elbo = elbo
        
        self._theta = theta
        self._beta = Beta
        self._Gamma = Gamma
        self._Sigma = Sigma
        
        # Compute covariate effects from Gamma
        self._compute_covariate_effects(Gamma, covariate_names)
        
        print(f"  STM training completed ({iteration + 1} iterations)")
    
    def _compute_covariate_effects(self, Gamma, covariate_names):
        """Compute interpretable covariate effects from Gamma matrix."""
        self._covariate_effects = {}
        # Gamma shape: (n_cov+1, K-1), first row is intercept
        n_cov = Gamma.shape[0] - 1  # exclude intercept
        
        for k in range(self.num_topics - 1):
            effects = {}
            for c in range(n_cov):
                name = covariate_names[c] if covariate_names and c < len(covariate_names) else f'cov_{c}'
                effects[name] = {
                    'coefficient': float(Gamma[c + 1, k]),  # +1 to skip intercept
                    'intercept': float(Gamma[0, k]),
                }
            self._covariate_effects[k] = effects
    
    def _bow_to_r_docs(self, bow_matrix):
        """Convert BOW matrix to R stm document format."""
        import rpy2.robjects as ro
        
        docs = []
        for d in range(bow_matrix.shape[0]):
            row = bow_matrix[d]
            if hasattr(row, 'toarray'):
                row = row.toarray().flatten()
            nonzero = np.nonzero(row)[0]
            if len(nonzero) == 0:
                continue
            # R stm format: matrix with 2 rows (word_index+1, count)
            indices = ro.IntVector(nonzero + 1)  # R is 1-indexed
            counts = ro.IntVector(row[nonzero].astype(int))
            doc = ro.r.rbind(indices, counts)
            docs.append(doc)
        
        return ro.ListVector({str(i): d for i, d in enumerate(docs)})
    
    def _covariates_to_r_df(self, covariates, covariate_names):
        """Convert covariates to R data.frame."""
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        import pandas as pd
        
        names = covariate_names or [f'V{i+1}' for i in range(covariates.shape[1])]
        df = pd.DataFrame(covariates, columns=names)
        
        pandas2ri.activate()
        r_df = pandas2ri.py2rpy(df)
        pandas2ri.deactivate()
        
        return r_df
    
    def _analyze_covariate_effects_r(self, result, stm_pkg, covariates, covariate_names):
        """Extract covariate effects from R stm result."""
        import rpy2.robjects as ro
        
        try:
            effect = stm_pkg.estimateEffect(
                ro.Formula("~ " + " + ".join(covariate_names or [f'V{i+1}' for i in range(covariates.shape[1])])),
                result,
                metadata=self._covariates_to_r_df(covariates, covariate_names)
            )
            self._covariate_effects = {'r_effect_object': effect}
        except Exception as e:
            warnings.warn(f"Could not estimate covariate effects: {e}")
    
    def transform(self, bow_matrix: np.ndarray) -> np.ndarray:
        """
        Transform documents to topic distributions.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
        
        Returns:
            theta: Document-topic distribution, shape (num_docs, num_topics)
        """
        if self._beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Simple projection using learned beta
        theta = bow_matrix @ self._beta.T
        if hasattr(theta, 'toarray'):
            theta = np.asarray(theta)
        theta = np.maximum(theta, 1e-10)
        theta = theta / theta.sum(axis=1, keepdims=True)
        return theta
    
    def get_theta(self, bow_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Get document-topic distribution."""
        if bow_matrix is not None:
            return self.transform(bow_matrix)
        return self._theta
    
    def get_beta(self) -> np.ndarray:
        """Get topic-word distribution."""
        return self._beta
    
    def get_covariate_effects(self, topic_id: Optional[int] = None) -> Dict:
        """
        Get covariate effects on topic prevalence.
        
        Args:
            topic_id: Specific topic (None for all topics)
        
        Returns:
            Dictionary of covariate effects
        """
        if self._covariate_effects is None:
            return {}
        
        if topic_id is not None:
            return self._covariate_effects.get(topic_id, {})
        
        return self._covariate_effects
    
    def estimate_effect(
        self,
        covariate_idx: int,
        values: Optional[np.ndarray] = None,
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Estimate the effect of a covariate on topic prevalence.
        
        Args:
            covariate_idx: Index of covariate
            values: Specific values to evaluate (None for range)
            n_points: Number of points if values not specified
        
        Returns:
            Dictionary with 'values', 'effects' for each topic
        """
        if self._covariate_effects is None:
            raise ValueError("No covariate effects computed. Fit with covariates first.")
        
        if self._covariates is None:
            raise ValueError("No covariates available.")
        
        # Get range of covariate values
        if values is None:
            cov_min = self._covariates[:, covariate_idx].min()
            cov_max = self._covariates[:, covariate_idx].max()
            values = np.linspace(cov_min, cov_max, n_points)
        
        # Compute effects using Gamma
        effects = {}
        if hasattr(self, '_Gamma'):
            for k in range(self.num_topics - 1):
                coef = self._Gamma[covariate_idx + 1, k]  # +1 for intercept
                intercept = self._Gamma[0, k]
                effects[k] = intercept + coef * values
        
        return {
            'values': values,
            'effects': effects,
            'covariate_name': self._covariate_names[covariate_idx] if self._covariate_names else f'covariate_{covariate_idx}'
        }
    
    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
        vocab: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top words for a topic.
        
        Args:
            topic_id: Topic index
            top_n: Number of top words
            vocab: Vocabulary list
        
        Returns:
            List of (word, probability) tuples
        """
        if self._beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        v = vocab or self._vocab
        topic_dist = self._beta[topic_id]
        top_indices = np.argsort(topic_dist)[::-1][:top_n]
        
        if v is not None:
            return [(v[i], topic_dist[i]) for i in top_indices]
        else:
            return [(str(i), topic_dist[i]) for i in top_indices]
    
    def find_topics_by_covariate(
        self,
        covariate_idx: int,
        direction: str = 'positive'
    ) -> List[Tuple[int, float]]:
        """
        Find topics most affected by a covariate.
        
        Args:
            covariate_idx: Index of covariate
            direction: 'positive', 'negative', or 'absolute'
        
        Returns:
            List of (topic_id, coefficient) sorted by effect size
        """
        if not hasattr(self, '_Gamma') or self._Gamma is None:
            raise ValueError("No covariate effects computed.")
        
        effects = []
        for k in range(self.num_topics - 1):
            coef = self._Gamma[covariate_idx + 1, k]
            effects.append((k, float(coef)))
        
        if direction == 'positive':
            effects.sort(key=lambda x: x[1], reverse=True)
        elif direction == 'negative':
            effects.sort(key=lambda x: x[1])
        else:  # absolute
            effects.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return effects


def create_stm(vocab_size: int, num_topics: int = 20, **kwargs) -> STM:
    """Create STM model."""
    return STM(vocab_size=vocab_size, num_topics=num_topics, **kwargs)

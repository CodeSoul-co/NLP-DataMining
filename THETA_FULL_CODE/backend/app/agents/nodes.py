"""
LangGraph Node Definitions
Each node corresponds to a stage in the ETM pipeline
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from scipy import sparse

# Add ETM to path
ETM_PATH = Path("/root/autodl-tmp/ETM")
sys.path.insert(0, str(ETM_PATH))

from ..schemas.agent import AgentState, StepStatus
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


def _update_state(state: AgentState, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to update state with timestamp"""
    updates["updated_at"] = datetime.now().isoformat()
    return updates


def _add_log(state: AgentState, step: str, status: str, message: str, **kwargs) -> Dict[str, Any]:
    """Add log entry to state"""
    logs = list(state.get("logs", []))
    logs.append({
        "step": step,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    })
    return {"logs": logs}


async def preprocess_node(state: AgentState, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Node 1: Preprocessor
    Generates BOW matrix and vocabulary from raw text data
    """
    step_name = "preprocess"
    logger.info(f"[{state['task_id']}] Starting preprocessing...")
    
    try:
        from config import PipelineConfig
        from bow.vocab_builder import VocabBuilder
        from bow.bow_generator import BOWGenerator
        from model.vocab_embedder import VocabEmbedder
        import pandas as pd
        
        dataset = state["dataset"]
        mode = state["mode"]
        
        config = PipelineConfig()
        config.data.dataset = dataset
        config.embedding.mode = mode
        config.bow.vocab_size = state.get("vocab_size", 5000)
        config.gpu_id = settings.GPU_ID
        config.dev_mode = state.get("dev_mode", False)
        
        result_dir = settings.get_result_path(dataset, mode)
        bow_dir = result_dir / "bow"
        bow_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = config.data.raw_data_path
        logger.info(f"Loading texts from {csv_path}")
        
        if callback:
            await callback(step_name, "in_progress", f"Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        text_col = None
        for col in ['cleaned_content', 'clean_text', 'text', 'content', 'Text']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")
        
        texts = df[text_col].fillna('').astype(str).tolist()
        logger.info(f"Loaded {len(texts)} documents")
        
        if callback:
            await callback(step_name, "in_progress", f"Building vocabulary from {len(texts)} documents")
        
        vocab_builder = VocabBuilder(
            max_vocab_size=config.bow.vocab_size,
            min_doc_freq=config.bow.min_doc_freq,
            max_doc_freq_ratio=config.bow.max_doc_freq_ratio,
            dev_mode=config.dev_mode
        )
        
        vocab_builder.add_documents(texts, dataset_name=dataset)
        vocab_builder.build_vocab()
        
        if callback:
            await callback(step_name, "in_progress", "Generating BOW matrix")
        
        bow_generator = BOWGenerator(vocab_builder, dev_mode=config.dev_mode)
        bow_output = bow_generator.generate_bow(texts, dataset_name=dataset)
        
        vocab = vocab_builder.get_vocab_list()
        bow_matrix = bow_output.bow_matrix
        
        logger.info(f"BOW shape: {bow_matrix.shape}, vocab_size: {len(vocab)}")
        
        if callback:
            await callback(step_name, "in_progress", f"Generating embeddings for {len(vocab)} vocabulary words")
        
        embedder = VocabEmbedder(
            model_path=str(settings.QWEN_MODEL_PATH),
            device=settings.DEVICE,
            batch_size=64,
            dev_mode=config.dev_mode
        )
        
        vocab_embeddings = embedder.embed_vocab(vocab)
        logger.info(f"Vocab embeddings shape: {vocab_embeddings.shape}")
        
        sparse.save_npz(str(bow_dir / "bow_matrix.npz"), bow_matrix)
        np.save(str(bow_dir / "vocab_embeddings.npy"), vocab_embeddings)
        with open(bow_dir / "vocab.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab))
        
        logger.info(f"BOW saved to {bow_dir}")
        
        if callback:
            await callback(step_name, "completed", f"Preprocessing complete. BOW: {bow_matrix.shape}")
        
        return _update_state(state, {
            "current_step": "embedding",
            "preprocess_completed": True,
            "bow_dir": str(bow_dir),
            "bow_shape": bow_matrix.shape,
            "vocab_size_actual": len(vocab),
            **_add_log(state, step_name, "completed", f"BOW matrix: {bow_matrix.shape}, vocab: {len(vocab)}")
        })
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        if callback:
            await callback(step_name, "failed", str(e))
        return _update_state(state, {
            "status": "failed",
            "error_message": str(e),
            **_add_log(state, step_name, "failed", str(e))
        })


async def embedding_node(state: AgentState, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Node 2: Embedder
    Loads pre-computed document embeddings
    """
    step_name = "embedding"
    logger.info(f"[{state['task_id']}] Loading document embeddings...")
    
    try:
        dataset = state["dataset"]
        mode = state["mode"]
        
        result_dir = settings.get_result_path(dataset, mode)
        embeddings_dir = result_dir / "embeddings"
        
        if callback:
            await callback(step_name, "in_progress", f"Loading embeddings from {embeddings_dir}")
        
        emb_path = embeddings_dir / f"{dataset}_{mode}_embeddings.npy"
        
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        
        embeddings = np.load(str(emb_path))
        logger.info(f"Loaded embeddings: {embeddings.shape}")
        
        if callback:
            await callback(step_name, "completed", f"Loaded embeddings: {embeddings.shape}")
        
        return _update_state(state, {
            "current_step": "training",
            "embedding_completed": True,
            "embeddings_dir": str(embeddings_dir),
            "doc_embeddings_shape": embeddings.shape,
            **_add_log(state, step_name, "completed", f"Doc embeddings: {embeddings.shape}")
        })
        
    except Exception as e:
        logger.error(f"Embedding loading failed: {e}")
        if callback:
            await callback(step_name, "failed", str(e))
        return _update_state(state, {
            "status": "failed",
            "error_message": str(e),
            **_add_log(state, step_name, "failed", str(e))
        })


async def training_node(state: AgentState, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Node 3: Trainer
    Trains the ETM model
    """
    step_name = "training"
    logger.info(f"[{state['task_id']}] Starting ETM training...")
    
    try:
        import torch
        from config import PipelineConfig
        from model.etm import ETM
        from data.dataloader import ETMDataset
        from torch.utils.data import DataLoader, random_split
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.GPU_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dataset = state["dataset"]
        mode = state["mode"]
        
        result_dir = settings.get_result_path(dataset, mode)
        bow_dir = result_dir / "bow"
        model_dir = result_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if callback:
            await callback(step_name, "in_progress", "Loading data matrices")
        
        bow_matrix = sparse.load_npz(str(bow_dir / "bow_matrix.npz"))
        vocab_embeddings = np.load(str(bow_dir / "vocab_embeddings.npy"))
        
        embeddings_dir = result_dir / "embeddings"
        doc_embeddings = np.load(str(embeddings_dir / f"{dataset}_{mode}_embeddings.npy"))
        
        logger.info(f"BOW: {bow_matrix.shape}, Doc emb: {doc_embeddings.shape}, Vocab emb: {vocab_embeddings.shape}")
        
        config = PipelineConfig()
        config.model.num_topics = state.get("num_topics", 20)
        config.model.epochs = state.get("epochs", 50)
        config.model.batch_size = state.get("batch_size", 64)
        config.model.learning_rate = state.get("learning_rate", 0.002)
        config.model.hidden_dim = state.get("hidden_dim", 512)
        config.dev_mode = state.get("dev_mode", False)
        
        if callback:
            await callback(step_name, "in_progress", f"Creating dataset and model (topics={config.model.num_topics})")
        
        etm_dataset = ETMDataset(
            doc_embeddings=doc_embeddings,
            bow_matrix=bow_matrix,
            normalize_bow=True,
            dev_mode=config.dev_mode
        )
        
        n_total = len(etm_dataset)
        n_train = int(n_total * config.model.train_ratio)
        n_val = int(n_total * config.model.val_ratio)
        n_test = n_total - n_train - n_val
        
        train_dataset, val_dataset, test_dataset = random_split(
            etm_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.model.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=False)
        
        vocab_size = bow_matrix.shape[1]
        word_emb_tensor = torch.tensor(vocab_embeddings, dtype=torch.float32)
        
        model = ETM(
            vocab_size=vocab_size,
            num_topics=config.model.num_topics,
            doc_embedding_dim=config.model.doc_embedding_dim,
            word_embedding_dim=config.model.word_embedding_dim,
            hidden_dim=config.model.hidden_dim,
            encoder_dropout=config.model.encoder_dropout,
            word_embeddings=word_emb_tensor,
            train_word_embeddings=config.model.train_word_embeddings,
            dev_mode=config.dev_mode
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        logger.info(f"Starting training: {config.model.epochs} epochs, device={device}")
        
        for epoch in range(config.model.epochs):
            kl_weight = min(
                config.model.kl_end,
                config.model.kl_start + (config.model.kl_end - config.model.kl_start) * (epoch + 1) / config.model.kl_warmup_epochs
            )
            
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)
                
                optimizer.zero_grad()
                loss, recon_loss, kl_loss = model(doc_emb, bow, kl_weight=kl_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * doc_emb.size(0)
            
            train_loss /= n_train
            
            model.eval()
            val_loss = 0.0
            val_recon = 0.0
            val_kl = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    doc_emb = batch['doc_embedding'].to(device)
                    bow = batch['bow'].to(device)
                    
                    loss, recon_loss, kl_loss = model(doc_emb, bow, kl_weight=kl_weight)
                    val_loss += loss.item() * doc_emb.size(0)
                    val_recon += recon_loss.item() * doc_emb.size(0)
                    val_kl += kl_loss.item() * doc_emb.size(0)
            
            val_loss /= n_val
            val_recon /= n_val
            val_kl /= n_val
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['recon_loss'].append(val_recon)
            history['kl_loss'].append(val_kl)
            
            improved = val_loss < best_val_loss - config.model.min_delta
            if improved:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            scheduler.step(val_loss)
            
            if callback and (epoch + 1) % 5 == 0:
                progress = (epoch + 1) / config.model.epochs * 100
                await callback(step_name, "in_progress", 
                    f"Epoch {epoch+1}/{config.model.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}",
                    progress=progress)
            
            logger.info(f"Epoch {epoch+1:3d}/{config.model.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            if config.model.early_stopping and patience_counter >= config.model.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)
                loss, _, _ = model(doc_emb, bow, kl_weight=1.0)
                test_loss += loss.item() * doc_emb.size(0)
        test_loss /= n_test
        
        history['test_loss'] = test_loss
        history['best_val_loss'] = best_val_loss
        history['epochs_trained'] = epoch + 1
        
        if callback:
            await callback(step_name, "in_progress", "Saving model and results")
        
        with open(bow_dir / "vocab.txt", 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        
        with torch.no_grad():
            doc_emb_tensor = torch.tensor(doc_embeddings, dtype=torch.float32).to(device)
            theta = model.get_theta(doc_emb_tensor).cpu().numpy()
            beta = model.get_beta().cpu().numpy()
            topic_emb = model.get_topic_embeddings().cpu().numpy()
            topic_words = model.get_topic_words(vocab, top_k=20)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        np.save(str(model_dir / f"theta_{timestamp}.npy"), theta)
        np.save(str(model_dir / f"beta_{timestamp}.npy"), beta)
        np.save(str(model_dir / f"topic_embeddings_{timestamp}.npy"), topic_emb)
        
        topic_words_dict = {str(k): [w for w, _ in words] for k, words in topic_words}
        with open(model_dir / f"topic_words_{timestamp}.json", 'w') as f:
            json.dump(topic_words_dict, f, indent=2, ensure_ascii=False)
        
        with open(model_dir / f"training_history_{timestamp}.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        torch.save(model.state_dict(), str(model_dir / f"etm_model_{timestamp}.pt"))
        
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}, Test loss: {test_loss:.4f}")
        
        if callback:
            await callback(step_name, "completed", f"Training complete. Best val: {best_val_loss:.4f}")
        
        return _update_state(state, {
            "current_step": "evaluation",
            "training_completed": True,
            "model_dir": str(model_dir),
            "theta_shape": theta.shape,
            "beta_shape": beta.shape,
            "topic_words": topic_words_dict,
            "training_history": history,
            **_add_log(state, step_name, "completed", 
                f"Trained {epoch+1} epochs. Best val: {best_val_loss:.4f}, Test: {test_loss:.4f}")
        })
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if callback:
            await callback(step_name, "failed", str(e))
        return _update_state(state, {
            "status": "failed",
            "error_message": str(e),
            **_add_log(state, step_name, "failed", str(e))
        })


async def evaluation_node(state: AgentState, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Node 4: Evaluator
    Computes topic coherence, diversity, and other metrics
    """
    step_name = "evaluation"
    logger.info(f"[{state['task_id']}] Running evaluation...")
    
    try:
        from evaluation.topic_metrics import compute_all_metrics
        
        dataset = state["dataset"]
        mode = state["mode"]
        
        result_dir = settings.get_result_path(dataset, mode)
        model_dir = result_dir / "model"
        bow_dir = result_dir / "bow"
        evaluation_dir = result_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        if callback:
            await callback(step_name, "in_progress", "Loading model outputs")
        
        result_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
        if not result_files:
            raise FileNotFoundError(f"No results found in {model_dir}")
        timestamp = result_files[0].stem.replace("theta_", "")
        
        theta = np.load(str(model_dir / f"theta_{timestamp}.npy"))
        beta = np.load(str(model_dir / f"beta_{timestamp}.npy"))
        bow_matrix = sparse.load_npz(str(bow_dir / "bow_matrix.npz"))
        
        logger.info(f"Loaded theta: {theta.shape}, beta: {beta.shape}")
        
        if callback:
            await callback(step_name, "in_progress", "Computing metrics")
        
        metrics = compute_all_metrics(
            beta=beta,
            theta=theta,
            doc_term_matrix=bow_matrix,
            top_k_coherence=10,
            top_k_diversity=25
        )
        
        logger.info(f"Coherence: {metrics['topic_coherence_avg']:.4f}, Diversity: {metrics['topic_diversity_td']:.4f}")
        
        metrics_path = evaluation_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if callback:
            await callback(step_name, "completed", 
                f"Coherence: {metrics['topic_coherence_avg']:.4f}, Diversity: {metrics['topic_diversity_td']:.4f}")
        
        return _update_state(state, {
            "current_step": "visualization",
            "evaluation_completed": True,
            "evaluation_dir": str(evaluation_dir),
            "metrics": metrics,
            **_add_log(state, step_name, "completed", 
                f"Coherence: {metrics['topic_coherence_avg']:.4f}, Diversity: {metrics['topic_diversity_td']:.4f}")
        })
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if callback:
            await callback(step_name, "failed", str(e))
        return _update_state(state, {
            "status": "failed",
            "error_message": str(e),
            **_add_log(state, step_name, "failed", str(e))
        })


async def visualization_node(state: AgentState, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Node 5: Visualizer
    Generates topic visualizations and reports
    
    Includes:
    - Tab 1: Topic overview (word clouds, proportions)
    - Tab 2: Temporal analysis (document volume, topic evolution)
    - Tab 3: Sankey diagram (topic flow over time)
    - Tab 4: Dimension heatmap (topic distribution by region/category)
    """
    step_name = "visualization"
    logger.info(f"[{state['task_id']}] Generating visualizations...")
    
    try:
        from visualization.topic_visualizer import TopicVisualizer, load_etm_results, generate_pyldavis_visualization
        from visualization.temporal_analysis import TemporalTopicAnalyzer
        from visualization.dimension_analysis import DimensionAnalyzer
        
        dataset = state["dataset"]
        mode = state["mode"]
        
        result_dir = settings.get_result_path(dataset, mode)
        model_dir = result_dir / "model"
        bow_dir = result_dir / "bow"
        viz_dir = result_dir / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        if callback:
            await callback(step_name, "in_progress", "Loading results")
        
        result_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
        timestamp = result_files[0].stem.replace("theta_", "")
        
        results = load_etm_results(str(model_dir), timestamp)
        
        visualizer = TopicVisualizer(output_dir=str(viz_dir))
        prefix = f"{dataset}_{mode}"
        
        visualization_paths = []
        
        if callback:
            await callback(step_name, "in_progress", "Generating topic word visualization")
        
        visualizer.visualize_topic_words(
            results['topic_words'],
            num_topics=20,
            num_words=10,
            as_wordcloud=True,
            filename=f"topic_words_{prefix}.png"
        )
        visualization_paths.append(str(viz_dir / f"topic_words_{prefix}.png"))
        
        if callback:
            await callback(step_name, "in_progress", "Generating topic similarity heatmap")
        
        visualizer.visualize_topic_similarity(
            results['beta'],
            results['topic_words'],
            filename=f"topic_similarity_{prefix}.png"
        )
        visualization_paths.append(str(viz_dir / f"topic_similarity_{prefix}.png"))
        
        if callback:
            await callback(step_name, "in_progress", "Generating document-topic visualization")
        
        visualizer.visualize_document_topics(
            results['theta'],
            method='tsne',
            filename=f"doc_topics_{prefix}.png"
        )
        visualization_paths.append(str(viz_dir / f"doc_topics_{prefix}.png"))
        
        visualizer.visualize_topic_proportions(
            results['theta'],
            results['topic_words'],
            filename=f"topic_proportions_{prefix}.png"
        )
        visualization_paths.append(str(viz_dir / f"topic_proportions_{prefix}.png"))
        
        if callback:
            await callback(step_name, "in_progress", "Generating interactive pyLDAvis")
        
        bow_path = bow_dir / "bow_matrix.npz"
        vocab_path = bow_dir / "vocab.txt"
        
        if bow_path.exists() and vocab_path.exists():
            bow_matrix = sparse.load_npz(str(bow_path))
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f]
            
            pyldavis_path = viz_dir / f"pyldavis_{prefix}.html"
            generate_pyldavis_visualization(
                theta=results['theta'],
                beta=results['beta'],
                bow_matrix=bow_matrix,
                vocab=vocab,
                output_path=str(pyldavis_path),
                mds='tsne',
                sort_topics=True
            )
            visualization_paths.append(str(pyldavis_path))
        
        # Tab 2 & 3: Temporal Analysis (if timestamps available)
        if callback:
            await callback(step_name, "in_progress", "Generating temporal analysis")
        
        temporal_data = None
        data_dir = settings.DATA_DIR / dataset
        csv_files = list(data_dir.glob("*.csv"))
        
        if csv_files:
            import pandas as pd
            try:
                df = pd.read_csv(csv_files[0])
                # Check for timestamp column
                time_cols = [c for c in df.columns if any(t in c.lower() for t in ['time', 'date', 'year', '时间', '日期', '年份'])]
                
                if time_cols and len(df) == len(results['theta']):
                    timestamps = df[time_cols[0]].values
                    
                    temporal_analyzer = TemporalTopicAnalyzer(
                        theta=results['theta'],
                        timestamps=timestamps,
                        topic_words=results['topic_words'],
                        output_dir=str(viz_dir)
                    )
                    
                    # Tab 2 Chart A: Document volume
                    temporal_analyzer.plot_document_volume(freq='M', filename=f"document_volume_{prefix}.png")
                    visualization_paths.append(str(viz_dir / f"document_volume_{prefix}.png"))
                    
                    # Tab 2 Chart B: Topic evolution
                    temporal_analyzer.plot_topic_evolution(time_bins=10, filename=f"topic_evolution_{prefix}.png")
                    visualization_paths.append(str(viz_dir / f"topic_evolution_{prefix}.png"))
                    
                    temporal_analyzer.plot_topic_stacked_area(time_bins=10, filename=f"topic_stacked_area_{prefix}.png")
                    visualization_paths.append(str(viz_dir / f"topic_stacked_area_{prefix}.png"))
                    
                    # Tab 3: Sankey diagram
                    sankey_path = temporal_analyzer.plot_topic_sankey(num_periods=3, filename=f"topic_sankey_{prefix}.html")
                    if sankey_path:
                        visualization_paths.append(sankey_path)
                    
                    # Get frontend data
                    temporal_data = temporal_analyzer.get_visualization_data_for_frontend()
                    
                    logger.info(f"Generated temporal analysis visualizations")
                else:
                    logger.info("No timestamp column found, skipping temporal analysis")
            except Exception as e:
                logger.warning(f"Could not generate temporal analysis: {e}")
        
        # Tab 4: Dimension Analysis (if dimension column available)
        if callback:
            await callback(step_name, "in_progress", "Generating dimension analysis")
        
        dimension_data = None
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                # Check for dimension column (region, province, category, etc.)
                dim_cols = [c for c in df.columns if any(d in c.lower() for d in ['region', 'province', 'city', 'category', 'type', '省', '市', '地区', '类型', '部门'])]
                
                if dim_cols and len(df) == len(results['theta']):
                    dimension_values = df[dim_cols[0]].values
                    dimension_name = dim_cols[0]
                    
                    dim_analyzer = DimensionAnalyzer(
                        theta=results['theta'],
                        dimension_values=dimension_values,
                        topic_words=results['topic_words'],
                        dimension_name=dimension_name,
                        output_dir=str(viz_dir)
                    )
                    
                    # Tab 4: Dimension heatmap
                    dim_analyzer.plot_dimension_heatmap(filename=f"dimension_heatmap_{prefix}.png")
                    visualization_paths.append(str(viz_dir / f"dimension_heatmap_{prefix}.png"))
                    
                    # Get frontend data
                    dimension_data = dim_analyzer.get_visualization_data_for_frontend()
                    
                    logger.info(f"Generated dimension analysis visualizations")
                else:
                    logger.info("No dimension column found, skipping dimension analysis")
            except Exception as e:
                logger.warning(f"Could not generate dimension analysis: {e}")
        
        logger.info(f"Visualizations saved to {viz_dir}")
        
        if callback:
            await callback(step_name, "completed", f"Generated {len(visualization_paths)} visualizations")
        
        return _update_state(state, {
            "current_step": "completed",
            "status": "completed",
            "visualization_completed": True,
            "visualization_dir": str(viz_dir),
            "visualization_paths": visualization_paths,
            "temporal_data": temporal_data,
            "dimension_data": dimension_data,
            "completed_at": datetime.now().isoformat(),
            **_add_log(state, step_name, "completed", f"Generated {len(visualization_paths)} visualizations")
        })
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        if callback:
            await callback(step_name, "failed", str(e))
        return _update_state(state, {
            "status": "failed",
            "error_message": str(e),
            **_add_log(state, step_name, "failed", str(e))
        })


def check_mode_requirements(state: AgentState) -> str:
    """
    Conditional edge: Check if mode requirements are met
    Returns next node name or error node
    """
    mode = state.get("mode", "zero_shot")
    dataset = state.get("dataset", "")
    
    if mode == "supervised":
        result_dir = settings.get_result_path(dataset, mode)
        embeddings_dir = result_dir / "embeddings"
        label_path = embeddings_dir / f"{dataset}_{mode}_labels.npy"
        
        if not label_path.exists():
            return "error"
    
    return "training"

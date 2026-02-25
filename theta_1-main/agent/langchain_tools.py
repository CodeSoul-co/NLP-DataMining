"""
LangChain Tools for THETA Topic Model Pipeline

Defines tools that the LangChain agent can invoke to:
- Run bash scripts (data cleaning, preparation, training, evaluation, visualization)
- Query analysis results (metrics, topics, charts)
- List available datasets and experiments
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("THETA_ROOT", "/root/autodl-tmp"))
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULT_DIR = Path(os.environ.get("RESULT_DIR", str(BASE_DIR / "result")))
DATA_DIR = BASE_DIR / "data"


def _run_script(script_name: str, args: str, timeout: int = 3600) -> str:
    """Run a bash script and return stdout/stderr."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return f"Error: script not found: {script_path}"
    cmd = f"bash {script_path} {args}"
    logger.info(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(BASE_DIR),
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[STDERR]\n{result.stderr}"
        # Truncate very long output
        if len(output) > 4000:
            output = output[:2000] + "\n...(truncated)...\n" + output[-1500:]
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: script timed out after {timeout}s"
    except Exception as e:
        return f"Error running script: {e}"


# ── Dataset & Experiment Discovery ──────────────────────────────────

@tool
def list_datasets() -> str:
    """List all available datasets in the data/ directory with basic info (row count, columns)."""
    results = []
    for ds_dir in sorted(DATA_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        csv_files = list(ds_dir.glob("*.csv"))
        if not csv_files:
            results.append(f"- {ds_name}: (no CSV files)")
            continue
        for csv_file in csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, nrows=0)
                row_count = sum(1 for _ in open(csv_file)) - 1
                results.append(f"- {ds_name}/{csv_file.name}: {row_count} rows, columns={list(df.columns)}")
            except Exception as e:
                results.append(f"- {ds_name}/{csv_file.name}: (error reading: {e})")
    return "\n".join(results) if results else "No datasets found in data/ directory."


@tool
def list_experiments(dataset: str, model_type: str = "theta") -> str:
    """List available data and model experiments for a dataset.

    Args:
        dataset: Dataset name (e.g. 'edu_data').
        model_type: 'theta' or 'baseline'.
    """
    lines = []

    if model_type == "theta":
        for size in ["0.6B", "4B", "8B"]:
            data_dir = RESULT_DIR / size / dataset / "data"
            model_dir = RESULT_DIR / size / dataset / "models"
            if data_dir.is_dir():
                for exp in sorted(data_dir.iterdir()):
                    if exp.is_dir() and exp.name.startswith("exp_"):
                        has_bow = (exp / "bow" / "bow_matrix.npy").exists()
                        has_emb = (exp / "embeddings" / "embeddings.npy").exists()
                        config_info = ""
                        cfg = exp / "config.json"
                        if cfg.exists():
                            try:
                                c = json.loads(cfg.read_text())
                                config_info = f" mode={c.get('mode','?')}, vocab={c.get('vocab_size','?')}"
                            except Exception:
                                pass
                        lines.append(f"[DATA {size}] {exp.name}  BOW:{'✓' if has_bow else '✗'} EMB:{'✓' if has_emb else '✗'}{config_info}")
            if model_dir.is_dir():
                for exp in sorted(model_dir.iterdir()):
                    if exp.is_dir() and exp.name.startswith("exp_"):
                        has_model = (exp / "model").is_dir()
                        has_eval = (exp / "evaluation").is_dir()
                        lines.append(f"[MODEL {size}] {exp.name}  model:{'✓' if has_model else '✗'} eval:{'✓' if has_eval else '✗'}")
    else:
        base = RESULT_DIR / "baseline" / dataset
        data_dir = base / "data"
        models_dir = base / "models"
        if data_dir.is_dir():
            for exp in sorted(data_dir.iterdir()):
                if exp.is_dir() and exp.name.startswith("exp_"):
                    lines.append(f"[DATA] {exp.name}")
        if models_dir.is_dir():
            for model_name in sorted(models_dir.iterdir()):
                if model_name.is_dir():
                    for exp in sorted(model_name.iterdir()):
                        if exp.is_dir() and exp.name.startswith("exp_"):
                            lines.append(f"[MODEL {model_name.name}] {exp.name}")

    return "\n".join(lines) if lines else f"No experiments found for dataset '{dataset}' ({model_type})."


# ── Script Execution Tools ──────────────────────────────────────────

@tool
def clean_data(
    input_path: str,
    language: str = "",
    text_column: str = "",
    label_columns: str = "",
    keep_all: bool = False,
    preview: bool = False,
    output_path: Optional[str] = None,
    min_words: int = 3,
) -> str:
    """Clean raw text data for topic modeling. Supports CSV files and directories.

    For CSV input, use --preview first to inspect columns, then specify --text_column.
    The text column is cleaned row-by-row; label columns are preserved as-is.

    Args:
        input_path: Path to input CSV file or directory containing docx/txt files.
        language: Data language: english, chinese, german, spanish. Not needed for preview.
        text_column: Name of the text column to clean (REQUIRED for CSV input).
        label_columns: Comma-separated label/metadata columns to keep as-is.
        keep_all: If True, keep ALL original columns (only text column is cleaned).
        preview: If True, show CSV columns and sample rows, then exit.
        output_path: Optional output CSV path. Auto-generated if not provided.
        min_words: Minimum words per document after cleaning (default: 3).
    """
    if preview:
        return _run_script("02_clean_data.sh", f"--input {input_path} --preview")
    args = f"--input {input_path}"
    if language:
        args += f" --language {language}"
    if text_column:
        args += f" --text_column '{text_column}'"
    if label_columns:
        args += f" --label_columns '{label_columns}'"
    if keep_all:
        args += " --keep_all"
    if output_path:
        args += f" --output {output_path}"
    if min_words != 3:
        args += f" --min_words {min_words}"
    return _run_script("02_clean_data.sh", args)


@tool
def prepare_data(
    dataset: str,
    model: str,
    language: str = "english",
    vocab_size: int = 5000,
    model_size: str = "0.6B",
    mode: str = "zero_shot",
    gpu: int = 0,
    bow_only: bool = False,
    time_column: str = "year",
    label_column: Optional[str] = None,
    emb_epochs: int = 10,
    emb_batch_size: int = 8,
) -> str:
    """Prepare data (BOW + embeddings) for a specific model.

    Args:
        dataset: Dataset name (must exist in data/ directory).
        model: Target model: lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta.
        language: Data language for BOW tokenization: english, chinese.
        vocab_size: Vocabulary size for BOW.
        model_size: Qwen model size for THETA: 0.6B, 4B, 8B.
        mode: Embedding mode for THETA: zero_shot, unsupervised, supervised.
        gpu: GPU device ID.
        bow_only: If True, only generate BOW, skip embeddings.
        time_column: Time column name for DTM.
        label_column: Label column for THETA supervised mode.
        emb_epochs: Embedding fine-tuning epochs (THETA unsupervised/supervised).
        emb_batch_size: Embedding fine-tuning batch size.
    """
    args = f"--dataset {dataset} --model {model} --language {language} --vocab_size {vocab_size} --gpu {gpu}"
    if model == "theta":
        args += f" --model_size {model_size} --mode {mode}"
        if mode == "supervised" and label_column:
            args += f" --label_column {label_column}"
        if mode != "zero_shot":
            args += f" --emb_epochs {emb_epochs} --emb_batch_size {emb_batch_size}"
    if model == "dtm":
        args += f" --time_column {time_column}"
    if bow_only:
        args += " --bow-only"
    return _run_script("03_prepare_data.sh", args, timeout=7200)


@tool
def train_theta(
    dataset: str,
    model_size: str = "0.6B",
    mode: str = "zero_shot",
    num_topics: int = 20,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 512,
    learning_rate: float = 0.002,
    kl_start: float = 0.0,
    kl_end: float = 1.0,
    kl_warmup: int = 50,
    patience: int = 10,
    gpu: int = 0,
    language: str = "en",
    skip_viz: bool = False,
    data_exp: Optional[str] = None,
) -> str:
    """Train the THETA topic model.

    Args:
        dataset: Dataset name.
        model_size: Qwen model size: 0.6B, 4B, 8B.
        mode: Embedding mode: zero_shot, unsupervised, supervised.
        num_topics: Number of topics K.
        epochs: Training epochs.
        batch_size: Training batch size.
        hidden_dim: Encoder hidden dimension.
        learning_rate: Learning rate.
        kl_start: KL annealing start weight.
        kl_end: KL annealing end weight.
        kl_warmup: KL warmup epochs.
        patience: Early stopping patience.
        gpu: GPU device ID.
        language: Visualization language: en, zh.
        skip_viz: Skip visualization generation.
        data_exp: Data experiment ID (auto-select if not provided).
    """
    args = (
        f"--dataset {dataset} --model_size {model_size} --mode {mode} "
        f"--num_topics {num_topics} --epochs {epochs} --batch_size {batch_size} "
        f"--hidden_dim {hidden_dim} --learning_rate {learning_rate} "
        f"--kl_start {kl_start} --kl_end {kl_end} --kl_warmup {kl_warmup} "
        f"--patience {patience} --gpu {gpu} --language {language}"
    )
    if skip_viz:
        args += " --skip-viz"
    if data_exp:
        args += f" --data_exp {data_exp}"
    return _run_script("04_train_theta.sh", args, timeout=7200)


@tool
def train_baseline(
    dataset: str,
    models: str,
    num_topics: int = 20,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 512,
    learning_rate: float = 0.002,
    gpu: int = 0,
    language: str = "en",
    skip_viz: bool = True,
    data_exp: Optional[str] = None,
) -> str:
    """Train baseline topic models for comparison with THETA.

    Args:
        dataset: Dataset name.
        models: Comma-separated model list: lda,hdp,stm,btm,nvdm,gsm,prodlda,ctm,etm,dtm,bertopic.
        num_topics: Number of topics (ignored for hdp/bertopic).
        epochs: Training epochs for neural models.
        batch_size: Batch size.
        hidden_dim: Hidden dimension.
        learning_rate: Learning rate.
        gpu: GPU device ID.
        language: Visualization language: en, zh.
        skip_viz: Skip visualization (default True for speed).
        data_exp: Data experiment ID (auto-select if not provided).
    """
    args = (
        f"--dataset {dataset} --models {models} "
        f"--num_topics {num_topics} --epochs {epochs} --batch_size {batch_size} "
        f"--hidden_dim {hidden_dim} --learning_rate {learning_rate} "
        f"--gpu {gpu} --language {language}"
    )
    if skip_viz:
        args += " --skip-viz"
    else:
        args += " --with-viz"
    if data_exp:
        args += f" --data_exp {data_exp}"
    return _run_script("05_train_baseline.sh", args, timeout=7200)


@tool
def visualize(
    dataset: str,
    baseline: bool = False,
    model: Optional[str] = None,
    model_size: str = "0.6B",
    mode: str = "zero_shot",
    num_topics: int = 20,
    language: str = "en",
    dpi: int = 300,
) -> str:
    """Generate visualizations for a trained model.

    Args:
        dataset: Dataset name.
        baseline: True for baseline models, False for THETA.
        model: Baseline model name (required if baseline=True).
        model_size: THETA model size: 0.6B, 4B, 8B.
        mode: THETA embedding mode: zero_shot, unsupervised, supervised.
        num_topics: Number of topics.
        language: Visualization language: en, zh.
        dpi: Image DPI.
    """
    args = f"--dataset {dataset} --language {language} --dpi {dpi}"
    if baseline:
        args += f" --baseline --model {model} --num_topics {num_topics}"
    else:
        args += f" --model_size {model_size} --mode {mode}"
    return _run_script("06_visualize.sh", args)


@tool
def evaluate_model(
    dataset: str,
    model: str,
    num_topics: int = 20,
    model_size: str = "0.6B",
    mode: str = "zero_shot",
) -> str:
    """Evaluate a trained topic model with 7 unified metrics (TD, iRBO, NPMI, C_V, UMass, Exclusivity, PPL).

    Args:
        dataset: Dataset name.
        model: Model name: lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta.
        num_topics: Number of topics.
        model_size: THETA model size (only for theta).
        mode: THETA mode (only for theta).
    """
    args = f"--dataset {dataset} --model {model} --num_topics {num_topics}"
    if model == "theta":
        args += f" --model_size {model_size} --mode {mode}"
    else:
        args += " --baseline"
    return _run_script("07_evaluate.sh", args)


@tool
def compare_models(
    dataset: str,
    models: str,
    num_topics: int = 20,
) -> str:
    """Compare evaluation metrics across multiple models.

    Args:
        dataset: Dataset name.
        models: Comma-separated model list to compare.
        num_topics: Number of topics.
    """
    args = f"--dataset {dataset} --models {models} --num_topics {num_topics}"
    return _run_script("08_compare_models.sh", args)


# ── Result Query Tools ──────────────────────────────────────────────

@tool
def get_training_results(dataset: str, model: str, model_size: str = "0.6B", mode: str = "zero_shot") -> str:
    """Get training results (metrics, topic words) for a trained model.

    Args:
        dataset: Dataset name.
        model: Model name (e.g. 'theta', 'lda', 'prodlda').
        model_size: THETA model size (only for theta).
        mode: THETA mode (only for theta).
    """
    results = []

    if model == "theta":
        models_dir = RESULT_DIR / model_size / dataset / "models"
    else:
        models_dir = RESULT_DIR / "baseline" / dataset / "models" / model

    if not models_dir.is_dir():
        return f"No model results found at {models_dir}"

    # Find latest experiment
    exps = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not exps:
        return f"No experiments found in {models_dir}"

    exp = exps[0]
    results.append(f"Latest experiment: {exp.name}")

    # Read metrics
    for metrics_file in exp.rglob("metrics*.json"):
        try:
            metrics = json.loads(metrics_file.read_text())
            results.append(f"\nMetrics ({metrics_file.name}):")
            for k, v in metrics.items():
                if isinstance(v, float):
                    results.append(f"  {k}: {v:.4f}")
                else:
                    results.append(f"  {k}: {v}")
        except Exception:
            pass

    # Read topic words
    for tw_file in exp.rglob("topic_words*.json"):
        try:
            topics = json.loads(tw_file.read_text())
            results.append(f"\nTopic Words ({tw_file.name}):")
            if isinstance(topics, dict):
                for tid, words in list(topics.items())[:10]:
                    if isinstance(words, list):
                        results.append(f"  Topic {tid}: {', '.join(str(w) for w in words[:8])}")
            elif isinstance(topics, list):
                for t in topics[:10]:
                    if isinstance(t, dict):
                        results.append(f"  Topic {t.get('id','?')}: {', '.join(str(w) for w in t.get('keywords', t.get('words', []))[:8])}")
        except Exception:
            pass

    return "\n".join(results) if results else "No results found."


@tool
def list_visualizations(dataset: str, model: str, model_size: str = "0.6B", mode: str = "zero_shot") -> str:
    """List available visualization files for a trained model.

    Args:
        dataset: Dataset name.
        model: Model name.
        model_size: THETA model size (only for theta).
        mode: THETA mode (only for theta).
    """
    if model == "theta":
        models_dir = RESULT_DIR / model_size / dataset / "models"
    else:
        models_dir = RESULT_DIR / "baseline" / dataset / "models" / model

    if not models_dir.is_dir():
        return f"No model directory found at {models_dir}"

    viz_files = []
    for f in models_dir.rglob("*"):
        if f.suffix in (".png", ".html", ".svg", ".pdf") and f.is_file():
            viz_files.append(str(f.relative_to(RESULT_DIR)))

    if not viz_files:
        return "No visualization files found. Run visualize tool first."

    return "Available visualizations:\n" + "\n".join(f"- {f}" for f in sorted(viz_files)[:50])


# ── All tools list ──────────────────────────────────────────────────

ALL_TOOLS = [
    list_datasets,
    list_experiments,
    clean_data,
    prepare_data,
    train_theta,
    train_baseline,
    visualize,
    evaluate_model,
    compare_models,
    get_training_results,
    list_visualizations,
]

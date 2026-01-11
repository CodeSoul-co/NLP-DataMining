<div align="center">

<img src="assets/THETA.png" width="40%" alt="THETA Logo"/>

<h1>THETA (θ)</h1>

Textual Hybrid Embedding–based Topic Analysis  

</div>

**[English](README.md) | [中文](README_CN.md)**

## Overview

THETA (θ) is an open-source, research-oriented platform for LLM-enhanced topic analysis in social science. It combines:

- Domain-adaptive document embeddings from a unified base model (e.g., Qwen-3)
  - Zero-shot embedding (no training), or
  - LoRA adapters (lightweight, modular domain fine-tuning)
- Generative topic models
  - ETM (Embedded Topic Model) for coherent and interpretable topics
  - DETM (Dynamic ETM) for topic evolution over time
- Scientific validation via intrinsic metrics (coherence/perplexity) and downstream tasks
- Agent-based orchestration to make the workflow accessible to non-technical researchers

THETA aims to move topic modeling from “clustering with pretty plots” to a reproducible, validated scientific workflow.

---

## Key Features

- Hybrid embedding topic analysis (Zero-shot / LoRA)
- LoRA “expert adapters” for multiple domains (plug-and-play)
- Generative topic modeling with ETM/DETM for interpretability and stability
- Fine-grained, domain-aware data governance (privacy mask normalization, emoji-to-text translation, metadata stripping)
- Validation layer with coherence/perplexity and downstream task metrics (F1/AUC)
- Agent-based interactive platform (Data Steward → Modeling Analyst → Domain Expert → Interpretive Report)

---

## Method Summary

THETA follows a six-layer pipeline:

1. Data source & governance: ingestion, de-identification, noise removal, dataset-specific cleaning rules
2. Semantic enhancement: base model embedding (zero-shot or LoRA) to obtain domain-adaptive vectors
3. Deep structure mining: ETM/DETM as the main topic backbone (optional UMAP/HDBSCAN for baseline exploration)
4. Validation: intrinsic metrics + downstream tasks
5. Agent platform: orchestration, parameter tuning, report generation
6. Interpretive report: topic maps, hierarchies, trend narratives

---

## Requirements

- Python 3.10+
- CUDA is optional but recommended for LoRA training and large-scale embedding
- Typical dependencies: PyTorch, Transformers, PEFT, datasets tooling, ETM/DETM implementation, evaluation + visualization tooling, Streamlit or Gradio for the app

---

## Installation

```bash
git clone https://github.com/<YOUR_ORG>/THETA.git
cd THETA

pip install -r requirements.txt
# or
pip install -e .
````

---

## **Quickstart**
The commands below are templates. Replace module paths/CLI names with your actual implementation.
### **A) Zero-shot Mode**

```
# 1) Data cleaning & semantic enhancement
python -m theta.clean \
  --dataset cfpb \
  --input data/raw/cfpb.csv \
  --output data/processed/cfpb.jsonl

# 2) Generate embeddings (zero-shot)
python -m theta.embed \
  --model Qwen3 \
  --mode zeroshot \
  --input data/processed/cfpb.jsonl \
  --output artifacts/embeddings/cfpb.npy

# 3) Train ETM (or DETM)
python -m theta.topic \
  --method etm \
  --embeddings artifacts/embeddings/cfpb.npy \
  --texts data/processed/cfpb.jsonl \
  --output artifacts/topics/cfpb_etm
```

### **B) LoRA Mode**

```
# 1) Train a domain LoRA adapter (MLM/CLM depending on your setup)
python -m theta.train_lora \
  --base Qwen3 \
  --dataset cfpb \
  --output adapters/adapter_finance

# 2) Generate embeddings with LoRA adapter
python -m theta.embed \
  --model Qwen3 \
  --mode lora \
  --adapter adapters/adapter_finance \
  --input data/processed/cfpb.jsonl \
  --output artifacts/embeddings/cfpb_lora.npy

# 3) Train ETM/DETM
python -m theta.topic \
  --method etm \
  --embeddings artifacts/embeddings/cfpb_lora.npy \
  --texts data/processed/cfpb.jsonl \
  --output artifacts/topics/cfpb_etm_lora
```
---

## **Data Governance & Preprocessing**
Data governance directly impacts topic coherence and interpretability. THETA uses domain-aware cleaning strategies to preserve semantic signals while removing artifacts that distort topic structure.
Examples of governance strategies:
- Finance/regulatory text (e.g., CFPB)
    - normalize privacy masks (e.g., “XXXX”, “[REDACTED]”) into special tokens      
    - remove template-only submissions without substantive content 
- Social media health or sensitive discourse (e.g., Reddit mental health, hate speech)   
    - avoid deleting emojis and slang; translate emojis into text (e.g., 😭 → “sadness”)     
    - normalize abbreviations when helpful for semantic grouping
     
- Political transcripts
    
    - remove procedural metadata (e.g., “applause”, “chair intervention”) while keeping policy content
        
    
- Honeypot/bot corpora
    
    - strip spam links and tracking tokens, filter extremely low-information posts, optionally tag suspected bot accounts
        
    

  

Governance outputs:

- Standardized corpora (JSONL/CSV), de-identified and format-aligned
    
- Semantically enhanced text copies (emoji-to-text, normalization)
    
- Cleaning report (rules, removed counts, noise categories)
    

---

## **Semantic Enhancement (Embeddings)**

  

THETA supports two embedding operation modes:

- Zero-shot: embed texts directly with the base model to establish a baseline
    
- LoRA: load a domain adapter to generate embeddings aligned with domain-specific semantics
    

  

Typical embedding artifacts:

- Embedding matrix (N x D) stored as .npy or .pt
    
- Index mapping from row number to document id
    
- Optional caching for incremental runs
    

  

Recommended adapter naming:

- adapters/adapter_finance
    
- adapters/adapter_mental_health
    
- adapters/adapter_politics
    
- adapters/adapter_hate_speech
    
- adapters/adapter_honeypot
    

---

## **Deep Structure Mining (ETM/DETM)**

  

THETA uses ETM as the primary topic modeling backbone, and DETM for dynamic topic evolution when timestamps are available.

  

Typical outputs:

- Document-topic distribution (theta_d)
    
- Topic descriptors (top keywords and weights per topic)
    
- Topic embeddings for distance maps and hierarchy discovery
    
- Topic evolution timelines (DETM) for trend analysis over time
    

---

## **Validation & Evaluation**

  

THETA treats topic modeling as a validated scientific workflow.

  

Intrinsic evaluation:

- Topic coherence (e.g., C_v)
    
- Perplexity
    
- Optional stability checks across seeds/bootstraps
    

  

Downstream validation examples:

- Finance risk detection: early discovery of illegal collection or hidden-fee themes; trend-based early warning
    
- Suicide ideation detection: topic-derived risk factors improving classifier F1/AUC
    
- Public opinion analysis: decomposing coarse sentiment into fine-grained themes; correlating topic prevalence with external indicators
    
- Bot/script mining: clustering templated narratives and link-heavy spam patterns
    
- Political discourse: tracking topic shifts aligned with events or policy cycles
    

---

## **Agent-Based Platform**

  

THETA can be deployed as an interactive app with three core agents:

- Data Steward: selects cleaning strategies and runs governance scripts
    
- Modeling Analyst: chooses embedding mode, runs ETM/DETM, tunes hyperparameters using metrics
    
- Domain Expert: produces interpretive reports grounded in topics and optional knowledge bases
    

  

Example commands:

```
streamlit run apps/streamlit_app.py
# or
python apps/gradio_app.py
```

---

## **Configuration**

  

YAML-based configuration is recommended:

```
model:
  base: "Qwen3"
  embedding_mode: "lora"
  adapter_path: "adapters/adapter_finance"

topic:
  method: "etm"
  num_topics: 50
  epochs: 200
  batch_size: 256

data:
  dataset: "cfpb"
  input_path: "data/processed/cfpb.jsonl"
  text_field: "text"
  id_field: "id"
```

Run with:

```
python -m theta.run --config configs/default.yaml
```

---

## **Baselines**

  

Recommended baselines for comparison:

- LDA
    
- TF-IDF + KMeans/HDBSCAN
    
- Zero-shot embeddings vs LoRA-adapted embeddings (core ablation)
    
- Optional BERTopic-style pipelines for additional comparison
    

---

## **Roadmap**

### 📅 Current Sprint (Sprint DDL: 1.14)

#### 🚀 Core Data Pipeline & NLP

* [ ] **Standardized Preprocessing Pipeline** `Assignee: pjq` `DDL: 1.14`
  * [ ] Implement data cleaning, preprocessing, tokenization, and full workflow.
  * [ ] Package as standardized modules for production readiness.

* [ ] **Model Training & Data Persistence** `Assignee: pjq` `DDL: 1.14`
  * [ ] Train Topic Models with Embedding inputs (covering multiple datasets).
  * [ ] Save model matrices and intermediate data required for visualization.

* [ ] **Output Artifacts Research** `Assignee: pjq` `DDL: 1.14`
  * [ ] Define specific output contents (chart types, table formats, metrics).

* [ ] **Basic Visualization & Export** `Assignee: pjq` `DDL: 1.14`
  * [ ] Implement basic plotting functionality.
  * [ ] Implement CSV table data export.

---

#### 🧩 Feature Development & Integration

##### 📊 Visualization & Open-Source Library Integration

* [ ] **Open-Source Visualization Library Integration** `Assignee: lixin`
  * [ ] Integrate visualization capabilities from `tmtoolkit` and `pyLDAvis`.
  * [ ] **Ref:** [tmtoolkit docs](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html), [pyLDAvis repo](https://github.com/bmabey/pyLDAvis).
  * [ ] Integrate with core data pipeline (pjq's part).

* [ ] **New Data Adapter** `Assignee: pjq`
  * [ ] Ensure Pipeline can handle new input data formats.

##### 🤖 AI Agent & Analysis Module

* [ ] **Chart/Data Analysis Agent Development** `Assignee: pjq, lixin`
  * [ ] Tech stack: `LangGraph` / `LangChain`.
  * [ ] Functionality: Single-pass analysis of generated charts/tables (no multi-turn dialogue required).
  * [ ] **Focus**: Prompt Engineering.
  * [ ] Interface: May involve multimodal API calls.

* [ ] **Agent Infrastructure** `Assignee: 李芳痕`
  * [ ] Build Agent foundational framework.

* [ ] **Conversation History Storage** `Assignee: pjq`
  * [ ] Implement cloud storage for conversation records.

* [ ] **Interpretation Feature Enhancement** `Assignee: 李国正` `Priority: Low`
  * [ ] (Time permitting) Assist with Agent interpretation features.

##### 🔌 Backend & API

* [ ] **Format-free Data Interface** `Assignee: 李芳痕`
  * [ ] Develop flexible interface supporting multiple input formats.

* [ ] **Scriptable Execution Interface** `Assignee: pjq`
  * [ ] Implement Bash-like invocation interface (receive parameters from frontend -> trigger visualization/analysis code).

---

#### 💻 Product & Frontend Integration

* [ ] **Product Design** `Assignee: 李国正`
  * [ ] Define overall product functionality and interaction flows.

* [ ] **Requirements Documentation** `Assignee: lixin`
  * [ ] Compile detailed functional requirements documentation.

* [ ] **Frontend-Backend Integration** `Assignee: 李国正, pjq`
  * [ ] Frontend integration after page design completion.
  * [ ] Product-level frontend-backend logic integration.

---

### 📋 Long-term Roadmap

- **v0.1**: unified dataset interface + zero-shot embeddings + ETM baseline
- **v0.2**: LoRA adapter training pipeline + coherence/perplexity reports
- **v0.3**: DETM topic evolution + interactive visualizations
- **v0.4**: agent-based one-click analysis + exportable reports
- **v1.0**: reproducible benchmark suite (datasets, baselines, downstream tasks)
    

---

## **Citation**

```
@software{theta_topic_analysis,
  title  = {THETA (θ): Textual Hybrid Embedding--based Topic Analysis},
  author = {TODO: Your Name},
  year   = {2026}
}
```

---

## **License**

  

Choose one and add it to LICENSE:

- Apache-2.0
    
- MIT
    

---

## **Contributing**

  

Contributions are welcome, including:

- new dataset adapters
    
- topic visualization modules
    
- evaluation and reproducibility scripts
    
- agent tools and reporting templates
    

  

Suggested workflow:

1. Fork the repo and create a feature branch
    
2. Add a minimal reproducible example or tests
    
3. Open a pull request
    

---

## **Ethics & Safety**

  

This project analyzes social text and may involve sensitive content.

- Do not include personally identifiable information (PII)
    
- Ensure dataset usage complies with platform terms and research ethics
    
- Interpret outputs cautiously; topic discovery does not replace scientific conclusions
    
- Be responsible with sensitive domains such as self-harm, hate speech, and political polarization
    

---

## **FAQ**

  

Q: Is this only for Qwen-3?

A: No. Qwen-3 is the reference backbone, but THETA is designed to be model-agnostic.

  

Q: What is the difference between ETM and DETM?

A: ETM learns static topics across the corpus; DETM models topic evolution over time and requires timestamps or time bins.

  

Q: Can I add my own dataset?

A: Yes. Export standardized JSONL with at least id and text fields, and implement a dataset adapter.

---
## **Contact me**
Please contact me duanzhenke@stu.zuel.edu.cn, panjiqun@stu.zuel.edu.cn or lixin@stu.zuel.edu.cn if you have any questions.

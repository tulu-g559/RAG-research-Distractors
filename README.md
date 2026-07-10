# Distractor Injection: A Robustness Probe for Retrieval-Augmented Generation Faithfulness under Retrieval Noise

A systematic framework for measuring how retrieval-augmented generation (RAG) systems withstand distracting passages injected into their context window. This repository provides a modular pipeline for injecting structured distractor passages—topical, hard-negative, and paraphrased-contradiction—and evaluating LLM answer consistency using exact match (EM), token-level F1, answer flip rate, and a novel Faithfulness Fragility Score (FFS).

> **Note on model comparison.** GPT-4.1 mini produced consistently low baseline scores under our prompting and evaluation setup, making robustness comparisons across models inconclusive. The analysis therefore focuses the robustness discussion primarily on Llama-3.1-8B, where baseline performance provided sufficient signal to measure degradation meaningfully. The framework itself is model-agnostic and supports any instruction-tuned LLM accessible via API.
>
> **Note on computational constraints.** This evaluation involved 2,000+ LLM inference calls across two API providers. Free-tier API quotas are insufficient for experiments at this scale — rate limits on Gemini (20 requests/day in the free tier) disrupted the paraphrased contradiction distractor generation, and aggregate latency exceeded several hours. Reproducing or extending this work at scale requires either institutional API credits, a local inference setup, or financial contributions to LLM access budgets.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Codebase Architecture](#codebase-architecture)
- [Datasets](#datasets)
- [Embedding & Retrieval](#embedding--retrieval)
- [Models Under Evaluation](#models-under-evaluation)
- [Distractor Injection Framework](#distractor-injection-framework)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiment Design](#experiment-design)
- [Results & Artifacts](#results--artifacts)
- [Reproducing Experiments](#reproducing-experiments)

---

## Research Motivation

Retrieval-Augmented Generation (RAG) grounds LLM responses in externally retrieved passages, reducing hallucination and enabling knowledge-intensive tasks. However, retrieval pipelines are imperfect: they frequently return irrelevant, topically adjacent, or even contradictory passages alongside the gold evidence. The central question of this research is:

> **How faithfully do LLMs rely on the correct passage when distractors are present in their context?**

This work introduces a controlled experimental framework that:

1. **Injects known distractor passages** into the LLM context at varying counts and semantic types.
2. **Measures answer degradation** via multiple metrics: exact match, token F1, answer flip rate, and a composite fragility score.
3. **Provides publication-ready visualizations** and summary tables for analysis.

The framework enables researchers to compare model robustness, identify failure modes, and benchmark progress toward distraction-robust RAG.

---

## Codebase Architecture

```
├── experiments/
│   └── config.yaml                 # Central experiment configuration
│
├── src/
│   ├── evaluation.py                # EM, F1 scoring
│   ├── embeddings.py                # Google / OpenAI / Local embedding providers
│   ├── vectorstore.py               # FAISS index wrapper
│   ├── retriever.py                 # Retriever with query cache
│   ├── cache.py                     # SHA-256 keyed disk cache for API responses
│   ├── rag_pipeline.py              # Orchestrator: retrieve → inject → generate
│   │
│   ├── loaders/
│   │   ├── squad_loader.py          # SQuAD v2.0 → LangChain Documents
│   │   └── hotpot_loader.py         # HotPotQA → LangChain Documents
│   │
│   ├── distractors/
│   │   ├── injector.py              # DistractorInjector orchestrator
│   │   └── generators.py            # Topical, hard-negative, contradiction generators
│   │
│   └── generators/
│       ├── base.py                  # Abstract BaseGenerator
│       ├── gemini.py                # Google Gemini (genai SDK)
│       ├── openrouter.py            # OpenRouter (ChatOpenAI)
│       └── groq.py                  # Groq (ChatGroq)
│
├── scripts/
│   ├── run_evaluation.py            # Main experiment runner
│   ├── analyze_results.py           # Statistical analysis & figure generation
│   ├── build_index.py               # Standalone FAISS index builder
│   ├── retrieval_audit.py           # Retrieval quality audit
│   └── test_distractors.py          # Manual distractor inspection
│
├── results/
│   ├── experiment_results.csv       # Clean results (requested column spec)
│   ├── baseline.csv                 # Full experimental data
│   ├── plots/                       # Publication-ready figures
│   │   ├── accuracy_vs_distractor_count.png
│   │   ├── f1_vs_distractor_count.png
│   │   ├── ffs_leaderboard.png
│   │   └── distractor_type_heatmap.png
│   └── analysis/                    # Summary statistics
│       ├── overall_metrics.csv
│       ├── answer_flip_rate.csv
│       ├── ffs_scores.csv
│       ├── ffs_leaderboard.csv
│       └── per_dataset_metrics.csv
│
├── data/
│   ├── faiss_intfloat_e5-small-v2/  # Pre-built 400-doc FAISS index
│   ├── raw/
│   │   ├── squad_v2/               # SQuAD v2.0 training data
│   │   └── hotpot/                  # HotPotQA (distractor setting)
│   └── processed/                   # Reserved for processed data
│
└── cache/
    ├── api_responses.json            # LLM response cache
    └── distractor_contradictions.json # Contradiction distractor cache
```

---

## Datasets

### SQuAD v2.0

[Stanford Question Answering Dataset v2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) combines 100,000+ answerable questions with 50,000 unanswerable questions written by crowdworkers on 500+ Wikipedia articles. This work uses the training split (`train-v2.0.json`) as a document corpus, treating each unique context paragraph as a retrievable document. Each document's metadata aggregates all Q/A pairs that share that paragraph.

**Loader:** `src/loaders/squad_loader.py` parses the standard SQuAD JSON, deduplicates by paragraph content, and attaches a `questions` list to each `Document` metadata.

### HotPotQA (Distractor Setting)

[HotPotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) is a multi-hop QA dataset where each question requires reasoning over multiple supporting passages. The *distractor setting* provides 10 passages per question—2 gold supporting facts and 8 distractor paragraphs. This work uses the Hugging Face `hotpotqa/hotpot_qa` dataset with the `"distractor"` configuration.

**Loader:** `src/loaders/hotpot_loader.py` loads from disk (via `datasets.load_from_disk`), expands each sample's 10 context paragraphs into individual `Document` objects, and deduplicates by paragraph text while aggregating all associated Q/A pairs.

---

## Embedding & Retrieval

### Embedding Models

Three embedding providers are available via the factory `src/embeddings.py:create_embedding_provider()`:

| Model | Provider Class | Backend | Use Case |
|-------|---------------|---------|----------|
| `intfloat/e5-small-v2` | `LocalEmbeddingProvider` | SentenceTransformers (local CPU/GPU) | **Default**—384-dim, fast, no API cost |
| `gemini-embedding-2` | `GoogleEmbeddingProvider` | Google Gemini API | Cloud-based, higher dimensionality |
| `text-embedding-3-small` | `OpenAIEmbeddingProvider` | OpenRouter → OpenAI API | Cloud-based via OpenRouter |

The E5 model uses task-specific prefixes (`"passage: "` for indexing, `"query: "` for retrieval) as recommended by the [E5 paper](https://arxiv.org/abs/2212.03533).

### Vector Store

`src/vectorstore.py` wraps a FAISS (Facebook AI Similarity Search) index:

- **Building:** `VectorStore.from_documents(docs, embedding_provider)` embeds all documents and builds a dense FAISS index in memory.
- **Persistence:** `save_local(path)` writes `index.faiss` (binary vectors) and `index.pkl` (serialized docstore) to disk.
- **Loading:** `load_local(path, provider)` restores a previously saved index.
- **Retrieval:** `similarity_search_with_score(query, k)` returns the top-k documents with L2 distances.

### Retriever

`src/retriever.py` provides a thin wrapper around the vector store with an in-memory query cache (`Dict[Tuple[str, int], List]`) to avoid redundant similarity searches across multiple distractor configurations for the same question.

---

## Models Under Evaluation

The experiment evaluates two LLMs accessible through API providers. All models receive identical prompts and context windows for controlled comparison.

### GPT-4.1 Mini (OpenRouter)

- **Provider:** OpenRouter API (`https://openrouter.ai/api/v1`)
- **Model ID:** `openai/gpt-4.1-mini`
- **Generator:** `src/generators/openrouter.py` — uses LangChain's `ChatOpenAI` pointed at OpenRouter's OpenAI-compatible endpoint.
- **Configuration:** `max_completion_tokens=512`, temperature defaults from `ChatOpenAI`.

### Llama 3.1 8B Instant (Groq)

- **Provider:** Groq API (`https://api.groq.com`)
- **Model ID:** `llama-3.1-8b-instant`
- **Generator:** `src/generators/groq.py` — uses LangChain's `ChatGroq` with `StrOutputParser`.
- **Configuration:** `max_tokens=512`, piped through LangChain's `|` operator.

### Common Prompt Template

Both generators use the same prompt (from `ChatPromptTemplate`):

```
Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer:
```

This instruction-level constraint ("based only on the provided context") is critical for measuring faithfulness—responses that use external knowledge are considered unfaithful.

### Response Caching

`src/cache.py` implements a JSON-persisted disk cache keyed by `SHA-256(question ||| context ||| model_name)`. When an identical (question, context, model) triple is encountered, the cached response is reused. This is essential for the experiment design: the baseline (0 distractors) context is identical across distractor-type runs, so cached responses are served instantly.

---

## Distractor Injection Framework

The distractor framework (`src/distractors/`) systematically contaminates the LLM context with passages designed to test specific failure modes.

### Architecture

`DistractorInjector` (`src/distractors/injector.py`) is the orchestrator. Given a question, gold passage, distractor count, and type, it:

1. Generates the requested number of distractor passages via the appropriate generator.
2. Concatenates the gold passage with distractors: `gold + "\n\n" + d1 + "\n\n" + d2 + ...`
3. Returns the augmented context and metadata about the injection.

### Distractor Types

#### Topical Distractors

Generated by `generate_topical()` in `src/distractors/generators.py`.

- **Strategy:** Retrieve passages that share the same Wikipedia `title` as the gold passage, using the gold passage itself as the similarity query.
- **Mechanism:** FAISS similarity search (`k=20`), filter by matching title metadata, exclude the gold passage.
- **Research purpose:** Tests whether the model can distinguish the correct passage from other passages about the same topic. This mimics the common RAG failure mode where the retriever returns several topically-relevant chunks, only one of which contains the answer.

#### Hard-Negative Distractors

Generated by `generate_hard_negative()` in `src/distractors/generators.py`.

- **Strategy:** Retrieve passages that are semantically similar to the **question** (not the gold passage) but whose associated answer differs from the gold answer.
- **Mechanism:** FAISS similarity search using the question as query, exclude the gold passage, exclude any passage whose answer metadata matches the gold answer.
- **Research purpose:** Tests whether the model can resist passages that appear relevant to the question but do not contain the correct answer. This is the most challenging distractor type—it simulates a retriever returning plausible but ultimately incorrect evidence.

#### Paraphrased Contradiction Distractors

Generated by `generate_paraphrased_contradiction()` in `src/distractors/generators.py`.

- **Strategy:** Prompt an LLM (Gemini by default) to generate a version of the gold passage that contradicts its key facts while preserving the same topic and writing style.
- **Prompt:** `"Generate a version of the passage below that contradicts its key facts. Keep the same topic and writing style."`
- **Caching:** Results are cached in `cache/distractor_contradictions.json` keyed by `SHA-256(question ||| gold_passage)`.
- **Research purpose:** Tests whether the model detects and disregards factual contradictions. This simulates the worst-case retrieval failure where the context contains directly conflicting information.

### Experimental Conditions

The experiment crosses three factors:

| Factor | Levels |
|--------|--------|
| **Distractor Count** | 0 (baseline), 1, 3, 5 |
| **Distractor Type** | topical, hard_negative, paraphrased_contradiction |
| **Model** | GPT-4.1 Mini, Llama 3.1 8B |

For `distractor_count=0`, the distractor type is `"none"` (gold passage only). For each `distractor_count > 0`, all three types are evaluated independently, yielding 10 experimental conditions plus the baseline per question per model.

---

## Evaluation Metrics

### Exact Match (EM)

```python
def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize(prediction) == normalize(ground_truth)
```

A binary indicator: 1 if the normalized prediction exactly equals the normalized ground truth, 0 otherwise. Normalization lowercases, strips punctuation, collapses whitespace. EM is the strictest metric—any deviation from the canonical answer counts as a failure.

**Formula (across N questions):**

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{normalize}(p_i) = \text{normalize}(g_i)]$$
```
EM = (number of predictions that match gold exactly, after normalization) / (total number of examples)
```

### F1 Score

```python
def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
```

Token-level F1 measures unigram overlap between the prediction and ground truth. Unlike EM, F1 rewards partial correctness—a prediction containing some but not all tokens of the answer receives a proportional score. This is the standard metric in extractive QA (Rajpurkar et al., 2016).

### Recall@k

A retrieval-side metric: 1 if the normalized answer string appears in any of the top-k retrieved documents, 0 otherwise. Evaluated at k=5 and k=10.

$$\text{Recall@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{answer}_i \in \text{top-k documents}_i]$$

### Answer Flip Rate

Measures how often a correct answer becomes incorrect when distractors are introduced:

$$\text{Flip Rate}(m, t, c) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{EM}_{\text{baseline}, i} = 1 \land \text{EM}_{(t,c), i} = 0]$$

where the baseline is the `distractor_count=0` condition for the same model and question. This metric captures the marginal harm of distractors.

### Faithfulness Fragility Score (FFS)

A composite metric quantifying the fractional F1 degradation attributable to distractors:

$$\text{FFS}_i = \max\left(0, \frac{\text{F1}_{\text{baseline}, i} - \text{F1}_{(t,c), i}}{\text{F1}_{\text{baseline}, i}}\right)$$

with the convention that $\text{FFS}_i = 0$ when $\text{F1}_{\text{baseline}, i} = 0$ (no correctness to lose). The aggregate FFS is the mean across all questions:

$$\text{FFS}(m, t, c) = \frac{1}{N} \sum_{i=1}^{N} \text{FFS}_i$$

**Interpretation:**
- **FFS = 0.0:** Distractors cause no degradation (perfect robustness).
- **FFS = 0.15:** On average, 15% of baseline F1 is lost when distractors are present.
- **FFS = 1.0:** Complete collapse—all baseline correctness is destroyed.

This metric is inspired by the concept of *fragility* in machine learning (Geirhos et al., 2020; Goodfellow et al., 2014) and adapts it to the RAG faithfulness setting.

---

## Experiment Design

### Configuration

All experiment parameters are specified declaratively in `experiments/config.yaml`:

```yaml
embedding_model: intfloat/e5-small-v2

generator_models:
  gpt41mini:
    provider: openrouter
    model: openai/gpt-4.1-mini
  llama31:
    provider: groq
    model: llama-3.1-8b-instant

top_k: 5
random_seed: 42

dataset_subset:
  squad:
    documents: 200
    questions: 50
  hotpot:
    documents: 200
    questions: 50

distractor_count: [0, 1, 3, 5]
distractor_type: [topical, hard_negative, paraphrased_contradiction]
```

### Pipeline

The experiment runner (`scripts/run_evaluation.py`) executes the following steps:

1. **Configuration Loading:** Parse `config.yaml`, load API keys from `.env`.
2. **Dataset Subset Construction:** Randomly sample 50 questions from each dataset (seeded for reproducibility). Collect the unique gold passages, then pad the document index to 200 documents per dataset with randomly sampled filler passages.
3. **Index Validation:** Verify every gold passage is present in the index and no duplicates exist.
4. **FAISS Index Building:** Embed all 400 documents with E5-small-v2, build a FAISS index, save to disk.
5. **Evaluation Loop:** For each of the 100 test questions, retrieve top-10 documents from the index. Then for each of the 10 experimental conditions (count 0 + 3 types × 3 counts):
   - Inject distractors into the context (or use gold-only for baseline).
   - Query each model (with caching).
   - Compute EM, F1, record latency.
6. **Output:** Write `results/baseline.csv` with 2,000 rows (100 questions × 2 models × 10 conditions).

### Total Experimental Design

| Factor | Levels | Count |
|--------|--------|-------|
| Datasets | SQuAD, HotPotQA | 2 |
| Questions per dataset | 50 | 100 |
| Models | GPT-4.1 Mini, Llama 3.1 8B | 2 |
| Distractor counts | 0, 1, 3, 5 | 4 |
| Distractor types (count > 0) | topical, hard_negative, paraphrased_contradiction | 3 |
| **Total runs** | | **2,000** |

---

## Results & Artifacts

### Output Structure

```
results/
├── experiment_results.csv        # Clean CSV with requested column spec
├── baseline.csv                  # Full experimental data (all metadata)
│
├── plots/
│   ├── accuracy_vs_distractor_count.png   # EM decay per dataset
│   ├── f1_vs_distractor_count.png         # F1 decay per dataset
│   ├── ffs_leaderboard.png               # Model-level FFS comparison
│   └── distractor_type_heatmap.png       # F1 drop by type × count
│
└── analysis/
    ├── overall_metrics.csv               # Mean EM, F1, latency, recall by cell
    ├── answer_flip_rate.csv              # Flip rate by cell
    ├── ffs_scores.csv                    # Per-cell FFS with baseline F1
    ├── ffs_leaderboard.csv              # Model-level FFS ranking
    └── per_dataset_metrics.csv           # Dataset-stratified means
```

### Figures

#### Accuracy vs. Distractor Count
Two-panel plot (SQuAD, HotPotQA) showing mean EM on the y-axis vs. distractor count on the x-axis. Separate series for each (model, distractor_type) combination. The baseline (count=0, `"none"` type) is plotted with a solid line and markers.

#### F1 vs. Distractor Count
Same layout as accuracy, but with F1 as the metric. Reveals degradation patterns that EM alone may miss, since F1 captures partial answer overlap.

#### FFS Leaderboard
Horizontal bar chart ranking models by their aggregate FFS (lower = more robust). Bars are color-coded by severity: green (FFS < 0.3), amber (0.3–0.6), red (> 0.6).

#### Distractor-Type Heatmap
Per-model heatmap with distractor_type on the y-axis, distractor_count on the x-axis, and mean F1 drop as the fill value. Enables rapid visual comparison of which distractor types cause the most degradation for each model.

### Analysis Script

`scripts/analyze_results.py` performs all statistical computations and figure generation:

```bash
python scripts/analyze_results.py --csv results/baseline.csv
```

The script:
1. Loads and validates the CSV, excluding cached rows to avoid double-counting.
2. Computes overall metrics, answer flip rate, and FFS per (model, dataset, distractor_type, distractor_count).
3. Saves all summary tables to `results/analysis/` as both CSV and TXT.
4. Generates all four publication-ready plots to `results/plots/`.
5. Uses `seaborn` and `matplotlib` with a whitegrid theme and 150 DPI for high-quality output.

---

## Reproducing Experiments

### Prerequisites

- Python 3.10+
- API keys for at least one LLM provider (OpenRouter, Groq) and optionally Gemini
- FAISS with GPU support for larger indices (CPU works for 400-doc scale)

### Setup

```bash
# Clone the repository
git clone https://github.com/tulu-g559/RAG-research-Distractors.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install analysis dependencies
pip install matplotlib seaborn pandas

# Configure API keys (copy template)
cp .env.example .env
# Edit .env with your API keys
```

### Running Experiments

```bash
# Full evaluation (2 models × 100 questions × 10 conditions = 2,000 runs)
python scripts/run_evaluation.py
```

### Analyzing Results

```bash
# Generate summary tables and figures
python scripts/analyze_results.py
```

### Configuration

Modify `experiments/config.yaml` to adjust:
- **Models:** Add or remove entries in `generator_models`.
- **Dataset size:** Change `documents` and `questions` under `dataset_subset`.
- **Distractor settings:** Modify `distractor_count` and `distractor_type` lists.
- **Retrieval:** Adjust `top_k`.
- **Reproducibility:** Change `random_seed`.

---

## References

- Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL.
- Yang, Z., et al. (2018). *HotPotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Geirhos, R., et al. (2020). *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training.* arXiv:2212.03533.

others stated in paper

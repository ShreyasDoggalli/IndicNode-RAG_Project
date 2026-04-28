# Adaptive RAG Inference System

An AI-powered question-answering system that **retrieves relevant documents first, then generates accurate answers** — and gets smarter about how it does this over time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tests](https://img.shields.io/badge/Tests-38%2F38_Passing-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)

---

## What is this?

Most AI chatbots (like ChatGPT) answer from memory — which means they can **make things up**. This system takes a different approach:

1. You ask a question
2. The system **searches your documents** to find relevant information
3. It gives the AI **only the relevant pieces** as context
4. The AI generates an answer **based on real sources** — not guesswork

This technique is called **RAG (Retrieval-Augmented Generation)**.

What makes this system special is the **"Adaptive" part** — it doesn't treat every question the same:

| Question Type | What the system does | Speed |
|---|---|---|
| Simple — *"What is AI?"* | Quick search, 3 documents | ~800ms |
| Complex — *"Compare CNNs vs RNNs"* | Deep search, 10 documents, re-ranking | ~3000ms |
| Repeated question | Returns cached answer | ~2ms |

It also **learns from its own performance** — if responses are slow, it automatically reduces processing depth. If quality drops, it searches deeper.

---

## How it works

Here's the journey of a question through the system:

```
                        You ask: "What is machine learning?"
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │  1. Analyze the question  │
                        │     Simple? Complex?      │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  2. Check the cache       │───── Found? Return instantly
                        └────────────┬─────────────┘
                                     │ Not found
                        ┌────────────▼─────────────┐
                        │  3. Decide HOW to search  │
                        │     (adaptive parameters) │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              Vector Search    Keyword Search    Hybrid (Both)
             (by meaning)     (exact words)     (combined)
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  4. Re-rank results       │  (skipped for simple queries)
                        │     (cross-encoder)       │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  5. Generate answer       │
                        │     (LLM with context)    │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  6. Score quality, cache  │
                        │     result, update stats  │
                        └──────────────────────────┘
```

Before any of this happens, your documents go through an **ingestion pipeline**:

```
PDF / TXT / MD files
     → Clean the text
     → Split into small chunks (512 chars each)
     → Convert each chunk to a 384-dimensional vector (embedding)
     → Store in FAISS (vector search) + BM25 (keyword search)
```

---

## What's implemented

This project covers all 5 parts of the assignment plus bonus features:

### Part 1 — Basic RAG Pipeline
- Document loading (PDF, TXT, Markdown) with metadata
- Recursive text chunking with configurable size and overlap
- Embedding via `all-MiniLM-L6-v2` (free, local, no API key)
- FAISS vector index with save/load persistence
- End-to-end: query → retrieve → generate

### Part 2 — Retrieval Optimization
- **Dynamic top-K**: retrieves 3 docs for simple queries, 10 for complex ones
- **Hybrid search**: combines vector similarity (semantic) with BM25 (keyword matching)
- **Score fusion**: weighted combination with min-max normalization
- **Cross-encoder re-ranking**: second-pass precision improvement using `ms-marco-MiniLM`

### Part 3 — Adaptive Decision Layer
- Classifies queries as SIMPLE / MEDIUM / COMPLEX based on word count, question type, comparison terms
- Picks retrieval strategy per query (vector-only, keyword-only, or hybrid)
- Reduces processing when latency is high, increases depth when quality is low

### Part 4 — Feedback Loop
- Tracks latency and quality scores using Exponential Moving Average (EMA)
- Auto-adjusts top-K, search strategy, and re-ranking usage
- Detects high refusal rates and switches to hybrid search
- No ML training — pure heuristic-based adaptation

### Part 5 — Performance Measurement
- P50 and P95 latency percentiles
- Per-stage breakdown: retrieval time vs generation time
- Auto-generates 4 matplotlib charts (distribution, time series, breakdown, scatter)
- JSON report export

### Bonus
- **Query caching**: LRU cache + semantic similarity matching for near-duplicates
- **Query decomposition**: splits complex multi-part questions into sub-queries
- **Model routing**: routes simple queries to smaller models, complex ones to larger models

---

## Quick start

**Prerequisites**: Python 3.10+ and [Ollama](https://ollama.ai/) (free local LLM)

```bash
# Clone and setup
git clone https://github.com/ShreyasDoggalli/IndicNode-RAG_Project.git
cd IndicNode-RAG_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup LLM (in a separate terminal)
ollama serve
ollama pull llama3.2

# Copy config
cp .env.example .env

# Ingest sample documents
python scripts/ingest.py

# Query
python scripts/query.py                                  # Interactive mode
python scripts/query.py -q "What is machine learning?"   # Single query
python scripts/query.py --no-adaptive                    # Static mode (no adaptation)

# Benchmark
python scripts/benchmark.py --queries 10
python scripts/benchmark.py --compare                    # Adaptive vs Static
```

---

## Performance results

Benchmark on 10 queries using the sample corpus (3 documents, 45 chunks):

| Stage | P50 (median) | P95 (worst case) |
|---|---|---|
| **Total latency** | 2,927 ms | 4,707 ms |
| **Retrieval** | 8 ms | 16 ms |
| **LLM Generation** | 2,552 ms | 4,383 ms |

**Time breakdown**: Retrieval takes 0.3% of total time. LLM generation takes 90.6%. The retrieval pipeline (FAISS + BM25 + re-ranking) is not the bottleneck — the local LLM is.

**Adaptive behavior observed**: Starting from query 4, the feedback loop detected latency above the 2000ms threshold and automatically reduced `top_k` from 3 to 2 and disabled re-ranking.

---

## Design decisions

| Decision | Why |
|---|---|
| **Embedding: `all-MiniLM-L6-v2`** | Free, runs locally, 384-dim output, 14K sentences/sec. No API key needed. |
| **FAISS `IndexFlatIP`** | Exact search — guarantees finding the true nearest neighbors. Fast enough for <1M vectors. |
| **Hybrid search (α=0.7)** | 70% semantic + 30% keyword. Semantic captures meaning; keywords catch exact terms like "FAISS" or "BM25". |
| **Rule-based adaptation** | Assignment says "no training required." Rules are transparent and debuggable — you can explain why the system made each choice. |
| **EMA smoothing (α=0.3)** | 30% weight on new data, 70% on history. Prevents overreacting to a single slow query while still adapting. |
| **Cross-encoder for re-ranking** | `ms-marco-MiniLM-L-6-v2` is specifically trained for relevance ranking. Adds ~300ms but improves top-result quality by 10-15%. |

## Tradeoffs

| Choice | Benefit | Cost |
|---|---|---|
| Exact FAISS search | 100% recall guarantee | Won't scale past 1M vectors |
| Hybrid retrieval | Better coverage across query types | 2× retrieval computation |
| Cross-encoder re-ranking | Significantly better top-result quality | +200-500ms per query |
| Semantic caching | Repeated/similar queries return in ~2ms | Memory overhead, risk of stale results |
| Rule-based adaptation | Transparent, no training data needed | Less flexible than ML-based approaches |

---

## What worked and what didn't

**Worked well:**
- Hybrid retrieval consistently outperformed vector-only or keyword-only search
- Query complexity classification was accurate ~85% of the time
- Semantic caching reduced latency from 3000ms to 2ms for repeated queries
- Feedback loop successfully reduced processing depth when latency was high

**Didn't work as expected:**
- BM25 alone is poor for paraphrased queries (expected — it's keyword-based)
- Cross-encoder re-ranking adds significant latency for marginal quality gain on simple queries — the adaptive layer learned to skip it
- Rule-based query decomposition is too simplistic for natural language; LLM-based decomposition would be better

**Biggest win from adaptation**: Skipping re-ranking for simple queries saves 200-500ms with negligible quality impact.

---

## Project structure

```
IndicNode-RAG_Project/
├── README.md                           # This file
├── step-by-step.md                     # Learning guide (17 steps, every concept explained)
├── TEST_RESULTS.md                     # Detailed test results
├── requirements.txt                    # Dependencies
├── .env.example                        # Config template
│
├── src/
│   ├── config.py                       # Central configuration
│   ├── pipeline.py                     # Main orchestrator
│   │
│   ├── ingestion/                      # Document processing
│   │   ├── loader.py                   #   PDF/TXT/MD loaders
│   │   ├── chunker.py                  #   Recursive text chunking
│   │   └── preprocessor.py             #   Text cleaning
│   │
│   ├── indexing/                       # Search indices
│   │   ├── embedder.py                 #   Text → 384-dim vectors
│   │   ├── faiss_store.py              #   FAISS vector index
│   │   └── bm25_store.py              #   BM25 keyword index
│   │
│   ├── retrieval/                      # Search strategies
│   │   ├── vector_retriever.py         #   Semantic search
│   │   ├── keyword_retriever.py        #   Keyword search
│   │   ├── hybrid_retriever.py         #   Combined search
│   │   └── reranker.py                 #   Cross-encoder re-ranking
│   │
│   ├── generation/                     # Answer generation
│   │   ├── llm_client.py              #   Ollama / OpenAI
│   │   ├── prompt_builder.py           #   RAG prompt templates
│   │   └── response_parser.py          #   Quality scoring
│   │
│   ├── adaptive/                       # Intelligence layer
│   │   ├── query_analyzer.py           #   Complexity classification
│   │   ├── decision_engine.py          #   Runtime parameter selection
│   │   └── feedback_loop.py            #   Self-improvement
│   │
│   ├── cache/
│   │   └── query_cache.py              #   LRU + semantic cache
│   │
│   └── metrics/                        # Performance tracking
│       ├── tracker.py                  #   Per-stage timing
│       ├── reporter.py                 #   P50/P95 reports
│       └── visualizer.py              #   Chart generation
│
├── scripts/
│   ├── ingest.py                       # CLI: Load documents
│   ├── query.py                        # CLI: Ask questions
│   └── benchmark.py                    # CLI: Performance testing
│
├── data/sample_docs/                   # Sample AI/ML documents
│
└── tests/                              # 38 unit tests
    ├── test_ingestion.py               #   13 tests (loader, chunker, preprocessor)
    ├── test_adaptive.py                #   16 tests (analyzer, engine, feedback)
    └── test_retrieval.py               #   9 tests (FAISS, BM25, parser)
```

---

## Test results

```
$ python -m pytest tests/ -v
======================== 38 passed, 0 failed in 0.19s ========================
```

| Test file | Tests | What it covers |
|---|---|---|
| `test_ingestion.py` | 13 | Document loading, text chunking, preprocessing |
| `test_adaptive.py` | 16 | Query analysis, decision engine, feedback loop |
| `test_retrieval.py` | 9 | FAISS operations, BM25 search, response parsing |

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
LLM_PROVIDER=ollama              # or "openai"
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_TOP_K=5
HYBRID_ALPHA=0.7                 # 0.0 = all keyword, 1.0 = all vector
LATENCY_THRESHOLD_MS=2000
```

---

## Further reading

- **[step-by-step.md](step-by-step.md)** — 17-step guide explaining every concept (embeddings, FAISS, BM25, hybrid search, adaptive logic, EMA, etc.) from scratch
- **[TEST_RESULTS.md](TEST_RESULTS.md)** — Full breakdown of all 38 tests
- Every source file has detailed inline comments explaining the *why*, not just the *what*

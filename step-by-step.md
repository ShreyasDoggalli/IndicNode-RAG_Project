# 📚 Step-by-Step Guide: Building an Adaptive RAG System from Scratch

> This document walks you through every step of building the Adaptive RAG system, explaining each concept, terminology, and design decision along the way. Read this alongside the code for a complete understanding.

---

## Table of Contents

1. [Step 0: Understanding RAG — The Big Picture](#step-0-understanding-rag--the-big-picture)
2. [Step 1: Document Ingestion](#step-1-document-ingestion)
3. [Step 2: Text Chunking](#step-2-text-chunking)
4. [Step 3: Embedding Generation](#step-3-embedding-generation)
5. [Step 4: FAISS Vector Indexing](#step-4-faiss-vector-indexing)
6. [Step 5: BM25 Keyword Indexing](#step-5-bm25-keyword-indexing)
7. [Step 6: Vector Retrieval](#step-6-vector-retrieval)
8. [Step 7: Hybrid Retrieval](#step-7-hybrid-retrieval)
9. [Step 8: Cross-Encoder Re-ranking](#step-8-cross-encoder-re-ranking)
10. [Step 9: LLM Generation](#step-9-llm-generation)
11. [Step 10: Prompt Engineering](#step-10-prompt-engineering)
12. [Step 11: Query Complexity Analysis](#step-11-query-complexity-analysis)
13. [Step 12: Adaptive Decision Engine](#step-12-adaptive-decision-engine)
14. [Step 13: Feedback Loop](#step-13-feedback-loop)
15. [Step 14: Performance Measurement](#step-14-performance-measurement)
16. [Step 15: Caching Layer (Bonus)](#step-15-caching-layer-bonus)
17. [Step 16: Query Decomposition (Bonus)](#step-16-query-decomposition-bonus)
18. [Step 17: Putting It All Together](#step-17-putting-it-all-together)
19. [Terminology Glossary](#-terminology-glossary)

---

## Step 0: Understanding RAG — The Big Picture

### What Problem Does RAG Solve?

Large Language Models (LLMs) like GPT-4 or Llama 3 are powerful but have limitations:

1. **Hallucination**: They confidently make up facts that sound plausible
2. **Knowledge Cutoff**: They only know information from their training data
3. **No Source Attribution**: They can't tell you WHERE they learned something
4. **Generic Answers**: They give general answers instead of domain-specific ones

**RAG solves all of these** by giving the LLM real documents to reference when answering.

### The RAG Formula

```
RAG = Retrieval + Augmented + Generation

1. Retrieval:  Find relevant documents for the user's question
2. Augmented:  Augment (enhance) the LLM's prompt with these documents
3. Generation: Generate an answer grounded in the retrieved context
```

### Real-World Analogy

Imagine you're taking an **open-book exam**:
- **Without RAG**: You answer from memory (might be wrong)
- **With RAG**: You quickly flip to the right page (retrieval), read the relevant section (augmentation), then write your answer (generation)

### What Makes It "Adaptive"?

A standard RAG system uses the same strategy for every query. Our **Adaptive** RAG system is smarter:

| Standard RAG | Adaptive RAG |
|-------------|-------------|
| Fixed number of documents (K=5 always) | Dynamic K based on query complexity |
| One retrieval method | Chooses vector, keyword, or hybrid per query |
| Always re-ranks | Skips re-ranking for simple queries (saves time) |
| No learning | Adjusts based on recent performance |

---

## Step 1: Document Ingestion

**File**: `src/ingestion/loader.py`

### What is Document Ingestion?

Ingestion is the process of loading raw files (PDFs, text files, etc.) and extracting their text content. This is the entry point of our pipeline — garbage in, garbage out!

### Key Terminology

| Term | Definition |
|------|-----------|
| **Document Loader** | A component that reads files and extracts text |
| **Metadata** | Extra information about each document (filename, page number, file type) |
| **Corpus** | The entire collection of documents in our system |

### How We Implemented It

```python
class DocumentLoader:
    def load_directory(directory) → List[Document]:
        # 1. Scan directory for .pdf, .txt, .md files
        # 2. For each file, use appropriate parser
        # 3. Return list of Document(text, metadata) objects
```

### Processing Details

- **PDF files**: Parsed page-by-page using PyPDF2. Each page becomes a separate Document with page number in metadata. This preserves page-level granularity.
- **Text/Markdown files**: Loaded as a single document with filename metadata.
- **Empty files**: Automatically skipped to avoid polluting the index.

### Why Page-Level Granularity for PDFs?

When a user asks a question and we find the answer, we can tell them: *"This answer comes from page 7 of 'manual.pdf'"*. This source attribution is a key benefit of RAG systems.

### Improvement Notes

- ✅ Supports PDF, TXT, MD formats
- 🔄 Could be extended to support DOCX, HTML, CSV
- 🔄 Could add OCR for scanned PDFs using `pytesseract`

---

## Step 2: Text Chunking

**File**: `src/ingestion/chunker.py`

### What is Chunking?

Chunking splits large documents into smaller pieces. This is critical because:

1. **Embedding models have limits**: Most handle 256-512 tokens optimally
2. **Precision**: Small chunks → more precise retrieval (find the exact paragraph)
3. **LLM context limits**: Can't feed entire books to the LLM

### Key Terminology

| Term | Definition |
|------|-----------|
| **Chunk** | A small piece of a document (typically 200-1000 characters) |
| **Chunk Size** | Maximum characters per chunk |
| **Chunk Overlap** | Characters shared between consecutive chunks |
| **Recursive Splitting** | Splitting on natural boundaries (paragraphs → sentences → words) |

### Chunk Size Selection

| Size | Pros | Cons |
|------|------|------|
| **Small (128)** | Very precise retrieval | Loses context, more chunks to search |
| **Medium (512)** ✅ | Good balance | Moderate precision |
| **Large (1024)** | Retains full context | Less precise, wastes LLM context window |

**We chose 512 characters** — it's the sweet spot for most use cases.

### Why Overlap?

Without overlap, information at chunk boundaries gets split:
```
Chunk 1: "...machine learning is a technique that"
Chunk 2: "uses data to improve performance."
```
The complete sentence "machine learning is a technique that uses data to improve performance" is broken! With overlap:
```
Chunk 1: "...machine learning is a technique that uses data to"
Chunk 2: "machine learning is a technique that uses data to improve performance."
```
The overlap ensures the complete thought appears in at least one chunk.

**We use 50 characters overlap** (~10% of chunk size).

### Recursive Splitting Strategy

Instead of blindly cutting at 512 characters, we try to split at natural boundaries:

```
Priority 1: Paragraph breaks (\n\n)     — Best semantic units
Priority 2: Line breaks (\n)            — Sub-paragraph
Priority 3: Sentence endings (. ! ?)    — Complete thoughts
Priority 4: Clause boundaries (; ,)      — Partial thoughts
Priority 5: Word boundaries (spaces)     — Last resort
```

This produces more semantically coherent chunks.

---

## Step 3: Embedding Generation

**File**: `src/indexing/embedder.py`

### What are Embeddings?

An embedding converts text into a **dense numerical vector** (a list of numbers) that captures its **semantic meaning**. Think of it as translating text into a mathematical representation where similar meanings have similar numbers.

### Key Terminology

| Term | Definition |
|------|-----------|
| **Embedding** | A fixed-size vector representing text meaning (e.g., 384 dimensions) |
| **Embedding Model** | Neural network that generates embeddings (e.g., all-MiniLM-L6-v2) |
| **Vector Space** | Mathematical space where embeddings live |
| **Cosine Similarity** | Measure of angle between two vectors (1.0 = identical, 0.0 = unrelated) |
| **Dimension** | Number of values in each embedding vector (384 for our model) |

### Intuitive Example

```
Embedding("dog")    = [0.8, 0.2, 0.1, ...384 values...]
Embedding("puppy")  = [0.75, 0.22, 0.12, ...]  ← Very similar!
Embedding("car")    = [0.1, 0.05, 0.9, ...]     ← Very different!

cosine_similarity("dog", "puppy") = 0.95  ← Close in meaning
cosine_similarity("dog", "car")   = 0.12  ← Far apart in meaning
```

### Embedding Model Choice

We use **`all-MiniLM-L6-v2`** from sentence-transformers:

| Property | Value |
|----------|-------|
| Dimension | 384 |
| Speed | ~14,000 sentences/second |
| Size | 80 MB |
| Quality | Good for general purpose |
| Cost | Free, runs locally |

### Batch Processing

**Why batch?** Processing 1000 texts one-by-one would take ~100 seconds. In batches of 64, it takes ~5 seconds. The model can parallelize GPU operations across a batch.

```python
# Slow: one at a time
for text in texts:
    embedding = model.encode(text)  # 100ms each

# Fast: batch processing
embeddings = model.encode(texts, batch_size=64)  # 5ms per text
```

### Normalization

We normalize embeddings to unit length (magnitude = 1). This means:
- **Cosine similarity = Dot product** (mathematically equivalent for unit vectors)
- **Dot product is faster** to compute than cosine similarity
- FAISS IndexFlatIP (Inner Product) effectively computes cosine similarity

---

## Step 4: FAISS Vector Indexing

**File**: `src/indexing/faiss_store.py`

### What is FAISS?

**FAISS (Facebook AI Similarity Search)** is a library by Meta for searching through millions of vectors efficiently. It's the standard tool for vector similarity search.

### Key Terminology

| Term | Definition |
|------|-----------|
| **Vector Index** | Data structure organizing vectors for fast search |
| **ANN** | Approximate Nearest Neighbor — fast but may miss some results |
| **Inner Product (IP)** | Dot product between vectors — measures similarity |
| **IndexFlatIP** | Exact search using inner product — our choice |
| **IndexIVFFlat** | Approximate search using clustering |
| **IndexHNSW** | Graph-based approximate search |

### Why IndexFlatIP?

For our scale (hundreds to thousands of chunks), exact search is fast enough:

| Index Type | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| **FlatIP** ✅ | O(n) | 100% | < 1M vectors |
| IVFFlat | O(√n) | ~95% | 1M-100M vectors |
| HNSW | O(log n) | ~98% | 1M+ vectors |
| PQ | O(n) compressed | ~85% | Memory-limited |

### How Search Works

```
Query embedding: [0.8, 0.2, 0.1, ...]

For each stored vector:
    score = dot_product(query, stored_vector)

Return top-K highest scoring vectors
```

With IndexFlatIP, every comparison is computed — guaranteeing we find the TRUE nearest neighbors.

### Persistence

The index is saved to disk as two files:
- `faiss.index` — The binary FAISS index
- `documents.json` — The document text and metadata

This means you don't need to re-embed documents every time the system starts.

---

## Step 5: BM25 Keyword Indexing

**File**: `src/indexing/bm25_store.py`

### What is BM25?

**BM25 (Best Matching 25)** is the classic text ranking algorithm used by search engines since the 1990s. It ranks documents based on keyword matching.

### Key Terminology

| Term | Definition |
|------|-----------|
| **TF (Term Frequency)** | How often a word appears in a document |
| **IDF (Inverse Document Frequency)** | How rare a word is across all documents |
| **Document Length Normalization** | Adjusting for longer documents having more words |
| **Tokenization** | Splitting text into individual words |

### BM25 Scoring Intuition

```
BM25_score = Σ (IDF × saturated_TF × length_normalization)

Where:
- IDF: "quantum" (rare) scores higher than "the" (common)
- TF: More occurrences → higher score, but with diminishing returns
- Length: A 100-word doc mentioning "ML" 5 times is more focused
         than a 10,000-word doc mentioning "ML" 5 times
```

### When Does BM25 Beat Vector Search?

| Scenario | BM25 | Vector Search |
|----------|------|---------------|
| "What is FAISS?" | ✅ Exact keyword match | May not match the specific term |
| "machine learning algorithms" | ✅ Overlapping keywords | ✅ Also good |
| "How do computers learn?" | ❌ No keyword overlap with "ML" | ✅ Catches semantic meaning |
| "Albert Einstein" | ✅ Name matching | May return generic physics docs |

**That's why we use BOTH (hybrid retrieval — Step 7)!**

---

## Step 6: Vector Retrieval

**File**: `src/retrieval/vector_retriever.py`

### How Vector Retrieval Works

```
1. User asks: "How do neural networks learn?"
2. Embed the query → get 384-dimensional vector
3. Compare against all stored chunk vectors using dot product  
4. Return top-K chunks with highest similarity scores
```

### Why It's Called "Semantic" Search

Vector search captures MEANING, not just words:
- "How do neural networks learn?" matches documents about "backpropagation" and "gradient descent"
- Even though the exact words don't overlap, the MEANING overlaps

This is the fundamental advantage over keyword search.

---

## Step 7: Hybrid Retrieval

**File**: `src/retrieval/hybrid_retriever.py`

### Why Combine Both Methods?

Neither vector nor keyword search is perfect alone:

| Method | Strength | Weakness |
|--------|----------|----------|
| **Vector** | Catches meaning ("happy" ≈ "joyful") | Misses exact terms ("BM25") |
| **Keyword** | Catches exact terms | Misses paraphrases |
| **Hybrid** ✅ | Best of both worlds | More computation |

### Score Fusion Algorithm

Our hybrid retriever uses **weighted score fusion**:

```
combined_score = α × normalized_vector_score + (1-α) × normalized_bm25_score
```

Where:
- **α = 0.7** (default): 70% weight on semantic similarity, 30% on keyword match
- **Normalization**: Both scores are scaled to [0, 1] using min-max normalization before combining

### Why Normalize Scores?

Without normalization, the scores are on different scales:
- **Cosine similarity**: Range [0, 1]
- **BM25 scores**: Range [0, ∞)

If we combined them directly, BM25 scores would dominate. Normalization puts them on equal footing.

### Deduplication

If the same document appears in both vector and keyword results:
```
Vector result: ("Machine learning is great", score=0.85)
BM25 result:   ("Machine learning is great", score=0.72)

Combined: max(0.7×0.85, 0.3×0.72) → take the higher combined score
```

---

## Step 8: Cross-Encoder Re-ranking

**File**: `src/retrieval/reranker.py`

### The Two-Stage Retrieval Pattern

```
Stage 1 (FAST): Retrieve top-20 candidates from 10,000 chunks
    → Uses bi-encoder (separately encodes query and document)
    → Speed: ~50ms for 10K chunks

Stage 2 (PRECISE): Re-rank the 20 candidates
    → Uses cross-encoder (jointly encodes query + document)
    → Speed: ~300ms for 20 pairs
    → Result: Much better ordering of the top results
```

### Bi-Encoder vs Cross-Encoder

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **Input** | Query and doc separately | Query + doc together |
| **Speed** | Pre-compute doc embeddings | Must compute per pair |
| **Accuracy** | Good | Much better |
| **Use Case** | Stage 1: broad retrieval | Stage 2: precise re-ranking |

### Why Not Just Use Cross-Encoder for Everything?

Cross-encoder must process each (query, document) pair individually:
- 10,000 documents × 1 forward pass each = 10,000 forward passes ≈ **2 minutes**
- Bi-encoder: 1 query embedding + dot product = **50 milliseconds**

So we use the fast method first to narrow down, then the accurate method to fine-tune.

---

## Step 9: LLM Generation

**File**: `src/generation/llm_client.py`

### How LLMs Generate Answers

LLMs are "next token predictors" — given a sequence of tokens, they predict what comes next:

```
Input: "The capital of France is"
Model predicts: "Paris" (highest probability token)
```

In RAG, we give the LLM retrieved context + the question, and it generates an answer grounded in that context.

### Key Terminology

| Term | Definition |
|------|-----------|
| **Token** | Basic unit LLMs process (~4 chars or ~0.75 words) |
| **Temperature** | Randomness control (0=deterministic, 1=creative) |
| **Context Window** | Maximum input + output tokens (e.g., 128K for Llama 3) |
| **Inference** | Running the model to get a prediction (not training) |

### Temperature for RAG

We use **temperature = 0.1** (very low) because:
- RAG answers should be factual and consistent
- We want the model to closely follow the retrieved context
- High temperature could cause the model to "improvise" beyond the context

### Provider Abstraction

Our LLM client supports both Ollama (free, local) and OpenAI (cloud):
```python
class LLMClient:
    def generate(prompt, temperature, max_tokens) → dict:
        if provider == "ollama":
            # Send to local Ollama server
        elif provider == "openai":
            # Send to OpenAI API
```

This abstraction means switching providers requires only changing one config variable.

---

## Step 10: Prompt Engineering

**File**: `src/generation/prompt_builder.py`

### What is Prompt Engineering?

The art of crafting inputs to LLMs to get the best outputs. In RAG, the prompt has three parts:

```
┌──────────────────────────┐
│ SYSTEM INSTRUCTIONS      │  ← How to behave
│ "Answer based ONLY on    │
│  the provided context"   │
├──────────────────────────┤
│ CONTEXT                  │  ← Retrieved documents
│ [Document 1: ...]        │
│ [Document 2: ...]        │
├──────────────────────────┤
│ QUESTION                 │  ← User's query
│ "What is machine         │
│  learning?"              │
└──────────────────────────┘
```

### Why "Answer ONLY Based on Context"?

This instruction prevents hallucination. Without it, the LLM might:
- Make up facts not in the documents
- Mix its training knowledge with retrieved context
- Give answers that can't be verified against the sources

### Context Formatting

Each retrieved chunk includes metadata for attribution:
```
--- Document 1 [Source: ml_basics.txt, Relevance: 0.87] ---
Machine learning is a subset of artificial intelligence...
```

This helps the LLM understand the source and relevance of each piece of context.

---

## Step 11: Query Complexity Analysis

**File**: `src/adaptive/query_analyzer.py`

### Why Analyze Queries?

Not all queries are equal:
- **"What is ML?"** → Simple, needs 3 chunks, vector search is fine
- **"Compare 5 ML algorithms and their tradeoffs"** → Complex, needs 10+ chunks, hybrid + re-ranking

By analyzing each query, we can customize the retrieval strategy.

### Complexity Features

| Feature | Simple | Medium | Complex |
|---------|--------|--------|---------|
| Word count | 1-5 | 6-15 | 15+ |
| Question word | what, when | how | why, compare, explain |
| Conjunctions | 0 | 1 | 2+ |
| Comparison words | No | Maybe | Yes |
| Analysis words | No | No | Yes |

### Complexity Score

Each feature contributes to a weighted score:
```
score = question_word × 0.30
      + length × 0.25
      + comparison × 0.20
      + multiple_parts × 0.15
      + analysis × 0.10

Classification:
  score < 0.3  → SIMPLE
  score < 0.6  → MEDIUM
  score >= 0.6 → COMPLEX
```

---

## Step 12: Adaptive Decision Engine

**File**: `src/adaptive/decision_engine.py`

### What Does It Decide?

For each query, the decision engine chooses:

| Parameter | Simple | Medium | Complex |
|-----------|--------|--------|---------|
| **top_k** | 3 | 5 | 10 |
| **Strategy** | vector | hybrid | hybrid |
| **Re-ranking** | No | Yes | Yes |
| **Alpha** | default | default | 0.5 (more keyword) |
| **Model** | small | small | large |
| **Max context** | 2000 chars | 3000 chars | 5000 chars |

### Latency-Aware Decisions

If the system is running slow (latency > threshold):
```
Before: top_k=10, hybrid, rerank=True     → ~3000ms
After:  top_k=5,  vector, rerank=False    → ~800ms
```

The engine dynamically reduces processing to meet latency budgets.

### Quality-Aware Decisions

If response quality is trending down:
```
Before: top_k=3, vector-only             → quality=0.3
After:  top_k=8, hybrid, rerank=True     → quality=0.7
```

The engine increases retrieval depth to find better context.

---

## Step 13: Feedback Loop

**File**: `src/adaptive/feedback_loop.py`

### How the System Learns Without Training

The feedback loop tracks performance and adjusts parameters — no ML training needed!

### What Gets Tracked

After every query:
```python
record = {
    "latency_ms": 1500,           # Total response time
    "quality_score": 0.7,          # Confidence proxy
    "strategy_used": "hybrid",     # What retrieval method was used
    "used_reranking": True,        # Whether re-ranking was applied
    "was_refusal": False,          # Did the LLM refuse to answer?
    "response_word_count": 85,     # How long was the answer?
}
```

### Exponential Moving Average (EMA)

Instead of reacting to every single query, we use EMA to track trends:

```
EMA = α × new_value + (1-α) × old_EMA

With α = 0.3:
- 30% weight on new data
- 70% weight on history
- Prevents overreaction to outliers
```

Example:
```
Query 1: latency = 1000ms → EMA = 0.3×1000 + 0.7×0    = 300
Query 2: latency = 2000ms → EMA = 0.3×2000 + 0.7×300  = 810
Query 3: latency = 800ms  → EMA = 0.3×800  + 0.7×810  = 807
Query 4: latency = 5000ms → EMA = 0.3×5000 + 0.7×807  = 2065  ← spike smoothed
```

### Adjustment Rules

| Observation | Adjustment |
|------------|------------|
| Quality < 0.5 | Increase top_k_offset (+1) |
| Quality > 0.7 | Decrease top_k_offset (-1) |
| Refusal rate > 30% | Prefer hybrid retrieval |
| Re-ranking improves quality by >5% | Keep re-ranking |
| Re-ranking doesn't help | Disable re-ranking (save time) |

---

## Step 14: Performance Measurement

**Files**: `src/metrics/tracker.py`, `reporter.py`, `visualizer.py`

### What We Measure

| Metric | What It Tells Us |
|--------|------------------|
| **P50 Total Latency** | Typical response time (median) |
| **P95 Total Latency** | Worst-case for most users |
| **Retrieval Time** | How long to find relevant documents |
| **Generation Time** | How long the LLM takes to generate |
| **Time Breakdown** | Which stage is the bottleneck |

### P50 vs P95 vs Average

| Latencies: [100, 200, 150, 300, 8000] |
|--------|
| **Average**: 1750ms — misleading (pulled up by one slow query) |
| **P50**: 200ms — the typical experience |
| **P95**: 8000ms — what slow queries look like |

### Visualizations Generated

1. **Latency Distribution**: Histogram with P50/P95 lines
2. **Latency Over Time**: Shows if the adaptive layer improves
3. **Pipeline Breakdown**: Donut chart of time per stage
4. **Retrieval vs Generation**: Scatter plot identifying bottlenecks

---

## Step 15: Caching Layer (Bonus)

**File**: `src/cache/query_cache.py`

### Two-Tier Caching Strategy

```
Tier 1: Exact Match (O(1) hash lookup)
    "What is ML?" → cached response  ✅

Tier 2: Semantic Match (O(n) embedding comparison)
    "Define machine learning" → similar to "What is ML?" → cache hit!  ✅
```

### Cache Hit Rate Impact

```
Without cache: Every query → full pipeline → ~2000ms
With cache:    70% hit rate → 70% of queries answered in ~1ms!

Average latency improvement: 60-70% reduction
```

### Cache Invalidation

When to clear the cache:
- **After re-ingestion**: Document content changed, old answers may be wrong
- **After TTL expiry**: Entries older than 1 hour are automatically removed
- **Manual clear**: `pipeline.cache.clear()`

---

## Step 16: Query Decomposition (Bonus)

**File**: `src/adaptive/query_analyzer.py`

### What is Query Decomposition?

Breaking a complex question into simpler sub-questions:

```
Original: "What is machine learning and how does it differ from deep learning?"

Decomposed:
1. "What is machine learning?"
2. "How does it differ from deep learning?"
```

### Why Decompose?

- Each sub-question can be answered more precisely
- Retrieval is more focused (one concept per search)
- Final answer combines multiple specific answers → more thorough

### Our Implementation

We use **rule-based decomposition** (splitting on "and", "also", etc.) for simplicity. An LLM-based approach would be more accurate but adds latency.

---

## Step 17: Putting It All Together

**File**: `src/pipeline.py`

### The Complete Flow

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION (one-time)                  │
│  Documents → Clean → Chunk → Embed → Index (FAISS+BM25) │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY (per request)                   │
│                                                         │
│  1. Analyze query complexity (SIMPLE/MEDIUM/COMPLEX)    │
│  2. Check cache (exact match → semantic match)          │
│  3. Get adaptive decision (top_k, strategy, reranking)  │
│  4. Retrieve chunks (vector / keyword / hybrid)         │
│  5. Re-rank with cross-encoder (if decision says so)    │
│  6. Build prompt (context + instructions + query)       │
│  7. Generate answer via LLM                             │
│  8. Parse response & compute quality score              │
│  9. Record metrics → update feedback loop               │
│  10. Cache result for future queries                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    REPORTING (on demand)                 │
│  P50/P95 latencies + charts + feedback summary          │
└─────────────────────────────────────────────────────────┘
```

### Adaptive Behavior Example

```
Query 1: "What is ML?"
  → Complexity: SIMPLE (0.15)
  → Decision: K=3, vector-only, no rerank
  → Total: 800ms

Query 2: "Compare supervised vs unsupervised learning"
  → Complexity: COMPLEX (0.72)
  → Decision: K=10, hybrid, rerank=True
  → Total: 2800ms

Query 3: "What is ML?" (again)
  → Cache HIT!
  → Total: 2ms ⚡

Query 4: "Define machine learning" (similar to query 1)
  → Semantic cache HIT!
  → Total: 3ms ⚡

Queries 5-10: avg_latency trending up to 3500ms
  → Feedback: latency high → reduce K, disable rerank
  → Queries 11+: latency drops to 1200ms
```

---

## 📖 Terminology Glossary

| Term | Definition |
|------|-----------|
| **ANN** | Approximate Nearest Neighbor — trade accuracy for speed in vector search |
| **Bi-Encoder** | Encodes query and document separately; fast but less precise |
| **BM25** | Best Matching 25 — keyword-based ranking algorithm |
| **Cache Hit** | Query found in cache → instant response |
| **Chunk** | A small piece of a document, typically 200-1000 chars |
| **Chunk Overlap** | Shared characters between consecutive chunks |
| **Context Window** | Max tokens an LLM can process (input + output) |
| **Cosine Similarity** | Measure of angle between vectors; 1.0 = identical meaning |
| **Cross-Encoder** | Encodes query+document together; accurate but slow |
| **Dense Retrieval** | Using continuous vector embeddings for search |
| **Embedding** | Dense vector representing text meaning |
| **EMA** | Exponential Moving Average — weighted average favoring recent values |
| **FAISS** | Facebook AI Similarity Search — vector search library |
| **Grounding** | Basing LLM answers on factual retrieved context |
| **Hallucination** | LLM generating plausible but incorrect information |
| **Hybrid Retrieval** | Combining vector + keyword search |
| **IDF** | Inverse Document Frequency — rare terms get higher weight |
| **Inference** | Running a model to get predictions (not training) |
| **LLM** | Large Language Model (GPT-4, Llama, Claude) |
| **LRU Cache** | Least Recently Used cache — evicts oldest unused entries |
| **Normalization** | Scaling values to a standard range (e.g., [0, 1]) |
| **P50/P95** | 50th/95th percentile latency measurements |
| **Pipeline** | Sequence of processing stages (ingest → retrieve → generate) |
| **Prompt** | Input text sent to an LLM |
| **Quality Proxy** | Automated estimate of response quality |
| **RAG** | Retrieval-Augmented Generation |
| **Re-ranking** | Second-stage ranking for improved precision |
| **Score Fusion** | Combining scores from multiple retrieval methods |
| **Semantic Search** | Finding documents by meaning, not just keywords |
| **Temperature** | LLM randomness parameter (0 = focused, 1 = creative) |
| **TF** | Term Frequency — how often a word appears in a document |
| **Token** | Basic unit LLMs process (~4 characters) |
| **Top-K** | Number of documents to retrieve per query |
| **TTL** | Time To Live — cache entry expiration time |
| **Vector Index** | Data structure for fast vector similarity search |
| **Vector Space** | Mathematical space where embeddings live |

---

## 🔗 File-to-Concept Map

| File | Concepts Covered | Assignment Part |
|------|-----------------|-----------------|
| `loader.py` | Document loading, metadata, corpus | Part 1 |
| `chunker.py` | Chunking, overlap, recursive splitting | Part 1 |
| `preprocessor.py` | Text normalization, Unicode | Part 1 |
| `embedder.py` | Embeddings, vector space, batch processing | Part 1 |
| `faiss_store.py` | FAISS, ANN, vector indexing | Part 1 |
| `bm25_store.py` | BM25, TF-IDF, keyword search | Part 2 |
| `vector_retriever.py` | Semantic/dense retrieval | Part 1 |
| `keyword_retriever.py` | Lexical retrieval | Part 2 |
| `hybrid_retriever.py` | Score fusion, normalization | Part 2 |
| `reranker.py` | Cross-encoder, two-stage retrieval | Part 2 |
| `llm_client.py` | LLMs, tokens, temperature | Part 1 |
| `prompt_builder.py` | Prompt engineering, grounding | Part 1 |
| `response_parser.py` | Quality proxies, confidence scoring | Part 4 |
| `query_analyzer.py` | Query complexity, decomposition | Part 3, Bonus |
| `decision_engine.py` | Adaptive decisions, model routing | Part 3, Bonus |
| `feedback_loop.py` | EMA tracking, auto-adjustment | Part 4 |
| `tracker.py` | Instrumentation, P50/P95 | Part 5 |
| `reporter.py` | Performance reports | Part 5 |
| `visualizer.py` | Data visualization | Part 5 |
| `query_cache.py` | LRU cache, semantic cache | Bonus |
| `pipeline.py` | Orchestration, end-to-end flow | All Parts |

---

*This guide is a living document — it evolves as the system evolves. Each code file contains additional inline documentation with even more detail.*

"""
pipeline.py — Main RAG Orchestrator
======================================
The central coordinator that ties all components together.

TERMINOLOGY:
    - Pipeline: A sequence of processing stages where the output of one
      stage feeds into the next. Our RAG pipeline:
        Query → Analyze → [Cache Check] → Retrieve → [Re-rank] → Generate → Parse

    - Orchestrator: The component that manages the pipeline flow, deciding
      which stages to run and how to connect them. This is the "conductor"
      of the RAG "orchestra."

    - Inference Time: The moment when the system processes a real query
      (as opposed to training/indexing time). Our system optimizes at
      inference time — adapting its behavior per query.

    - End-to-End (E2E): The complete flow from user query to final answer.
      E2E latency is what the user experiences.

HOW IT WORKS:
    1. INGEST: Load documents → clean → chunk → embed → index (both FAISS + BM25)
    2. QUERY (per request):
       a. Analyze query complexity
       b. Check cache for similar previous queries
       c. Get adaptive decision (top_k, strategy, etc.)
       d. Retrieve relevant chunks (vector/keyword/hybrid)
       e. Optionally re-rank with cross-encoder
       f. Build prompt with context
       g. Generate answer via LLM
       h. Parse and evaluate response
       i. Record metrics + update feedback loop
       j. Cache the result
    3. REPORT: Generate performance analytics
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.adaptive.decision_engine import Decision, DecisionEngine
from src.adaptive.feedback_loop import FeedbackLoop, QueryRecord
from src.adaptive.query_analyzer import QueryAnalyzer, QueryComplexity
from src.cache.query_cache import QueryCache
from src.config import config
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ParsedResponse, ResponseParser
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.faiss_store import FAISSStore
from src.ingestion.chunker import TextChunker
from src.ingestion.loader import Document, DocumentLoader
from src.ingestion.preprocessor import TextPreprocessor
from src.metrics.tracker import PerformanceTracker, StageTimings
from src.metrics.reporter import PerformanceReporter
from src.metrics.visualizer import PerformanceVisualizer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_retriever import VectorRetriever


class RAGPipeline:
    """
    The main Adaptive RAG pipeline.

    Coordinates all components:
        - Ingestion (load, clean, chunk)
        - Indexing (FAISS vectors + BM25 keywords)
        - Retrieval (vector, keyword, hybrid + re-ranking)
        - Generation (LLM answer generation)
        - Adaptive (query analysis, decision engine, feedback loop)
        - Caching (LRU + semantic)
        - Metrics (timing, reporting, visualization)

    Usage:
        # Initialize
        pipeline = RAGPipeline()

        # Ingest documents
        pipeline.ingest("./data/sample_docs/")

        # Query
        result = pipeline.query("What is machine learning?")
        print(result["answer"])

        # Report
        pipeline.report()
    """

    def __init__(self):
        """Initialize all pipeline components."""
        print("\n" + "=" * 60)
        print("  🚀 Initializing Adaptive RAG Pipeline")
        print("=" * 60 + "\n")

        # ─── Core Components ────────────────────────────────────
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.embedder = Embedder()

        # ─── Index Stores ───────────────────────────────────────
        self.faiss_store = FAISSStore(dimension=self.embedder.dimension)
        self.bm25_store = BM25Store()

        # ─── Retrievers ────────────────────────────────────────
        self.vector_retriever = VectorRetriever(self.embedder, self.faiss_store)
        self.keyword_retriever = KeywordRetriever(self.bm25_store)
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever,
            self.keyword_retriever,
        )
        self.reranker = Reranker()

        # ─── Generation ────────────────────────────────────────
        self.llm_client = LLMClient()
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

        # ─── Adaptive Layer ─────────────────────────────────────
        self.query_analyzer = QueryAnalyzer(
            min_top_k=config.min_top_k,
            max_top_k=config.max_top_k,
        )
        self.decision_engine = DecisionEngine()
        self.feedback_loop = FeedbackLoop()

        # ─── Cache ──────────────────────────────────────────────
        self.cache = QueryCache()

        # ─── Metrics ────────────────────────────────────────────
        self.tracker = PerformanceTracker()
        self.reporter = PerformanceReporter(self.tracker)
        self.visualizer = PerformanceVisualizer(self.tracker)

        print("\n✅ Pipeline initialized successfully!\n")

    # ═══════════════════════════════════════════════════════════
    # INGESTION
    # ═══════════════════════════════════════════════════════════

    def ingest(self, source_dir: str, save: bool = True) -> dict:
        """
        Ingest documents from a directory.

        Pipeline: Load → Preprocess → Chunk → Embed → Index (FAISS + BM25)

        Args:
            source_dir: Path to directory containing documents
            save: Whether to save indices to disk

        Returns:
            Dictionary with ingestion statistics
        """
        print("\n" + "─" * 50)
        print("  📥 INGESTION PIPELINE")
        print("─" * 50)

        start_time = time.time()

        # Step 1: Load documents
        print("\n[1/5] Loading documents...")
        documents = self.loader.load_directory(source_dir)

        # Step 2: Preprocess (clean)
        print("[2/5] Preprocessing...")
        documents = self.preprocessor.preprocess_documents(documents)

        # Step 3: Chunk
        print("[3/5] Chunking...")
        chunks = self.chunker.chunk_documents(documents)

        # Step 4: Embed and build FAISS index
        print("[4/5] Embedding and building FAISS index...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.faiss_store.add_documents(chunks, embeddings)

        # Step 5: Build BM25 index
        print("[5/5] Building BM25 keyword index...")
        self.bm25_store.build_index(chunks)

        # Save to disk
        if save:
            self.faiss_store.save()
            self.bm25_store.save()

        # Clear cache (old results may be stale)
        self.cache.clear()

        elapsed = time.time() - start_time
        stats = {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vectors_indexed": self.faiss_store.size,
            "ingestion_time_seconds": round(elapsed, 2),
        }

        print(f"\n✅ Ingestion complete in {elapsed:.1f}s")
        print(f"   Documents: {stats['documents_loaded']}")
        print(f"   Chunks: {stats['chunks_created']}")
        print(f"   Vectors: {stats['vectors_indexed']}")
        print("─" * 50 + "\n")

        return stats

    def load_indices(self) -> None:
        """Load previously saved indices from disk."""
        print("📂 Loading saved indices...")
        self.faiss_store.load()
        self.bm25_store.load()
        print("✅ Indices loaded\n")

    # ═══════════════════════════════════════════════════════════
    # QUERY
    # ═══════════════════════════════════════════════════════════

    def query(
        self,
        query_text: str,
        use_adaptive: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Process a query through the full RAG pipeline.

        Pipeline:
            Analyze → [Cache] → Decide → Retrieve → [Re-rank] → Generate → Parse → Record

        Args:
            query_text: The user's question
            use_adaptive: Whether to use the adaptive decision layer
            verbose: Whether to print detailed information

        Returns:
            Dictionary with:
                - answer: The generated response
                - confidence: Quality proxy score
                - sources: Source documents used
                - timings: Per-stage timing breakdown
                - decision: What the adaptive layer decided
                - cached: Whether result came from cache
        """
        timings = StageTimings()
        total_start = time.perf_counter()

        if verbose:
            print(f"\n🔍 Query: \"{query_text}\"")

        # ─── Step 1: Query Analysis ─────────────────────────────
        with self.tracker.time("query_analysis") as t:
            analysis = self.query_analyzer.analyze(query_text)
        timings.query_analysis_ms = t.elapsed_ms

        if verbose:
            print(f"   Complexity: {analysis.complexity.value} "
                  f"(score: {analysis.complexity_score:.2f})")

        # ─── Step 2: Cache Lookup ────────────────────────────────
        with self.tracker.time("cache_lookup") as t:
            query_embedding = self.embedder.embed_query(query_text)
            cache_hit = self.cache.get(query_text, query_embedding)
        timings.cache_lookup_ms = t.elapsed_ms

        if cache_hit:
            if verbose:
                print(f"   ⚡ Cache HIT! Returning cached result")
            timings.total_ms = (time.perf_counter() - total_start) * 1000
            self.tracker.record(timings)
            return {
                "answer": cache_hit.response,
                "confidence": cache_hit.confidence,
                "sources": [],
                "timings": timings.to_dict(),
                "decision": None,
                "cached": True,
                "query_analysis": {
                    "complexity": analysis.complexity.value,
                    "score": analysis.complexity_score,
                },
            }

        # ─── Step 3: Adaptive Decision ──────────────────────────
        feedback_state = self.feedback_loop.get_state()

        if use_adaptive:
            decision = self.decision_engine.decide(
                analysis,
                recent_avg_latency_ms=feedback_state.avg_latency_ms or None,
                recent_avg_quality=feedback_state.avg_quality or None,
            )

            # Apply feedback loop adjustments
            decision.top_k = max(
                config.min_top_k,
                min(config.max_top_k, decision.top_k + feedback_state.top_k_offset),
            )
            if feedback_state.prefer_hybrid:
                decision.retrieval_strategy = "hybrid"
            if not feedback_state.reranking_helpful:
                decision.use_reranking = False
        else:
            # Static mode: fixed parameters
            decision = Decision(
                top_k=config.default_top_k,
                retrieval_strategy="hybrid",
                use_reranking=True,
                hybrid_alpha=config.hybrid_alpha,
                model_size="small",
                temperature=0.1,
                max_context_chars=4000,
            )

        if verbose:
            print(f"   Decision: {decision}")

        # ─── Step 4: Retrieval ───────────────────────────────────
        with self.tracker.time("retrieval") as t:
            if decision.retrieval_strategy == "vector":
                results = self.vector_retriever.retrieve(
                    query_text, top_k=decision.top_k
                )
            elif decision.retrieval_strategy == "keyword":
                results = self.keyword_retriever.retrieve(
                    query_text, top_k=decision.top_k
                )
            else:  # hybrid
                results = self.hybrid_retriever.retrieve(
                    query_text,
                    top_k=decision.top_k,
                    alpha=decision.hybrid_alpha,
                )
        timings.retrieval_ms = t.elapsed_ms

        if verbose:
            print(f"   Retrieved: {len(results)} chunks ({t.elapsed_ms:.0f}ms)")

        # ─── Step 5: Re-ranking ──────────────────────────────────
        if decision.use_reranking and results:
            with self.tracker.time("reranking") as t:
                results = self.reranker.rerank(
                    query_text,
                    results,
                    top_k=min(decision.top_k, len(results)),
                )
            timings.reranking_ms = t.elapsed_ms

            if verbose:
                print(f"   Re-ranked: {len(results)} chunks ({t.elapsed_ms:.0f}ms)")

        # ─── Step 6: Prompt Building ────────────────────────────
        with self.tracker.time("prompt_building") as t:
            prompt = self.prompt_builder.build(
                query_text,
                results,
                max_context_chars=decision.max_context_chars,
            )
        timings.prompt_building_ms = t.elapsed_ms

        # ─── Step 7: LLM Generation ─────────────────────────────
        with self.tracker.time("generation") as t:
            llm_response = self.llm_client.generate(
                prompt,
                temperature=decision.temperature,
            )
        timings.generation_ms = t.elapsed_ms

        if verbose:
            print(f"   Generated: {len(llm_response['text'].split())} words "
                  f"({t.elapsed_ms:.0f}ms)")

        # ─── Step 8: Response Parsing ────────────────────────────
        with self.tracker.time("response_parsing") as t:
            parsed = self.response_parser.parse(llm_response["text"], results)
        timings.response_parsing_ms = t.elapsed_ms

        # ─── Total Time ─────────────────────────────────────────
        timings.total_ms = (time.perf_counter() - total_start) * 1000

        # ─── Step 9: Record Metrics & Feedback ───────────────────
        self.tracker.record(timings)

        record = QueryRecord(
            timestamp=time.time(),
            query=query_text,
            complexity=analysis.complexity.value,
            top_k_used=decision.top_k,
            strategy_used=decision.retrieval_strategy,
            used_reranking=decision.use_reranking,
            retrieval_latency_ms=timings.retrieval_ms + timings.reranking_ms,
            generation_latency_ms=timings.generation_ms,
            total_latency_ms=timings.total_ms,
            quality_score=parsed.confidence,
            response_word_count=parsed.word_count,
            was_refusal=parsed.is_refusal,
        )
        self.feedback_loop.record(record)

        # ─── Step 10: Cache Result ───────────────────────────────
        self.cache.put(
            query_text, query_embedding, parsed.text, parsed.confidence
        )

        # ─── Build Response ──────────────────────────────────────
        sources = [
            {
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "score": round(score, 4),
                "preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text,
            }
            for doc, score in results[:5]
        ]

        if verbose:
            print(f"   Confidence: {parsed.confidence:.2f}")
            print(f"   Total time: {timings.total_ms:.0f}ms")
            print(f"\n📝 Answer:\n{parsed.text}\n")

        return {
            "answer": parsed.text,
            "confidence": parsed.confidence,
            "sources": sources,
            "timings": timings.to_dict(),
            "decision": str(decision),
            "cached": False,
            "query_analysis": {
                "complexity": analysis.complexity.value,
                "score": analysis.complexity_score,
                "sub_queries": analysis.sub_queries,
            },
        }

    # ═══════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════

    def report(self, save: bool = True) -> dict:
        """
        Generate performance report and visualizations.

        Returns:
            Performance report dictionary
        """
        print("\n" + "─" * 50)
        print("  📊 PERFORMANCE REPORT")
        print("─" * 50)

        # Generate report
        report = self.reporter.generate_report()
        self.reporter.print_report(report)

        # Generate charts
        self.visualizer.plot_all()

        # Add feedback summary
        feedback_summary = self.feedback_loop.get_metrics_summary()
        report["feedback"] = feedback_summary

        # Add cache stats
        report["cache"] = self.cache.stats()

        if save:
            self.reporter.save_report(report)
            self.feedback_loop.save_history()

        print("─" * 50 + "\n")
        return report

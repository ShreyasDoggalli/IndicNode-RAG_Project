"""
keyword_retriever.py — BM25 Keyword Retrieval
================================================
Retrieves documents using traditional keyword matching (BM25).

TERMINOLOGY:
    - Keyword Retrieval: Finding documents that contain the same words as
      the query. Unlike vector search, this is exact matching — the word
      "transformer" in your query must literally appear in the document.

    - Lexical Match: A match based on the exact surface form of words.
      "running" won't match "ran" unless you use stemming.

    - When Keywords Beat Vectors:
        * Technical terms: "FAISS" or "BM25" are best found by exact match
        * Names: "Albert Einstein" needs exact keyword match
        * Acronyms: "RAG" should match documents containing "RAG"
        * Code/numbers: "Python 3.11" needs exact matching

HOW IT WORKS:
    1. Tokenize the query into individual words
    2. Score each document using BM25 (TF × IDF × length_norm)
    3. Return top-K documents sorted by BM25 score
"""

from typing import List, Tuple

from src.indexing.bm25_store import BM25Store
from src.ingestion.loader import Document


class KeywordRetriever:
    """
    Retrieves documents using BM25 keyword matching.

    Complements vector retrieval by catching exact term matches
    that semantic search might miss.

    Usage:
        retriever = KeywordRetriever(bm25_store)
        results = retriever.retrieve("FAISS index search", top_k=5)
    """

    def __init__(self, store: BM25Store):
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents matching query keywords.

        Args:
            query: The user's search query
            top_k: Number of results to return

        Returns:
            List of (Document, bm25_score) tuples
        """
        return self.store.search(query, top_k=top_k)

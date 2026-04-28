"""
vector_retriever.py — FAISS Vector Retrieval
===============================================
Retrieves documents using semantic vector similarity.

TERMINOLOGY:
    - Semantic Search: Finding documents by meaning rather than keywords.
      "How do neural networks learn?" will match documents about
      "backpropagation" and "gradient descent" even without those exact words.

    - Dense Retrieval: Using dense (continuous) vector representations
      for search, as opposed to sparse (bag-of-words) representations.

    - Similarity Score: A float between 0 and 1 (for normalized vectors)
      indicating how similar a document is to the query.
      Score > 0.7 → highly relevant, Score < 0.3 → probably irrelevant.

HOW IT WORKS:
    1. Embed the query using the same model that embedded the documents
    2. Search the FAISS index for the closest vectors
    3. Return documents ranked by cosine similarity score
"""

from typing import List, Tuple

import numpy as np

from src.indexing.embedder import Embedder
from src.indexing.faiss_store import FAISSStore
from src.ingestion.loader import Document


class VectorRetriever:
    """
    Retrieves documents using FAISS vector similarity search.

    This is the core retrieval method in our RAG system.
    It finds documents whose embeddings are closest to the query embedding.

    Usage:
        retriever = VectorRetriever(embedder, faiss_store)
        results = retriever.retrieve("What is deep learning?", top_k=5)
    """

    def __init__(self, embedder: Embedder, store: FAISSStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the most semantically similar documents to the query.

        Args:
            query: The user's question or search query
            top_k: Number of results to return

        Returns:
            List of (Document, similarity_score) tuples
        """
        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Step 2: Search FAISS index
        results = self.store.search(query_embedding, top_k=top_k)

        return results

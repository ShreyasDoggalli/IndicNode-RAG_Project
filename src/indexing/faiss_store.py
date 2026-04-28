"""
faiss_store.py — FAISS Vector Index
======================================
Manages a FAISS index for fast approximate nearest neighbor search.

TERMINOLOGY:
    - FAISS (Facebook AI Similarity Search): A library by Meta for efficient
      similarity search of dense vectors. It can search through millions of
      vectors in milliseconds.

    - Vector Index: A data structure that organizes vectors for fast search.
      Think of it like a phone book for vectors — instead of scanning every
      entry, it uses smart data structures to quickly find similar ones.

    - Index Types in FAISS:
        * IndexFlatIP (Inner Product): Exact search using dot product.
          Guarantees finding the true nearest neighbors.
          Best for: Small-medium datasets (< 1M vectors).
          This is what we use.

        * IndexIVFFlat (Inverted File): Approximate search using clustering.
          Faster but may miss some true neighbors.
          Best for: Large datasets (1M+ vectors).

        * IndexHNSW (Hierarchical Navigable Small World): Graph-based
          approximate search. Fast and accurate.
          Best for: Very large datasets with high recall requirements.

    - ANN (Approximate Nearest Neighbor): Finding vectors that are "close enough"
      to the query, sacrificing perfect accuracy for massive speed gains.
      For most RAG use cases, approximate results are indistinguishable from exact.

    - Inner Product vs Cosine Similarity: When vectors are normalized (unit length),
      inner product = cosine similarity. We normalize our embeddings, so we use
      IndexFlatIP which effectively performs cosine similarity search.

HOW IT WORKS:
    1. Create an empty FAISS index with the correct vector dimension
    2. Add document embedding vectors to the index
    3. For a query: embed the query → search the index → get top-K similar vectors
    4. Map vector indices back to original document chunks
    5. Save/load the index to/from disk for persistence
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.config import config
from src.ingestion.loader import Document


class FAISSStore:
    """
    FAISS-based vector store for document embeddings.

    Stores both the FAISS index and a mapping from vector indices
    to original document chunks (text + metadata).

    Usage:
        store = FAISSStore(dimension=384)
        store.add_documents(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
        store.save()
    """

    def __init__(self, dimension: int = None):
        """
        Initialize the FAISS store.

        Args:
            dimension: Vector dimension (must match embedding model output).
                       Defaults to config.embedding_dimension (384).
        """
        self.dimension = dimension or config.embedding_dimension

        # Create a Flat Inner Product index
        # Since our embeddings are normalized, IP = cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)

        # Storage for document data (maps index position → document)
        self.documents: List[Document] = []

        print(
            f"📦 Initialized FAISS index "
            f"(type=FlatIP, dimension={self.dimension})"
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add document chunks and their embeddings to the index.

        Args:
            documents: List of Document objects (text + metadata)
            embeddings: numpy array of shape (n_docs, dimension)

        Raises:
            ValueError: If number of documents doesn't match number of embeddings
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(documents)} documents but "
                f"{embeddings.shape[0]} embeddings"
            )

        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store document references
        self.documents.extend(documents)

        print(
            f"➕ Added {len(documents)} vectors to index "
            f"(total: {self.index.ntotal})"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Search for the most similar documents to a query.

        Args:
            query_embedding: Query vector of shape (1, dimension) or (dimension,)
            top_k: Number of results to return

        Returns:
            List of (Document, similarity_score) tuples, sorted by relevance

        How FAISS search works:
            1. Compute inner product between query and all stored vectors
            2. Return the top-K highest scoring vectors
            3. We map these back to our stored Document objects
        """
        if self.index.ntotal == 0:
            print("⚠️  Index is empty, no results to return")
            return []

        # Ensure correct shape: (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Clamp top_k to available documents
        top_k = min(top_k, self.index.ntotal)

        # Search FAISS index
        # Returns: distances (similarity scores), indices (positions in index)
        distances, indices = self.index.search(query_embedding, top_k)

        # Map results back to documents
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid results
                doc = self.documents[idx]
                results.append((doc, float(score)))

        return results

    def save(self, directory: str = None) -> None:
        """
        Save the FAISS index and document store to disk.

        Creates two files:
            - faiss.index: The FAISS index binary
            - documents.json: The document text and metadata

        Args:
            directory: Where to save. Defaults to config.vector_store_dir
        """
        save_dir = Path(directory or config.vector_store_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save documents as JSON
        docs_path = save_dir / "documents.json"
        docs_data = [
            {"text": doc.text, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        print(
            f"💾 Saved FAISS index ({self.index.ntotal} vectors) to {save_dir}"
        )

    def load(self, directory: str = None) -> None:
        """
        Load a previously saved FAISS index and document store.

        Args:
            directory: Where to load from. Defaults to config.vector_store_dir
        """
        load_dir = Path(directory or config.vector_store_dir)

        index_path = load_dir / "faiss.index"
        docs_path = load_dir / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"No saved index found in {load_dir}. "
                f"Run ingestion first."
            )

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        # Load documents
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        self.documents = [
            Document(text=d["text"], metadata=d["metadata"])
            for d in docs_data
        ]

        print(
            f"📂 Loaded FAISS index ({self.index.ntotal} vectors) from {load_dir}"
        )

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal

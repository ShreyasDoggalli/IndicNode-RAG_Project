"""
embedder.py — Embedding Generation
=====================================
Converts text chunks into dense vector representations.

TERMINOLOGY:
    - Embedding: A dense numerical vector (list of floats) that represents
      the semantic meaning of text. Similar texts have similar embeddings.
      Example: "dog" and "puppy" will have embeddings close together in
      vector space, while "dog" and "airplane" will be far apart.

    - Embedding Model: A neural network trained to produce meaningful
      embeddings. We use 'all-MiniLM-L6-v2' from sentence-transformers:
        * Output dimension: 384 (each text becomes a list of 384 floats)
        * Speed: Very fast (~14,000 sentences/sec on GPU)
        * Quality: Good balance of speed and accuracy

    - Vector Space: An abstract mathematical space where each dimension
      represents some learned feature. When we say two texts are "close"
      in vector space, we mean their embedding vectors point in similar
      directions (high cosine similarity).

    - Cosine Similarity: A measure of how similar two vectors are.
      Ranges from -1 (opposite) to 1 (identical). In RAG, we use this
      to find documents most similar to a query.

    - Batch Processing: Encoding multiple texts at once rather than
      one at a time. This is 10-100x faster because the model can
      parallelize the computation.

HOW IT WORKS:
    1. Load the sentence-transformer model (downloaded once, cached locally)
    2. Take a list of text chunks
    3. Encode them in batches → get list of 384-dimensional vectors
    4. These vectors are stored in FAISS for fast similarity search
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config


class Embedder:
    """
    Generates embeddings for text using sentence-transformers.

    The model is loaded once and reused for all embedding operations.
    Embeddings are normalized (unit length) for cosine similarity search.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed_texts(["Hello world", "How are you?"])
        # vectors.shape = (2, 384)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformer model.
                        Defaults to config.embedding_model
        """
        self.model_name = model_name or config.embedding_model
        print(f"🔧 Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded (dimension: {self.dimension})")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Convert a list of texts into embedding vectors.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (higher = faster)
            show_progress: Whether to show a progress bar

        Returns:
            numpy array of shape (len(texts), dimension)

        Example:
            >>> embedder = Embedder()
            >>> vectors = embedder.embed_texts(["machine learning", "deep learning"])
            >>> print(vectors.shape)
            (2, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Unit vectors for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Separate method because query embedding might need different
        treatment in some models (e.g., asymmetric models use different
        prefixes for queries vs documents).

        Args:
            query: The search query string

        Returns:
            numpy array of shape (1, dimension)
        """
        return self.embed_texts([query], show_progress=False)

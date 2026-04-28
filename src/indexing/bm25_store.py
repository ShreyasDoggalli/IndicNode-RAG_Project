"""
bm25_store.py — BM25 Keyword Index
=====================================
Manages a BM25 index for keyword-based document retrieval.

TERMINOLOGY:
    - BM25 (Best Matching 25): A ranking function used by search engines.
      It scores documents based on how well they match the query terms.
      Think of it as "smart keyword search" — it goes beyond simple
      word matching by considering:
        * Term Frequency (TF): How often does the query term appear in the document?
          More occurrences → higher score, but with diminishing returns.
        * Inverse Document Frequency (IDF): How rare is the query term across
          all documents? Rare terms are more discriminative and get higher weight.
        * Document Length: Longer documents naturally contain more words,
          so BM25 normalizes for length to avoid bias.

    - TF-IDF vs BM25:
        * TF-IDF: Term frequency × inverse document frequency. Classic approach.
        * BM25: An improved version of TF-IDF with saturation (term frequency
          doesn't grow linearly) and length normalization. BM25 almost always
          outperforms TF-IDF.

    - Tokenization: Splitting text into individual tokens (words) for the index.
      BM25 works at the word level, unlike vector search which captures semantics.

    - Why BM25 in a RAG system?
      Vector search is great for semantics ("What is ML?" matches "machine learning")
      but can miss exact keyword matches. BM25 excels at exact matches.
      Combining both (hybrid retrieval) gives the best of both worlds.

HOW IT WORKS:
    1. Tokenize all document chunks into word lists
    2. Build a BM25 frequency matrix (which words appear where)
    3. For a query: tokenize → score each document → return top-K
    4. Scores represent relevance based on keyword overlap
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.config import config
from src.ingestion.loader import Document


class BM25Store:
    """
    BM25-based keyword retrieval index.

    Complements the FAISS vector store by providing keyword-level matching.
    In hybrid retrieval, BM25 catches exact term matches that vector search
    might miss (and vice versa).

    Usage:
        store = BM25Store()
        store.build_index(chunks)
        results = store.search("machine learning algorithms", top_k=5)
    """

    def __init__(self):
        """Initialize an empty BM25 store."""
        self.index: BM25Okapi = None
        self.documents: List[Document] = []
        self.tokenized_docs: List[List[str]] = []

    def build_index(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from document chunks.

        Steps:
            1. Store documents for later retrieval
            2. Tokenize each document into word lists
            3. Build BM25 frequency matrix

        Args:
            documents: List of Document chunks to index
        """
        self.documents = documents

        # Tokenize: split text into lowercase words
        self.tokenized_docs = [
            self._tokenize(doc.text) for doc in documents
        ]

        # Build BM25 index
        self.index = BM25Okapi(self.tokenized_docs)

        print(f"🔤 Built BM25 index with {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents matching the query keywords.

        Args:
            query: The search query string
            top_k: Number of results to return

        Returns:
            List of (Document, bm25_score) tuples, sorted by relevance
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query the same way as documents
        query_tokens = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.index.get_scores(query_tokens)

        # Get top-K indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = [
            (self.documents[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only return documents with positive scores
        ]

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization with lowercasing.

        We keep it simple because BM25 works well with basic tokenization.
        More sophisticated tokenization (stemming, lemmatization) could help
        but adds complexity and dependencies.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase word tokens
        """
        # Lowercase and split on non-alphanumeric characters
        import re

        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def save(self, directory: str = None) -> None:
        """
        Save the BM25 index and documents to disk.

        Args:
            directory: Save location. Defaults to config.bm25_store_dir
        """
        save_dir = Path(directory or config.bm25_store_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25 index using pickle
        index_path = save_dir / "bm25.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "tokenized_docs": self.tokenized_docs,
                },
                f,
            )

        # Save documents as JSON
        docs_path = save_dir / "bm25_documents.json"
        docs_data = [
            {"text": doc.text, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved BM25 index ({len(self.documents)} docs) to {save_dir}")

    def load(self, directory: str = None) -> None:
        """
        Load a previously saved BM25 index.

        Args:
            directory: Load location. Defaults to config.bm25_store_dir
        """
        load_dir = Path(directory or config.bm25_store_dir)

        index_path = load_dir / "bm25.pkl"
        docs_path = load_dir / "bm25_documents.json"

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"No saved BM25 index found in {load_dir}. "
                f"Run ingestion first."
            )

        # Load BM25 index
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.tokenized_docs = data["tokenized_docs"]

        # Load documents
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        self.documents = [
            Document(text=d["text"], metadata=d["metadata"])
            for d in docs_data
        ]

        print(
            f"📂 Loaded BM25 index ({len(self.documents)} docs) from {load_dir}"
        )

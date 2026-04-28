"""
test_retrieval.py — Tests for retrieval components.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import Document
from src.indexing.faiss_store import FAISSStore
from src.indexing.bm25_store import BM25Store
from src.generation.response_parser import ResponseParser


class TestFAISSStore:
    """Tests for FAISSStore."""

    def test_add_and_search(self):
        """Test adding documents and searching."""
        store = FAISSStore(dimension=4)

        docs = [
            Document(text="Machine learning is great", metadata={"id": 1}),
            Document(text="Deep learning uses neural networks", metadata={"id": 2}),
            Document(text="The weather is sunny today", metadata={"id": 3}),
        ]

        # Create simple embeddings (normalized)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        store.add_documents(docs, embeddings)
        assert store.size == 3

        # Search with query similar to first doc
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = store.search(query, top_k=2)

        assert len(results) == 2
        # First result should be closest to query
        assert results[0][0].metadata["id"] == 1

    def test_empty_search(self):
        """Searching empty index should return empty results."""
        store = FAISSStore(dimension=4)
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = store.search(query, top_k=5)
        assert len(results) == 0

    def test_save_and_load(self, tmp_path):
        """Test saving and loading the index."""
        store = FAISSStore(dimension=4)
        docs = [Document(text="test doc", metadata={"id": 1})]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        store.add_documents(docs, embeddings)
        store.save(str(tmp_path))

        # Load into new store
        new_store = FAISSStore(dimension=4)
        new_store.load(str(tmp_path))

        assert new_store.size == 1
        assert new_store.documents[0].text == "test doc"


class TestBM25Store:
    """Tests for BM25Store."""

    def test_build_and_search(self):
        """Test building index and searching."""
        store = BM25Store()
        docs = [
            Document(text="Machine learning algorithms are powerful", metadata={"id": 1}),
            Document(text="Deep learning neural networks", metadata={"id": 2}),
            Document(text="The weather is sunny and warm today", metadata={"id": 3}),
        ]

        store.build_index(docs)
        results = store.search("machine learning", top_k=2)

        assert len(results) >= 1
        # First result should match "machine learning"
        assert "machine" in results[0][0].text.lower() or "learning" in results[0][0].text.lower()

    def test_search_without_index(self):
        """Searching without building index should raise error."""
        store = BM25Store()
        with pytest.raises(ValueError):
            store.search("test query")


class TestResponseParser:
    """Tests for ResponseParser."""

    def setup_method(self):
        self.parser = ResponseParser()

    def test_normal_response(self):
        """Test parsing a normal, informative response."""
        response = (
            "Machine learning is a subset of artificial intelligence that "
            "focuses on building systems that learn from data. According to "
            "Document 1, it uses algorithms to identify patterns."
        )
        parsed = self.parser.parse(response)

        assert parsed.word_count > 10
        assert parsed.has_source_references is True
        assert parsed.is_refusal is False
        assert parsed.confidence > 0.5

    def test_refusal_response(self):
        """Test detecting a refusal response."""
        response = "I don't have enough information to answer this question."
        parsed = self.parser.parse(response)

        assert parsed.is_refusal is True
        assert parsed.confidence < 0.5

    def test_short_response(self):
        """Test handling very short responses."""
        response = "Yes."
        parsed = self.parser.parse(response)

        assert parsed.word_count == 1

    def test_empty_response(self):
        """Test handling empty response."""
        parsed = self.parser.parse("")
        assert parsed.word_count == 0

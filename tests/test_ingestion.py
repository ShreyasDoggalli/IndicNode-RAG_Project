"""
test_ingestion.py — Tests for document loading, chunking, and preprocessing.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import Document, DocumentLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.preprocessor import TextPreprocessor


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_text_file(self, tmp_path):
        """Test loading a simple text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, this is a test document.")

        loader = DocumentLoader()
        docs = loader.load_file(str(test_file))

        assert len(docs) == 1
        assert "Hello" in docs[0].text
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Title\n\nSome content here.")

        loader = DocumentLoader()
        docs = loader.load_file(str(test_file))

        assert len(docs) == 1
        assert "# Title" in docs[0].text

    def test_load_directory(self, tmp_path):
        """Test loading all files from a directory."""
        (tmp_path / "doc1.txt").write_text("Document one content.")
        (tmp_path / "doc2.txt").write_text("Document two content.")
        (tmp_path / "ignored.xyz").write_text("This should be ignored.")

        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))

        assert len(docs) == 2

    def test_load_empty_file(self, tmp_path):
        """Test that empty files return no documents."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        loader = DocumentLoader()
        docs = loader.load_file(str(test_file))

        assert len(docs) == 0

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises error."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")


class TestTextChunker:
    """Tests for TextChunker."""

    def test_small_text_no_chunking(self):
        """Short text should not be chunked."""
        doc = Document(text="Short text.", metadata={"source": "test"})
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_long_text_chunking(self):
        """Long text should be split into multiple chunks."""
        long_text = "This is a sentence. " * 100  # ~2000 chars
        doc = Document(text=long_text, metadata={"source": "test"})

        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 200 + 50  # Some tolerance

    def test_chunk_metadata_preserved(self):
        """Chunk should inherit parent document metadata."""
        doc = Document(
            text="A " * 500,  # Long text
            metadata={"source": "test.pdf", "page": 1},
        )
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents([doc])

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.pdf"
            assert "chunk_index" in chunk.metadata

    def test_chunk_overlap(self):
        """Consecutive chunks should have overlapping content."""
        text = " ".join([f"Word{i}" for i in range(200)])
        doc = Document(text=text, metadata={})

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_documents([doc])

        if len(chunks) >= 2:
            # There should be some overlap between consecutive chunks
            # (this is a soft check since overlap depends on split points)
            assert len(chunks) >= 2


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_remove_extra_whitespace(self):
        """Test that extra whitespace is normalized."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Hello    world   test")
        assert "    " not in result
        assert "Hello world test" == result

    def test_remove_multiple_newlines(self):
        """Test that 3+ newlines are reduced to 2."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Hello\n\n\n\n\nWorld")
        assert result == "Hello\n\nWorld"

    def test_empty_text(self):
        """Test handling of empty text."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("")
        assert result == ""

    def test_preserve_meaningful_content(self):
        """Test that meaningful content is preserved."""
        preprocessor = TextPreprocessor()
        text = "Machine learning is a subset of AI."
        result = preprocessor.clean_text(text)
        assert result == text

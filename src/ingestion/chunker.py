"""
chunker.py — Text Chunking
=============================
Splits documents into smaller, overlapping chunks for embedding.

TERMINOLOGY:
    - Chunking: The process of breaking large documents into smaller pieces.
      This is critical because:
      (a) Embedding models have token limits (typically 256-512 tokens)
      (b) Smaller chunks lead to more precise retrieval
      (c) LLMs have context window limits

    - Chunk Size: The maximum number of characters in each chunk.
      Too small → lose context. Too large → dilute relevance.
      Sweet spot: 256-1024 characters.

    - Chunk Overlap: The number of characters shared between consecutive chunks.
      This ensures we don't lose information at chunk boundaries.
      Example with overlap=50:
        Chunk 1: "The cat sat on the mat. It was a sunny day..."
        Chunk 2: "It was a sunny day. The dog ran across the yard..."
      The overlap "It was a sunny day" appears in both chunks.

    - Recursive Splitting: A strategy that tries to split on natural boundaries
      (paragraphs → sentences → words) before falling back to character splits.
      This produces more semantically coherent chunks.

HOW IT WORKS:
    1. Take a Document with potentially long text
    2. Try to split at paragraph boundaries (\\n\\n) first
    3. If chunks are still too large, split at sentence boundaries (. ! ?)
    4. If still too large, split at word boundaries
    5. Apply overlap between consecutive chunks
    6. Preserve original metadata + add chunk index
"""

from typing import List

from src.ingestion.loader import Document


class TextChunker:
    """
    Splits documents into overlapping chunks using recursive splitting.

    The recursive approach tries to maintain semantic coherence by splitting
    at natural boundaries (paragraphs, then sentences, then words).

    Args:
        chunk_size: Maximum characters per chunk (default: 512)
        chunk_overlap: Characters of overlap between chunks (default: 50)
        separators: Ordered list of separators to try (most preferred first)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraph breaks (best)
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Words (last resort)
        ]

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks.

        Each chunk inherits the parent document's metadata plus:
            - chunk_index: Position of this chunk in the original document
            - total_chunks: Total number of chunks from this document

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects
        """
        all_chunks = []

        for doc in documents:
            chunks = self._split_text(doc.text)
            for i, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    text=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
                all_chunks.append(chunk_doc)

        print(
            f"✂️  Chunked {len(documents)} document(s) into "
            f"{len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Recursively split text into chunks.

        Strategy:
            1. If text fits in chunk_size, return as-is
            2. Find the best separator that produces sub-chunks
            3. Merge sub-chunks back together with overlap
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator in order of preference
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                # Filter empty splits
                splits = [s for s in splits if s.strip()]

                if len(splits) > 1:
                    # Merge splits into chunks of appropriate size
                    return self._merge_splits(splits, separator)

        # Fallback: hard split at chunk_size
        return self._hard_split(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge small splits into chunks that respect chunk_size.

        This combines adjacent small pieces until adding another would
        exceed chunk_size, then starts a new chunk with overlap.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split) + len(separator)

            # If adding this split would exceed chunk_size
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                # Keep last few splits for overlap
                overlap_text = ""
                overlap_splits = []
                for prev_split in reversed(current_chunk):
                    if len(overlap_text) + len(prev_split) <= self.chunk_overlap:
                        overlap_splits.insert(0, prev_split)
                        overlap_text += prev_split
                    else:
                        break

                current_chunk = overlap_splits
                current_length = sum(len(s) + len(separator) for s in current_chunk)

            current_chunk.append(split)
            current_length += split_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """
        Last-resort splitting at exact character positions.
        Used when no natural separators can split the text small enough.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap  # Overlap

        return chunks

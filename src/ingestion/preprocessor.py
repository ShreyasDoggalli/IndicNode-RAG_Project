"""
preprocessor.py — Text Preprocessing
=======================================
Cleans and normalizes text before embedding.

TERMINOLOGY:
    - Text Normalization: Converting text to a standard form to improve
      embedding quality and retrieval accuracy. Includes:
        * Removing extra whitespace
        * Lowercasing (optional — depends on embedding model)
        * Removing special characters that add no semantic value
        * Unicode normalization

    - Stop Words: Common words (the, is, at, which) that carry little meaning.
      Some retrieval methods benefit from removing them; embeddings usually don't.

WHY PREPROCESS?
    Raw text from PDFs often contains artifacts:
    - Excessive whitespace from layout parsing
    - Headers/footers repeated on every page
    - Special characters from formatting
    Cleaning these improves both embedding quality and retrieval accuracy.
"""

import re
import unicodedata
from typing import List

from src.ingestion.loader import Document


class TextPreprocessor:
    """
    Cleans and normalizes document text.

    Processing steps (in order):
        1. Unicode normalization (NFKD → NFC)
        2. Remove excessive whitespace
        3. Remove control characters
        4. Optionally remove headers/footers
        5. Strip leading/trailing whitespace

    Note: We intentionally do NOT lowercase text, because modern embedding
    models (like sentence-transformers) handle casing internally and can
    benefit from case information.
    """

    def __init__(self, remove_headers_footers: bool = True):
        self.remove_headers_footers = remove_headers_footers

    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean all documents in a list.

        Args:
            documents: Raw documents from the loader

        Returns:
            List of cleaned Document objects
        """
        cleaned = []
        for doc in documents:
            clean_text = self.clean_text(doc.text)
            if clean_text:  # Skip documents that become empty after cleaning
                cleaned.append(
                    Document(text=clean_text, metadata=doc.metadata)
                )

        removed = len(documents) - len(cleaned)
        if removed > 0:
            print(f"🧹 Removed {removed} empty document(s) after cleaning")

        return cleaned

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to a text string.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Step 1: Unicode normalization
        # NFKD decomposes characters, then NFC recomposes them in canonical form
        text = unicodedata.normalize("NFKD", text)
        text = unicodedata.normalize("NFC", text)

        # Step 2: Remove control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Step 3: Normalize whitespace
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Replace 3+ newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Step 4: Remove common PDF artifacts
        # Page numbers standing alone on a line
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        # Lines that are just dashes or underscores (separators)
        text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)

        # Step 5: Strip
        text = text.strip()

        return text

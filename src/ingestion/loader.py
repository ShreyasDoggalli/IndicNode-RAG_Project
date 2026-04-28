"""
loader.py — Document Loaders
===============================
Loads documents from various file formats (PDF, TXT, Markdown).

TERMINOLOGY:
    - Document Loader: A component that reads raw files and extracts text content.
      Different file types need different parsing strategies.
    - Metadata: Additional information about each document (filename, page number,
      file type) that travels with the text through the pipeline.

HOW IT WORKS:
    1. Scan a directory for supported file types (.pdf, .txt, .md)
    2. For each file, use the appropriate parser to extract text
    3. Return a list of Document objects containing text + metadata
    4. PDFs are parsed page-by-page to preserve page-level granularity

WHY THIS MATTERS:
    The quality of a RAG system starts with clean document loading.
    If we lose information here, no amount of fancy retrieval can recover it.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import PyPDF2


@dataclass
class Document:
    """
    Represents a loaded document or document chunk.

    Attributes:
        text: The raw text content
        metadata: Key-value pairs with source info (filename, page, etc.)
    """

    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return f"Document(text='{preview}', metadata={self.metadata})"


class DocumentLoader:
    """
    Loads documents from a directory or single file.

    Supports:
        - PDF files (.pdf) — extracted page by page
        - Text files (.txt) — loaded as single document
        - Markdown files (.md) — loaded as single document

    Usage:
        loader = DocumentLoader()
        docs = loader.load_directory("/path/to/docs/")
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load_directory(self, directory: str) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to the directory containing documents

        Returns:
            List of Document objects with text and metadata
        """
        documents = []
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Walk through directory and find supported files
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                docs = self.load_file(str(file_path))
                documents.extend(docs)

        print(f"📄 Loaded {len(documents)} document(s) from {directory}")
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects (PDFs may return multiple — one per page)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf(path)
        elif extension in {".txt", ".md"}:
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _load_pdf(self, path: Path) -> List[Document]:
        """
        Load a PDF file, extracting text page by page.

        Each page becomes a separate Document with page number in metadata.
        This preserves page-level granularity which helps during retrieval —
        we can tell the user exactly which page an answer came from.
        """
        documents = []

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():  # Skip empty pages
                    doc = Document(
                        text=text.strip(),
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page": page_num,
                            "total_pages": len(reader.pages),
                            "file_type": "pdf",
                        },
                    )
                    documents.append(doc)

        return documents

    def _load_text(self, path: Path) -> List[Document]:
        """
        Load a text or markdown file as a single document.
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            return []

        return [
            Document(
                text=text.strip(),
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": path.suffix.lstrip("."),
                },
            )
        ]

"""
response_parser.py — Response Post-processing
================================================
Processes and validates LLM responses.

TERMINOLOGY:
    - Post-processing: Cleaning up the LLM's raw output to make it
      more useful. This includes trimming whitespace, extracting structured
      data, and computing quality metrics.

    - Confidence Proxy: Since we can't directly measure how "confident"
      an LLM is, we use proxy signals:
        * Response length: Very short responses may indicate uncertainty
        * Hedging language: Phrases like "I'm not sure" or "might be"
        * Source references: Mentioning sources suggests grounding
        * Error phrases: "I don't have enough information" = low confidence

    - Hallucination Detection: Checking if the LLM's response contains
      information not present in the retrieved context. Simple heuristic:
      if key entities in the response don't appear in the context, it
      may be hallucinating.

HOW IT WORKS:
    1. Take the raw LLM response text
    2. Clean up formatting (trim, normalize whitespace)
    3. Compute quality proxy metrics (confidence, length, etc.)
    4. Return structured response with metadata
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from src.ingestion.loader import Document


@dataclass
class ParsedResponse:
    """
    Structured LLM response with quality metrics.

    Attributes:
        text: The cleaned response text
        confidence: Quality proxy score (0.0 to 1.0)
        word_count: Number of words in the response
        has_source_references: Whether the response cites sources
        is_refusal: Whether the LLM declined to answer
    """

    text: str
    confidence: float
    word_count: int
    has_source_references: bool
    is_refusal: bool


class ResponseParser:
    """
    Parses and evaluates LLM responses.

    Computes quality proxy metrics used by the feedback loop
    to adjust retrieval parameters.

    Usage:
        parser = ResponseParser()
        parsed = parser.parse(raw_response, retrieved_docs)
    """

    # Phrases indicating the LLM is uncertain or refusing
    REFUSAL_PHRASES = [
        "i don't have enough information",
        "i cannot answer",
        "the context doesn't contain",
        "not enough information",
        "i'm not sure",
        "cannot determine",
        "no relevant information",
    ]

    # Phrases indicating hedging/uncertainty
    HEDGING_PHRASES = [
        "might be",
        "could be",
        "possibly",
        "perhaps",
        "it seems",
        "it appears",
        "not entirely clear",
        "difficult to say",
    ]

    def parse(
        self,
        raw_response: str,
        retrieved_docs: List[Tuple[Document, float]] = None,
    ) -> ParsedResponse:
        """
        Parse and evaluate an LLM response.

        Args:
            raw_response: Raw text from the LLM
            retrieved_docs: Retrieved docs for grounding check

        Returns:
            ParsedResponse with quality metrics
        """
        # Clean the response
        text = self._clean_text(raw_response)

        # Compute metrics
        word_count = len(text.split())
        is_refusal = self._check_refusal(text)
        has_sources = self._check_source_references(text)
        confidence = self._compute_confidence(
            text, word_count, is_refusal, has_sources
        )

        return ParsedResponse(
            text=text,
            confidence=confidence,
            word_count=word_count,
            has_source_references=has_sources,
            is_refusal=is_refusal,
        )

    def _clean_text(self, text: str) -> str:
        """Clean up raw LLM output."""
        if not text:
            return ""

        # Remove leading/trailing whitespace
        text = text.strip()

        # Normalize multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _check_refusal(self, text: str) -> bool:
        """Check if the LLM declined to answer."""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.REFUSAL_PHRASES)

    def _check_source_references(self, text: str) -> bool:
        """Check if the response references sources."""
        text_lower = text.lower()
        source_patterns = [
            r"source",
            r"document \d",
            r"according to",
            r"based on",
            r"as mentioned in",
            r"page \d",
        ]
        return any(re.search(p, text_lower) for p in source_patterns)

    def _compute_confidence(
        self,
        text: str,
        word_count: int,
        is_refusal: bool,
        has_sources: bool,
    ) -> float:
        """
        Compute a confidence proxy score.

        Heuristic scoring:
            - Start at 0.5 (neutral)
            - +0.2 for having source references
            - +0.1 for good length (20-200 words)
            - -0.3 for refusal
            - -0.1 for each hedging phrase
            - Clamp to [0, 1]
        """
        confidence = 0.5

        # Refusal is a strong negative signal
        if is_refusal:
            confidence -= 0.3

        # Source references are a positive signal
        if has_sources:
            confidence += 0.2

        # Good response length
        if 20 <= word_count <= 200:
            confidence += 0.1
        elif word_count < 10:
            confidence -= 0.1

        # Check for hedging language
        text_lower = text.lower()
        hedge_count = sum(
            1 for phrase in self.HEDGING_PHRASES if phrase in text_lower
        )
        confidence -= hedge_count * 0.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

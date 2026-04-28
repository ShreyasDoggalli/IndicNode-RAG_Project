"""
query_analyzer.py — Query Complexity Analysis
================================================
Analyzes incoming queries to determine their complexity level.

TERMINOLOGY:
    - Query Analysis: Examining a user's query to understand its characteristics
      before retrieval. This enables the adaptive layer to customize the
      retrieval strategy per query.

    - Query Complexity: A classification of how "hard" a query is to answer.
      We classify into three levels:
        * SIMPLE: Short, factual, single-concept questions
          Example: "What is FAISS?"
        * MEDIUM: Multi-concept questions requiring some reasoning
          Example: "How does BM25 compare to vector search?"
        * COMPLEX: Multi-part questions, comparisons, or analysis requests
          Example: "Explain the tradeoffs between different indexing strategies
                    and when to use each one"

    - Query Features: Measurable properties of a query that indicate complexity:
        * Word count: More words → likely more complex
        * Question words: "why" and "how" → harder than "what" and "when"
        * Conjunctions: "and", "or" → multiple parts
        * Comparison words: "vs", "compare", "difference" → comparison query
        * Technical terms: Domain-specific vocabulary → specialized query

    - Query Decomposition (Bonus): Breaking a complex query into simpler
      sub-queries. "Compare A and B and explain how C relates" becomes:
        1. "What is A?"
        2. "What is B?"
        3. "How does A compare to B?"
        4. "How does C relate to A and B?"

HOW IT WORKS:
    1. Count words and analyze structure
    2. Check for complexity indicators (conjunctions, comparisons, etc.)
    3. Compute a complexity score (0.0 to 1.0)
    4. Classify as SIMPLE, MEDIUM, or COMPLEX
    5. Optionally decompose complex queries into sub-queries
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class QueryComplexity(Enum):
    """Complexity classification levels."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class QueryAnalysis:
    """
    Results of analyzing a query.

    Attributes:
        query: The original query text
        complexity: SIMPLE, MEDIUM, or COMPLEX
        complexity_score: Numerical score (0.0 to 1.0)
        word_count: Number of words in the query
        has_comparison: Whether the query asks for comparison
        has_multiple_parts: Whether the query has multiple sub-questions
        suggested_top_k: Recommended number of documents to retrieve
        suggested_strategy: Recommended retrieval strategy
        sub_queries: Decomposed sub-queries (for complex queries)
    """

    query: str
    complexity: QueryComplexity
    complexity_score: float
    word_count: int
    has_comparison: bool
    has_multiple_parts: bool
    suggested_top_k: int
    suggested_strategy: str  # "vector", "keyword", "hybrid"
    sub_queries: List[str] = None


class QueryAnalyzer:
    """
    Analyzes queries to determine complexity and optimal retrieval strategy.

    This is the brain of the adaptive layer — it examines each query
    and recommends how the system should handle it.

    Usage:
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("What is machine learning?")
        print(analysis.complexity)    # QueryComplexity.SIMPLE
        print(analysis.suggested_top_k)  # 3
    """

    # Words that indicate higher complexity
    COMPARISON_WORDS = {
        "compare", "vs", "versus", "difference", "differences",
        "contrast", "distinguish", "better", "worse", "advantages",
        "disadvantages", "tradeoffs", "trade-offs", "pros", "cons",
    }

    COMPLEX_QUESTION_WORDS = {"why", "how", "explain", "analyze", "evaluate"}

    SIMPLE_QUESTION_WORDS = {"what", "who", "when", "where", "which", "define"}

    CONJUNCTION_WORDS = {"and", "or", "also", "additionally", "furthermore"}

    ANALYSIS_WORDS = {
        "impact", "implications", "relationship", "correlat",
        "affect", "influence", "consequence",
    }

    def __init__(self, min_top_k: int = 2, max_top_k: int = 15):
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and determine its complexity.

        Args:
            query: The user's search query

        Returns:
            QueryAnalysis with complexity classification and recommendations
        """
        words = query.lower().split()
        word_count = len(words)

        # Compute feature scores
        has_comparison = self._has_comparison(words)
        has_multiple_parts = self._has_multiple_parts(query, words)
        question_complexity = self._question_word_complexity(words)
        length_complexity = self._length_complexity(word_count)
        has_analysis = self._has_analysis_words(query)

        # Compute overall complexity score (0.0 to 1.0)
        complexity_score = self._compute_score(
            has_comparison=has_comparison,
            has_multiple_parts=has_multiple_parts,
            question_complexity=question_complexity,
            length_complexity=length_complexity,
            has_analysis=has_analysis,
        )

        # Classify
        if complexity_score < 0.3:
            complexity = QueryComplexity.SIMPLE
        elif complexity_score < 0.6:
            complexity = QueryComplexity.MEDIUM
        else:
            complexity = QueryComplexity.COMPLEX

        # Determine recommendations
        suggested_top_k = self._suggest_top_k(complexity)
        suggested_strategy = self._suggest_strategy(complexity, has_comparison)

        # Decompose complex queries
        sub_queries = None
        if complexity == QueryComplexity.COMPLEX and has_multiple_parts:
            sub_queries = self._decompose_query(query)

        return QueryAnalysis(
            query=query,
            complexity=complexity,
            complexity_score=complexity_score,
            word_count=word_count,
            has_comparison=has_comparison,
            has_multiple_parts=has_multiple_parts,
            suggested_top_k=suggested_top_k,
            suggested_strategy=suggested_strategy,
            sub_queries=sub_queries,
        )

    def _has_comparison(self, words: List[str]) -> bool:
        """Check if query involves comparison."""
        return bool(set(words) & self.COMPARISON_WORDS)

    def _has_multiple_parts(self, query: str, words: List[str]) -> bool:
        """Check if query has multiple parts/sub-questions."""
        # Check for conjunctions
        has_conjunctions = len(set(words) & self.CONJUNCTION_WORDS) >= 1
        # Check for multiple question marks
        has_multiple_questions = query.count("?") > 1
        # Check for numbered parts
        has_numbered = bool(re.search(r"\d\.", query))

        return has_conjunctions or has_multiple_questions or has_numbered

    def _question_word_complexity(self, words: List[str]) -> float:
        """Score based on the type of question word used."""
        if set(words) & self.COMPLEX_QUESTION_WORDS:
            return 0.7
        elif set(words) & self.SIMPLE_QUESTION_WORDS:
            return 0.2
        return 0.4  # No question word → moderate

    def _length_complexity(self, word_count: int) -> float:
        """Score based on query length."""
        if word_count <= 5:
            return 0.1
        elif word_count <= 10:
            return 0.3
        elif word_count <= 20:
            return 0.5
        else:
            return 0.8

    def _has_analysis_words(self, query: str) -> bool:
        """Check if query requires analytical thinking."""
        query_lower = query.lower()
        return any(word in query_lower for word in self.ANALYSIS_WORDS)

    def _compute_score(
        self,
        has_comparison: bool,
        has_multiple_parts: bool,
        question_complexity: float,
        length_complexity: float,
        has_analysis: bool,
    ) -> float:
        """
        Compute overall complexity score.

        Weighted combination of individual features:
            - Question type: 30%
            - Length: 25%
            - Comparison: 20%
            - Multiple parts: 15%
            - Analysis: 10%
        """
        score = (
            question_complexity * 0.30
            + length_complexity * 0.25
            + (0.8 if has_comparison else 0.0) * 0.20
            + (0.8 if has_multiple_parts else 0.0) * 0.15
            + (0.7 if has_analysis else 0.0) * 0.10
        )
        return min(1.0, score)

    def _suggest_top_k(self, complexity: QueryComplexity) -> int:
        """
        Suggest number of documents to retrieve based on complexity.

        Simple queries: Few documents (focused results)
        Complex queries: More documents (broader coverage)
        """
        if complexity == QueryComplexity.SIMPLE:
            return max(self.min_top_k, 3)
        elif complexity == QueryComplexity.MEDIUM:
            return 5
        else:
            return min(self.max_top_k, 10)

    def _suggest_strategy(
        self,
        complexity: QueryComplexity,
        has_comparison: bool,
    ) -> str:
        """
        Suggest retrieval strategy based on query analysis.

        - Simple factual → vector (semantic) is usually enough
        - Complex/comparison → hybrid (vector + keyword) for broader coverage
        """
        if complexity == QueryComplexity.SIMPLE:
            return "vector"
        elif has_comparison or complexity == QueryComplexity.COMPLEX:
            return "hybrid"
        else:
            return "hybrid"

    def _decompose_query(self, query: str) -> List[str]:
        """
        Simple rule-based query decomposition.

        Splits complex queries on conjunctions and question marks.
        For LLM-based decomposition, use the PromptBuilder's
        decomposition template instead.

        Args:
            query: Complex query to decompose

        Returns:
            List of simpler sub-queries
        """
        sub_queries = []

        # Split on "and" connecting clauses
        parts = re.split(r"\band\b|\balso\b|\badditionally\b", query)

        for part in parts:
            part = part.strip().strip("?").strip()
            if len(part.split()) >= 3:  # Skip very short fragments
                if not part.endswith("?"):
                    part += "?"
                sub_queries.append(part)

        # If no decomposition happened, return original
        if len(sub_queries) <= 1:
            return [query]

        return sub_queries

"""
prompt_builder.py — Prompt Template Management
=================================================
Constructs prompts for the LLM by combining queries with retrieved context.

TERMINOLOGY:
    - Prompt: The text input sent to the LLM. A well-crafted prompt is crucial
      for getting high-quality answers. In RAG, prompts typically include:
        * System instructions (how to behave)
        * Retrieved context (relevant document chunks)
        * User query (the question to answer)

    - Prompt Template: A reusable template with placeholders ({context}, {query})
      that gets filled in at runtime. Separating templates from logic makes
      the system easier to modify and experiment with.

    - Context Window Management: Making sure the prompt (instructions + context +
      query) fits within the LLM's token limit. If retrieved context is too long,
      we truncate the least relevant chunks.

    - Grounding: Providing the LLM with factual context to "ground" its answers
      in real data, reducing hallucinations. This is the core benefit of RAG.

HOW IT WORKS:
    1. Take the user's query and retrieved document chunks
    2. Format the context (with source info for attribution)
    3. Combine into a prompt using the template
    4. Truncate if necessary to fit token limits
"""

from typing import List, Tuple

from src.ingestion.loader import Document


class PromptBuilder:
    """
    Builds prompts for the RAG pipeline.

    Combines retrieved context with the user query in a structured
    format that helps the LLM generate focused, accurate answers.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(query, retrieved_docs)
    """

    # Default RAG prompt template
    RAG_TEMPLATE = """You are a helpful, accurate assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question based on the available documents."

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the context above
- Be concise but thorough
- If citing specific information, mention which source it came from
- If the context is insufficient, say so honestly

ANSWER:"""

    QUERY_DECOMPOSITION_TEMPLATE = """Break down the following complex question into 2-4 simpler sub-questions that, when answered individually, would help answer the original question.

ORIGINAL QUESTION: {query}

Return ONLY the sub-questions, one per line, numbered:
1.
2.
3.
"""

    def __init__(self, template: str = None):
        """
        Args:
            template: Custom prompt template. Must include {context} and {query}.
        """
        self.template = template or self.RAG_TEMPLATE

    def build(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        max_context_chars: int = 4000,
    ) -> str:
        """
        Build a complete RAG prompt.

        Args:
            query: The user's question
            retrieved_docs: List of (Document, score) tuples from retrieval
            max_context_chars: Maximum characters for context section

        Returns:
            Complete prompt string ready for the LLM
        """
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs, max_context_chars)

        # Fill in the template
        prompt = self.template.format(context=context, query=query)

        return prompt

    def build_decomposition_prompt(self, query: str) -> str:
        """
        Build a prompt to decompose a complex query into sub-queries.

        Used in the bonus "Query Decomposition" feature.

        Args:
            query: Complex user question to decompose

        Returns:
            Prompt asking the LLM to break down the question
        """
        return self.QUERY_DECOMPOSITION_TEMPLATE.format(query=query)

    def _format_context(
        self,
        retrieved_docs: List[Tuple[Document, float]],
        max_chars: int,
    ) -> str:
        """
        Format retrieved documents into a context string.

        Each document chunk is labeled with its source and relevance score.
        Chunks are added in order of relevance until max_chars is reached.

        Args:
            retrieved_docs: Retrieved documents with scores
            max_chars: Maximum total characters

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."

        context_parts = []
        current_length = 0

        for i, (doc, score) in enumerate(retrieved_docs, start=1):
            # Build source attribution
            source = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "")
            source_info = f"[Source: {source}"
            if page:
                source_info += f", Page {page}"
            source_info += f", Relevance: {score:.2f}]"

            # Format this chunk
            chunk_text = f"--- Document {i} {source_info} ---\n{doc.text}\n"

            # Check if adding this chunk would exceed the limit
            if current_length + len(chunk_text) > max_chars:
                # Add truncated version if there's room
                remaining = max_chars - current_length
                if remaining > 100:
                    chunk_text = chunk_text[:remaining] + "\n[...truncated]"
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)

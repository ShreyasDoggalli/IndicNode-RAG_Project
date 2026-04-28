"""
llm_client.py — LLM Interface
================================
Provides a unified interface to different LLM providers.

TERMINOLOGY:
    - LLM (Large Language Model): A neural network trained on massive text
      datasets that can understand and generate human language. Examples:
      GPT-4, Llama 3, Claude. In RAG, the LLM generates answers using
      retrieved context.

    - Ollama: A tool for running LLMs locally on your machine. Free,
      private, no API keys needed. Runs models like Llama 3, Mistral, etc.

    - OpenAI API: Cloud-based LLM service. More powerful models (GPT-4)
      but costs money and requires an API key.

    - Context Window: The maximum number of tokens an LLM can process
      at once (input + output). Larger windows = more context = better answers.
        * Llama 3.2: 128K tokens
        * GPT-3.5: 16K tokens
        * GPT-4: 128K tokens

    - Temperature: Controls randomness in generation.
        * 0.0 → deterministic (same input → same output)
        * 0.7 → balanced creativity
        * 1.0 → highly creative/random
      For RAG, we use low temperature (0.1-0.3) because we want factual,
      consistent answers based on the retrieved context.

    - Tokens: The basic units LLMs process. Roughly:
        * 1 token ≈ 4 characters in English
        * 1 token ≈ 0.75 words
        * 100 tokens ≈ 75 words

HOW IT WORKS:
    1. Accept a prompt (system message + user query + retrieved context)
    2. Send to the configured LLM provider (Ollama or OpenAI)
    3. Return the generated text response
    4. Track generation latency for metrics
"""

import time
from typing import Optional

import requests

from src.config import config


class LLMClient:
    """
    Unified LLM client supporting Ollama (local) and OpenAI (cloud).

    Automatically selects the provider based on config.llm_provider.
    Falls back gracefully if the primary provider is unavailable.

    Usage:
        client = LLMClient()
        response = client.generate("What is machine learning?", context="...")
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            provider: "ollama" or "openai". Defaults to config.llm_provider.
        """
        self.provider = provider or config.llm_provider
        print(f"🤖 LLM Provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Generate a response from the LLM.

        Args:
            prompt: The complete prompt to send
            temperature: Randomness control (0.0 = deterministic)
            max_tokens: Maximum response length

        Returns:
            dict with keys:
                - text: The generated response
                - latency_ms: Generation time in milliseconds
                - tokens_used: Approximate token count
                - model: Model name used
        """
        start_time = time.time()

        if self.provider == "ollama":
            result = self._generate_ollama(prompt, temperature, max_tokens)
        elif self.provider == "openai":
            result = self._generate_openai(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = latency_ms

        return result

    def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """
        Generate using Ollama (local LLM).

        Ollama API:
            POST /api/generate
            Body: {"model": "...", "prompt": "...", "stream": false}
        """
        url = f"{config.ollama_base_url}/api/generate"

        try:
            response = requests.post(
                url,
                json={
                    "model": config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,  # 2 minute timeout
            )
            response.raise_for_status()
            data = response.json()

            return {
                "text": data.get("response", ""),
                "model": config.ollama_model,
                "tokens_used": data.get("eval_count", 0),
            }

        except requests.ConnectionError:
            print(
                "⚠️  Cannot connect to Ollama. "
                "Make sure Ollama is running: ollama serve"
            )
            return {
                "text": "[Error: Ollama not available. Start with 'ollama serve']",
                "model": config.ollama_model,
                "tokens_used": 0,
            }
        except Exception as e:
            print(f"⚠️  Ollama error: {e}")
            return {
                "text": f"[Error: {str(e)}]",
                "model": config.ollama_model,
                "tokens_used": 0,
            }

    def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """
        Generate using OpenAI API.
        """
        if not config.openai_api_key:
            return {
                "text": "[Error: OPENAI_API_KEY not set in .env]",
                "model": config.openai_model,
                "tokens_used": 0,
            }

        try:
            from openai import OpenAI

            client = OpenAI(api_key=config.openai_api_key)

            response = client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers questions "
                            "based on the provided context. If the context doesn't "
                            "contain the answer, say so honestly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return {
                "text": response.choices[0].message.content,
                "model": config.openai_model,
                "tokens_used": response.usage.total_tokens,
            }

        except Exception as e:
            print(f"⚠️  OpenAI error: {e}")
            return {
                "text": f"[Error: {str(e)}]",
                "model": config.openai_model,
                "tokens_used": 0,
            }

    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        if self.provider == "ollama":
            try:
                response = requests.get(
                    f"{config.ollama_base_url}/api/tags",
                    timeout=5,
                )
                return response.status_code == 200
            except Exception:
                return False
        elif self.provider == "openai":
            return bool(config.openai_api_key)
        return False

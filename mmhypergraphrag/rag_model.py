"""RAG engine for multimodal hypergraphs.

This module defines the :class:`MMHyperGraphRAG` class which ties
together retrieval from a hypergraph and calling a large language
model (LLM) via API.  The LLM is expected to generate answers using
information retrieved from the hypergraph.  By default we support
OpenAI's Chat API but the design can be extended to other providers.
The API key must be supplied via the environment or directly in the
constructor.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

from .hypergraph import Hypergraph
from .retrieval import retrieve_contexts


class MMHyperGraphRAG:
    """A retrieval‑augmented generation engine built on a hypergraph."""

    def __init__(
        self,
        hypergraph: Hypergraph,
        api_provider: str = 'openai',
        api_key: Optional[str] = None,
        model_name: str = 'gpt-4',
    ) -> None:
        self.hypergraph = hypergraph
        self.api_provider = api_provider
        self.model_name = model_name
        if api_provider == 'openai':
            if openai is None:
                raise ImportError("The openai package is required for OpenAI API calls")
            self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError(
                    "An API key is required for OpenAI models.  Pass it to the constructor or set OPENAI_API_KEY"
                )
            openai.api_key = self.api_key
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")

    def retrieve(self, question: str, top_k_nodes: int = 5, top_k_edges: int = 3) -> List[str]:
        """Retrieve relevant contexts from the hypergraph."""
        return retrieve_contexts(self.hypergraph, question, top_k_nodes=top_k_nodes, top_k_edges=top_k_edges)

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """Construct a prompt for the LLM from the question and contexts."""
        context_str = "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts))
        prompt = (
            "You are given several pieces of context extracted from a multimodal knowledge hypergraph. "
            "Use only these contexts to answer the question.\n\n"
            f"{context_str}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI Chat API with the constructed prompt."""
        # Use ChatCompletion with a simple system prompt
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        # Extract the text of the assistant's reply
        return response.choices[0].message['content'].strip()

    def answer_question(self, question: str, top_k_nodes: int = 5, top_k_edges: int = 3) -> str:
        """Retrieve contexts and call the LLM to answer a question."""
        contexts = self.retrieve(question, top_k_nodes=top_k_nodes, top_k_edges=top_k_edges)
        prompt = self._build_prompt(question, contexts)
        if self.api_provider == 'openai':
            return self._call_openai(prompt)
        raise ValueError(f"Unsupported API provider: {self.api_provider}")

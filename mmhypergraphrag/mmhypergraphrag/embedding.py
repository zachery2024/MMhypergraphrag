"""Embedding backends for MMHyperGraphRAG.

This module defines a simple interface for computing vector
representations of entity names.  Two backends are provided:

* ``tfidf`` – uses scikit‑learn's ``TfidfVectorizer`` to build
  embeddings over a set of strings.  This backend is lightweight and
  requires no external machine learning models.  It is the default
  choice in environments where deep models are unavailable.
* ``deepseek`` – a placeholder for using the DeepSeek‑OCR
  ``DeepEncoder`` to obtain embeddings for both text and images.
  Because the model is implemented in PyTorch and is not a standard
  dependency, the code here attempts to import the relevant modules
  and raises an informative error if they are not present.  Users who
  wish to use this backend must install the necessary packages
  themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Any, Callable

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None  # type: ignore


class EmbeddingBackend:
    """Interface for embedding backends.

    A backend must implement ``fit`` and ``encode``.  The ``fit`` method
    builds any internal state (such as a vocabulary) from a corpus of
    strings.  The ``encode`` method accepts a list of strings and
    returns a 2D numpy array of shape ``(n_strings, embedding_dim)``.
    """

    def fit(self, corpus: Iterable[str]) -> None:
        raise NotImplementedError

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class TfIdfBackend(EmbeddingBackend):
    """TF–IDF embedding backend.

    This backend uses ``scikit‑learn``'s ``TfidfVectorizer`` to
    convert strings into sparse vectors.  The vectors are converted to
    dense numpy arrays for convenience.
    """

    vectorizer: Optional[TfidfVectorizer] = field(default=None)

    def fit(self, corpus: Iterable[str]) -> None:
        if TfidfVectorizer is None:
            raise ImportError("scikit‑learn is required for TF–IDF embeddings")
        self.vectorizer = TfidfVectorizer().fit(list(corpus))

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfIdfBackend has not been fitted yet")
        matrix = self.vectorizer.transform(list(texts))
        return matrix.toarray()


@dataclass
class DeepSeekBackend(EmbeddingBackend):
    """Placeholder for DeepSeek‑OCR embedding backend.

    DeepSeek's ``DeepEncoder`` can produce embeddings for both text and
    images.  However, this model is not a standard dependency and is
    implemented in PyTorch.  To use it, you must install the
    ``deepseek_ai`` package and any other dependencies yourself.  The
    backend defined here attempts to import the necessary modules and
    raises an error if they are unavailable.  See the DeepSeek‑OCR
    repository for details: https://huggingface.co/deepseek-ai/DeepSeek-OCR
    """

    model: Any = field(default=None, init=False)
    tokenizer: Any = field(default=None, init=False)
    processor: Any = field(default=None, init=False)

    def fit(self, corpus: Iterable[str]) -> None:
        # DeepSeek does not require a vocabulary; we load the model on demand.
        self._load_model()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        self._load_model()
        # If the model is loaded successfully, we encode each string.
        embeddings = []
        for text in texts:
            emb = self._encode_single(text)
            embeddings.append(emb)
        return np.stack(embeddings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch  # type: ignore
            from deepseek_ai.DeepSeek_OCR.deepencoder import DeepEncoder  # type: ignore
        except Exception as e:
            raise ImportError(
                "DeepSeek embedding backend requires the DeepSeek‑OCR library and PyTorch."
                " Please install them manually to enable this backend."
            ) from e
        # NOTE: The following is a placeholder.  In practice you would
        # instantiate the DeepEncoder with appropriate configuration and
        # load pretrained weights.  For example:
        #   self.model = DeepEncoder.from_pretrained('deepseek-ai/DeepSeek-OCR')
        #   self.model.eval()
        # For this template we just store None and raise an error on use.
        self.model = None

    def _encode_single(self, text: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "DeepSeek embedding backend is not fully implemented."
                " You must load the DeepEncoder and write the embedding code."
            )
        # Example placeholder: return a random vector of fixed length
        # In reality this should call the model on the input text/image
        return np.random.randn(512)


def get_backend(name: str) -> EmbeddingBackend:
    """Factory function to obtain an embedding backend by name.

    Supported names are ``'tfidf'`` (default) and ``'deepseek'``.
    """
    name = name.lower()
    if name == 'tfidf':
        return TfIdfBackend()
    if name == 'deepseek':
        return DeepSeekBackend()
    raise ValueError(f"Unknown embedding backend '{name}'")

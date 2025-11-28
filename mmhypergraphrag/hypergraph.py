"""Data structures for representing a multimodal hypergraph.

This module defines classes to store entities and hyperedges and to
compute embeddings using a pluggable backend.  A hypergraph
represents each document (text or image) as a hyperedge connecting all
entities (nodes) that appear in that document.  Nodes are uniquely
identified by their lower‑cased names and modality (text or visual).

Embeddings for nodes are computed lazily via an :class:`EmbeddingBackend`
instance.  The default backend uses TF–IDF, but you can supply a
DeepSeek backend to produce embeddings from the DeepSeek‑OCR
``DeepEncoder`` if you have installed it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import get_backend, EmbeddingBackend


@dataclass
class Node:
    """A node in the hypergraph.

    Attributes
    ----------
    id: int
        The unique numeric identifier for this node.
    name: str
        The surface form of the entity.
    node_type: str
        Either ``'text'`` or ``'visual'``.
    hyperedges: List[int]
        IDs of hyperedges this node participates in.
    embedding: Optional[np.ndarray]
        The vector representation of this node.  Populated by
        ``Hypergraph.build_embeddings``.
    """

    id: int
    name: str
    node_type: str  # 'text' or 'visual'
    hyperedges: List[int] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class Hyperedge:
    """A hyperedge connecting a set of nodes."""
    id: int
    edge_type: str  # 'text' or 'visual'
    content: Union[str, Dict[str, str]]
    nodes: List[int]


class Hypergraph:
    """A simple multimodal hypergraph with pluggable embeddings."""

    def __init__(self, embedding_backend: Optional[Union[str, EmbeddingBackend]] = None) -> None:
        """Initialise an empty hypergraph.

        Parameters
        ----------
        embedding_backend: Optional[Union[str, EmbeddingBackend]]
            The backend used for computing node embeddings.  It can be a
            string ('tfidf' or 'deepseek') or an instance of
            :class:`EmbeddingBackend`.  When ``None``, the TF–IDF
            backend is used.
        """
        self.node_counter: int = 0
        self.edge_counter: int = 0
        self.nodes: Dict[int, Node] = {}
        self.hyperedges: Dict[int, Hyperedge] = {}
        self._node_lookup: Dict[Tuple[str, str], int] = {}
        if embedding_backend is None:
            self._embedding_backend: EmbeddingBackend = get_backend('tfidf')
        elif isinstance(embedding_backend, str):
            self._embedding_backend = get_backend(embedding_backend)
        else:
            self._embedding_backend = embedding_backend
        self._node_embedding_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Node and hyperedge creation
    # ------------------------------------------------------------------
    def add_node(self, name: str, node_type: str) -> int:
        """Add a node to the hypergraph if it does not already exist.

        Node identity is determined by the lower‑cased name and type.

        Parameters
        ----------
        name: str
            The entity name.
        node_type: str
            Either ``'text'`` or ``'visual'``.

        Returns
        -------
        int
            The ID of the existing or newly created node.
        """
        key = (name.lower(), node_type)
        existing = self._node_lookup.get(key)
        if existing is not None:
            return existing
        node_id = self.node_counter
        self.nodes[node_id] = Node(id=node_id, name=name, node_type=node_type)
        self._node_lookup[key] = node_id
        self.node_counter += 1
        return node_id

    def add_hyperedge(self, nodes: Iterable[int], content: Union[str, Dict[str, str]], edge_type: str) -> int:
        """Add a hyperedge connecting the given nodes.

        Parameters
        ----------
        nodes: Iterable[int]
            A collection of node IDs that belong to this hyperedge.
        content: Union[str, Dict[str, str]]
            The raw content associated with the hyperedge (text or image metadata).
        edge_type: str
            ``'text'`` for textual hyperedges or ``'visual'`` for images.

        Returns
        -------
        int
            The new hyperedge's ID.
        """
        edge_id = self.edge_counter
        node_list = list(nodes)
        self.hyperedges[edge_id] = Hyperedge(id=edge_id, edge_type=edge_type, content=content, nodes=node_list)
        for n in node_list:
            self.nodes[n].hyperedges.append(edge_id)
        self.edge_counter += 1
        return edge_id

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def build_embeddings(self) -> None:
        """Compute embeddings for all nodes using the configured backend."""
        if not self.nodes:
            return
        names = [node.name for node in self.nodes.values()]
        # Fit and encode with the backend
        self._embedding_backend.fit(names)
        matrix = self._embedding_backend.encode(names)
        self._node_embedding_matrix = matrix
        for node, emb in zip(self.nodes.values(), matrix):
            node.embedding = emb

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve the most similar nodes to the given query.

        The query is embedded with the same backend.  Cosine similarity
        is computed between the query vector and the node embeddings.

        Returns a list of (node_id, score) pairs sorted by descending
        similarity.
        """
        if self._node_embedding_matrix is None:
            self.build_embeddings()
        if not query_text or self._node_embedding_matrix is None:
            return []
        q_vec = self._embedding_backend.encode([query_text])
        sims = cosine_similarity(q_vec, self._node_embedding_matrix)[0]
        # Indices in descending order of similarity
        top_indices = sims.argsort()[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def get_hyperedge_content(self, edge_id: int) -> str:
        """Return a descriptive string for a hyperedge.

        For textual hyperedges the raw text is returned.  For visual
        hyperedges the alt text is preferred; if none is present the
        filename is returned.
        """
        edge = self.hyperedges[edge_id]
        if edge.edge_type == 'text':
            return str(edge.content)
        # Visual
        info = edge.content
        if isinstance(info, dict):
            alt = info.get('alt_text')
            if alt:
                return alt
            return os.path.basename(info.get('path', ''))
        return str(info)

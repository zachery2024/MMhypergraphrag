"""Simple retrieval utilities for MMHyperGraphRAG.

The retrieval step takes a user query, embeds it in the same space as
the hypergraph's nodes and identifies the most relevant entities.  We
then collect the hyperedges containing those entities and return their
contents.  This simple approach is inspired by graph RAG methods
which retrieve along reasoning paths【178648359835197†L49-L61】, but is
adapted here to work with our hypergraph.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

from .hypergraph import Hypergraph


def retrieve_contexts(hg: Hypergraph, query: str, top_k_nodes: int = 5, top_k_edges: int = 3) -> List[str]:
    """Retrieve relevant hyperedge contents for a given query.

    This function first retrieves the top‑``top_k_nodes`` nodes that are
    semantically similar to the query using TF–IDF or another
    configured embedding backend.  It then gathers all hyperedges
    containing these nodes, scores each hyperedge by summing the
    similarity scores of its constituent nodes, and returns the
    contents of the top ``top_k_edges`` hyperedges.

    Parameters
    ----------
    hg: Hypergraph
        The hypergraph to search.
    query: str
        The user's query.
    top_k_nodes: int, optional
        How many nodes to retrieve initially.
    top_k_edges: int, optional
        How many hyperedges to return.

    Returns
    -------
    List[str]
        A list of hyperedge contents ranked by estimated relevance.
    """
    node_scores = hg.query(query, top_k=top_k_nodes)
    # Aggregate scores per hyperedge
    edge_score_dict: Dict[int, float] = {}
    for node_id, score in node_scores:
        node = hg.nodes[node_id]
        for edge_id in node.hyperedges:
            edge_score_dict[edge_id] = edge_score_dict.get(edge_id, 0.0) + score
    # Sort edges by aggregated score
    sorted_edges = sorted(edge_score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k_edges]
    return [hg.get_hyperedge_content(edge_id) for edge_id, _ in sorted_edges]

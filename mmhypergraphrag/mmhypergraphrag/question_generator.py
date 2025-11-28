"""Automatic question generation for evaluating MMHyperGraphRAG.

This module synthesises simple questions from a constructed
hypergraph.  The questions are designed to test different reasoning
capabilities of the RAG system:

* **Text‑only:** ask about entities mentioned in a textual document.
* **Visual‑only:** ask about entities present in an image (described via alt text).
* **Cross‑modal:** ask which entities appear in both a text and an
  image, encouraging the system to link the two modalities.
* **Multi‑hop:** ask about the distribution of an entity across
  multiple documents.

Each generated example includes the question text, the answer and a
label indicating the type.  Questions are generated from the
hypergraph structure; they do not require external knowledge.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any

from .hypergraph import Hypergraph


def generate_questions(hg: Hypergraph, max_per_type: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a suite of questions from a hypergraph.

    Parameters
    ----------
    hg: Hypergraph
        The hypergraph from which to generate questions.  It should
        have embeddings built, but this is not strictly necessary for
        question generation.
    max_per_type: int
        Maximum number of questions to produce for each type.
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    List[Dict[str, Any]]
        A list of question dictionaries with keys ``type``, ``question``
        and ``answer``.
    """
    random.seed(seed)
    questions: List[Dict[str, Any]] = []
    # Build reverse index: for each node, list hyperedges by type
    node_to_text_edges: Dict[int, List[int]] = {}
    node_to_visual_edges: Dict[int, List[int]] = {}
    for node_id, node in hg.nodes.items():
        text_edges = []
        visual_edges = []
        for e in node.hyperedges:
            if hg.hyperedges[e].edge_type == 'text':
                text_edges.append(e)
            else:
                visual_edges.append(e)
        node_to_text_edges[node_id] = text_edges
        node_to_visual_edges[node_id] = visual_edges

    # Text‑only questions
    count = 0
    for edge_id, edge in hg.hyperedges.items():
        if edge.edge_type != 'text':
            continue
        # Entities in this text
        ent_names = [hg.nodes[n_id].name for n_id in edge.nodes]
        if not ent_names:
            continue
        # Construct question
        question = f"Which entities are mentioned in the following text: '{edge.content[:100]}...'?"
        answer = ", ".join(ent_names)
        questions.append({"type": "text_only", "question": question, "answer": answer})
        count += 1
        if count >= max_per_type:
            break

    # Visual‑only questions
    count = 0
    for edge_id, edge in hg.hyperedges.items():
        if edge.edge_type != 'visual':
            continue
        ent_names = [hg.nodes[n_id].name for n_id in edge.nodes]
        if not ent_names:
            continue
        ctx = hg.get_hyperedge_content(edge_id)
        question = f"What entities can you identify in the image described as '{ctx}'?"
        answer = ", ".join(ent_names)
        questions.append({"type": "visual_only", "question": question, "answer": answer})
        count += 1
        if count >= max_per_type:
            break

    # Cross‑modal questions
    cross_pairs = []
    for node_id, node in hg.nodes.items():
        if node_to_text_edges[node_id] and node_to_visual_edges[node_id]:
            cross_pairs.append((node_id, node_to_text_edges[node_id], node_to_visual_edges[node_id]))
    random.shuffle(cross_pairs)
    for node_id, t_edges, v_edges in cross_pairs[:max_per_type]:
        # Take first pair of text and visual edge
        t_edge = t_edges[0]
        v_edge = v_edges[0]
        text_snippet = hg.get_hyperedge_content(t_edge)[:100]
        image_desc = hg.get_hyperedge_content(v_edge)
        ent_name = hg.nodes[node_id].name
        question = (
            f"In the text '{text_snippet}...' and the image described as '{image_desc}', "
            f"which entity appears in both?"
        )
        answer = ent_name
        questions.append({"type": "cross_modal", "question": question, "answer": answer})

    # Multi‑hop questions: choose entities appearing in multiple texts
    multi_entities = []
    for node_id, text_edges in node_to_text_edges.items():
        if len(text_edges) >= 2:
            multi_entities.append((node_id, text_edges))
    random.shuffle(multi_entities)
    for node_id, t_edges in multi_entities[:max_per_type]:
        ent_name = hg.nodes[node_id].name
        question = f"In how many different texts does the entity '{ent_name}' appear?"
        answer = str(len(t_edges))
        questions.append({"type": "multi_hop", "question": question, "answer": answer})

    return questions

"""Hypergraph construction from datasets.

This module provides functions to read a dataset of multimodal
examples and construct a hypergraph.  The expected dataset format
matches the JSONL files used in MMLongBench【438279919083872†L82-L94】:

Each line is a JSON object with the following fields:

* ``id`` (str): unique identifier for the example.
* ``text`` (str): a paragraph or document containing the textual
  content.
* ``images`` (list): a list of objects with ``path`` (str) and
  optional ``alt_text`` (str).  The ``path`` should be relative to
  some root directory containing the images.

The hypergraph builder extracts entities from the text and images
using the functions defined in ``entity_extraction.py``.  It then
creates one hyperedge for the text and one hyperedge per image.  If
the same entity name occurs in both modalities, the resulting nodes
will be shared, naturally linking the textual and visual subgraphs.
"""

from __future__ import annotations

import json
import os
from typing import Optional, List, Dict, Any

from .entity_extraction import extract_text_entities, extract_visual_entities
from .hypergraph import Hypergraph


def build_hypergraph_from_dataset(
    dataset_path: str,
    image_root: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Hypergraph:
    """Build a hypergraph from a JSONL dataset file.

    Parameters
    ----------
    dataset_path: str
        Path to a JSONL file containing examples.
    image_root: Optional[str]
        If provided, image paths from the dataset will be joined
        relative to this directory.  Otherwise they are used as is.
    embedding_backend: Optional[str]
        Name of the embedding backend ('tfidf' or 'deepseek').  If
        ``None``, the default TF–IDF backend is used.
    max_samples: Optional[int]
        If provided, stop processing after this many examples.  Useful
        for debugging on smaller subsets of the data.

    Returns
    -------
    Hypergraph
        The constructed hypergraph.  Embeddings are not computed
        until you call ``hypergraph.build_embeddings``.
    """
    hg = Hypergraph(embedding_backend=embedding_backend)
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = example.get("text", "")
            images: List[Dict[str, Any]] = example.get("images", [])
            # Extract text entities and add nodes
            text_entities = extract_text_entities(text)
            text_node_ids = [hg.add_node(name=ent, node_type="text") for ent in text_entities]
            # Create a textual hyperedge if there are any nodes
            if text_node_ids:
                hg.add_hyperedge(nodes=text_node_ids, content=text, edge_type="text")
            # Process each image
            for img_info in images:
                # Normalise path
                path = img_info.get("path")
                if path and image_root:
                    img_info = dict(img_info)
                    img_info["path"] = os.path.join(image_root, path)
                visual_entities = extract_visual_entities(img_info)
                visual_node_ids = [hg.add_node(name=ent, node_type="visual") for ent in visual_entities]
                if visual_node_ids:
                    hg.add_hyperedge(nodes=visual_node_ids, content=img_info, edge_type="visual")
    return hg

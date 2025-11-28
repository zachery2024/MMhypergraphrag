"""Top‑level package for MMHyperGraphRAG.

This package provides classes and functions to build a multimodal
hypergraph from a dataset, perform retrieval over the hypergraph and
interface with large language models via API.  See README.md for
usage instructions.
"""

from .entity_extraction import extract_text_entities, extract_visual_entities
from .hypergraph import Hypergraph
from .graph_builder import build_hypergraph_from_dataset
from .retrieval import retrieve_contexts
from .rag_model import MMHyperGraphRAG
from .question_generator import generate_questions

__all__ = [
    "extract_text_entities",
    "extract_visual_entities",
    "Hypergraph",
    "build_hypergraph_from_dataset",
    "retrieve_contexts",
    "MMHyperGraphRAG",
    "generate_questions",
]
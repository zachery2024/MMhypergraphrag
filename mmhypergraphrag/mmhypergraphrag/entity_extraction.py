"""Utilities for extracting entities from text and images.

The hypergraph uses entities as its basic nodes.  For textual
hyperedges we rely on named‑entity recognition to identify people,
locations, organisations and other proper nouns.  We use the NLTK
library's `ne_chunk` function because it is lightweight and does not
require large external downloads.  If the NLTK models are not
available, we fall back to a simple heuristic that treats capitalised
words and long nouns as entities.  Visual hyperedges are processed by
looking at the `alt_text` associated with an image; if none is
provided we derive entity names from the filename.
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Any

import nltk


def _ensure_nltk_resources() -> None:
    """Attempt to download required NLTK resources.

    When running in offline environments the download will silently
    fail; the calling functions will then use fallback heuristics.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("chunkers/maxent_ne_chunker")
    except LookupError:
        try:
            nltk.download("maxent_ne_chunker", quiet=True)
            nltk.download("words", quiet=True)
        except Exception:
            pass


def extract_text_entities(text: str) -> List[str]:
    """Extract named entities from a text string.

    The function first attempts to use NLTK's built‑in named entity
    recogniser (`ne_chunk`).  If that fails due to missing models, a
    heuristic fallback is applied that treats capitalised tokens and
    nouns longer than 3 characters as entities.

    Parameters
    ----------
    text: str
        The input text.

    Returns
    -------
    List[str]
        A list of entity names.  Duplicate names are removed and the
        original casing is preserved.
    """
    if not text:
        return []
    _ensure_nltk_resources()
    entities: List[str] = []
    try:
        # Use NLTK's named entity recogniser
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(words)
            chunked = nltk.ne_chunk(tagged, binary=False)
            for subtree in chunked:
                if hasattr(subtree, "label"):
                    entity = " ".join(token for token, pos in subtree.leaves())
                    entities.append(entity)
    except Exception:
        # Fallback heuristic: extract capitalised words and long words as entities
        # Avoid relying on NLTK tokenisers; split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)
        for w in tokens:
            if (w[0].isupper() and len(w) > 2) or len(w) > 6:
                entities.append(w)
    # Deduplicate and preserve order
    seen = set()
    unique_entities: List[str] = []
    for ent in entities:
        ent_stripped = ent.strip()
        if ent_stripped and ent_stripped.lower() not in seen:
            unique_entities.append(ent_stripped)
            seen.add(ent_stripped.lower())
    return unique_entities


def extract_visual_entities(image_info: Dict[str, Any]) -> List[str]:
    """Extract entity names from image metadata.

    We primarily use the ``alt_text`` field, which is often a short
    description of the image, to derive entities.  If no alt text is
    available, the filename (without extension) is split on common
    separators and digits are removed.  The resulting tokens are
    passed through the same text entity extractor.

    Parameters
    ----------
    image_info: Dict[str, Any]
        A dictionary containing at least ``path`` (the filename) and
        optionally ``alt_text``.

    Returns
    -------
    List[str]
        A list of entity names extracted from the image description.
    """
    alt_text = image_info.get("alt_text", "")
    if alt_text:
        return extract_text_entities(alt_text)
    # Use filename as a last resort
    path = image_info.get("path", "")
    name = os.path.basename(path)
    # Remove extension
    name = os.path.splitext(name)[0]
    # Replace common separators with spaces
    name = re.sub(r"[_\-]+", " ", name)
    # Remove digits
    name = re.sub(r"\d+", "", name)
    return extract_text_entities(name)

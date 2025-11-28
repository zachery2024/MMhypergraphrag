# MMHyperGraphRAG: A Multimodal Hypergraph Retrieval‑Augmented Generation Framework

**MMHyperGraphRAG** is a research prototype that extends the ideas of
retrieval‑augmented generation (RAG) to the multimodal domain.  It
follows the design of **multimodal graph RAG** systems such as
MMGraphRAG, which refine visual content into structured entities and
combine them with text‑based knowledge graphs【178648359835197†L49-L61】.
Unlike prior work that relies on standard graphs, this project
constructs a **hypergraph** where each document (text or image) is
represented as a hyperedge linking all of its entities.  Visual
hyperedges are created from images, while textual hyperedges come
from paragraphs or documents.  Entities appearing in both modalities
are connected across the two hypergraphs, forming a unified
knowledge base.

The code in this repository is a clean re‑implementation of the
benchmarking framework from
MMLongBench【438279919083872†L33-L39】.  It provides utilities to parse the
MMLongBench dataset, extract entities from both text and images, build
a multimodal hypergraph, perform retrieval over the hypergraph and
call a large language model (LLM) via API.  Additionally, it contains
scripts to automatically generate different types of evaluation
questions (text‑only, visual‑only, cross‑modal and multi‑hop) to
assess the effectiveness of the hypergraph RAG.

## Key Features

- **Hypergraph construction:** Each document (image or text) becomes a
  hyperedge.  For textual hyperedges we extract named entities using
  NLTK; for visual hyperedges we analyse the accompanying alt text or
  filename to derive object names.  Entities are represented as
  nodes.  When the same entity name appears in both an image and a
  paragraph, a cross‑modal link is created.
- **Pluggable embeddings and retrieval:** By default, a TF–IDF
  vectoriser from `scikit‑learn` is used to embed entity names.  You
  can switch to the `deepseek` backend, which calls DeepSeek‑OCR’s
  **DeepEncoder** to produce vision and text embeddings when you have
  installed the relevant PyTorch modules.  Queries are embedded in
  the same space and the most similar entities and their hyperedges
  are retrieved.  The TF–IDF backend is lightweight and requires no
  GPU support; the DeepSeek backend unlocks high‑fidelity embeddings.
- **LLM integration via API key:** A generic RAG engine connects
  retrieved contexts to an external LLM.  By default it expects an
  OpenAI API key in the `OPENAI_API_KEY` environment variable, but the
  implementation is modular and can be extended to other providers.
- **Question generation:** To evaluate the system we provide a
  `question_generator` that synthesises different question types
  automatically from the constructed hypergraph.  These questions
  exercise text‑only reasoning, visual‑only reasoning, cross‑modal
  linking and multi‑hop chaining.
- **Dataset adapter:** Utilities are included to parse the
  MMLongBench dataset files and prepare them for hypergraph building.
  Although MMLongBench comprises over 13k examples and five task
  categories【438279919083872†L33-L39】, you can also supply your own
  dataset with the same JSONL structure described below.

## Installation

1. **Clone this repository** and install the dependencies.  You can
   create a virtual environment if desired.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download the MMLongBench data.**  Follow the instructions in the
   original repo to obtain the image and text splits
   (see `scripts/download_text_data.sh` and
   `scripts/download_image_data.sh`【438279919083872†L82-L94】).  Place the
   decompressed JSONL files into a directory and specify it via
   command‑line when running the build script.  Each JSON line should
   contain at least the following fields:

   ```json
   {
     "id": "example_1",
     "text": "A paragraph of text ...",
     "images": [
       {"path": "relative/path/to/image.jpg", "alt_text": "A man and a dog"},
       ...
     ]
   }
   ```

3. **Build the hypergraph** using the provided script.  You can
   specify the embedding backend with `--embedding_backend tfidf` or
   `--embedding_backend deepseek`.  For example:

   ```bash
   python main.py build \
     --data_file data/vrag.jsonl \
     --image_root path/to/images \
     --embedding_backend tfidf \
     --output_file hypergraph.pkl
   ```

4. **Generate questions** and evaluate the RAG system.  For example:

   ```bash
   python main.py generate --hypergraph hypergraph.pkl --output_file questions.jsonl
   python main.py answer --hypergraph hypergraph.pkl --questions questions.jsonl
   ```

## File Structure

```
mmhypergraphrag_project/
│   README.md           # this file
│   requirements.txt    # list of Python dependencies
│   main.py             # command‑line interface for building and evaluating
└───mmhypergraphrag/
    │   __init__.py
    │   entity_extraction.py  # entity extraction for text and images
    │   hypergraph.py          # hypergraph data structure and embeddings
    │   graph_builder.py       # build hypergraph from dataset
    │   retrieval.py           # simple TF–IDF based retrieval
    │   rag_model.py           # RAG engine calling an LLM via API
    │   question_generator.py  # automatic question synthesis

```

## Command‑Line Usage

The `main.py` script exposes a simple interface with sub‑commands:

- `build`: parse a JSONL dataset, extract entities and build a hypergraph.
- `generate`: create evaluation questions from an existing hypergraph.
- `answer`: run retrieval and call an LLM to answer questions.

Run `python main.py --help` for more details on available options.

## Citation

If you use this code in your research, please cite the following
works that inspired it.  MMLongBench provided the large dataset of
vision‑language tasks【438279919083872†L33-L39】, and MMGraphRAG
introduced the idea of a multimodal knowledge graph with cross‑modal
entity linking【178648359835197†L49-L61】.

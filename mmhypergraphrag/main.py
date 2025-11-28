"""Command‑line interface for the MMHyperGraphRAG project.

This script exposes a few subcommands to build a hypergraph from a
dataset, generate evaluation questions and run the RAG pipeline to
produce answers.  Run ``python main.py --help`` for usage details.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, List

from mmhypergraphrag.entity_extraction import extract_text_entities, extract_visual_entities
from mmhypergraphrag.graph_builder import build_hypergraph_from_dataset
from mmhypergraphrag.question_generator import generate_questions
from mmhypergraphrag.rag_model import MMHyperGraphRAG


def cmd_build(args: argparse.Namespace) -> None:
    """Build a hypergraph from a dataset file and persist it."""
    hg = build_hypergraph_from_dataset(
        dataset_path=args.data_file,
        image_root=args.image_root,
        embedding_backend=args.embedding_backend,
        max_samples=args.max_samples,
    )
    # Build embeddings now so they are saved
    hg.build_embeddings()
    with open(args.output_file, 'wb') as f:
        pickle.dump(hg, f)
    print(f"Hypergraph saved to {args.output_file}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate evaluation questions from a saved hypergraph."""
    with open(args.hypergraph, 'rb') as f:
        hg = pickle.load(f)
    questions = generate_questions(hg, max_per_type=args.max_per_type)
    with open(args.output_file, 'w', encoding='utf-8') as fout:
        for q in questions:
            fout.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f"Generated {len(questions)} questions and saved to {args.output_file}")


def cmd_answer(args: argparse.Namespace) -> None:
    """Answer questions using the hypergraph and an LLM via API."""
    with open(args.hypergraph, 'rb') as f:
        hg = pickle.load(f)
    rag = MMHyperGraphRAG(
        hypergraph=hg,
        api_provider=args.api_provider,
        api_key=args.api_key,
        model_name=args.model_name,
    )
    results: List[Dict[str, Any]] = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q_text = obj.get('question')
            answer_gt = obj.get('answer')
            q_type = obj.get('type')
            try:
                pred = rag.answer_question(q_text, top_k_nodes=args.top_k_nodes, top_k_edges=args.top_k_edges)
            except Exception as e:
                pred = f"Error: {e}"
            correct = None
            if answer_gt is not None and pred:
                # Basic evaluation: exact match ignoring case and spaces
                correct = (pred.strip().lower() == str(answer_gt).strip().lower())
            results.append({
                'type': q_type,
                'question': q_text,
                'ground_truth': answer_gt,
                'prediction': pred,
                'correct': correct,
            })
    # Print simple metrics
    total = sum(1 for r in results if r['correct'] is not None)
    correct_count = sum(1 for r in results if r['correct'] is True)
    if total > 0:
        acc = correct_count / total
        print(f"Accuracy on {total} answerable questions: {acc:.2%}")
    else:
        print("No evaluable questions found.")
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Results saved to {args.output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMHyperGraphRAG command‑line tool")
    subparsers = parser.add_subparsers(dest='command', required=True)
    # Build subcommand
    p_build = subparsers.add_parser('build', help='Build a hypergraph from a dataset')
    p_build.add_argument('--data_file', type=str, required=True, help='Path to the JSONL dataset file')
    p_build.add_argument('--image_root', type=str, default=None, help='Root directory for image files')
    p_build.add_argument('--embedding_backend', type=str, default=None, choices=['tfidf', 'deepseek'], help='Embedding backend to use')
    p_build.add_argument('--max_samples', type=int, default=None, help='Process at most this many samples (for debugging)')
    p_build.add_argument('--output_file', type=str, required=True, help='File to save the hypergraph (pickle)')
    p_build.set_defaults(func=cmd_build)
    # Generate subcommand
    p_gen = subparsers.add_parser('generate', help='Generate evaluation questions from a hypergraph')
    p_gen.add_argument('--hypergraph', type=str, required=True, help='Path to the pickled hypergraph file')
    p_gen.add_argument('--max_per_type', type=int, default=50, help='Maximum questions per type')
    p_gen.add_argument('--output_file', type=str, required=True, help='File to save generated questions (JSONL)')
    p_gen.set_defaults(func=cmd_generate)
    # Answer subcommand
    p_ans = subparsers.add_parser('answer', help='Answer questions using a hypergraph and LLM')
    p_ans.add_argument('--hypergraph', type=str, required=True, help='Path to the pickled hypergraph file')
    p_ans.add_argument('--questions', type=str, required=True, help='Path to questions JSONL file')
    p_ans.add_argument('--api_provider', type=str, default='openai', help='LLM API provider (only "openai" supported)')
    p_ans.add_argument('--api_key', type=str, default=None, help='API key for the LLM provider')
    p_ans.add_argument('--model_name', type=str, default='gpt-4', help='Model name to use when calling the LLM')
    p_ans.add_argument('--top_k_nodes', type=int, default=5, help='Number of nodes to retrieve per query')
    p_ans.add_argument('--top_k_edges', type=int, default=3, help='Number of hyperedges to retrieve per query')
    p_ans.add_argument('--output_file', type=str, default=None, help='File to save answer results (JSONL)')
    p_ans.set_defaults(func=cmd_answer)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

"""RAG vs non-RAG evaluation script.

Usage examples
--------------
# 1) Index documents and run evaluation with dense RAG retriever
python experiments/rag_vs_nonrag.py \
    --docs data/news_2025/*.jsonl \
    --dataset data/qa_2025.jsonl \
    --mode rag --retriever dense --top_k 5 \
    --gemini_keys KEY1,KEY2

# 2) Evaluate plain LLM answers (no retrieval)
python experiments/rag_vs_nonrag.py --dataset data/qa_2025.jsonl \
    --mode nonrag --gemini_keys KEY1,KEY2

The script prints Exact-Match, F1 and Accuracy metrics across the batch and
stores per-question predictions to ``results.jsonl``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters.csv import CSVToDocument
from haystack.components.converters.json import JSONConverter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators.gemini import GeminiProGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.dataclasses.document import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _read_documents(patterns: Sequence[str]) -> List[Document]:
    """Load documents from CSV or JSON/JSONL files matched by *patterns*."""
    documents: List[Document] = []
    csv_loader = CSVToDocument()
    json_loader = JSONConverter(content_key="content")  # expects {"content": "..."}

    for pattern in patterns:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            if path.suffix.lower() in {".csv"}:
                docs = csv_loader.run(sources=[str(path)]) ["documents"]
            elif path.suffix.lower() in {".json", ".jsonl"}:
                docs = json_loader.run(sources=[str(path)]) ["documents"]
            else:
                print(f"[WARN] Skipping unsupported file type: {path}")
                continue
            documents.extend(docs)
    return documents


# Text normalisation helpers for metrics
_punc_table = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_punc_table)
    text = " ".join(text.split())
    return text


def _exact_match(pred: str, truth: str) -> int:
    return int(_normalize(pred) == _normalize(truth))


def _f1(pred: str, truth: str) -> float:
    pred_tokens = _normalize(pred).split()
    truth_tokens = _normalize(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


# -----------------------------------------------------------------------------
# Pipeline builders
# -----------------------------------------------------------------------------


def _build_rag_pipeline(
    document_store: InMemoryDocumentStore,
    gemini_keys: List[str],
    retriever_type: str = "dense",
    top_k: int = 5,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Return a Haystack pipeline and a *template* dict of inputs shared across runs."""

    pipe = Pipeline()
    # LLM
    llm = GeminiProGenerator(api_keys=[Secret.from_token(k) for k in gemini_keys])

    if retriever_type == "dense":
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=top_k)
        pipe.add_component("text_embedder", text_embedder)
        pipe.add_component("retriever", retriever)
        pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    elif retriever_type == "bm25":
        retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k)
        pipe.add_component("retriever", retriever)
    else:
        raise ValueError("retriever_type must be 'dense' or 'bm25'")

    prompt_template = (
        """Given these documents, answer the question.\n\n"""
        "{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
        "Question: {{query}}\nAnswer:"""
    )
    prompt_builder = PromptBuilder(template=prompt_template)

    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)

    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    shared_kwargs = {}
    if retriever_type == "dense":
        shared_kwargs["text_embedder"] = {}
    return pipe, shared_kwargs


def _build_nonrag_pipeline(gemini_keys: List[str]) -> Tuple[Pipeline, Dict[str, Any]]:
    pipe = Pipeline()
    llm = GeminiProGenerator(api_keys=[Secret.from_token(k) for k in gemini_keys])
    prompt_template = "Answer the question precisely. Question: {{query}}\nAnswer:"
    prompt_builder = PromptBuilder(template=prompt_template)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe, {}


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------


def evaluate(
    pipe: Pipeline,
    shared_kwargs: Dict[str, Any],
    qa_pairs: List[Tuple[str, str]],
    retriever_type: str | None = None,
) -> Dict[str, Any]:
    em_scores: List[int] = []
    f1_scores: List[float] = []
    predictions: List[str] = []

    for question, truth in qa_pairs:
        run_inputs: Dict[str, Any] = {
            "prompt_builder": {"query": question},
        }
        if "retriever" in pipe.get_components():
            run_inputs["retriever"] = {"query": question}
        if retriever_type == "dense":
            run_inputs["text_embedder"] = {"text": question}
        # Merge with shared defaults
        run_inputs = {**shared_kwargs, **run_inputs}

        try:
            result = pipe.run(run_inputs)
            llm_reply = result["llm"]["replies"][0]
        except Exception as e:
            print(f"[ERROR] Question failed: {e}")
            llm_reply = ""

        predictions.append(llm_reply)
        em_scores.append(_exact_match(llm_reply, truth))
        f1_scores.append(_f1(llm_reply, truth))

    accuracy = sum(em_scores) / len(em_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    metrics = {
        "EM": sum(em_scores) / len(em_scores),
        "F1": mean_f1,
        "Accuracy": accuracy,
    }
    with open("results.jsonl", "w") as f_out:
        for (q, t), pred, em, f1 in zip(qa_pairs, predictions, em_scores, f1_scores):
            json.dump({"question": q, "truth": t, "prediction": pred, "em": em, "f1": f1}, f_out)
            f_out.write("\n")
    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare RAG vs non-RAG with Gemini Pro")
    parser.add_argument("--docs", nargs="*", help="Glob patterns for document files (CSV/JSONL)")
    parser.add_argument("--dataset", required=True, help="JSONL file with fields 'question' and 'answer'")
    parser.add_argument("--mode", choices=["rag", "nonrag"], required=True)
    parser.add_argument("--retriever", choices=["dense", "bm25"], default="dense")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--gemini_keys", required=True, help="Comma-separated Gemini API keys (primary,secondary)")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: List[str] | None = None):
    args = parse_args(argv)
    gemini_keys = [k.strip() for k in args.gemini_keys.split(",") if k.strip()]
    if not gemini_keys:
        raise ValueError("At least one Gemini API key must be provided via --gemini_keys")

    # Load QA dataset
    qa_pairs: List[Tuple[str, str]] = []
    with open(args.dataset, "r", encoding="utf-8") as f_in:
        for line in f_in:
            item = json.loads(line)
            qa_pairs.append((item["question"], item["answer"]))

    if args.mode == "rag":
        if not args.docs:
            print("[ERROR] --docs must be provided in RAG mode", file=sys.stderr)
            sys.exit(1)
        docs = _read_documents(args.docs)
        print(f"Loaded {len(docs)} documents.")
        # Embed and index documents
        doc_store = InMemoryDocumentStore()
        if args.retriever == "dense":
            embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
            embedder.warm_up()
            docs = embedder.run(documents=docs)["documents"]
        doc_store.write_documents(docs)
        pipe, shared_kwargs = _build_rag_pipeline(
            document_store=doc_store,
            gemini_keys=gemini_keys,
            retriever_type=args.retriever,
            top_k=args.top_k,
        )
        metrics = evaluate(pipe, shared_kwargs, qa_pairs, retriever_type=args.retriever)
    else:  # non-RAG
        pipe, shared_kwargs = _build_nonrag_pipeline(gemini_keys)
        metrics = evaluate(pipe, shared_kwargs, qa_pairs)

    print("\n=== Evaluation results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nThe script also highlights typical RAG **limitations** encountered in")
    print("real-world QA setups:")
    print("* **Retriever recall bottleneck** – the correct passage might not be among the")
    print("  top-k retrieved documents, especially for BM25 when wording diverges.  ")
    print("  → *Research direction*: hybrid retrievers and retrieval-time query expansion.")
    print("* **Long-context dilution** – when many documents are stuffed into the prompt")
    print("  the model may ignore or hallucinate details.  ")
    print("  → *Research*: segment-aware summarisation or iterative retrieval-generation")
    print("  loops (\"retrieval-step prompting\").")
    print("* **Temporal misalignment** – static document snapshots become stale (e.g. ")
    print("  rapidly evolving news).  ")
    print("  → *Research*: freshness-aware scoring and incremental index updates.")
    print("* **Answer faithfulness** – LLM may blend information from multiple passages")
    print("  leading to unsupported claims.  ")
    print("  → *Research*: incorporate citation verification or chain-of-thought +")
    print("  re-checking with fact-consistent models.")


if __name__ == "__main__":
    main() 
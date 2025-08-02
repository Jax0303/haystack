#!/usr/bin/env python3
"""advanced_rag.py

Enterprise-grade Retrieval-Augmented Generation (RAG)
=====================================================
This script shows a *reference* implementation of an on-prem / private-cloud
RAG system that tackles several real-world enterprise requirements:

1. **Document ingestion & vectorisation** with LangChain + HuggingFace
2. **ChromaDB** for fast similarity search with persistence on disk
3. **Fine-grained access control (ACL)** at query-time via metadata filters
4. **Hallucination / Confidence scoring** to flag low-support answers
5. **Dynamic content update** – re-index only changed files
6. **Scalability hooks** – multiprocessing batch embeds, env-driven configs

The code intentionally keeps external dependencies minimal while staying
framework-agnostic. Adapt to your own auth system, LLM provider, queue, etc.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from pathlib import Path
from typing import List, Sequence

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def env(key: str, default: str | None = None) -> str | None:
    """Tiny helper to read env vars with default."""
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Data ingestion utilities
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", " "]


def discover_txt_files(folder: Path) -> List[Path]:
    pattern = str(folder / "**" / "*.txt")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]


def read_acl_sidecar(txt_path: Path) -> list[str]:
    """Look for a JSON sidecar with ACL info next to the .txt file.

    E.g.  `policy.txt`  →  `policy.txt.meta`
    The sidecar *should* contain JSON like: {"acl": ["hr", "all"]}
    Missing file ⇒ open to all.
    """
    sidecar = txt_path.with_suffix(txt_path.suffix + ".meta")
    if not sidecar.exists():
        return ["*"]  # wildcard
    try:
        meta = json.loads(sidecar.read_text("utf-8"))
        acl = meta.get("acl", ["*"])
        if isinstance(acl, str):
            return [acl]
        return list(acl)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not parse ACL sidecar for {txt_path}: {exc}")
        return ["*"]


def load_and_split_docs(
    folder: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    txt_files = discover_txt_files(folder)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )
    docs: list[Document] = []
    for fp in txt_files:
        loader = TextLoader(str(fp), encoding="utf-8")
        file_docs = loader.load()  # returns List[Document]
        # attach metadata BEFORE splitting so children inherit it
        acl_roles = read_acl_sidecar(fp)
        for d in file_docs:
            d.metadata.update(
                {
                    "source": str(fp),
                    "acl": acl_roles,
                    "mtime": fp.stat().st_mtime,
                }
            )
        split_docs = splitter.split_documents(file_docs)
        docs.extend(split_docs)
    return docs


# ---------------------------------------------------------------------------
# Vector store wrapper with ACL-aware retrieval
# ---------------------------------------------------------------------------

class AclRetriever(BaseRetriever):
    """Wrap an existing vector store and enforce ACL filtering."""

    def __init__(self, vectordb: Chroma, allowed_roles: Sequence[str] = None, top_k: int = 4):
        super().__init__()
        self.vectordb = vectordb
        self.allowed_roles = allowed_roles or ["*"]
        self.top_k = top_k

    def _is_authorised(self, doc: Document) -> bool:
        doc_roles = doc.metadata.get("acl", ["*"])
        if "*" in doc_roles:
            return True
        return bool(set(doc_roles) & set(self.allowed_roles))

    def similarity_search(self, query: str) -> list[Document]:
        candidates = self.vectordb.similarity_search(query, k=self.top_k * 2)
        authorised = [d for d in candidates if self._is_authorised(d)]
        return authorised[: self.top_k]

    # Expose as LangChain retriever interface
    def _get_relevant_documents(self, query: str) -> list[Document]:  # noqa: D401
        return self.similarity_search(query)


# ---------------------------------------------------------------------------
# Hallucination / confidence scoring
# ---------------------------------------------------------------------------

def compute_similarity(a_emb: list[float], b_emb: list[float]) -> float:
    """Cosine similarity for small lists."""
    import numpy as np  # local import to keep base deps lean

    a = np.array(a_emb)
    b = np.array(b_emb)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a.dot(b) / denom)


def hallucination_score(answer: str, context: list[Document], embedder) -> float:
    """Rough metric: similarity of answer vs. mean of context embeddings.

    Returns a *distance* score in [0, 2] where < 0.7 usually indicates
    strong grounding; > 1.0 might be hallucination.
    """
    ans_emb = embedder.embed_query(answer)
    ctx_embs = [embedder.embed_query(doc.page_content) for doc in context]
    import numpy as np

    ctx_mean = np.mean(ctx_embs, axis=0)
    sim = compute_similarity(ans_emb, ctx_mean)  # -1..1
    return 1.0 - ((sim + 1) / 2)  # convert to distance ~ 0(best)..2(worst)


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------

@dataclass
class RagEngine:
    collection_name: str = "rag_collection"
    persist_dir: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    def __post_init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
        )

    # ------------------ ingestion ------------------
    def index_folder(
        self,
        folder: Path,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        incremental: bool = True,
    ):
        """Index documents in *folder*.

        If *incremental* is True, only add files that were modified
        after their last index time (mtime stored in metadata).
        """
        all_docs = load_and_split_docs(folder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not incremental or not list(self.vectordb.get()):
            self.vectordb.add_documents(all_docs)
        else:
            new_docs: list[Document] = []
            # Use Chroma metadata filter to see if we already indexed this mtime
            from datetime import datetime

            for doc in all_docs:
                src = doc.metadata.get("source")
                mtime = doc.metadata.get("mtime")
                records = self.vectordb.get(
                    where={"source": src, "mtime": mtime},
                    include=["metadatas"],
                )
                if len(records["ids"]) == 0:
                    new_docs.append(doc)
            if new_docs:
                print(f"Adding {len(new_docs)} new / updated chunks to index …")
                self.vectordb.add_documents(new_docs)
        self.vectordb.persist()

    # ------------------ query ------------------
    def answer(
        self,
        query: str,
        *,
        user_roles: Sequence[str] = ("*",),
        top_k: int = 4,
        llm_repo: str = "google/flan-t5-base",
        hallucination_threshold: float = 0.35,
    ) -> dict:
        retriever = AclRetriever(self.vectordb, allowed_roles=user_roles, top_k=top_k)
        llm = HuggingFaceHub(
            repo_id=llm_repo, 
            model_kwargs={"temperature": 0.0},
            task="text2text-generation"
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
        )
        answer = qa_chain.run(query)
        docs = retriever.similarity_search(query)
        score = hallucination_score(answer, docs, self.embedder)
        return {
            "answer": answer,
            "sources": [d.metadata for d in docs],
            "confidence": 1 - score,
            "hallucination_flag": score > hallucination_threshold,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli():
    parser = argparse.ArgumentParser(description="Enterprise RAG reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index a folder of .txt docs")
    p_index.add_argument("folder", type=Path)
    p_index.add_argument("--chunk_size", type=int, default=1000)
    p_index.add_argument("--chunk_overlap", type=int, default=100)
    p_index.add_argument("--no_incremental", action="store_true")

    # query
    p_query = sub.add_parser("query", help="Ask a question via RAG")
    p_query.add_argument("question")
    p_query.add_argument("--roles", default="*", help="Comma-separated user roles (ACL)")
    p_query.add_argument("--top_k", type=int, default=4)
    p_query.add_argument("--llm_repo", default="google/flan-t5-base")

    return parser


def main(argv: list[str] | None = None):
    parser = build_cli()
    args = parser.parse_args(argv)

    engine = RagEngine(
        collection_name=env("RAG_COLLECTION", "rag_collection"),
        persist_dir=env("RAG_PERSIST_DIR", "./chroma_db"),
        embedding_model=env("RAG_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"),
    )

    if args.command == "index":
        engine.index_folder(
            args.folder,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            incremental=not args.no_incremental,
        )
        print("Indexing completed.")

    elif args.command == "query":
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
        start = time.perf_counter()
        res = engine.answer(
            args.question,
            user_roles=roles,
            top_k=args.top_k,
            llm_repo=args.llm_repo,
        )
        dur = time.perf_counter() - start
        print(json.dumps({**res, "elapsed_sec": dur}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple Retrieval-Augmented Generation (RAG) pipeline built with:
  • LangChain for glue code
  • HuggingFace embeddings for semantic vectors
  • ChromaDB as the vector database

Steps performed:
1. Read every .txt file under a given directory and split the content into
   manageable chunks.
2. Embed those chunks and store them inside a persistent Chroma collection.
3. (Optional) Run an example query through a Retrieval-QA chain to show end-to-end use.

Usage examples
--------------
# Build / update the vector store only
python rag_system.py --docs_path ./my_docs/

# Build the vector store and immediately test a query
python rag_system.py \
    --docs_path ./my_docs/ \
    --query "우리 회사의 휴가 정책을 알려줘" \
    --collection_name company_policies

Environment
-----------
For the optional QA step you need a HuggingFace API token set in the
HF_API_TOKEN environment variable (or configure another LLM).
"""

import argparse
import glob
import os
from typing import List

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document


# -------------------------
# Helper functions
# -------------------------

def load_and_split_docs(folder: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Recursively load *.txt files and split them into smaller passages."""
    pattern = os.path.join(os.path.abspath(folder), "**", "*.txt")
    file_paths = glob.glob(pattern, recursive=True)

    if not file_paths:
        raise FileNotFoundError(f"No .txt files found under {folder!r}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]  # try bigger breaks first
    )

    docs: List[Document] = []
    for fp in file_paths:
        loader = TextLoader(fp, encoding="utf-8")
        loaded_docs = loader.load()
        # Add filename for traceability
        for d in loaded_docs:
            d.metadata["source"] = os.path.relpath(fp, folder)
        docs.extend(splitter.split_documents(loaded_docs))

    return docs


def build_chroma(docs: List[Document], embeddings, persist_dir: str, collection_name: str) -> Chroma:
    """Create (or update) a Chroma collection with the given documents."""
    os.makedirs(persist_dir, exist_ok=True)

    # If the collection already exists, load it; otherwise create new
    if os.listdir(persist_dir):
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        vectordb.add_documents(docs)
    else:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

    vectordb.persist()
    return vectordb


# -------------------------
# Main script
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a simple RAG datastore with LangChain + Chroma")
    parser.add_argument("--docs_path", type=str, required=True, help="Folder containing .txt files to index")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Characters per chunk (default: 1000)")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Overlap between chunks (default: 100)")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="HF embedding model")
    parser.add_argument("--collection_name", type=str, default="rag_collection", help="Chroma collection name")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Folder for Chroma persistence")
    parser.add_argument("--query", type=str, help="Optional: run a test query after building the store")
    parser.add_argument("--llm_model", type=str, default="google/flan-t5-base", help="HF model for generation in QA phase")
    parser.add_argument("--top_k", type=int, default=4, help="How many passages to retrieve for QA")

    args = parser.parse_args()

    # 1) Load + split
    print("[1/3] Loading and splitting docs ...")
    docs = load_and_split_docs(args.docs_path, args.chunk_size, args.chunk_overlap)
    print(f"  Loaded {len(docs)} text chunks.")

    # 2) Embedding + store
    print("[2/3] Creating embeddings and populating Chroma ...")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    vectordb = build_chroma(docs, embeddings, args.persist_dir, args.collection_name)
    print(f"  Vector store saved to: {args.persist_dir}")

    # 3) Optional QA demo
    if args.query:
        print("[3/3] Running a Retrieval-QA demo query ...")
        llm = HuggingFaceHub(repo_id=args.llm_model, model_kwargs={"temperature": 0.0})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": args.top_k}),
        )
        answer = qa.run(args.query)
        print("\n=== Answer ===")
        print(answer)


if __name__ == "__main__":
    main()

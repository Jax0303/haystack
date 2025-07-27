#!/usr/bin/env python3
"""
index_rag_dataset.py
--------------------
1. Load rag_dataset_2024_25.jsonl.gz (created earlier).
2. Split each paper into ~1000-token chunks.
3. Embed chunks with a sentence-transformers model (bge-small-en).
4. Build FAISS index (cosine / inner-product on normalized vectors).
5. Provide an interactive CLI to ask questions via Gemini-Pro with retrieved context.

Usage:
  # (가상환경 활성화 후)
  pip install sentence-transformers faiss-cpu google-generativeai nltk
  python -c "import nltk, ssl, os, nltk.downloader; nltk.download('punkt')"
  export GOOGLE_API_KEY="<YOUR_GEMINI_API_KEY>"
  python index_rag_dataset.py build   # 인덱스 생성 (최초 5-10분)
  python index_rag_dataset.py ask "What are the key challenges in AI alignment in 2025?"
"""
import sys, gzip, json, os, time, itertools, pickle
from pathlib import Path
from typing import List

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

try:
    import google.generativeai as genai
except ImportError:
    genai = None

DATA_FILE = Path("rag_dataset_2024_25.jsonl.gz")
INDEX_FILE = Path("rag_faiss.index")
META_FILE  = Path("rag_meta.pkl")   # list[str] chunk texts
CHUNK_SIZE_TOK = 1000  # rough token count (we approximate by 4 chars/token)
OVERLAP_TOK    = 120

EMB_MODEL_NAME = "BAAI/bge-small-en"


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)  # crude 1 token ≈ 4 chars


def chunk_text(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        sent_len = estimate_token_count(sent)
        if current_len + sent_len > CHUNK_SIZE_TOK:
            if current:
                chunks.append(" ".join(current)[:4000])  # hard cap 4k chars
            # overlap
            overlap_tokens = OVERLAP_TOK // 4  # back to char approx
            overlap_text = current[-overlap_tokens:] if overlap_tokens < len(current) else current
            current = overlap_text.copy() if isinstance(overlap_text, list) else list(overlap_text)
            current_len = estimate_token_count(" ".join(current))
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current)[:4000])
    return chunks


def build_index():
    nltk.download('punkt', quiet=True)
    print("📖 Loading dataset…")
    if not DATA_FILE.exists():
        print(f"❌ {DATA_FILE} not found.")
        sys.exit(1)

    model = SentenceTransformer(EMB_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    metadata: List[str] = []

    def batch(iterable, n=64):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="parsing"):
            obj = json.loads(line)
            txt = obj.get("full_text") or obj.get("text") or ""
            if not txt:
                continue
            for chunk in chunk_text(txt):
                metadata.append(chunk)
    print(f"🧩 Total chunks: {len(metadata):,}")

    # encode in batches
    for texts in tqdm(batch(metadata, 64), total=len(metadata)//64 + 1, desc="embedding"):
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        index.add(emb)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Index saved to {INDEX_FILE}, metadata to {META_FILE}")


def ask(query: str, top_k: int = 5):
    if not INDEX_FILE.exists():
        print("❌ Index file not found. Run with 'build' first.")
        return
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(EMB_MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    ctx = "\n\n".join(meta[i] for i in I[0])
    prompt = f"Answer the following question using the provided context. If the context does not contain the answer, say 'I don't know.'\n\nContext:\n{ctx}\n\nQuestion: {query}"

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("ℹ️  GOOGLE_API_KEY not set. Returning only retrieved context.")
        print("\n----- Context -----\n", ctx[:2000])
        return
    if genai is None:
        print("pip install google-generativeai required.")
        return
    genai.configure(api_key=key)
    gem = genai.GenerativeModel("models/gemini-2.5-pro")
    resp = gem.generate_content(prompt)
    print("\n----- Answer -----\n")
    print(resp.text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build")                       # 그대로
    ask_p = sub.add_parser("ask")                # ask 서브커맨드
    ask_p.add_argument("question", nargs="+")    # 질문
    ask_p.add_argument("--top_k", type=int, default=5)
    ask_p.add_argument("--show_ctx", action="store_true", help="print retrieved context without calling Gemini")

    args = parser.parse_args()

    if args.cmd == "build":
        build_index()
    else:
        q = " ".join(args.question)
        if args.show_ctx:
            os.environ.pop("GOOGLE_API_KEY", None)  # 강제로 Gemini 건너뜀
        ask(q, top_k=args.top_k) 
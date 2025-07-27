#!/usr/bin/env python3
"""
merge_and_rebuild.py
====================
1. gzip concat rag_dataset_2024_25.jsonl.gz + cv_papers_2024_25.jsonl.gz → rag_dataset_cv.jsonl.gz
2. Rebuild FAISS index using index_rag_dataset.build
   (imports build_index from index_rag_dataset)
"""
import gzip, shutil, os
from pathlib import Path
import index_rag_dataset as rag

OLD = Path('rag_dataset_2024_25.jsonl.gz')
NEW = Path('cv_papers_2024_25.jsonl.gz')
MERGED = Path('rag_dataset_cv.jsonl.gz')

def concat():
    if MERGED.exists():
        MERGED.unlink()
    with gzip.open(MERGED, 'wb') as out:
        for src in (OLD, NEW):
            if not src.exists():
                continue
            with gzip.open(src, 'rb') as f:
                shutil.copyfileobj(f, out)
    print(f"✅ Merged -> {MERGED}")

def rebuild():
    rag.DATA_FILE = MERGED  # monkey patch path
    rag.INDEX_FILE = Path('rag_cv_faiss.index')
    rag.META_FILE  = Path('rag_cv_meta.pkl')
    rag.build_index()

if __name__ == '__main__':
    concat()
    rebuild() 
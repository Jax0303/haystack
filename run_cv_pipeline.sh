#!/usr/bin/env bash
# run_cv_pipeline.sh
# ------------------
# 1. installs deps (only if not present)
# 2. downloads & parses 2024-25 cs.CV PDFs (max 500)
# 3. merges with existing dataset & rebuilds FAISS index
#
# usage:
#   bash run_cv_pipeline.sh [MAX_PDF]
#     MAX_PDF - number of PDFs to fetch (default 500)
#
set -euo pipefail

MAX_PDF=${1:-500}

echo "[1/3] installing python deps (requests tqdm pdfminer.six)"
pip install --upgrade -q requests tqdm pdfminer.six

echo "[2/3] downloading & parsing ${MAX_PDF} cs.CV PDFs from arXiv (this may take ~20 min)"
python get_cv_papers_arxiv.py --max_pdf "$MAX_PDF"

echo "[3/3] merging new corpus and rebuilding FAISS index"
python merge_and_rebuild.py

echo "✅ Pipeline finished. New dataset: rag_dataset_cv.jsonl.gz, new index: rag_cv_faiss.index" 
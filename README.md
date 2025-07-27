# RAG Pipeline (2024-2025 Experimental Corpus)

This repo contains scripts to build a Retrieval-Augmented Generation (RAG) experiment with latest 2024-2025 papers.

## Contents
1. **Dataset generation** – `create_realistic_dataset.py` (400 MB synthetic baseline)
2. **PDF pipeline** – `get_cv_papers_arxiv.py` + `run_cv_pipeline.sh` (download & parse 500 cs.CV papers)
3. **Indexing** – `index_rag_dataset.py` (FAISS + Gemini-2.5-Pro)

Run the full CV pipeline:
```bash
bash run_cv_pipeline.sh 500
```
Then ask questions:
```bash
export GOOGLE_API_KEY="<Your_Gemini_Key>"
python index_rag_dataset.py ask --top_k 15 "What progress was made in computer vision in 2025?"
```

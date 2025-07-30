# RAG Pipeline (2024-2025 Experimental Corpus)

End-to-end Retrieval-Augmented Generation pipeline built around **latest (2024-2025) papers** + Google Gemini.

## 🚀 Latest Updates (v2.0)
### New Features
- **🧠 Advanced Chunking Methods**: Semantic & Agentic chunking for better context coherence
- **🌏 Korean Language Support**: Native Korean Q&A with multilingual BGE embeddings  
- **🔧 Intelligent Text Segmentation**: LLM-guided chunking using Gemini for logical breakpoints
- **📊 Configurable Chunking**: Choose between simple, semantic, or agentic chunking methods
- **🔍 Enhanced Search Quality**: Semantic similarity-based chunk grouping for better retrieval

### Technical Improvements
- Added comprehensive Korean comments throughout codebase
- Implemented fallback strategies for PyTorch compatibility issues
- Enhanced error handling and dependency management
- Improved CLI with chunking method selection

## Contents
1. Synthetic baseline dataset (≈400 MB)  
2. Computer-Vision PDF pipeline (500 cs.CV papers)  
3. FAISS index builder & Gemini query CLI  
4. One-shot shell pipeline  
5. **NEW**: Advanced chunking methods (semantic/agentic)
6. **NEW**: Korean language query support

---
## 0. Quick Start (10 commands)
```bash
# clone & enter
 git clone https://github.com/Jax0303/haystack.git
 cd haystack

# create / activate venv (Python ≥3.9)
 python3 -m venv venv_rag
 source venv_rag/bin/activate

# install minimal deps and run full CV pipeline (≈20 min)
 bash run_cv_pipeline.sh 500

# (optional) set Gemini key once per shell
 export GOOGLE_API_KEY="<YOUR_GEMINI_KEY>"
 python index_rag_dataset.py ask "What progress was made in computer vision in 2025?"
```

---
## 1. Scripts overview
| file | role |
|------|------|
| `create_realistic_dataset.py` | synthetic 400 MB corpus (already generated) |
| `get_cv_papers_arxiv.py`      | download & parse 2024-25 **cs.CV** PDFs (PyMuPDF + multiprocessing) |
| `run_cv_pipeline.sh`          | installs deps → run `get_cv_papers_arxiv.py` → merge & rebuild index |
| `merge_and_rebuild.py`        | concat new JSONL with baseline and rebuild FAISS |
| `index_rag_dataset.py`        | build / query FAISS index (Gemini 2.5-Pro back-end) |

Generated artefacts (large, Git-ignored):
```
cv_pdfs/                         # raw PDFs
cv_papers_2024_25.jsonl.gz       # parsed CV corpus
rag_dataset_2024_25.jsonl.gz     # 400 MB baseline
rag_dataset_cv.jsonl.gz          # merged baseline+CV
rag_cv_faiss.index / rag_cv_meta.pkl
```

---
## 2. Re-running pipeline step-by-step
### 2-1. Install/activate venv
```bash
python3 -m venv venv_rag
source venv_rag/bin/activate
pip install --upgrade requests tqdm pymupdf sentence-transformers faiss-cpu google-generativeai nltk
python -c "import nltk, nltk.downloader; nltk.downloader.download('punkt')"
```

### 2-2. Synthetic dataset (optional)
Already created (`rag_dataset_2024_25.jsonl.gz`). Re-generate if needed:
```bash
python create_realistic_dataset.py
```

### 2-3. Download & parse PDFs
```bash
python get_cv_papers_arxiv.py --max_pdf 500      # resume safe; skips existing PDFs
```

### 2-4. Merge & rebuild index
```bash
python merge_and_rebuild.py                      # produces rag_cv_faiss.index
```

### 2-5. Query with Gemini
```bash
export GOOGLE_API_KEY="<YOUR_KEY>"

# English queries
python index_rag_dataset.py ask --top_k 20 "progress in computer vision in 2025?"

# Korean queries (NEW!)
python index_rag_dataset.py ask "RAG관련 최신 연구는?"
python index_rag_dataset.py ask "컴퓨터 비전에서 트랜스포머 모델은?"
```

### 2-6. Advanced Chunking Options (NEW!)
```bash
# Semantic chunking (default) - groups semantically similar sentences
python index_rag_dataset.py build --chunking semantic --similarity-threshold 0.6

# Agentic chunking - uses LLM to identify logical breakpoints  
python index_rag_dataset.py build --chunking agentic

# Simple chunking - traditional token-based splitting
python index_rag_dataset.py build --chunking simple
```

---
## 3. Resume / continue download
The PDF downloader is **idempotent**:
* Existing files in `cv_pdfs/` are skipped.
* `get_cv_papers_arxiv.py` can be rerun with the same `--max_pdf` to fetch only missing items.

After additional downloads, run `merge_and_rebuild.py` (or simply `bash run_cv_pipeline.sh 500`) to update the index.

---
## 4. Git hygiene
Large binaries & local env are ignored via `.gitignore`:
```
venv_rag/
cv_pdfs/
*.gz
*.index
*.pkl
```

---
## 5. Chunking Methods Explained

### 🔧 Simple Chunking
- **Method**: Fixed token-based splitting with overlap
- **Use case**: Fast processing, consistent chunk sizes
- **Pros**: Predictable, efficient
- **Cons**: May split semantically related content

### 🧠 Semantic Chunking  
- **Method**: Groups sentences by embedding similarity
- **Use case**: Better context coherence, topic continuity
- **Pros**: Maintains semantic relationships
- **Cons**: Slightly slower due to embedding computation

### 🤖 Agentic Chunking
- **Method**: LLM analyzes text structure for logical breakpoints
- **Use case**: Complex documents with clear sections
- **Pros**: Respects document structure, highest quality
- **Cons**: Requires LLM calls, slower processing

---
## 6. Troubleshooting
* **Slow parsing?** PyMuPDF + multiprocessing already enabled (≈10× faster than pdfminer). Adjust `--max_pdf` or `--workers` inside script if needed.
* **Gemini returns _I don't know_**: likely no relevant chunk – add more papers and rebuild.
* **PyTorch import errors**: Script auto-installs compatible versions. Set `CUDA_VISIBLE_DEVICES=''` for CPU-only mode.
* **Korean text issues**: Ensure UTF-8 encoding and BGE-Small-EN model supports your language.
* **Authentication**: Git pushes require PAT (set once, then cached).

---
© 2025 Jax0303  
Feel free to fork / PR.

# Enterprise RAG with HERB Dataset

기업 특화 Retrieval-Augmented Generation 시스템 with **HERB (Heterogeneous Enterprise RAG Benchmark)** 데이터셋.

## 🚀 Latest Updates (v3.0 - Enterprise Edition)
### 🏢 Enterprise Features
- **📊 HERB Dataset Integration**: Salesforce AI Research의 39K+ 엔터프라이즈 아티팩트 활용
- **🔐 Role-Based Access Control (RBAC)**: 사용자 역할별 문서 접근 제어
- **🎯 Hallucination Detection**: 신뢰도 점수 & 환각 탐지 메커니즘
- **📝 Audit Logging**: 모든 질의-응답 불변 로그 저장 (컴플라이언스)
- **🔍 Multi-Artifact Support**: Slack, 문서, 회의록, PR 등 다양한 소스 통합

### 🛠️ Technical Improvements  
- **Chrome Vector DB**: 임베딩 저장 & 유사도 검색 최적화
- **LangChain Integration**: 모듈식 RAG 파이프라인 구조
- **CLI Interface**: 간편한 인덱싱 & 질의 명령어 지원
- **Enterprise JSON Parsing**: HERB 실제 구조에 맞춘 다중 아티팩트 파싱

## 📁 Project Structure
```
├── index_herb_dataset.py      # HERB 데이터셋 다운로드 & 인덱싱
├── advanced_rag.py            # 엔터프라이즈 RAG 엔진 (RBAC, 환각탐지)
├── requirements_enterprise_rag.txt  # 필요 패키지 목록
├── HERB/                      # GitHub clone된 HERB 리포지토리
└── chroma_db/                 # 벡터 인덱스 저장소
```

## 🚀 Quick Start (Enterprise RAG)
```bash
# 1. 리포지토리 클론
git clone https://github.com/Jax0303/haystack-2.git
cd haystack-2

# 2. 가상환경 설정
python3 -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치
pip install -r requirements_enterprise_rag.txt

# 4. HuggingFace 토큰 설정 (무료 계정 가능)
export HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# 5. HERB 데이터셋 인덱싱 (최초 1회, 약 5분)
python index_herb_dataset.py --max_files 0  # 전체 또는 --max_files 100 (빠른 테스트)

# 6. 엔터프라이즈 RAG 질의 실행
python advanced_rag.py query "What features did customers complain about?" --top_k 6 --roles "*"
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

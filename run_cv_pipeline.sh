#!/usr/bin/env bash
# run_cv_pipeline.sh
# ------------------
# Computer Vision 논문 RAG 파이프라인 자동화 스크립트
#
# 전체 과정:
# 1. Python 의존성 패키지 설치 (requests, tqdm, pymupdf, nltk 등)
# 2. arXiv에서 2024-25년 cs.CV 논문 PDF 다운로드 및 텍스트 추출
# 3. 기존 합성 데이터셋과 병합하여 FAISS 인덱스 재구축
#
# 사용법:
#   bash run_cv_pipeline.sh [MAX_PDF]
#     MAX_PDF - 다운로드할 PDF 수 (기본값: 500)
#
# 예시:
#   bash run_cv_pipeline.sh 1000  # 1000개 PDF 처리
#
# 소요 시간: 약 20-30분 (네트워크 속도 및 CPU 성능에 따라 변동)
# 디스크 사용량: PDF ~2GB, 처리된 데이터 ~1GB
#
set -euo pipefail  # 엄격한 오류 처리: 명령 실패 시 즉시 종료

# 커맨드라인 인자로 최대 PDF 수 설정 (기본값: 500)
MAX_PDF=${1:-500}

echo "[1/3] installing python deps (requests tqdm pymupdf pdfminer.six nltk)"
# 필수 Python 패키지 설치
# - requests: HTTP 요청 (arXiv API, PDF 다운로드)
# - tqdm: 진행률 표시
# - pymupdf: PDF 텍스트 추출 (get_cv_papers_arxiv.py에서 사용)
# - pdfminer.six: 대체 PDF 파서 (호환성용)
# - nltk: 자연어 처리 (문장 분할용)
pip install --upgrade -q requests tqdm pymupdf pdfminer.six nltk

echo "[2/3] downloading & parsing ${MAX_PDF} cs.CV PDFs from arXiv (this may take ~20 min)"
# arXiv Computer Vision 논문 수집 및 파싱
# - 2024-2025년 제출된 cs.CV 카테고리 논문 대상
# - PyMuPDF를 사용한 멀티프로세싱 텍스트 추출
# - 결과: cv_papers_2024_25.jsonl.gz (각 줄: {arxiv_id, title, published, text})
python get_cv_papers_arxiv.py --max_pdf "$MAX_PDF"

echo "[3/3] merging new corpus and rebuilding FAISS index"
# 데이터셋 병합 및 인덱스 재구축
# - 기존 합성 데이터셋(400MB) + 새 CV 논문 데이터 병합
# - Sentence-Transformer (BGE-Small-EN)로 임베딩 생성
# - FAISS IndexFlatIP로 코사인 유사도 검색 인덱스 구축
# - 결과: rag_cv_faiss.index, rag_cv_meta.pkl
python merge_and_rebuild.py

echo "✅ Pipeline finished. New dataset: rag_dataset_cv.jsonl.gz, new index: rag_cv_faiss.index"
# 성공 메시지 및 생성된 파일 안내
# 이제 다음 명령으로 RAG 질의 가능:
# python index_rag_dataset.py ask "질문 내용" 
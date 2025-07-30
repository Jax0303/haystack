#!/usr/bin/env python3
"""
merge_and_rebuild.py
====================
데이터셋 병합 및 FAISS 인덱스 재구축 스크립트

주요 기능:
1. 기존 합성 데이터셋(rag_dataset_2024_25.jsonl.gz)과 
   새로 수집한 CV 논문(cv_papers_2024_25.jsonl.gz)을 병합
2. 병합된 데이터셋으로 FAISS 인덱스 재구축
3. CV 전용 인덱스 파일 생성 (rag_cv_faiss.index)

사용법:
  python merge_and_rebuild.py
  
결과:
  - rag_dataset_cv.jsonl.gz: 병합된 전체 데이터셋
  - rag_cv_faiss.index: CV 데이터 포함 FAISS 인덱스
  - rag_cv_meta.pkl: 인덱스에 대응하는 텍스트 청크 메타데이터
"""
import gzip, shutil, os
from pathlib import Path
import index_rag_dataset as rag

# === 파일 경로 정의 ===
OLD = Path('rag_dataset_2024_25.jsonl.gz')    # 기존 합성 데이터셋 (400MB)
NEW = Path('cv_papers_2024_25.jsonl.gz')      # 새로 수집한 CV 논문 데이터
MERGED = Path('rag_dataset_cv.jsonl.gz')      # 병합 결과 파일

def concat():
    """
    두 개의 gzip 압축된 JSONL 파일을 효율적으로 병합
    
    메모리 효율성을 위해 파일을 전체 로드하지 않고
    바이너리 스트림 복사를 통해 병합 수행
    """
    # 기존 병합 파일이 있으면 삭제 (새로 생성)
    if MERGED.exists():
        MERGED.unlink()
    
    # gzip 압축 상태로 파일 병합
    with gzip.open(MERGED, 'wb') as out:
        # 기존 데이터셋과 새 데이터셋을 순차적으로 병합
        for src in (OLD, NEW):
            if not src.exists():  # 파일이 없으면 건너뛰기
                continue
            # 바이너리 스트림 복사 (메모리 효율적)
            with gzip.open(src, 'rb') as f:
                shutil.copyfileobj(f, out)
    
    print(f"✅ Merged -> {MERGED}")

def rebuild():
    """
    병합된 데이터셋으로 FAISS 인덱스 재구축
    
    index_rag_dataset 모듈의 설정을 동적으로 변경하여
    CV 전용 인덱스 파일 경로로 빌드 수행
    """
    # 인덱스 빌더 모듈의 파일 경로를 CV 전용으로 변경 (monkey patch)
    rag.DATA_FILE = MERGED                          # 입력: 병합된 데이터셋
    rag.INDEX_FILE = Path('rag_cv_faiss.index')    # 출력: CV 인덱스 파일
    rag.META_FILE  = Path('rag_cv_meta.pkl')       # 출력: CV 메타데이터 파일
    
    # 기존 build_index 함수 호출하여 인덱스 생성
    rag.build_index()

if __name__ == '__main__':
    # 1단계: 데이터셋 병합
    concat()
    # 2단계: FAISS 인덱스 재구축
    rebuild() 
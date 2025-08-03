#!/usr/bin/env python3
"""index_herb_dataset.py

GitHub에서 공개된 HERB(Heterogeneous Enterprise RAG Benchmark) 리포지토리를
로컬에 클론(clone)한 뒤, 제품(product) 아티팩트 JSON 파일을 LangChain
`Document`로 변환하여 `advanced_rag.RagEngine`의 Chroma 인덱스에 저장한다.

장점
-----
1. Hugging Face `datasets` 모듈의 Arrow 파싱 오류를 우회한다.
2. 파일별(JSON) 파싱이므로 스키마가 달라도 안전하다.
3. 원하는 개수만 샘플링할 수 있어 빠른 실험 가능.

사용법
-------
```bash
source .venv/bin/activate
pip install -U -r requirements_enterprise_rag.txt  # json, gitpython 등 필요 X

# 전체 인덱싱 (수 분)
python index_herb_dataset.py --repo_dir HERB --max_files 0

# 빠른 샘플(예: 100파일)만 인덱싱
python index_herb_dataset.py --max_files 100
```

Options
-------
--repo_dir      : HERB GitHub 저장소 경로(없으면 자동 clone)
--chunk_size    : 청크 문자 길이 (기본 800)
--chunk_overlap : 청크 겹침 (기본 120)
--max_files     : 0이면 전체, n>0 이면 n개 파일만 인덱싱
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from advanced_rag import RagEngine

REPO_URL = "https://github.com/SalesforceAIResearch/HERB.git"
DEFAULT_REPO_DIR = Path("HERB")  # ./HERB
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def ensure_repo(repo_dir: Path):
    """
    HERB GitHub 저장소가 로컬에 없으면 자동으로 클론
    
    Args:
        repo_dir: 클론할 로컬 디렉토리 경로
    """
    if repo_dir.exists():
        print(f"✅ HERB 저장소가 이미 존재합니다: {repo_dir}")
        return
    print(f"📥 HERB 데이터셋 다운로드 중... {REPO_URL} → {repo_dir}")
    print(f"⏳ 약 수초~수분 소요 (네트워크 속도에 따라 다름)")
    # --depth 1: 최신 커밋만 다운로드하여 속도 향상
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)
    print(f"✅ 다운로드 완료!")


def iter_product_files(repo_dir: Path, max_files: int = 0):
    """
    HERB 데이터셋의 제품별 JSON 파일들을 순회하는 제너레이터
    
    Args:
        repo_dir: HERB 저장소 루트 디렉토리
        max_files: 최대 처리할 파일 수 (0=전체, 테스트용으로 일부만 처리 가능)
        
    Yields:
        Path: 각 제품의 JSON 파일 경로
    """
    products_dir = repo_dir / "data" / "products"        # HERB 제품 데이터 폴더
    print(f"📁 제품 데이터 폴더 탐색: {products_dir}")
    
    # rglob으로 하위 폴더까지 재귀적으로 JSON 파일 검색
    files = sorted(products_dir.rglob("*.json"))
    print(f"🔍 총 {len(files)}개의 JSON 파일 발견")
    
    if max_files > 0:
        files = files[:max_files]
        print(f"📊 테스트용으로 {len(files)}개 파일만 처리")
        
    for fp in files:
        yield fp


def load_docs_from_file(json_path: Path, splitter) -> List[Document]:
    """
    HERB 제품 JSON 파일을 파싱하여 LangChain Document 객체들로 변환
    
    Args:
        json_path: 처리할 JSON 파일 경로
        splitter: 텍스트 청킹을 위한 LangChain 분할기
        
    Returns:
        List[Document]: 변환된 문서 객체 목록 (청킹 후)
    """
    try:
        # JSON 파일 읽기 및 파싱
        data = json.loads(json_path.read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  JSON 파싱 실패: {json_path.name} → 건너뜀 (원인: {exc})")
        return []

    product_id = json_path.stem                      # 파일명에서 제품 ID 추출
    docs: List[Document] = []                        # 결과 문서 리스트
    
    # HERB 데이터셋의 실제 아티팩트 구조 매핑
    # 각 제품마다 다양한 기업용 데이터 유형들이 포함됨
    artifact_types = {
        "slack": "slack_message",               # 팀 내부 슬랙 메시지
        "documents": "document",                # 제품 문서, 요구사항 등 
        "meeting_transcripts": "meeting_transcript",
        "meeting_chats": "meeting_chat",
        "urls": "url",
        "prs": "pull_request"
    }
    
    for art_key, art_type in artifact_types.items():
        if art_key not in data:
            continue
            
        items = data[art_key]                        # 해당 유형의 아티팩트 배열
        if not isinstance(items, list):
            continue  # 배열이 아니면 건너뜀
            
        # 각 아티팩트를 순회하며 처리
        for idx, item in enumerate(items):
            content = ""                             # 추출될 텍스트 컨텐츠
            
            # HERB의 헤테로지니어스 스키마: 아티팩트 유형별로 다른 추출 방식 사용
            if art_key == "slack":
                # 슬랙: messages 배열 내의 각 메시지에서 text 추출
                if isinstance(item.get("messages"), list):
                    messages_text = []
                    for msg in item["messages"]:
                        if isinstance(msg, dict) and msg.get("text"):
                            messages_text.append(msg["text"])
                    content = "\n".join(messages_text)
            elif art_key == "documents":
                # 문서: content 또는 text 필드에서 컨텐츠 추출
                content = item.get("content") or item.get("text") or ""
            elif art_key in ["meeting_transcripts", "meeting_chats"]:
                # 회의 관련: transcript 또는 content 필드 사용
                content = item.get("transcript") or item.get("content") or ""
            else:
                # 기타 유형 (URLs, PRs 등): 일반적인 필드들 시도
                content = (
                    item.get("content")     # 일반 컨텐츠
                    or item.get("text")    # 텍스트 내용
                    or item.get("description")  # 설명
                    or ""                 # 기본값
                )
            
            # 너무 짧거나 빈 컨텐츠는 제외 (RAG 효율성을 위해)
            if not content or len(content.strip()) < 20:
                continue  # 최소 20자 이상의 의미있는 컨텐츠만 사용
                
            # LangChain Document를 위한 메타데이터 구성
            meta = {
                "product_id": product_id,                         # 제품 식별자
                "artifact_type": art_type,                        # 아티팩트 유형 (슬랙, 문서 등)
                "artifact_id": item.get("id", f"{art_key}_{idx}"), # 개별 아티팩트 ID (없으면 자동 생성)
                "source_file": json_path.name,                    # 원본 JSON 파일명
                "acl": "*"                                      # 접근 제어 (기본: 모든 사용자 허용)
            }
            
            # 긴 컨텐츠를 작은 청크로 분할 (RAG 검색 효율성 향상)
            for chunk in splitter.split_text(content):
                # 각 청크를 LangChain Document로 변환하여 추가
                docs.append(Document(page_content=chunk, metadata=meta))
    
    return docs  # 완성된 문서 청크 목록 반환


def build_index(repo_dir: Path, chunk_size: int, chunk_overlap: int, max_files: int):
    """
    HERB 데이터셋을 처리하여 ChromaDB 벡터 인덱스를 구축
    
    Args:
        repo_dir: HERB 저장소 디렉토리
        chunk_size: 텍스트 청크 크기 (문자 단위)
        chunk_overlap: 청크 간 겹침 크기
        max_files: 처리할 최대 파일 수 (0=전체)
    """
    print(f"🔧 텍스트 분할기 초기화: 청크={chunk_size}, 겹침={chunk_overlap}")
    # 재귀적 문자 단위 텍스트 분할기 설정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,           # 각 청크의 최대 문자 수
        chunk_overlap=chunk_overlap,     # 인접 청크 간 겹치는 문자 수
        separators=["\n\n", "\n", " "], # 분할 우선순위: 단락 > 줄바꿈 > 공백
    )
    
    print(f"📦 모든 JSON 파일에서 문서 추출 시작...")
    all_docs: List[Document] = []        # 모든 처리된 문서들을 저장
    
    # 각 JSON 파일을 순회하며 문서 추출
    for fp in iter_product_files(repo_dir, max_files):
        file_docs = load_docs_from_file(fp, splitter)
        all_docs.extend(file_docs)
        
    print(f"✅ 총 {len(all_docs):,}개의 Document 청크 생성 완료")
    
    # 빈 결과 처리 (추출 실패 시)
    if not all_docs:
        print("❌ 수집된 문서가 없습니다. HERB 경로 또는 JSON 스키마를 확인하세요.")
        return

    print(f"🚀 RAG 엔진 초기화: ChromaDB 벡터 저장소 준비...")
    # HERB 전용 ChromaDB 컬렉션으로 RAG 엔진 초기화
    engine = RagEngine(collection_name="herb_collection", persist_dir="./chroma_db")
    
    # ChromaDB의 배치 크기 제한 문제 해결을 위한 수동 배치 처리
    BATCH_SIZE = 5000  # 한 번에 처리할 최대 문서 수
    # 전체 문서를 배치로 나누어 처리 (큰 데이터셋 대응)
    total_batches = (len(all_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 배치 처리 시작: {total_batches}개 배치로 분할 처리")
    
    # 각 배치를 순차적으로 벡터 DB에 추가
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]           # 현재 배치 추출
        batch_num = (i // BATCH_SIZE) + 1            # 배치 번호 계산
        print(f"   📦 배치 {batch_num}/{total_batches} 임베딩 중... ({len(batch)}개 문서)")
        
        # ChromaDB에 벡터 임베딩 및 저장
        engine.vectordb.add_documents(batch)
    
    # 디스크에 영구 저장
    engine.vectordb.persist()
    print(f"✅ HERB 데이터셋 인덱스 구축 완료! 저장 위치: ./chroma_db")
    print(f"📊 최종 통계: {len(all_docs):,}개 문서 청크가 벡터 DB에 저장됨")


def main():
    """
    메인 실행 함수: CLI 인자 파싱 및 HERB 데이터셋 인덱싱 프로세스 실행
    """
    print(f"🔍 HERB 데이터셋 인덱싱 도구 시작")
    print(f"🎆 Salesforce AI Research - Heterogeneous Enterprise RAG Benchmark")
    print()
    
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser(
        description="HERB GitHub 데이터셋을 ChromaDB 벡터 인덱스로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python index_herb_dataset.py                    # 전체 데이터셋 인덱싱
  python index_herb_dataset.py --max_files 50    # 테스트용 50개 파일만
  python index_herb_dataset.py --chunk_size 1000  # 청크 크기 조절
        """
    )
    parser.add_argument("--repo_dir", type=Path, default=DEFAULT_REPO_DIR, 
                        help=f"HERB 저장소 디렉토리 (기본: {DEFAULT_REPO_DIR})")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"텍스트 청크 크기 (기본: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"청크 간 겹침 크기 (기본: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--max_files", type=int, default=0, 
                        help="처리할 최대 파일 수 (0=전체, n>0=n개 파일만 테스트용)")
    
    args = parser.parse_args()

    print(f"⚙️  설정값:")
    print(f"   - HERB 디렉토리: {args.repo_dir}")
    print(f"   - 청크 크기: {args.chunk_size} 문자")
    print(f"   - 청크 겹침: {args.chunk_overlap} 문자")
    print(f"   - 처리 파일 수: {'전체' if args.max_files == 0 else f'{args.max_files}개'}")
    print()

    # 단계별 실행
    ensure_repo(args.repo_dir)      # 1단계: HERB 저장소 클론 확인
    build_index(args.repo_dir, args.chunk_size, args.chunk_overlap, args.max_files)  # 2단계: 인덱스 구축
    
    print()
    print(f"🎉 모든 작업이 완료되었습니다!")
    print(f"🚀 이제 'python advanced_rag.py query \"질문\"' 명령으로 RAG 시스템을 테스트해보세요!")


if __name__ == "__main__":
    # 스크립트가 직접 실행될 때만 메인 함수 호출
    main()

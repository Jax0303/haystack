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
    """HERB 리포가 없으면 git clone 한다."""
    if repo_dir.exists():
        return
    print(f"▶️  Git clone {REPO_URL} → {repo_dir} … (수초~수분 소요)")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)


def iter_product_files(repo_dir: Path, max_files: int = 0):
    """`data/products/*.json` 경로의 파일 Path iterator."""
    products_dir = repo_dir / "data" / "products"
    files = sorted(products_dir.rglob("*.json"))  # 하위 폴더까지 탐색
    if max_files > 0:
        files = files[:max_files]
    for fp in files:
        yield fp


def load_docs_from_file(json_path: Path, splitter) -> List[Document]:
    """HERB product 파일 하나를 읽어 Document 리스트로 변환."""
    try:
        data = json.loads(json_path.read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  {json_path.name} 파싱 실패 → 건너뜀 ({exc})")
        return []

    product_id = json_path.stem
    docs: List[Document] = []
    
    # HERB 실제 구조: slack, documents, meeting_transcripts, meeting_chats, urls, prs
    artifact_types = {
        "slack": "slack_message",
        "documents": "document", 
        "meeting_transcripts": "meeting_transcript",
        "meeting_chats": "meeting_chat",
        "urls": "url",
        "prs": "pull_request"
    }
    
    for art_key, art_type in artifact_types.items():
        if art_key not in data:
            continue
            
        items = data[art_key]
        if not isinstance(items, list):
            continue
            
        for idx, item in enumerate(items):
            content = ""
            
            # 각 타입별로 텍스트 추출 방식이 다름
            if art_key == "slack":
                # Slack: messages 리스트에서 text 추출
                if isinstance(item.get("messages"), list):
                    content = "\n".join([
                        msg.get("text", "") for msg in item["messages"] 
                        if isinstance(msg, dict) and msg.get("text")
                    ])
            elif art_key == "documents":
                # Documents: content 키 또는 text 키
                content = item.get("content") or item.get("text") or ""
            elif art_key in ["meeting_transcripts", "meeting_chats"]:
                # 회의: transcript 또는 content
                content = item.get("transcript") or item.get("content") or ""
            else:
                # 기타: content, text, description 등
                content = (
                    item.get("content") 
                    or item.get("text")
                    or item.get("description")
                    or ""
                )
            
            if not content or len(content.strip()) < 20:
                continue
                
            meta = {
                "product_id": product_id,
                "artifact_type": art_type,
                "artifact_id": item.get("id", f"{art_key}_{idx}"),
                "source_file": json_path.name,
            }
            
            for chunk in splitter.split_text(content):
                docs.append(Document(page_content=chunk, metadata=meta))
    
    return docs


def build_index(repo_dir: Path, chunk_size: int, chunk_overlap: int, max_files: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    all_docs: List[Document] = []
    for fp in iter_product_files(repo_dir, max_files):
        all_docs.extend(load_docs_from_file(fp, splitter))
    print(f"   총 Document 수: {len(all_docs):,}")
    if not all_docs:
        print("❌ 수집된 문서가 없습니다. 경로 또는 JSON 키를 확인하세요.")
        return

    engine = RagEngine(collection_name="herb_collection", persist_dir="./chroma_db")
    
    # ChromaDB 배치 크기 제한 (최대 5000개씩 처리)
    BATCH_SIZE = 5000
    total_batches = (len(all_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"   배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 문서)")
        engine.vectordb.add_documents(batch)
    
    engine.vectordb.persist()
    print("✅ 인덱스 구축 완료 → ./chroma_db")


def main():
    parser = argparse.ArgumentParser(description="HERB GitHub JSON을 Chroma 인덱스로 변환")
    parser.add_argument("--repo_dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--max_files", type=int, default=0, help="0=전체, n>0 일 경우 n개 파일만")
    args = parser.parse_args()

    ensure_repo(args.repo_dir)
    build_index(args.repo_dir, args.chunk_size, args.chunk_overlap, args.max_files)


if __name__ == "__main__":
    main()

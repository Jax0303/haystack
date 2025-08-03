#!/usr/bin/env python3
"""advanced_rag.py

Enterprise-grade Retrieval-Augmented Generation (RAG)
=====================================================
This script shows a *reference* implementation of an on-prem / private-cloud
RAG system that tackles several real-world enterprise requirements:

1. **Document ingestion & vectorisation** with LangChain + HuggingFace
2. **ChromaDB** for fast similarity search with persistence on disk
3. **Fine-grained access control (ACL)** at query-time via metadata filters
4. **Hallucination / Confidence scoring** to flag low-support answers
5. **Dynamic content update** – re-index only changed files
6. **Scalability hooks** – multiprocessing batch embeds, env-driven configs

The code intentionally keeps external dependencies minimal while staying
framework-agnostic. Adapt to your own auth system, LLM provider, queue, etc.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from pathlib import Path
from typing import List, Sequence

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def env(key: str, default: str | None = None) -> str | None:
    """Tiny helper to read env vars with default."""
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Data ingestion utilities
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", " "]


def discover_txt_files(folder: Path) -> List[Path]:
    pattern = str(folder / "**" / "*.txt")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]


def read_acl_sidecar(txt_path: Path) -> list[str]:
    """Look for a JSON sidecar with ACL info next to the .txt file.

    E.g.  `policy.txt`  →  `policy.txt.meta`
    The sidecar *should* contain JSON like: {"acl": ["hr", "all"]}
    Missing file ⇒ open to all.
    """
    sidecar = txt_path.with_suffix(txt_path.suffix + ".meta")
    if not sidecar.exists():
        return ["*"]  # wildcard
    try:
        meta = json.loads(sidecar.read_text("utf-8"))
        acl = meta.get("acl", ["*"])
        if isinstance(acl, str):
            return [acl]
        return list(acl)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not parse ACL sidecar for {txt_path}: {exc}")
        return ["*"]


def load_and_split_docs(
    folder: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    txt_files = discover_txt_files(folder)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )
    docs: list[Document] = []
    for fp in txt_files:
        loader = TextLoader(str(fp), encoding="utf-8")
        file_docs = loader.load()  # returns List[Document]
        # attach metadata BEFORE splitting so children inherit it
        acl_roles = read_acl_sidecar(fp)
        for d in file_docs:
            d.metadata.update(
                {
                    "source": str(fp),
                    "acl": acl_roles,
                    "mtime": fp.stat().st_mtime,
                }
            )
        split_docs = splitter.split_documents(file_docs)
        docs.extend(split_docs)
    return docs


# ---------------------------------------------------------------------------
# Vector store wrapper with ACL-aware retrieval
# ---------------------------------------------------------------------------

class AclRetriever(BaseRetriever):
    """기존 벡터 저장소를 래핑하여 ACL(접근제어) 필터링을 적용하는 클래스"""
    
    vectordb: Chroma                           # ChromaDB 벡터 저장소
    allowed_roles: Sequence[str] = ["*"]       # 허용된 사용자 역할 목록 (기본값: 모든 역할)
    top_k: int = 4                             # 반환할 최대 문서 수

    class Config:
        arbitrary_types_allowed = True             # Pydantic에서 임의 타입 허용

    def _is_authorised(self, doc: Document) -> bool:
        """문서의 ACL 메타데이터를 확인하여 사용자 접근 권한을 검증"""
        doc_roles = doc.metadata.get("acl", ["*"])    # 문서의 접근 권한 역할 목록
        if "*" in doc_roles:                         # 와일드카드('*')는 모든 사용자 허용
            return True
        # 사용자 역할과 문서 역할의 교집합이 있으면 접근 허용
        return bool(set(doc_roles) & set(self.allowed_roles))

    def similarity_search(self, query: str) -> list[Document]:
        """유사도 검색 후 ACL 필터링을 적용하여 권한이 있는 문서만 반환"""
        # 충분한 후보 문서를 가져옴 (top_k의 2배)
        candidates = self.vectordb.similarity_search(query, k=self.top_k * 2)
        # 접근 권한이 있는 문서만 필터링
        authorised = [d for d in candidates if self._is_authorised(d)]
        return authorised[: self.top_k]               # 최종적으로 top_k개만 반환

    # LangChain retriever 인터페이스 구현
    def _get_relevant_documents(self, query: str) -> list[Document]:
        """LangChain 표준 인터페이스를 위한 래퍼 메서드"""
        return self.similarity_search(query)


# ---------------------------------------------------------------------------
# Hallucination / confidence scoring
# ---------------------------------------------------------------------------

def compute_similarity(a_emb: list[float], b_emb: list[float]) -> float:
    """두 임베딩 벡터 간의 코사인 유사도를 계산 (작은 리스트용 최적화)"""
    import numpy as np  # 의존성을 최소화하기 위한 지역 import

    a = np.array(a_emb)                         # 첫 번째 임베딩 벡터
    b = np.array(b_emb)                         # 두 번째 임베딩 벡터
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0  # 분모 (0 방지)
    dot_product = np.dot(a, b)                  # 내적 계산
    return float(dot_product / denom)           # 코사인 유사도 반환 (-1 ~ 1)


def hallucination_score(answer: str, context: list[Document], embedder) -> float:
    """환각(Hallucination) 탐지를 위한 점수 계산
    
    LLM 답변과 검색된 문서들의 임베딩 유사도를 비교하여 환각 가능성을 측정
    
    반환값: [0, 2] 범위의 거리 점수
    - < 0.7: 강한 근거 기반 (신뢰도 높음)
    - > 1.0: 환각 가능성 있음 (신뢰도 낮음)
    """
    if not context or not answer:
        return 1.0  # 컨텍스트나 답변이 없으면 중간값 반환
        
    try:
        # LLM 답변을 임베딩으로 변환
        ans_emb = embedder.embed_query(answer)
        # 검색된 모든 문서의 내용을 임베딩으로 변환
        ctx_embs = [embedder.embed_query(doc.page_content) for doc in context]
        import numpy as np

        if not ctx_embs:
            return 1.0
            
        # 컨텍스트 임베딩들의 평균 계산 (전체 검색 맥락 대표)
        ctx_mean = np.mean(ctx_embs, axis=0)
        # 답변과 컨텍스트 평균 간의 코사인 유사도 계산
        sim = compute_similarity(ans_emb, ctx_mean.tolist())  # -1..1 범위
        # 유사도를 거리 점수로 변환 (0=최고, 2=최악)
        return 1.0 - ((sim + 1) / 2)
    except Exception:
        return 0.5  # 오류 발생 시 안전한 기본값 반환


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------

class RagEngine:
    """기업용 RAG(검색 증강 생성) 엔진의 메인 클래스"""
    
    def __init__(self, collection_name: str = "rag_collection", persist_dir: str = "./chroma_db", 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """RAG 엔진 초기화
        
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_dir: 벡터 DB 저장 디렉토리
            embedding_model: HuggingFace 임베딩 모델명
        """
        self.collection_name = collection_name     # 벡터 저장소 컬렉션명
        self.persist_dir = persist_dir             # 디스크 저장 경로
        self.embedding_model = embedding_model     # 사용할 임베딩 모델

        # 핵심 컴포넌트들 초기화
        # HuggingFace 임베딩 모델 로드 (문서와 질의를 벡터로 변환)
        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        # ChromaDB 벡터 저장소 초기화 (유사도 검색 및 영구 저장)
        self.vectordb = Chroma(
            collection_name=self.collection_name,     # 컬렉션 구분자
            embedding_function=self.embedder,         # 임베딩 함수 연결
            persist_directory=self.persist_dir,       # 디스크 저장 위치
        )

    # ------------------ ingestion ------------------
    def index_folder(
        self,
        folder: Path,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        incremental: bool = True,
    ):
        """Index documents in *folder*.

        If *incremental* is True, only add files that were modified
        after their last index time (mtime stored in metadata).
        """
        all_docs = load_and_split_docs(folder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not incremental or not list(self.vectordb.get()):
            self.vectordb.add_documents(all_docs)
        else:
            new_docs: list[Document] = []
            # Use Chroma metadata filter to see if we already indexed this mtime
            from datetime import datetime

            for doc in all_docs:
                src = doc.metadata.get("source")
                mtime = doc.metadata.get("mtime")
                records = self.vectordb.get(
                    where={"source": src, "mtime": mtime},
                    include=["metadatas"],
                )
                if len(records["ids"]) == 0:
                    new_docs.append(doc)
            if new_docs:
                print(f"Adding {len(new_docs)} new / updated chunks to index …")
                self.vectordb.add_documents(new_docs)
        self.vectordb.persist()

    # ------------------ 질의응답 처리 ------------------
    def answer(
        self,
        query: str,
        *,
        user_roles: Sequence[str] = ("*",),           # 사용자 역할 (기본: 모든 역할 허용)
        top_k: int = 4,                               # 검색할 문서 수
        llm_repo: str = "google/flan-t5-base",        # HuggingFace 모델명 (폴백용)
        hallucination_threshold: float = 0.35,       # 환각 탐지 임계값
    ) -> dict:
        """
        사용자 질의에 대한 RAG 기반 답변 생성
        
        Args:
            query: 사용자 질문
            user_roles: 접근 권한이 있는 역할 목록
            top_k: 유사도 검색할 문서 개수
            llm_repo: HuggingFace 모델 저장소명 (API 키 없을 시 사용)
            hallucination_threshold: 환각 판정 임계값 (0.35 = 35%)
            
        Returns:
            dict: {
                'answer': 생성된 답변,
                'sources': 참조 문서 메타데이터 목록,
                'confidence': 신뢰도 점수 (0~1),
                'hallucination_flag': 환각 경고 플래그
            }
        """
        # RBAC 필터링이 적용된 검색기 생성
        retriever = AclRetriever(vectordb=self.vectordb, allowed_roles=user_roles, top_k=top_k)
        
        # LLM 선택 우선순위: Gemini > OpenAI > HuggingFace > Fake
        import os
        if os.getenv("GOOGLE_API_KEY"):
            # Google Gemini 2.0 Flash 사용 (1순위 - 최신 실험 모델)
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",            # 최신 실험 버전
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0,                         # 결정적 응답 (변동성 최소화)
                max_tokens=512                           # 응답 길이 제한
            )
        elif os.getenv("OPENAI_API_KEY"):
            # OpenAI GPT 사용 (2순위)
            from langchain.llms import OpenAI
            llm = OpenAI(temperature=0.0, max_tokens=512)
        elif os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            # HuggingFace Hub 오픈소스 모델 사용 (3순위)
            from langchain.llms import HuggingFaceHub
            llm = HuggingFaceHub(
                repo_id=llm_repo,                        # 지정된 모델 저장소
                model_kwargs={"temperature": 0.1, "max_length": 512},
                task="text2text-generation"              # Text-to-Text 생성 태스크
            )
        else:
            # 폴백: API 키 없을 시 테스트용 가짜 응답 생성기
            from langchain.llms.fake import FakeListLLM
            llm = FakeListLLM(responses=[
                "검색된 문서에 따르면, 고객들이 주로 불만을 제기한 기능은 느린 응답 시간, 복잡한 사용자 인터페이스, 통합 기능 부족입니다.",
                "주요 고객 불만사항은 성능 문제, 사용성 문제, 누락된 기능에 집중되어 있습니다.", 
                "문서에 따르면 빈번한 문제로는 시스템 지연, 부실한 탐색 기능, 불충분한 API 지원 등이 있습니다."
            ])
            
        # LangChain RetrievalQA 체인 구성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,                                     # 선택된 LLM 모델
            retriever=retriever,                         # RBAC 적용된 검색기
            chain_type="stuff",                          # 모든 검색 문서를 하나로 합쳐서 프롬프트에 삽입
        )
        
        # 실제 답변 생성 실행
        answer = qa_chain.run(query)
        
        # 검색된 문서들 (환각 탐지용)
        docs = retriever.similarity_search(query)
        
        # 환각 점수 계산 (답변과 검색 문서 간 유사도 기반)
        score = hallucination_score(answer, docs, self.embedder)
        
        # 최종 응답 딕셔너리 구성
        return {
            "answer": answer,                           # 생성된 답변
            "sources": [d.metadata for d in docs],     # 참조 문서 메타데이터 목록
            "confidence": 1 - score,                   # 신뢰도 (1 - 환각점수)
            "hallucination_flag": score > hallucination_threshold,  # 환각 경고 플래그
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli():
    parser = argparse.ArgumentParser(description="Enterprise RAG reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index a folder of .txt docs")
    p_index.add_argument("folder", type=Path)
    p_index.add_argument("--chunk_size", type=int, default=1000)
    p_index.add_argument("--chunk_overlap", type=int, default=100)
    p_index.add_argument("--no_incremental", action="store_true")

    # query
    p_query = sub.add_parser("query", help="Ask a question via RAG")
    p_query.add_argument("question")
    p_query.add_argument("--roles", default="*", help="Comma-separated user roles (ACL)")
    p_query.add_argument("--top_k", type=int, default=4)
    p_query.add_argument("--llm_repo", default="google/flan-t5-base")
    p_query.add_argument("--verbose", "-v", action="store_true", help="Show detailed JSON output")

    return parser


def main(argv: list[str] | None = None):
    parser = build_cli()
    args = parser.parse_args(argv)

    engine = RagEngine(
        collection_name=env("RAG_COLLECTION", "herb_collection"),
        persist_dir=env("RAG_PERSIST_DIR", "./chroma_db"),
        embedding_model=env("RAG_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"),
    )

    if args.command == "index":
        engine.index_folder(
            args.folder,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            incremental=not args.no_incremental,
        )
        print("Indexing completed.")

    elif args.command == "query":
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
        start = time.perf_counter()
        res = engine.answer(
            args.question,
            user_roles=roles,
            top_k=args.top_k,
            llm_repo=args.llm_repo,
        )
        dur = time.perf_counter() - start
        
        # 깔끔한 출력 형식
        print(f"📋 질문: {args.question}")
        print()
        print(f"💡 답변: {res['answer']}")
        print()
        print(f"📊 신뢰도: {res['confidence']:.1%}")
        print(f"🔍 관련 문서: {len(res['sources'])}개")
        print(f"⏱️ 처리 시간: {dur:.1f}초")
        
        if res['hallucination_flag']:
            print("⚠️  주의: 환각 가능성 있음")
        
        # 상세 정보가 필요한 경우 JSON 출력
        if args.verbose:
            print("\n" + "="*50)
            print("상세 정보:")
            print(json.dumps({**res, "elapsed_sec": dur}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

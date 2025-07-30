#!/usr/bin/env python3
"""
index_rag_dataset.py
--------------------
RAG (Retrieval-Augmented Generation) 파이프라인 핵심 스크립트

주요 기능:
1. JSONL 데이터셋 로드 (논문 텍스트)
2. 텍스트를 ~1000토큰 청크로 분할 (중첩 포함)
3. Sentence-Transformers로 임베딩 벡터 생성
4. FAISS 인덱스 구축 (코사인 유사도/내적 검색)
5. Gemini-Pro를 통한 RAG 질의응답 CLI 제공

사용법:
  # 의존성 설치
  pip install sentence-transformers faiss-cpu google-generativeai nltk
  python -c "import nltk; nltk.download('punkt')"
  
  # 인덱스 생성 (최초 5-10분 소요)
  export GOOGLE_API_KEY="<YOUR_GEMINI_API_KEY>"
  python index_rag_dataset.py build
  
  # 질의응답
  python index_rag_dataset.py ask "What are the key challenges in AI alignment in 2025?"
"""
import sys, gzip, json, os, time, itertools, pickle
from pathlib import Path
from typing import List

import numpy as np
import faiss
from tqdm import tqdm
# PyTorch/Transformers 충돌 해결을 위한 안전한 임포트
try:
    # CPU 전용 모드로 PyTorch 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_DISABLE_CUDA'] = '1'
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SentenceTransformers import failed: {e}")
    print("📦 Installing compatible versions...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch==2.1.0", "transformers==4.35.0", "sentence-transformers==2.2.2"], check=False)
    try:
        from sentence_transformers import SentenceTransformer
        TRANSFORMERS_AVAILABLE = True
    except:
        TRANSFORMERS_AVAILABLE = False
        print("❌ Failed to load SentenceTransformers. Using fallback embedding.")

import nltk
from nltk.tokenize import sent_tokenize

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# === 파일 경로 및 설정 상수 ===
DATA_FILE = Path("rag_dataset_cv.jsonl.gz")  # 입력 데이터셋 (gzip 압축 JSONL)
INDEX_FILE = Path("rag_faiss.index")              # FAISS 인덱스 저장 파일
META_FILE  = Path("rag_meta.pkl")                 # 청크 텍스트 메타데이터 (pickle)

# === 텍스트 청킹 설정 ===
CHUNK_SIZE_TOK = 1000  # 청크당 대략적인 토큰 수 (4자 ≈ 1토큰으로 추정)
OVERLAP_TOK    = 120   # 청크 간 중첩 토큰 수 (문맥 연속성 보장)

# === 청킹 방법 설정 ===
CHUNKING_METHOD = "semantic"  # "simple", "semantic", "agentic" 중 선택

# === 임베딩 모델 설정 ===
EMB_MODEL_NAME = "BAAI/bge-small-en"  # BGE-Small-EN: 경량화된 다국어 임베딩 모델 (384차원)


def estimate_token_count(text: str) -> int:
    """
    텍스트의 대략적인 토큰 수 추정
    
    간단한 휴리스틱: 4문자 ≈ 1토큰 (영어 기준)
    실제 토크나이저보다 빠르지만 근사치임
    
    Args:
        text: 입력 텍스트
    
    Returns:
        추정 토큰 수
    """
    return max(1, len(text) // 4)  # 최소 1토큰 보장


def chunk_text(text: str) -> List[str]:
    """
    긴 텍스트를 중첩되는 청크로 분할
    
    NLTK의 sent_tokenize로 문장 단위 분할 후,
    지정된 토큰 수에 맞춰 청크를 생성하며
    청크 간 중첩을 통해 문맥 연속성 보장
    
    Args:
        text: 분할할 원본 텍스트
    
    Returns:
        청크 텍스트 리스트 (각 청크는 최대 4000자로 제한)
    """
    # NLTK로 문장 단위 분할
    sentences = sent_tokenize(text)
    chunks = []
    current = []      # 현재 청크에 포함될 문장들
    current_len = 0   # 현재 청크의 토큰 수
    
    for sent in sentences:
        sent_len = estimate_token_count(sent)
        
        # 토큰 수 초과 시 현재 청크 완성하고 새 청크 시작
        if current_len + sent_len > CHUNK_SIZE_TOK:
            if current:
                # 현재 청크를 최대 4000자로 제한하여 저장
                chunks.append(" ".join(current)[:4000])
            
            # 중첩(overlap) 처리: 이전 청크의 마지막 부분을 새 청크 시작으로 복사
            overlap_tokens = OVERLAP_TOK // 4  # 토큰을 문자 수로 근사 변환
            overlap_text = current[-overlap_tokens:] if overlap_tokens < len(current) else current
            current = overlap_text.copy() if isinstance(overlap_text, list) else list(overlap_text)
            current_len = estimate_token_count(" ".join(current))
        
        # 현재 문장을 청크에 추가
        current.append(sent)
        current_len += sent_len
    
    # 마지막 청크 처리
    if current:
        chunks.append(" ".join(current)[:4000])
    
    return chunks


def semantic_chunk_text(text: str, model: SentenceTransformer, similarity_threshold: float = 0.5) -> List[str]:
    """
    의미적 유사도 기반 텍스트 청킹
    
    문장 간 임베딩 유사도를 계산하여 의미적으로 연관된 문장들을 
    같은 청크로 그룹화. 유사도가 임계값 이하로 떨어지면 새 청크 시작.
    
    Args:
        text: 분할할 원본 텍스트
        model: 임베딩 생성용 SentenceTransformer 모델
        similarity_threshold: 청크 분리 임계값 (0-1, 낮을수록 더 세분화)
    
    Returns:
        의미적으로 일관성 있는 청크 텍스트 리스트
    """
    # 문장 단위 분할
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return [text[:4000]]
    
    # 모든 문장의 임베딩 생성 (배치 처리로 효율성 향상)
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    
    chunks = []
    current_chunk = [sentences[0]]  # 첫 번째 문장으로 시작
    current_tokens = estimate_token_count(sentences[0])
    
    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_tokens = estimate_token_count(sent)
        
        # 이전 문장과의 코사인 유사도 계산 (정규화된 벡터의 내적)
        similarity = np.dot(embeddings[i-1], embeddings[i])
        
        # 유사도가 임계값 이상이고 토큰 수가 허용 범위 내이면 현재 청크에 추가
        should_continue = (
            similarity >= similarity_threshold and 
            current_tokens + sent_tokens <= CHUNK_SIZE_TOK
        )
        
        if should_continue:
            current_chunk.append(sent)
            current_tokens += sent_tokens
        else:
            # 현재 청크 완성하고 새 청크 시작
            if current_chunk:
                chunks.append(" ".join(current_chunk)[:4000])
            current_chunk = [sent]
            current_tokens = sent_tokens
    
    # 마지막 청크 처리
    if current_chunk:
        chunks.append(" ".join(current_chunk)[:4000])
    
    return chunks


def agentic_chunk_text(text: str, model_name: str = "models/gemini-2.5-flash") -> List[str]:
    """
    LLM 기반 에이전틱 청킹
    
    Gemini 모델을 사용하여 텍스트의 자연스러운 논리적 단위를 식별하고
    주제 전환점, 섹션 경계, 논증 구조 등을 고려한 지능적 분할 수행.
    
    Args:
        text: 분할할 원본 텍스트
        model_name: 사용할 Gemini 모델명
    
    Returns:
        논리적 일관성을 갖춘 청크 텍스트 리스트
    """
    if genai is None:
        print("⚠️ Agentic chunking requires google-generativeai. Falling back to simple chunking.")
        return chunk_text(text)
    
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("⚠️ GOOGLE_API_KEY not set. Falling back to simple chunking.")
        return chunk_text(text)
    
    # 텍스트가 너무 짧으면 단순 청킹 사용
    if len(text) < 2000:
        return chunk_text(text)
    
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        
        # LLM에게 텍스트 분할 지시하는 프롬프트
        prompt = f"""
다음 학술 논문 텍스트를 의미적으로 일관된 청크들로 분할해주세요.

분할 기준:
1. 주제 전환점 식별
2. 논리적 단위 (서론, 방법론, 결과, 결론 등) 고려  
3. 각 청크는 500-1000 토큰 정도가 적절
4. 문장 중간에서 자르지 말고 완전한 문장 단위로 분할

출력 형식:
각 청크를 "---CHUNK---" 구분자로 분리하여 출력

텍스트:
{text[:8000]}
"""
        
        response = model.generate_content(prompt)
        result_text = response.text
        
        # LLM 응답에서 청크 추출
        if "---CHUNK---" in result_text:
            raw_chunks = result_text.split("---CHUNK---")
            chunks = []
            
            for chunk in raw_chunks:
                cleaned = chunk.strip()
                if len(cleaned) > 100:  # 너무 짧은 청크 제외
                    # 토큰 수 확인하여 너무 크면 재분할
                    if estimate_token_count(cleaned) > CHUNK_SIZE_TOK * 1.5:
                        chunks.extend(chunk_text(cleaned))
                    else:
                        chunks.append(cleaned[:4000])
            
            if chunks:  # LLM이 성공적으로 분할했으면 반환
                return chunks
    
    except Exception as e:
        print(f"⚠️ Agentic chunking failed: {e}. Falling back to simple chunking.")
    
    # LLM 분할 실패 시 기본 청킹으로 폴백
    return chunk_text(text)


def get_chunking_function(method: str):
    """
    청킹 방법에 따른 함수 반환
    
    Args:
        method: "simple", "semantic", "agentic" 중 하나
    
    Returns:
        선택된 청킹 함수
    """
    if method == "semantic":
        return semantic_chunk_text
    elif method == "agentic":
        return agentic_chunk_text
    else:
        return chunk_text  # 기본값: simple chunking


def build_index(chunking_method: str = "semantic", similarity_threshold: float = 0.5):
    """
    FAISS 인덱스 구축 메인 함수
    
    처리 과정:
    1. NLTK 리소스 다운로드 (punkt, punkt_tab)
    2. 데이터셋 로드 및 텍스트 청킹
    3. Sentence-Transformer로 임베딩 생성
    4. FAISS 인덱스 구축 (코사인 유사도용 내적 인덱스)
    5. 인덱스와 메타데이터 저장
    """
    # NLTK 리소스 다운로드 (문장 분할용)
    nltk.download('punkt', quiet=True)
    # NLTK 4.x 호환성: punkt_tab 리소스도 필요할 수 있음
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    print("📖 Loading dataset…")
    
    # 데이터셋 파일 존재 확인
    if not DATA_FILE.exists():
        print(f"❌ {DATA_FILE} not found.")
        sys.exit(1)

    # Sentence-Transformer 모델 로드
    model = SentenceTransformer(EMB_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()  # 임베딩 차원 수 (384)
    
    # 청킹 방법 선택 및 설정
    print(f"🔧 Using {chunking_method} chunking method")
    chunking_func = get_chunking_function(chunking_method)
    
    # FAISS 인덱스 초기화 (내적/코사인 유사도용)
    # IndexFlatIP: 정규화된 벡터에 대해 내적 = 코사인 유사도
    index = faiss.IndexFlatIP(dim)
    metadata: List[str] = []  # 각 임베딩에 대응하는 텍스트 청크 저장

    def batch(iterable, n=64):
        """
        이터러블을 지정된 크기의 배치로 분할하는 제너레이터
        메모리 효율적인 배치 처리를 위함
        """
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    # JSONL 데이터셋을 스트리밍으로 처리 (메모리 절약)
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="parsing"):
            obj = json.loads(line)
            # 다양한 필드명 지원 (full_text, text 등)
            txt = obj.get("full_text") or obj.get("text") or ""
            if not txt:
                continue
            
            # 선택된 방법으로 텍스트를 청크로 분할하여 메타데이터에 추가
            if chunking_method == "semantic":
                # 의미적 청킹: 임베딩 모델과 유사도 임계값 전달
                chunks = chunking_func(txt, model, similarity_threshold=similarity_threshold)
            elif chunking_method == "agentic":
                # 에이전틱 청킹: LLM 모델명 전달
                chunks = chunking_func(txt, model_name="models/gemini-2.5-flash")
            else:
                # 단순 청킹: 텍스트만 전달
                chunks = chunking_func(txt)
            
            for chunk in chunks:
                metadata.append(chunk)
    
    print(f"🧩 Total chunks: {len(metadata):,}")

    # 배치 단위로 임베딩 생성 및 인덱스에 추가
    # 64개씩 처리하여 GPU 메모리 효율성 확보
    for texts in tqdm(batch(metadata, 64), total=len(metadata)//64 + 1, desc="embedding"):
        # 임베딩 생성 후 정규화 (코사인 유사도 계산을 위해)
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # FAISS 인덱스에 추가
        index.add(emb)

    # 인덱스와 메타데이터를 디스크에 저장
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Index saved to {INDEX_FILE}, metadata to {META_FILE}")


def ask(query: str, top_k: int = 5):
    """
    RAG 질의응답 함수
    
    처리 과정:
    1. 질의 텍스트를 임베딩으로 변환
    2. FAISS에서 유사한 청크 top_k개 검색
    3. 검색된 청크들을 컨텍스트로 Gemini에 전달
    4. LLM이 생성한 답변 출력
    
    Args:
        query: 사용자 질문
        top_k: 검색할 상위 청크 수
    """
    # 인덱스 파일 존재 확인
    if not INDEX_FILE.exists():
        print("❌ Index file not found. Run with 'build' first.")
        return
    
    # 저장된 인덱스와 메타데이터 로드
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    
    # 질의 임베딩 생성 (동일한 모델 사용)
    model = SentenceTransformer(EMB_MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True)
    
    # FAISS에서 유사한 청크 검색
    # D: 유사도 점수, I: 인덱스 번호
    D, I = index.search(q_emb, top_k)
    
    # 검색된 청크들을 컨텍스트로 결합
    ctx = "\n\n".join(meta[i] for i in I[0])
    
    # RAG 프롬프트 구성
    prompt = f"Answer the following question using the provided context. If the context does not contain the answer, say 'I don't know.'\n\nContext:\n{ctx}\n\nQuestion: {query}"

    # Gemini API 키 확인
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("ℹ️  GOOGLE_API_KEY not set. Returning only retrieved context.")
        print("\n----- Context -----\n", ctx[:2000])  # 컨텍스트만 출력
        return
    
    # google-generativeai 패키지 확인
    if genai is None:
        print("pip install google-generativeai required.")
        return
    
    # Gemini API 호출
    genai.configure(api_key=key)
    gem = genai.GenerativeModel("models/gemini-2.5-pro")  # 최신 Gemini Pro 모델
    resp = gem.generate_content(prompt)
    
    # 답변 출력
    print("\n----- Answer -----\n")
    print(resp.text)


if __name__ == "__main__":
    import argparse

    # 커맨드라인 인터페이스 설정
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 'build' 서브커맨드: 인덱스 구축
    build_p = sub.add_parser("build")
    build_p.add_argument("--chunking", choices=["simple", "semantic", "agentic"], 
                        default="semantic", help="청킹 방법 선택 (기본값: semantic)")
    build_p.add_argument("--similarity-threshold", type=float, default=0.5,
                        help="의미적 청킹 유사도 임계값 (0.0-1.0, 기본값: 0.5)")
    
    # 'ask' 서브커맨드: 질의응답
    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question", nargs="+")  # 질문 (여러 단어 가능)
    ask_p.add_argument("--top_k", type=int, default=5, help="검색할 상위 청크 수")
    ask_p.add_argument("--show_ctx", action="store_true", help="Gemini 호출 없이 컨텍스트만 출력")

    args = parser.parse_args()

    if args.cmd == "build":
        # 빌드 시 청킹 방법 설정
        build_index(chunking_method=args.chunking, similarity_threshold=args.similarity_threshold)
    else:
        # 질문 문자열 재구성
        q = " ".join(args.question)
        if args.show_ctx:
            # 컨텍스트만 보기 모드: Gemini 키를 제거하여 LLM 호출 건너뛰기
            os.environ.pop("GOOGLE_API_KEY", None)
        ask(q, top_k=args.top_k) 
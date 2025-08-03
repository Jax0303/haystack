# Enterprise RAG with HERB Dataset

> **2025-08-04: 코드가 최신 LangChain 패키지(`langchain-huggingface`, `langchain-chroma`)로 마이그레이션되었습니다. 기존 `langchain_community`/`langchain_core` 대신 새로운 패키지와 import 경로를 사용합니다. 자세한 내용은 커밋 로그 참고!**

기업 특화 Retrieval-Augmented Generation 시스템 with **HERB (Heterogeneous Enterprise RAG Benchmark)** 데이터셋과 **Google Gemini 2.5 Flash 및 기타 LLM** 연동.

## 🚀 프로젝트 개요

본 프로젝트는 **조직 내 RAG 활용**을 위한 기업 특화 기능들을 구현해본 실무 기반 RAG 시스템

### 🏢 핵심 Enterprise 기능
- **🔐 Role-Based Access Control (RBAC)**: 사용자 역할별 문서 접근 제어
- **🎯 Hallucination Detection**: 신뢰도 점수 & 환각 탐지 메커니즘 (80%+ 신뢰도)
- **📝 Source Tracking**: 답변의 출처 문서 추적 및 메타데이터 제공
- **⚡ Performance Monitoring**: 응답 시간, 처리 성능 모니터링
- **🔍 Multi-Artifact Support**: Slack, 문서, 회의록, PR 등 다양한 소스 통합

### 🛠️ 기술 스택
- **LLM**: Google Gemini 2.5 Flash (최신 실험 모델)
- **Vector DB**: ChromaDB (5,487개 문서 인덱싱)
- **Embeddings**: HuggingFace sentence-transformers/all-mpnet-base-v2
- **Framework**: LangChain최신 패키지 + Python 3.12
- **Dataset**: HERB (Heterogeneous Enterprise RAG Benchmark)

## 📊 HERB Dataset 정보

**HERB (Heterogeneous Enterprise RAG Benchmark)**는 Salesforce AI Research에서 개발한 기업용 RAG 벤치마크 데이터셋입니다.

### 데이터셋 구성:
- **총 문서 수**: 5,487개 (본 프로젝트 인덱싱 기준)
- **아티팩트 유형**:
  - 📄 **Documents**: 제품 비전, 요구사항 문서
  - 💬 **Slack Messages**: 팀 내부 커뮤니케이션
  - 🗣️ **Meeting Transcripts**: 회의록 및 대화 내용
  - 🔗 **URLs**: 웹 리소스 및 외부 링크
  - 🛠️ **Pull Requests**: 코드 변경사항 및 개발 논의

### 제품별 데이터:
- **CoachForce**: AI 코칭 플랫폼
- **ConnectForce**: 통합 연결 솔루션
- **ExplainabilityForce**: AI 설명가능성 도구

## 📁 프로젝트 구조

```
├── advanced_rag.py                 #  메인 RAG 엔진 (RBAC, 환각탐지, CLI)
├── index_herb_dataset.py           #  HERB 데이터셋 다운로드 & 인덱싱
├── requirements_enterprise_rag.txt #  의존성 패키지 목록
├── HERB/                          #  HERB GitHub 클론 (자동 다운로드)
├── chroma_db/                     #  벡터 DB 저장소 (5,487개 문서)
└── README.md                      #  본 문서
```

## 🚀 빠른 시작 가이드

### 1. 환경 설정
```bash
# 리포지토리 클론
git clone https://github.com/Jax0303/haystack-2.git
cd haystack-2

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements_enterprise_rag.txt
```

### 2. API 키 설정 (필수)
```bash
# Google Gemini API 키 (https://aistudio.google.com/)
export GOOGLE_API_KEY="your_gemini_api_key_here"

# HuggingFace 토큰 (https://huggingface.co/settings/tokens)
export HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"

# 영구 설정 (권장)
echo 'export GOOGLE_API_KEY="your_key"' >> ~/.bashrc
echo 'export HUGGINGFACEHUB_API_TOKEN="hf_your_token"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 데이터셋 인덱싱 (최초 1회)
```bash
# HERB 전체 데이터셋 인덱싱 (약 3-5분 소요)
python index_herb_dataset.py

# 빠른 테스트용 (일부 파일만)
python index_herb_dataset.py --max_files 50
```

### 4. RAG 쿼리 실행
```bash
# 기본 질의
python advanced_rag.py query "What features did customers complain about the most?" 2>/dev/null

# 옵션 사용
python advanced_rag.py query "What are the main security concerns?" --top_k 6 --verbose 2>/dev/null

# 역할 기반 접근 제어
python advanced_rag.py query "Sensitive information" --roles "admin,manager" 2>/dev/null
```

## 💻 사용법 상세

### 🎯 편리한 별칭 설정 (권장)
```bash
# RAG 명령어 별칭 추가
echo "alias rag='cd $(pwd) && source .venv/bin/activate && python advanced_rag.py'" >> ~/.bashrc
source ~/.bashrc

# 이제 간단하게 사용 가능
rag query "How can we improve user experience?"
rag query "What are the biggest technical challenges?" --top_k 10
```

### 📋 CLI 옵션들
```bash
# 기본 쿼리
rag query "질문 내용"

# 상세 정보 출력 (JSON 포맷)
rag query "질문" --verbose

# 검색할 문서 수 조정
rag query "질문" --top_k 10

# 사용자 역할 지정 (접근 제어)
rag query "질문" --roles "admin,user,manager"

# 다른 LLM 모델 사용 (폴백용)
rag query "질문" --llm_repo "google/flan-t5-large"
```

### 📊 출력 형식 예시
```
📋 질문: What are the main security concerns?

💡 답변: Based on the provided context, the main security concerns are:
   • Data privacy: Ensuring compliance with regulations like GDPR and CCPA.
   • Data security: Protecting user data through encryption and secure access controls.
   • Vulnerabilities: Regular security assessments and penetration testing.

📊 신뢰도: 88.2%
🔍 관련 문서: 4개
⏱️ 처리 시간: 2.1초
```

## 🔧 고급 설정

### 환경 변수
```bash
# 컬렉션 이름 변경
export RAG_COLLECTION="custom_collection"

# 저장 디렉토리 변경
export RAG_PERSIST_DIR="./custom_db"

# 임베딩 모델 변경
export RAG_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### 프로그래밍 방식 사용
```python
from advanced_rag import RagEngine

# RAG 엔진 초기화
engine = RagEngine(collection_name='herb_collection')

# 쿼리 실행
result = engine.answer(
    query="What integration challenges exist?",
    user_roles=["admin", "developer"],
    top_k=5
)

print(f"답변: {result['answer']}")
print(f"신뢰도: {result['confidence']:.1%}")
print(f"소스: {len(result['sources'])}개 문서")
```

## 🏗️ 아키텍처 및 구현 특징

### RAG 파이프라인
1. **Document Loading**: HERB JSON 파일 파싱 및 텍스트 추출
2. **Text Splitting**: 1000자 단위 청크 분할 (100자 오버랩)
3. **Embedding**: HuggingFace all-mpnet-base-v2 모델 사용
4. **Vector Storage**: ChromaDB 영구 저장
5. **Retrieval**: 코사인 유사도 기반 상위 K개 문서 검색
6. **Generation**: Gemini 2.0 Flash로 컨텍스트 기반 답변 생성
7. **Post-processing**: 환각 점수 계산 및 신뢰도 평가

### 기업용 보안 기능
- **Access Control Lists (ACL)**: 메타데이터 기반 문서 접근 제어
- **Confidence Scoring**: 답변-컨텍스트 유사도 기반 신뢰도 측정
- **Source Attribution**: 모든 답변에 대한 원본 문서 추적
- **Audit Trail**: 쿼리 로그 및 성능 메트릭 수집

## 🎓 교육적 가치 및 기업 RAG 고려사항

### 구현하면서 발견한 조직 내 RAG 핵심 요구사항:

1. **데이터 거버넌스**
   - 다양한 소스(Slack, 문서, 회의록)의 통합적 관리
   - 메타데이터 기반 분류 및 접근 제어

2. **신뢰성 및 검증가능성**
   - 환각 탐지 및 신뢰도 점수 제공
   - 답변의 출처 추적 및 원본 확인 가능

3. **보안 및 컴플라이언스**
   - 역할 기반 접근 제어 (RBAC)
   - 민감 정보 필터링 및 감사 로그

4. **확장성 및 성능**
   - 배치 처리를 통한 대용량 데이터 인덱싱
   - 실시간 응답 성능 모니터링

5. **사용자 경험**
   - 직관적인 CLI 인터페이스
   - 명확한 신뢰도 및 소스 정보 제공

## 🔧 문제 해결

### 일반적인 문제들
```bash
# 1. 환경변수 설정 확인
echo $GOOGLE_API_KEY
echo $HUGGINGFACEHUB_API_TOKEN

# 2. 의존성 재설치
pip install --upgrade -r requirements_enterprise_rag.txt

# 3. 데이터베이스 재구축
rm -rf chroma_db/
python index_herb_dataset.py

# 4. 경고 메시지 숨기기
python advanced_rag.py query "질문" 2>/dev/null
```

### ChromaDB 배치 크기 오류
```bash
# 대용량 데이터셋 처리 시 자동으로 5000개씩 배치 처리됨
# 별도 설정 불필요
```

## 📚 참고 자료

- **HERB Dataset**: [Stanford CRFM](https://crfm.stanford.edu/helm/latest/)
- **Google Gemini API**: [AI Studio](https://aistudio.google.com/)
- **LangChain Documentation**: [python.langchain.com](https://python.langchain.com/)
- **ChromaDB**: [docs.trychroma.com](https://docs.trychroma.com/)

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

## 🔄 변경 이력

### v3.0 
- ✅ Google Gemini 2.0 Flash 연동
- ✅ HERB 데이터셋 완전 인덱싱 (5,487개 문서)
- ✅ 깔끔한 출력 형식 및 사용자 경험 개선
- ✅ 환경변수 영구 설정 및 별칭 지원
- ✅ 배치 처리 ChromaDB 인덱싱 안정화

### v2.0 
- ✅ HERB 데이터셋 통합
- ✅ 기업용 보안 기능 (RBAC, 환각 탐지)
- ✅ CLI 인터페이스 구현

### v1.0 
- ✅ 기본 RAG 파이프라인 구현
- ✅ HuggingFace + ChromaDB 연동

---

© 2025 Jax0303 

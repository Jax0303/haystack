#!/usr/bin/env python3
"""
get_cv_papers_arxiv.py
======================
다운로드 대상
  • arXiv 카테고리: cs.CV (Computer Vision)
  • 제출 날짜: 2024-01-01 – 2025-12-31
  • PDF 400–600개 ≈ 2 GB (원본), 해제 텍스트 ≈ 3–4 GB

결과
  ./cv_papers_2024_25.jsonl.gz  (각 줄: {title, published, arxiv_id, text})

실행 전 의존 패키지
  pip install requests tqdm pymupdf

사용
  python get_cv_papers_arxiv.py --max_pdf 500
"""
import argparse, json, gzip, requests, xml.etree.ElementTree as ET, concurrent.futures, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF

# arXiv API 엔드포인트
aapi = "https://export.arxiv.org/api/query"
# PDF 다운로드 저장 폴더
PDF_DIR = Path("cv_pdfs")
# 파싱 결과 저장 파일 (gzip 압축된 JSONL)
OUT_FILE = Path("cv_papers_2024_25.jsonl.gz")
# PDF 저장 폴더가 없으면 생성
PDF_DIR.mkdir(exist_ok=True)

def arxiv_search(start=0, max_results=100):
    """
    arXiv API를 통해 Computer Vision 논문 검색
    
    Args:
        start: 검색 시작 인덱스 (페이지네이션용)
        max_results: 한 번에 가져올 논문 수
    
    Returns:
        XML 엔트리 리스트 (논문 메타데이터)
    """
    query = {
        'search_query': 'cat:cs.CV AND submittedDate:[2024-01-01 TO 2025-12-31]',  # CV 카테고리, 2024-25년
        'start': str(start),
        'max_results': str(max_results),
        'sortBy': 'submittedDate',  # 제출일자 순 정렬
        'sortOrder': 'descending'   # 최신순
    }
    # HTTP GET 요청으로 arXiv API 호출
    resp = requests.get(aapi, params=query, timeout=60)
    resp.raise_for_status()  # HTTP 에러 시 예외 발생
    # XML 응답 파싱
    root = ET.fromstring(resp.content)
    # 각 논문 엔트리 반환
    return root.findall('{http://www.w3.org/2005/Atom}entry')

def pdf_url_from_id(id_url: str) -> str:
    """
    arXiv ID URL에서 PDF 다운로드 URL 생성
    
    Args:
        id_url: arXiv 논문 ID URL (예: http://arxiv.org/abs/2401.12345)
    
    Returns:
        PDF 직접 다운로드 URL
    """
    arxiv_id = id_url.split('/')[-1]  # URL에서 논문 ID 추출
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def download_pdf(task):
    """
    PDF 파일 다운로드 (멀티스레딩용 함수)
    
    Args:
        task: (url, dest_path) 튜플
    
    Returns:
        성공 시 저장 경로, 실패 시 None
    """
    url, dest = task
    # 이미 파일이 존재하면 건너뛰기 (이어받기 기능)
    if dest.exists():
        return dest
    try:
        # 스트리밍 다운로드로 메모리 절약
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                return None
            # 8KB 청크 단위로 파일 저장
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return dest
    except Exception:
        # 다운로드 실패 시 None 반환 (오류 무시)
        return None

def parse_pdf(pdf_path: Path, max_chars=18000):
    """
    PyMuPDF를 사용한 PDF 텍스트 추출
    
    Args:
        pdf_path: PDF 파일 경로
        max_chars: 최대 추출 문자 수 (메모리 절약)
    
    Returns:
        추출된 텍스트 문자열
    """
    try:
        # PyMuPDF로 PDF 열기
        doc = fitz.open(pdf_path)
        # 앞쪽 최대 15페이지만 텍스트 추출 (논문 핵심 내용)
        text = "".join(page.get_text() for page in doc[:15])
        # 지정된 문자 수로 자르기
        return text[:max_chars]
    except Exception:
        # 파싱 실패 시 빈 문자열 반환
        return ""

def main(max_pdf: int):
    """
    메인 실행 함수: PDF 다운로드 → 파싱 → JSONL 저장
    
    Args:
        max_pdf: 처리할 최대 PDF 수
    """
    collected = 0  # 수집된 논문 수 카운터
    start = 0      # arXiv API 페이지네이션 시작점
    
    # gzip으로 압축하여 JSONL 파일에 순차 저장 (메모리 효율적)
    with gzip.open(OUT_FILE, 'at', encoding='utf-8') as fout:
        while collected < max_pdf:
            # arXiv에서 논문 메타데이터 100개씩 가져오기
            entries = arxiv_search(start=start, max_results=100)
            if not entries:  # 더 이상 논문이 없으면 종료
                break
            
            # 다운로드 작업 리스트와 메타데이터 준비
            tasks = []
            meta_list = []
            
            # 각 논문 엔트리 처리
            for e in entries:
                if collected + len(tasks) >= max_pdf:  # 목표 수량 도달 시 중단
                    break
                
                # XML에서 논문 정보 추출
                arxiv_id = e.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                title = e.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
                published = e.find('{http://www.w3.org/2005/Atom}published').text[:10]  # YYYY-MM-DD 형식
                
                # PDF URL과 저장 경로 생성
                pdf_url = pdf_url_from_id(e.find('{http://www.w3.org/2005/Atom}id').text)
                pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
                
                # 다운로드 작업과 메타데이터 추가
                tasks.append((pdf_url, pdf_path))
                meta_list.append((arxiv_id, title, published, pdf_path))

            # 멀티스레딩으로 PDF 병렬 다운로드 (최대 6개 동시)
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as dl_pool:
                list(tqdm(dl_pool.map(download_pdf, tasks), total=len(tasks), desc="downloading", leave=False))

            # 멀티프로세싱으로 PDF 병렬 파싱 (CPU 코어 수 - 1)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, mp.cpu_count()-1)) as pp:
                parse_results = list(tqdm(pp.map(parse_pdf, [m[3] for m in meta_list]), total=len(meta_list), desc="parsing", leave=False))

            # 파싱 결과를 JSONL 형태로 파일에 저장
            for meta, txt in zip(meta_list, parse_results):
                if collected >= max_pdf:  # 목표 수량 도달 시 중단
                    break
                if len(txt) < 500:  # 너무 짧은 텍스트는 제외 (파싱 실패 가능성)
                    continue
                
                arxiv_id, title, published, _ = meta
                # JSON 레코드 생성
                record = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'published': published,
                    'text': txt
                }
                # JSONL 파일에 한 줄씩 저장 (한국어 유니코드 보존)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected += 1
            
            start += 100  # 다음 페이지로 이동
    
    print(f"✅ Saved {collected} papers to {OUT_FILE}")

if __name__ == "__main__":
    # 커맨드라인 인자 파싱
    p = argparse.ArgumentParser()
    p.add_argument('--max_pdf', type=int, default=500, help='number of PDFs to download')
    args = p.parse_args()
    main(args.max_pdf) 
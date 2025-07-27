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
  pip install requests tqdm pdfminer.six

사용
  python get_cv_papers_arxiv.py --max_pdf 500
"""
import argparse, json, gzip, requests, xml.etree.ElementTree as ET, concurrent.futures, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF

aapi = "https://export.arxiv.org/api/query"
PDF_DIR = Path("cv_pdfs")
OUT_FILE = Path("cv_papers_2024_25.jsonl.gz")
PDF_DIR.mkdir(exist_ok=True)

def arxiv_search(start=0, max_results=100):
    query = {
        'search_query': 'cat:cs.CV AND submittedDate:[2024-01-01 TO 2025-12-31]',
        'start': str(start),
        'max_results': str(max_results),
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    resp = requests.get(aapi, params=query, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    return root.findall('{http://www.w3.org/2005/Atom}entry')

def pdf_url_from_id(id_url: str) -> str:
    arxiv_id = id_url.split('/')[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def download_pdf(task):
    url, dest = task
    if dest.exists():
        return dest
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                return None
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return dest
    except Exception:
        return None

def parse_pdf(pdf_path: Path, max_chars=18000):
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc[:15])
        return text[:max_chars]
    except Exception:
        return ""

def main(max_pdf: int):
    collected = 0
    start = 0
    with gzip.open(OUT_FILE, 'at', encoding='utf-8') as fout:
        while collected < max_pdf:
            entries = arxiv_search(start=start, max_results=100)
            if not entries:
                break
            tasks = []
            meta_list = []
            for e in entries:
                if collected + len(tasks) >= max_pdf:
                    break
                arxiv_id = e.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                title = e.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
                published = e.find('{http://www.w3.org/2005/Atom}published').text[:10]
                pdf_url = pdf_url_from_id(e.find('{http://www.w3.org/2005/Atom}id').text)
                pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
                tasks.append((pdf_url, pdf_path))
                meta_list.append((arxiv_id, title, published, pdf_path))

            # parallel download
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as dl_pool:
                list(tqdm(dl_pool.map(download_pdf, tasks), total=len(tasks), desc="downloading", leave=False))

            # parallel parse
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, mp.cpu_count()-1)) as pp:
                parse_results = list(tqdm(pp.map(parse_pdf, [m[3] for m in meta_list]), total=len(meta_list), desc="parsing", leave=False))

            for meta, txt in zip(meta_list, parse_results):
                if collected >= max_pdf:
                    break
                if len(txt) < 500:
                    continue
                arxiv_id, title, published, _ = meta
                record = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'published': published,
                    'text': txt
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected += 1
            start += 100
    print(f"✅ Saved {collected} papers to {OUT_FILE}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--max_pdf', type=int, default=500, help='number of PDFs to download')
    args = p.parse_args()
    main(args.max_pdf) 
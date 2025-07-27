#!/usr/bin/env python3
"""
Pipeline: S2ORC → PDF → text  (≈30-minute parsing dataset)

Steps
1. Get latest S2ORC release ID from Semantic Scholar API.
2. Download shard_0 – shard_3 tar.gz (≈2.6 GB) into ./s2orc_raw
3. Extract PDF files to ./s2orc_pdf
4. Parse each PDF with pdfminer.six → text (cap 20 000 chars)
5. Keep docs mentioning 2024 or 2025 to maximise freshness.
6. Save into ./s2orc_2024_25_fulltext.jsonl.gz

Before running, export your API key:
  export S2_API_KEY="<YOUR_KEY>"

And install deps once:
  pip install requests tqdm pdfminer.six
"""
import os, re, json, tarfile, gzip, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List

import requests
from tqdm import tqdm
from pdfminer.high_level import extract_text

API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ 환경 변수 S2_API_KEY 가 설정되지 않았습니다. Semantic Scholar API 키를 export 해주세요.")

RAW_DIR   = Path("s2orc_raw")
PDF_DIR   = Path("s2orc_pdf")
OUT_FILE  = Path("s2orc_2024_25_fulltext.jsonl.gz")
TARGET_SHARDS = [0, 1, 2, 3]  # shard indices to download (총 4개)
MAX_TXT_LEN   = 20000          # 각 논문 텍스트 최대 길이

RAW_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

HEADERS = {"x-api-key": API_KEY}
BASE = "https://api.semanticscholar.org/datasets/v1"


def get_latest_release() -> str:
    url = f"{BASE}/release/latest"
    return requests.get(url, headers=HEADERS, timeout=30).json()["release_id"]


def list_s2orc_files(release_id: str) -> List[str]:
    url = f"{BASE}/release/{release_id}/dataset/s2orc/"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    return data["files"]


def download_file(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def step1_download_shards():
    print("📥 1) S2ORC shard 다운로드 중...")
    release = get_latest_release()
    print(f"   최신 release: {release}")
    files = list_s2orc_files(release)
    pattern = re.compile(r"shard_(\d+)\.tar\.gz$")
    for url in files:
        m = pattern.search(url)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in TARGET_SHARDS:
            dest = RAW_DIR / f"shard_{idx}.tar.gz"
            if dest.exists():
                print(f"   ✅ {dest.name} 이미 존재")
                continue
            download_file(url, dest)
    print("   ✔ shard 다운로드 완료\n")


def step2_extract_pdfs():
    print("📂 2) PDF 추출 중...")
    tars = list(RAW_DIR.glob("shard_*.tar.gz"))
    for tar_path in tars:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".pdf")]
            for m in tqdm(members, desc=f"extract {tar_path.name}"):
                out_path = PDF_DIR / Path(m.name).name
                if out_path.exists():
                    continue
                try:
                    f = tar.extractfile(m)
                    if f:
                        with open(out_path, "wb") as out_f:
                            out_f.write(f.read())
                except Exception:
                    continue
    print("   ✔ PDF 추출 완료\n")


def parse_one(pdf_path: Path):
    try:
        text = extract_text(pdf_path, maxpages=20)  # 제한: 첫 20p 정도만
        if not text:
            return None
        if "2024" not in text and "2025" not in text:
            return None
        return {
            "file": pdf_path.name,
            "text": text[:MAX_TXT_LEN]
        }
    except Exception:
        return None


def step3_parse_pdfs():
    print("📝 3) PDF → 텍스트 변환 중 (병렬)...")
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"   대상 PDF: {len(pdf_files):,} 개")
    n_proc = max(1, cpu_count() - 1)
    with Pool(n_proc) as pool, gzip.open(OUT_FILE, "wt", encoding="utf-8") as fout:
        for result in tqdm(pool.imap_unordered(parse_one, pdf_files, chunksize=10), total=len(pdf_files)):
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("   ✔ 텍스트 추출 & 저장 완료\n")


def main():
    t0 = time.time()
    step1_download_shards()
    step2_extract_pdfs()
    step3_parse_pdfs()
    elapsed = (time.time() - t0) / 60
    size_mb = OUT_FILE.stat().st_size / (1024**2)
    print(f"🏁 파이프라인 완료! 경과 {elapsed:.1f} 분, 결과 {size_mb:.1f} MB → {OUT_FILE}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# get_recent_papers.py
import requests
import json
import gzip
import time
from datetime import datetime
import os

# Semantic Scholar API 설정
API_BASE = "https://api.semanticscholar.org/graph/v1"
PAPERS_PER_BATCH = 100  # API 제한에 맞춰 조정
TOTAL_TARGET = 5000     # 목표 논문 수 (400MB 정도)
OUT_PATH = "./recent_papers_2024_25.jsonl.gz"

# 필요한 필드
FIELDS = [
    "paperId", "title", "abstract", "year", "publicationDate", 
    "venue", "authors", "fieldsOfStudy", "citationCount"
]

def get_papers_batch(offset=0, limit=100):
    """최신 논문을 배치로 가져오기"""
    url = f"{API_BASE}/paper/search"
    params = {
        "query": "machine learning OR artificial intelligence OR deep learning",
        "year": "2024-2025",  # 2024-2025년만
        "fields": ",".join(FIELDS),
        "offset": offset,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ API 오류 (offset={offset}): {e}")
        return None

def main():
    print("🔍 Semantic Scholar API로 2024-2025 논문 수집 중...")
    print(f"📁 저장 경로: {OUT_PATH}")
    
    papers_collected = 0
    offset = 0
    
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f_out:
        while papers_collected < TOTAL_TARGET:
            print(f"📊 배치 요청 중... (offset={offset}, 수집된 논문={papers_collected})")
            
            result = get_papers_batch(offset, PAPERS_PER_BATCH)
            if not result or "data" not in result:
                print("❌ 더 이상 데이터가 없거나 API 오류")
                break
            
            papers = result["data"]
            if not papers:
                print("📝 검색 결과가 없습니다")
                break
                
            for paper in papers:
                # 2024-2025년 논문만 확실히 필터링
                pub_date = paper.get("publicationDate", "")
                year = paper.get("year", 0)
                
                if year >= 2024 or (pub_date and pub_date.startswith(("2024", "2025"))):
                    # 필요한 데이터만 저장
                    clean_paper = {
                        "paper_id": paper.get("paperId", ""),
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "year": year,
                        "publication_date": pub_date,
                        "venue": paper.get("venue", ""),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])],
                        "fields": paper.get("fieldsOfStudy", []),
                        "citations": paper.get("citationCount", 0)
                    }
                    
                    f_out.write(json.dumps(clean_paper, ensure_ascii=False) + "\n")
                    papers_collected += 1
                    
                    if papers_collected % 100 == 0:
                        size_mb = os.path.getsize(OUT_PATH) / (1024**2)
                        print(f"  ✅ {papers_collected}편 수집 완료, {size_mb:.1f} MB")
            
            offset += PAPERS_PER_BATCH
            
            # API 제한 준수 (1초 대기)
            time.sleep(1)
            
            # 총 논문 수가 목표에 도달하면 종료
            if papers_collected >= TOTAL_TARGET:
                break
    
    size_mb = os.path.getsize(OUT_PATH) / (1024**2)
    print(f"\n🎉 수집 완료!")
    print(f"📊 총 {papers_collected}편의 논문을 {size_mb:.1f} MB로 저장")
    print(f"💡 평균 논문당 크기: {size_mb*1024/papers_collected:.1f} KB")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# get_arxiv_papers.py
import requests
import xml.etree.ElementTree as ET
import json
import gzip
import time
import re
from datetime import datetime, date
import os

# arXiv API 설정
ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 100
TARGET_SIZE_MB = 400
OUT_PATH = "./arxiv_papers_2024_25.jsonl.gz"

# 관심 카테고리 (AI/ML 관련)
CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", 
    "stat.ML", "cs.IR", "cs.RO", "cs.HC"
]

def parse_arxiv_entry(entry):
    """arXiv XML entry를 파싱해서 dict로 변환"""
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    # 기본 정보 추출
    paper = {}
    paper["arxiv_id"] = entry.find('atom:id', ns).text.split('/')[-1]
    paper["title"] = entry.find('atom:title', ns).text.strip()
    paper["abstract"] = entry.find('atom:summary', ns).text.strip()
    
    # 날짜 추출
    published = entry.find('atom:published', ns).text
    paper["published_date"] = published[:10]  # YYYY-MM-DD
    
    # 저자 추출
    authors = []
    for author in entry.findall('atom:author', ns):
        name = author.find('atom:name', ns)
        if name is not None:
            authors.append(name.text)
    paper["authors"] = authors
    
    # 카테고리 추출
    categories = []
    for cat in entry.findall('atom:category', ns):
        term = cat.get('term')
        if term:
            categories.append(term)
    paper["categories"] = categories
    
    return paper

def get_papers_by_category(category, start_date="2024-01-01", max_results=1000):
    """특정 카테고리에서 논문 수집"""
    papers = []
    start = 0
    
    while len(papers) < max_results:
        params = {
            'search_query': f'cat:{category} AND submittedDate:[{start_date} TO 2025-12-31]',
            'start': start,
            'max_results': min(BATCH_SIZE, max_results - len(papers)),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        print(f"  📄 {category}: 요청 중... (start={start})")
        
        try:
            response = requests.get(ARXIV_API, params=params, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            if not entries:
                print(f"  ✅ {category}: 더 이상 결과 없음")
                break
                
            batch_papers = []
            for entry in entries:
                paper = parse_arxiv_entry(entry)
                # 2024-2025년만 필터링
                if paper["published_date"].startswith(("2024", "2025")):
                    batch_papers.append(paper)
            
            papers.extend(batch_papers)
            print(f"  📊 {category}: {len(batch_papers)}편 추가 (총 {len(papers)}편)")
            
            start += BATCH_SIZE
            time.sleep(1)  # API 제한 준수
            
        except Exception as e:
            print(f"  ❌ {category} 오류: {e}")
            break
    
    return papers

def main():
    print("📚 arXiv에서 2024-2025 AI/ML 논문 수집 중...")
    print(f"🎯 목표: {TARGET_SIZE_MB} MB")
    print(f"📁 저장: {OUT_PATH}")
    
    all_papers = []
    
    for category in CATEGORIES:
        print(f"\n🔍 카테고리: {category}")
        papers = get_papers_by_category(category, max_results=500)
        all_papers.extend(papers)
        
        # 중간 체크
        if len(all_papers) > 5000:  # 너무 많으면 중단
            print(f"⚠️  충분한 데이터 수집됨 ({len(all_papers)}편)")
            break
    
    # 중복 제거 (arxiv_id 기준)
    unique_papers = {}
    for paper in all_papers:
        unique_papers[paper["arxiv_id"]] = paper
    
    final_papers = list(unique_papers.values())
    print(f"\n🧹 중복 제거 후: {len(final_papers)}편")
    
    # 날짜순 정렬 (최신순)
    final_papers.sort(key=lambda x: x["published_date"], reverse=True)
    
    # gzip으로 저장
    print(f"💾 {OUT_PATH}에 저장 중...")
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f_out:
        for i, paper in enumerate(final_papers):
            f_out.write(json.dumps(paper, ensure_ascii=False) + "\n")
            
            if (i + 1) % 500 == 0:
                size_mb = os.path.getsize(OUT_PATH) / (1024**2)
                print(f"  📊 {i+1}편 저장 완료, {size_mb:.1f} MB")
    
    size_mb = os.path.getsize(OUT_PATH) / (1024**2)
    print(f"\n🎉 수집 완료!")
    print(f"📊 총 {len(final_papers)}편의 논문을 {size_mb:.1f} MB로 저장")
    print(f"💡 평균 논문당 크기: {size_mb*1024/len(final_papers):.1f} KB")
    
    # 연도별 분포 출력
    year_count = {}
    for paper in final_papers:
        year = paper["published_date"][:4]
        year_count[year] = year_count.get(year, 0) + 1
    
    print(f"📅 연도별 분포: {year_count}")

if __name__ == "__main__":
    main() 
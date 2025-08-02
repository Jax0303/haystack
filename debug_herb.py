#!/usr/bin/env python3
"""HERB JSON 구조 디버깅 스크립트"""
import json
from pathlib import Path

# 첫 번째 제품 파일 확인
herb_file = Path("HERB/data/products/ActionGenie.json")
if not herb_file.exists():
    print("❌ HERB/data/products/ActionGenie.json 파일이 없습니다.")
    exit(1)

data = json.loads(herb_file.read_text("utf-8"))
print("=== JSON 최상위 키 ===")
print(list(data.keys()))

print("\n=== artifacts 개수 ===")
artifacts = data.get("artifacts", [])
print(f"총 {len(artifacts)}개")

if artifacts:
    print("\n=== 첫 번째 artifact 키 ===")
    first_art = artifacts[0]
    print(list(first_art.keys()))
    
    print("\n=== artifact_type ===")
    print(first_art.get("artifact_type"))
    
    print("\n=== 텍스트 내용 찾기 ===")
    for key in ["content", "doc_content", "text", "body", "messages", "paragraphs"]:
        if key in first_art:
            val = first_art[key]
            if isinstance(val, str) and len(val) > 50:
                print(f"✅ '{key}' 키에 텍스트 발견: {val[:100]}...")
            elif isinstance(val, list) and val:
                print(f"✅ '{key}' 키에 리스트 발견: {len(val)}개 항목")
                if isinstance(val[0], str):
                    print(f"   첫 번째 항목: {val[0][:80]}...")
    
    print("\n=== 전체 artifact 샘플 (처음 500자) ===")
    print(str(first_art)[:500])
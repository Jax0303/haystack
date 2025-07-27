# make_peS2o_2024_25.py
from datasets import load_dataset
import gzip, json, os, tqdm

CACHE_DIR = "./hf_cache"                 # Hugging Face 캐시
OUT_PATH  = "./peS2o_2024_25.jsonl.gz"  # 최종 저장 파일

print("📚 Loading peS2o dataset in streaming mode...")

# ------------ 1) 스트리밍 로드 ------------
ds = load_dataset(
    "allenai/peS2o", "v2",
    split="train",
    streaming=True,
    cache_dir=CACHE_DIR
)

# ------------ 2) 연도 필터 정의 ------------
def is_2024_25(ex):
    date = ex.get("created", "")         # YYYY-MM-DD
    return date.startswith("2024") or date.startswith("2025")

filtered = ds.filter(is_2024_25)

print(f"🔍 Filtering for 2024-2025 papers...")
print(f"💾 Saving to {OUT_PATH}")

# ------------ 3) gzip으로 저장 ------------
count = 0
with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f_out:
    for ex in tqdm.tqdm(filtered, desc="Processing papers"):
        # 필요한 필드만 저장 (용량 절약)
        paper = {
            "paper_id": ex.get("paper_id", ""),
            "title": ex.get("title", ""),
            "abstract": ex.get("abstract", ""),
            "body_text": ex.get("body_text", ""),
            "created": ex.get("created", ""),
            "venue": ex.get("venue", ""),
            "authors": ex.get("authors", [])
        }
        f_out.write(json.dumps(paper, ensure_ascii=False) + "\n")
        count += 1
        
        # 진행상황 중간 체크
        if count % 1000 == 0:
            size_mb = os.path.getsize(OUT_PATH) / (1024**2)
            print(f"  📊 {count} papers processed, {size_mb:.1f} MB so far")

size_mb = os.path.getsize(OUT_PATH) / (1024**2)
print(f"\n✅ Complete! Saved {count} papers ({size_mb:.1f} MB) to {OUT_PATH}")
print(f"📈 Average paper size: {size_mb*1024/count:.1f} KB per paper") 
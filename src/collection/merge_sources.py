"""
merge_sources.py
=================
Gộp toàn bộ file JSON danh mục đã crawl từ 3 sàn (Shopee, Lazada, Tiki)
thành 1 file raw duy nhất: data/raw/merged_raw_data.json

Đây là bước cuối của COLLECTION DATA, chuẩn bị input cho bước CLEANING.
Lưu ý: đây KHÔNG phải là bước Integration trong sơ đồ 5 bước
(Cleaning -> Integration -> Transformation -> Normalization -> Encoding).
Integration thực sự (đồng bộ schema 3 platform về 1 chuẩn chung) được
thực hiện ở src/integration/integrate_data.py, SAU bước Cleaning.

Chạy: python -m src.collection.merge_sources
"""

import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def merge_raw_sources() -> list:
    """Đọc toàn bộ file JSON trong data/raw/<platform>/categories/, gộp thành 1 list."""
    print("=" * 60)
    print("GỘP DỮ LIỆU THÔ TỪ 3 SÀN (Shopee, Lazada, Tiki)")
    print("=" * 60)

    pattern = os.path.join(config.RAW_DIR, "**", "categories", "*.json")
    source_files = sorted(glob.glob(pattern, recursive=True))

    if not source_files:
        raise FileNotFoundError(
            f"Không tìm thấy file JSON nào trong {config.RAW_DIR}/<platform>/categories/. "
            "Hãy chạy crawler trước (crawl_shopee.py / crawl_lazada.py / crawl_tiki.py)."
        )

    merged = []
    platform_counts: dict[str, int] = {}

    for fpath in source_files:
        rel_path = os.path.relpath(fpath, config.RAW_DIR)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                merged.extend(data)
                platform = data[0].get("platform", "unknown")
                platform_counts[platform] = platform_counts.get(platform, 0) + len(data)
                print(f"  + {rel_path:<70} {len(data):>6,} records")
            else:
                print(f"  ~ {rel_path} (rỗng, bỏ qua)")
        except Exception as e:
            print(f"  ! {rel_path} - lỗi đọc: {e}")

    print("-" * 60)
    print("Thống kê theo platform:")
    for platform, count in sorted(platform_counts.items()):
        print(f"  {platform}: {count:,} records")
    print(f"\nTổng cộng: {len(merged):,} records")

    os.makedirs(os.path.dirname(config.RAW_MERGED_FILE), exist_ok=True)
    with open(config.RAW_MERGED_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu: {config.RAW_MERGED_FILE}")
    return merged


if __name__ == "__main__":
    config.ensure_all_dirs()
    merge_raw_sources()
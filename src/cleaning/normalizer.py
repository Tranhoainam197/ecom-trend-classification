"""
normalizer.py
==============
Chuẩn hóa tên cột (schema) từ 3 platform (Shopee, Lazada, Tiki) khác nhau
về một schema chung duy nhất, để bước Cleaning có thể xử lý đồng nhất.

Lưu ý: Đây là bước "chuẩn hóa schema" nội tại của Cleaning, KHÁC với bước
INTEGRATION trong sơ đồ 5 bước. Integration (ở module riêng) thực hiện việc
kiểm tra và hợp nhất tính toàn vẹn dữ liệu giữa các nguồn SAU khi mỗi nguồn
đã được làm sạch riêng lẻ.
"""

import pandas as pd

# Cả 3 platform đều dùng cùng tên field gốc (do crawler đã được viết thống
# nhất ở bước Collection) nên mapping giống nhau. Vẫn khai báo riêng theo
# platform để dễ mở rộng nếu sau này thêm sàn mới có field khác.
PLATFORM_FIELD_MAPPING = {
    "shopee": {
        "current_price": "price",
        "original_price": "original_price",
        "discount_rate": "discount_rate",
        "rating_average": "rating_average",
        "num_reviews": "review_count",
        "quantity_sold": "quantity_sold_value",
        "quantity_sold_text": "quantity_sold_text",
        "brand": "brand",
        "seller_location": "location",
    },
    "lazada": {
        "current_price": "price",
        "original_price": "original_price",
        "discount_rate": "discount_rate",
        "rating_average": "rating_average",
        "num_reviews": "review_count",
        "quantity_sold": "quantity_sold_value",
        "quantity_sold_text": "quantity_sold_text",
        "brand": "brand",
        "seller_location": "location",
    },
    "tiki": {
        "current_price": "price",
        "original_price": "original_price",
        "discount_rate": "discount_rate",
        "rating_average": "rating_average",
        "num_reviews": "review_count",
        "quantity_sold": "quantity_sold_value",
        "quantity_sold_text": "quantity_sold_text",
        "brand": "brand",
        "seller_location": "location",
    },
}

# Schema chuẩn dùng chung cho toàn bộ pipeline sau bước này
STANDARD_SCHEMA = [
    "crawl_date", "platform", "category", "id", "product_name",
    "current_price", "original_price", "discount_rate",
    "rating_average", "num_reviews", "quantity_sold", "quantity_sold_text",
    "brand", "seller_location", "product_url",
]


def normalize_product(item: dict) -> dict:
    """Chuẩn hóa một sản phẩm (dict) từ bất kỳ platform nào về schema chung."""
    platform = str(item.get("platform", "")).lower()

    normalized = {
        "crawl_date": item.get("crawl_date"),
        "platform": item.get("platform"),
        "category": item.get("category_name"),
        "id": item.get("id"),
        "product_name": item.get("name"),
        "current_price": None,
        "original_price": None,
        "discount_rate": None,
        "rating_average": None,
        "num_reviews": None,
        "quantity_sold": None,
        "quantity_sold_text": None,
        "brand": None,
        "seller_location": None,
        "product_url": item.get("url"),
    }

    mapping = PLATFORM_FIELD_MAPPING.get(platform)
    if mapping:
        for standard_key, source_key in mapping.items():
            normalized[standard_key] = item.get(source_key)

    return normalized


def normalize_dataset(raw_records: list[dict]) -> pd.DataFrame:
    """Chuẩn hóa toàn bộ danh sách record thô thành DataFrame theo schema chung."""
    return pd.DataFrame([normalize_product(item) for item in raw_records])
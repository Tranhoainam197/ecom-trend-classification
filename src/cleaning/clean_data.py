"""
clean_data.py
==============
BƯỚC 1 - CLEANING
==================
Orchestrator chính cho việc làm sạch dữ liệu thương mại điện tử thô
(đã gộp từ 3 sàn ở bước Collection) thành dữ liệu sạch, sẵn sàng cho
bước Integration.

Các công việc thực hiện theo đúng trình tự:
    1. Chuẩn hóa schema 3 platform về 1 chuẩn chung
    2. Trích xuất / ép kiểu các trường số (giá, discount, rating, sold)
    3. Chuẩn hóa trường brand
    4. Xử lý missing values (drop record thiếu trường quan trọng, fill trường phụ)
    5. Loại bỏ trùng lặp
    6. Xử lý outlier
    7. Chọn cột cuối cùng và lưu kết quả

Chạy: python -m src.cleaning.clean_data
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

import config
from src.cleaning.normalizer import normalize_dataset
from src.cleaning.value_extractor import (
    extract_price, extract_discount_rate, extract_quantity_sold, safe_to_numeric,
)
from src.cleaning.outlier_handler import OutlierHandler

# Các trường bắt buộc phải có giá trị — thiếu 1 trong số này thì record
# không đủ thông tin để tính feature ở bước Transformation, nên drop luôn
# ở bước Cleaning (drop sớm để các bước sau không phải xử lý NaN nữa).
CRITICAL_COLUMNS = ["quantity_sold", "num_reviews", "rating_average", "discount_rate"]

# Các trường phụ, thiếu thì fill giá trị mặc định thay vì drop record
TEXT_FILL_DEFAULTS = {
    "brand": "No Brand",
    "seller_location": "Unknown Location",
    "quantity_sold_text": "Chưa có thông tin bán",
}

FINAL_COLUMNS = [
    "id", "crawl_date", "platform", "category", "product_name",
    "current_price", "discount_rate", "rating_average", "num_reviews",
    "quantity_sold", "brand", "seller_location", "product_url",
]

DEDUP_KEYS = ["platform", "id"]


def _clean_numeric_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 2: Trích xuất các trường số từ chuỗi thô."""
    df["current_price"] = df["current_price"].apply(extract_price)
    df["original_price"] = df["original_price"].apply(extract_price)
    df[["current_price", "original_price"]] = df[["current_price", "original_price"]].apply(
        pd.to_numeric, errors="coerce"
    )

    df["discount_rate"] = df["discount_rate"].apply(extract_discount_rate)
    df["rating_average"] = df["rating_average"].apply(safe_to_numeric)
    df["num_reviews"] = df["num_reviews"].apply(safe_to_numeric)
    df["quantity_sold"] = df["quantity_sold_text"].apply(
        lambda x: extract_quantity_sold(x) if isinstance(x, str) else None
    )
    return df


def _clean_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 3: Chuẩn hóa các biến thể khác nhau của 'không có brand' về 1 giá trị."""
    no_brand_variants = {"no brand", "no.brand", "nobrand", "none", "n/a", ""}

    def normalize(value):
        if value is None or pd.isna(value):
            return "No Brand"
        value_str = str(value).strip()
        return "No Brand" if value_str.lower() in no_brand_variants or not value_str else value_str

    df["brand"] = df["brand"].apply(normalize)
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 4: Xử lý missing values.

    - Drop record thiếu bất kỳ trường CRITICAL_COLUMNS nào (không đủ dữ liệu
      để tính feature đáng tin cậy ở bước Transformation).
    - Sau khi đã drop hết NaN ở các trường critical, ép logic nghiệp vụ
      (vd: chưa bán thì không thể có review) trên dữ liệu đã đầy đủ.
    - Fill các trường text phụ (brand, location, ...) bằng giá trị mặc định.

    Lưu ý quan trọng (đã sửa so với bản gốc): vì record thiếu rating_average
    đã bị DROP ở trên, nên sau bước này rating_average không còn NaN nữa.
    Vì vậy KHÔNG cần (và không nên) fillna(0) cho rating_average — làm vậy
    sẽ sai logic vì cột đã sạch, gọi fillna thêm chỉ gây nhiễu code.
    """
    before = len(df)
    critical_present = [c for c in CRITICAL_COLUMNS if c in df.columns]
    missing_mask = df[critical_present].isna().any(axis=1)
    df = df[~missing_mask].copy()
    print(f"  Drop {missing_mask.sum():,} record thiếu ít nhất 1 trường quan trọng "
          f"({before:,} -> {len(df):,})")

    # Ép kiểu số nguyên, không âm
    df["quantity_sold"] = df["quantity_sold"].clip(lower=0).astype("int64")

    # Logic nghiệp vụ: chưa bán (sold=0) thì không thể có review
    mask_not_sold = df["quantity_sold"] == 0
    conflict = (mask_not_sold & (df["num_reviews"] > 0)).sum()
    if conflict:
        df.loc[mask_not_sold, "num_reviews"] = 0
        print(f"  Sửa {conflict:,} record: sold=0 nhưng review>0 -> ép review=0")
    df["num_reviews"] = df["num_reviews"].clip(lower=0).astype("int64")

    # Logic nghiệp vụ: không có review thì rating phải là "chưa có" (NaN),
    # không thể có rating hợp lệ. record này lẽ ra phải bị loại ở bước drop
    # critical phía trên rồi (rating_average nằm trong CRITICAL_COLUMNS),
    # nhưng vẫn kiểm tra lại để phát hiện dữ liệu mâu thuẫn nếu có.
    mask_no_review = df["num_reviews"] == 0
    invalid = (mask_no_review & df["rating_average"].notna() & (df["rating_average"] > 0)).sum()
    if invalid:
        print(f"  Cảnh báo: {invalid:,} record có num_reviews=0 nhưng rating_average>0 "
              f"(giữ nguyên, không tự suy diễn vì rating gốc có thể tính theo cách khác)")

    df["discount_rate"] = df["discount_rate"].clip(0, 100)
    df["rating_average"] = df["rating_average"].clip(1, 5)

    # Fill các trường text phụ
    for col, default_value in TEXT_FILL_DEFAULTS.items():
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[col] = df[col].fillna(default_value)
                if df[col].dtype == object:
                    df[col] = df[col].str.strip()
                print(f"  Fill {n_missing:,} giá trị thiếu ở '{col}' -> '{default_value}'")

    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 5: Loại bỏ trùng lặp, ưu tiên giữ record có chất lượng dữ liệu cao hơn."""
    before = len(df)
    df = df.sort_values(by=["quantity_sold", "num_reviews"], ascending=[False, False])
    df = df.drop_duplicates(subset=DEDUP_KEYS, keep="first").reset_index(drop=True)
    print(f"  Loại trùng lặp theo {DEDUP_KEYS}: {before:,} -> {len(df):,}")
    return df


def clean_data(input_file: str, output_file: str) -> pd.DataFrame:
    """Hàm chính: chạy toàn bộ quy trình Cleaning theo thứ tự."""
    print("=" * 70)
    print("BƯỚC 1: CLEANING - LÀM SẠCH DỮ LIỆU")
    print("=" * 70)

    print(f"\n[1/7] Đọc dữ liệu thô: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  Đã đọc {len(raw_data):,} records")

    print("\n[2/7] Chuẩn hóa schema 3 platform...")
    df = normalize_dataset(raw_data)
    print(f"  Đã chuẩn hóa {len(df):,} records, {len(df.columns)} cột")

    print("\n[3/7] Trích xuất các trường số (giá, discount, rating, sold)...")
    df = _clean_numeric_fields(df)

    print("\n[4/7] Chuẩn hóa trường brand...")
    df = _clean_brand(df)

    print("\n[5/7] Xử lý missing values...")
    df = _handle_missing_values(df)

    print("\n[6/7] Loại bỏ trùng lặp...")
    df = _remove_duplicates(df)

    print("\n[7/7] Xử lý outlier...")
    outlier_handler = OutlierHandler(df)
    outlier_handler.handle_price(upper_percentile=99.5) \
        .handle_quantity_sold(upper_percentile=99.0) \
        .handle_discount(max_discount=80) \
        .handle_rating() \
        .handle_num_reviews(upper_percentile=99.7, max_review_per_sold_ratio=1.0, strategy="drop")
    outlier_handler.save_comparison_report(config.OUTLIER_REPORT_DIR)
    df = outlier_handler.get_cleaned_data()
    print(f"  Sau xử lý outlier: {len(df):,} records")

    available_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[available_cols]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu dữ liệu sạch ({len(df):,} records, {len(df.columns)} cột): {output_file}")
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("THỐNG KÊ TÓM TẮT DỮ LIỆU SAU CLEANING")
    print("=" * 70)
    print(f"Tổng records: {len(df):,}")
    print(f"Giá: {df['current_price'].min():,.0f} - {df['current_price'].max():,.0f} VNĐ "
          f"(trung bình {df['current_price'].mean():,.0f})")
    print(f"Rating trung bình: {df['rating_average'].mean():.2f}")
    print(f"Review trung bình: {df['num_reviews'].mean():.0f}")
    print(f"\nPhân bố theo platform:\n{df['platform'].value_counts()}")
    print(f"\nTop 5 category:\n{df['category'].value_counts().head()}")


if __name__ == "__main__":
    config.ensure_all_dirs()
    clean_data(input_file=config.RAW_MERGED_FILE, output_file=config.CLEAN_FILE)
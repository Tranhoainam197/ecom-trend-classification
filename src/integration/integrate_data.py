"""
integrate_data.py
===================
BƯỚC 2 - INTEGRATION
=====================
Sau khi dữ liệu từng nguồn đã được làm sạch (bước Cleaning), bước
Integration chịu trách nhiệm hợp nhất và đảm bảo tính toàn vẹn dữ liệu
giữa các nguồn (Shopee, Lazada, Tiki) trước khi đưa vào Transformation:

    1. Kiểm tra schema thống nhất giữa các record (mọi record phải có
       đầy đủ các cột chuẩn, đúng kiểu dữ liệu)
    2. Phát hiện và giải quyết xung đột định danh sản phẩm giữa các
       platform (vd: cùng 1 "id" nhưng khác platform thì KHÔNG phải trùng,
       phải có cặp khóa platform+id mới định danh duy nhất 1 sản phẩm)
    3. Thống nhất đơn vị đo (tiền VNĐ, tỷ lệ %, số lượng nguyên)
    4. Gắn nhãn nguồn gốc (provenance) rõ ràng cho từng record để truy
       vết được khi cần kiểm tra chất lượng dữ liệu sau này

Input : data/clean/cleaned_data.json   (output của bước Cleaning)
Output: data/integrated/integrated_data.json

Chạy: python -m src.integration.integrate_data
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import config

REQUIRED_SCHEMA = {
    "id": object,
    "platform": object,
    "category": object,
    "product_name": object,
    "current_price": "number",
    "discount_rate": "number",
    "rating_average": "number",
    "num_reviews": "number",
    "quantity_sold": "number",
    "brand": object,
    "seller_location": object,
    "product_url": object,
}

VALID_PLATFORMS = {"Shopee", "Lazada", "Tiki"}


def _validate_schema(df: pd.DataFrame) -> list[str]:
    """Kiểm tra schema: cột thiếu, kiểu dữ liệu sai. Trả về list lỗi (rỗng nếu hợp lệ)."""
    issues = []

    missing_cols = [c for c in REQUIRED_SCHEMA if c not in df.columns]
    if missing_cols:
        issues.append(f"Thiếu cột bắt buộc: {missing_cols}")

    for col, expected_type in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            continue
        if expected_type == "number":
            non_numeric = df[col].apply(lambda v: not isinstance(v, (int, float)) or pd.isna(v))
            if non_numeric.any():
                issues.append(f"Cột '{col}': {non_numeric.sum():,} giá trị không phải số hợp lệ")

    invalid_platforms = set(df["platform"].unique()) - VALID_PLATFORMS
    if invalid_platforms:
        issues.append(f"Phát hiện platform không hợp lệ: {invalid_platforms}")

    return issues


def _check_cross_platform_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo khóa định danh duy nhất là (platform, id), KHÔNG phải chỉ id.
    Vì id do mỗi sàn tự sinh độc lập, "id=123" trên Shopee và Tiki là 2 sản
    phẩm hoàn toàn khác nhau — đây là vấn đề kinh điển khi tích hợp dữ liệu
    đa nguồn (record linkage). Tạo composite key rõ ràng để các bước sau
    không bị nhầm.
    """
    df["global_product_id"] = df["platform"].str.lower() + "_" + df["id"].astype(str)

    duplicate_global_ids = df["global_product_id"].duplicated().sum()
    if duplicate_global_ids:
        print(f"  Cảnh báo: {duplicate_global_ids:,} global_product_id trùng lặp sau khi tạo "
              f"composite key (không nên xảy ra nếu Cleaning đã dedup đúng) -> loại bỏ")
        df = df.drop_duplicates(subset="global_product_id", keep="first")

    return df


def _unify_units(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo các trường số dùng cùng đơn vị và kiểu dữ liệu trên toàn bộ dataset."""
    df["current_price"] = df["current_price"].astype(float).round(0)
    df["discount_rate"] = df["discount_rate"].astype(float).round(1)
    df["rating_average"] = df["rating_average"].astype(float).round(2)
    df["num_reviews"] = df["num_reviews"].astype("int64")
    df["quantity_sold"] = df["quantity_sold"].astype("int64")
    return df


def integrate_data(input_file: str, output_file: str) -> pd.DataFrame:
    """Hàm chính: hợp nhất và kiểm tra tính toàn vẹn dữ liệu đã làm sạch."""
    print("=" * 70)
    print("BƯỚC 2: INTEGRATION - HỢP NHẤT & KIỂM TRA TOÀN VẸN DỮ LIỆU")
    print("=" * 70)

    print(f"\n[1/4] Đọc dữ liệu đã làm sạch: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"  Đã đọc {len(df):,} records")

    print("\n[2/4] Kiểm tra schema thống nhất...")
    issues = _validate_schema(df)
    if issues:
        print("  Phát hiện vấn đề schema:")
        for issue in issues:
            print(f"    - {issue}")
        raise ValueError(
            "Dữ liệu không đạt chuẩn schema thống nhất. Hãy kiểm tra lại bước Cleaning."
        )
    print("  Schema hợp lệ trên toàn bộ records")

    print("\n[3/4] Kiểm tra định danh duy nhất giữa các platform...")
    df = _check_cross_platform_identity(df)
    print(f"  Số sản phẩm duy nhất (global_product_id): {df['global_product_id'].nunique():,}")

    print("\n[4/4] Thống nhất đơn vị đo và kiểu dữ liệu...")
    df = _unify_units(df)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu dữ liệu đã hợp nhất ({len(df):,} records): {output_file}")
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("THỐNG KÊ TÓM TẮT SAU INTEGRATION")
    print("=" * 70)
    print(f"Tổng số sản phẩm duy nhất: {df['global_product_id'].nunique():,}")
    print("\nPhân bố theo platform:")
    print(df["platform"].value_counts())
    print(f"\nSố category duy nhất (toàn bộ platform): {df['category'].nunique()}")


if __name__ == "__main__":
    config.ensure_all_dirs()
    integrate_data(input_file=config.CLEAN_FILE, output_file=config.INTEGRATED_FILE)
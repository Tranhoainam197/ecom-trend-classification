"""
encode_data.py
================
BƯỚC 5 - ENCODING
===================
Chuyển đổi các trường dữ liệu dạng chữ (categorical) sang dạng số mà mô
hình Machine Learning có thể học được:

    1. One-Hot Encoding cho categorical features (popularity_category,
       price_segment, quality_tier, discount_intensity, product_age)
    2. Label Encoding cho target variable (label: Hot Trend / Best Seller /
       Best Deal / Normal)

Input  : data/normalization/normalized_train.json, normalized_test.json
         (đã qua Normalization ở bước 4 — numerical feature đã được scale)
Output : data/encoding/encoded_train.json, encoded_test.json
         (sẵn sàng đưa thẳng vào mô hình Machine Learning)

Lưu ý thứ tự: One-hot encoding KHÔNG gây data leakage (vì không học
thống kê gì từ dữ liệu, chỉ là phép biến đổi 1-1 dựa trên tập categories
đã biết), nên việc encoding categorical trước hay sau khi split không
ảnh hưởng kết quả — khác với Normalization (StandardScaler) bắt buộc
phải fit sau khi split. Encoding ở đây xử lý train và test riêng nhưng
dùng chung danh sách categories phát hiện được trên TRAIN, để đảm bảo
2 tập có cùng cấu trúc cột one-hot.

Chạy: python -m src.encoding.encode_data
"""

import sys
import os
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config

CATEGORICAL_FEATURES = [
    "popularity_category", "price_segment", "quality_tier", "discount_intensity", "product_age",
]

# Cột không phải feature đầu vào — lấy từ config.py (nguồn DUY NHẤT), cộng
# thêm target column, để đếm/báo cáo số feature nhất quán với feature_selector.py
# (bug cũ: dùng list thiếu platform/category/brand/product_name -> đếm nhầm
# 44 feature "20 scaled + 24 one-hot" thay vì đúng 40 "20 scaled + 20 one-hot").
NON_FEATURE_COLUMNS = config.NON_FEATURE_COLUMNS + config.TARGET_COLUMNS


def _one_hot_encode(df_train: pd.DataFrame, df_test: pd.DataFrame, categorical_cols: list[str]):
    """One-hot encode categorical columns, đảm bảo train/test có cùng cấu trúc cột."""
    train_categories = {col: sorted(df_train[col].dropna().unique().tolist()) for col in categorical_cols}

    def encode(df):
        df = df.copy()
        for col in categorical_cols:
            one_hot = pd.get_dummies(df[col], prefix=col, prefix_sep="_")
            df = pd.concat([df.drop(columns=[col]), one_hot], axis=1)
        return df

    df_train_encoded = encode(df_train)
    df_test_encoded = encode(df_test)

    # Đảm bảo test có đủ cột one-hot như train (category nào không xuất hiện
    # trong test thì cột tương ứng = 0); và bỏ cột one-hot lạ chỉ có ở test.
    df_test_encoded = df_test_encoded.reindex(columns=df_train_encoded.columns, fill_value=0)

    return df_train_encoded, df_test_encoded, train_categories


def _label_encode_target(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Label encode cột 'label' (target), fit trên train, áp dụng cho cả 2 tập."""
    encoder = LabelEncoder()
    encoder.fit(df_train["label"])

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["label_encoded"] = encoder.transform(df_train["label"])
    df_test["label_encoded"] = encoder.transform(df_test["label"])

    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
    return df_train, df_test, encoder, mapping


def encode_data(train_file: str, test_file: str):
    print("=" * 70)
    print("BƯỚC 5: ENCODING - MÃ HÓA DỮ LIỆU")
    print("=" * 70)

    print(f"\nĐọc dữ liệu đã normalization: \n  {train_file}\n  {test_file}")
    with open(train_file, "r", encoding="utf-8") as f:
        df_train = pd.DataFrame(json.load(f))
    with open(test_file, "r", encoding="utf-8") as f:
        df_test = pd.DataFrame(json.load(f))
    print(f"Train: {len(df_train):,} records | Test: {len(df_test):,} records")

    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df_train.columns]
    print(f"\n[1/3] One-Hot Encoding cho {len(categorical_cols)} categorical features: {categorical_cols}")
    df_train, df_test, train_categories = _one_hot_encode(df_train, df_test, categorical_cols)
    for col, cats in train_categories.items():
        print(f"  {col}: {len(cats)} categories -> {cats}")

    print("\n[2/3] Label Encoding cho target 'label'...")
    df_train, df_test, label_encoder, label_mapping = _label_encode_target(df_train, df_test)
    print(f"  Mapping: {label_mapping}")

    print("\n[3/3] Lưu kết quả...")
    os.makedirs(config.ENCODING_DIR, exist_ok=True)

    with open(config.ENCODED_TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(df_train.to_dict("records"), f, ensure_ascii=False, indent=2)
    with open(config.ENCODED_TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(df_test.to_dict("records"), f, ensure_ascii=False, indent=2)
    with open(config.LABEL_ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoder, f)

    feature_mapping = {
        "categorical_features": train_categories,
        "label_encoding": {str(k): v for k, v in label_mapping.items()},
        "non_feature_columns": NON_FEATURE_COLUMNS,
    }
    with open(config.FEATURE_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=2)

    print(f"  Đã lưu: {config.ENCODED_TRAIN_FILE}")
    print(f"  Đã lưu: {config.ENCODED_TEST_FILE}")
    print(f"  Đã lưu: {config.LABEL_ENCODER_FILE}")
    print(f"  Đã lưu: {config.FEATURE_MAPPING_FILE}")

    _print_summary(df_train, df_test, label_mapping)
    return df_train, df_test, label_encoder


def _print_summary(df_train: pd.DataFrame, df_test: pd.DataFrame, label_mapping: dict):
    print("\n" + "=" * 70)
    print("THỐNG KÊ ENCODING")
    print("=" * 70)
    feature_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLUMNS]
    print(f"Tổng số input feature cho mô hình: {len(feature_cols)}")
    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

    print("\nPhân bố nhãn (Train):")
    for label_name, label_id in label_mapping.items():
        count = (df_train["label_encoded"] == label_id).sum()
        print(f"  {label_name}: {count:,} ({count / len(df_train) * 100:.1f}%)")

    print("\nPhân bố nhãn (Test):")
    for label_name, label_id in label_mapping.items():
        count = (df_test["label_encoded"] == label_id).sum()
        print(f"  {label_name}: {count:,} ({count / len(df_test) * 100:.1f}%)")


if __name__ == "__main__":
    config.ensure_all_dirs()
    encode_data(train_file=config.NORMALIZED_TRAIN_FILE, test_file=config.NORMALIZED_TEST_FILE)
"""
normalize_features.py
========================
BƯỚC 4 - NORMALIZATION
========================
Chuẩn hóa (chuẩn hóa thang đo) các đặc trưng số học bằng StandardScaler
(z-score: trung bình 0, độ lệch chuẩn 1), để các mô hình Machine Learning
nhạy với thang đo (KNN, SVM, Neural Network, Logistic Regression...) hoạt
động đúng — tránh việc các feature có giá trị lớn (vd: current_price hàng
trăm nghìn) lấn át các feature có giá trị nhỏ (vd: rating_average 1-5).

ĐIỂM QUAN TRỌNG VỀ DATA LEAKAGE:
Bước Normalization PHẢI được thực hiện SAU khi chia train/test, và scaler
chỉ được fit (học mean, std) trên tập TRAIN. Nếu fit scaler trên toàn bộ
dataset trước khi chia, thông tin của tập test sẽ "rò" vào tập train qua
mean/std, khiến đánh giá mô hình sau này không khách quan.

Vì vậy thứ tự đúng là:
    1. Split train/test (trên dữ liệu đã labeling, CHƯA scale)
    2. Fit scaler trên X_train
    3. Transform cả X_train và X_test bằng scaler đã fit

Input : data/transformation/labeled_data.json
Output: data/normalization/normalized_train.json, normalized_test.json,
         scaler.pkl

Chạy: python -m src.normalization.normalize_features
"""

import sys
import os
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config

NUMERICAL_FEATURES = [
    "current_price", "original_price", "absolute_saving",
    "discount_rate", "rating_average", "num_reviews", "quantity_sold", "days_active",
    "sales_velocity", "sales_velocity_normalized",
    "review_velocity", "review_velocity_normalized",
    "popularity_score", "engagement_score", "trend_momentum",
    "discount_score", "value_score", "deal_quality_score",
    "category_popularity_rank", "category_price_percentile",
]

# Các cột không phải feature đầu vào mô hình (metadata, nhãn, cột phụ trợ
# sinh ra trong quá trình labeling) — không tham gia normalization/encoding.
# Re-export từ config.py (nguồn DUY NHẤT) để encode_data.py / viz_encoded.py
# import lại mà không bị lệch danh sách giữa các module (bug cũ đã sửa: trước
# đây list này thiếu platform/category/brand/product_name, khiến bước Encoding
# đếm nhầm 44 feature thay vì đúng 40 feature thực sự đưa vào mô hình).
NON_FEATURE_COLUMNS = config.NON_FEATURE_COLUMNS


def normalize_features(input_file: str, test_size: float = 0.2, random_state: int = 42):
    """
    Chia train/test trước, sau đó fit StandardScaler CHỈ trên train rồi
    transform cả 2 tập (đúng thứ tự để tránh data leakage).

    Returns
    -------
    df_train, df_test : DataFrame đã thay numerical cột gốc bằng cột *_scaled
    scaler : StandardScaler đã fit, dùng lại khi cần inverse_transform
    """
    print("=" * 70)
    print("BƯỚC 4: NORMALIZATION - CHUẨN HÓA THANG ĐO (StandardScaler)")
    print("=" * 70)

    print(f"\nĐọc dữ liệu đã labeling: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Đã đọc {len(df):,} records")

    print("\n[1/3] Chia train/test (PHẢI làm trước khi fit scaler)...")
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    print(f"  Train: {len(df_train):,} ({(1 - test_size) * 100:.0f}%) | "
          f"Test: {len(df_test):,} ({test_size * 100:.0f}%)")

    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    print(f"\n[2/3] Fit StandardScaler trên {len(numeric_cols)} numerical features (CHỈ trên train)...")

    scaler = StandardScaler()
    scaler.fit(df_train[numeric_cols].values)

    df_train = df_train.copy()
    df_test = df_test.copy()

    train_scaled = scaler.transform(df_train[numeric_cols].values)
    test_scaled = scaler.transform(df_test[numeric_cols].values)

    for i, col in enumerate(numeric_cols):
        df_train[f"{col}_scaled"] = train_scaled[:, i]
        df_test[f"{col}_scaled"] = test_scaled[:, i]

    df_train = df_train.drop(columns=numeric_cols)
    df_test = df_test.drop(columns=numeric_cols)

    print(f"  Đã chuẩn hóa xong. X_train shape: {df_train.shape} | X_test shape: {df_test.shape}")

    print("\n[3/3] Lưu kết quả...")
    os.makedirs(config.NORMALIZATION_DIR, exist_ok=True)

    with open(config.NORMALIZED_TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(df_train.to_dict("records"), f, ensure_ascii=False, indent=2)
    with open(config.NORMALIZED_TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(df_test.to_dict("records"), f, ensure_ascii=False, indent=2)
    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  Đã lưu: {config.NORMALIZED_TRAIN_FILE}")
    print(f"  Đã lưu: {config.NORMALIZED_TEST_FILE}")
    print(f"  Đã lưu: {config.SCALER_FILE}")

    _print_summary(numeric_cols, scaler)
    return df_train, df_test, scaler


def _print_summary(numeric_cols: list[str], scaler: StandardScaler):
    print("\n" + "=" * 70)
    print("THỐNG KÊ NORMALIZATION")
    print("=" * 70)
    print(f"Số feature được chuẩn hóa: {len(numeric_cols)}")
    print("Mean / Std học được từ tập train (5 feature đầu):")
    for i, col in enumerate(numeric_cols[:5]):
        print(f"  {col}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")


if __name__ == "__main__":
    config.ensure_all_dirs()
    normalize_features(input_file=config.LABELED_FILE, test_size=0.2)
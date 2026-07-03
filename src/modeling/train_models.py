"""
train_models.py
==================
BƯỚC 6a - MODELING (Train)
=============================
Train nhiều mô hình phân loại trên dữ liệu đã Encoding, để phân loại sản
phẩm vào 4 nhóm: Hot Trend, Best Seller, Best Deal, Normal.

Các thuật toán dùng (phổ biến trong môn Khai thác Dữ liệu):
    - Decision Tree
    - Random Forest
    - Logistic Regression (đa lớp)
    - K-Nearest Neighbors (KNN)
    - Naive Bayes (GaussianNB)

Mỗi thuật toán được train trên 3 bộ feature (xem feature_selector.py):
    - "full"              : đầy đủ feature, bao gồm composite score (CÓ label leakage)
    - "realistic"         : đã loại composite score (KHÔNG leakage numeric,
                             phản ánh đúng khả năng áp dụng thực tế cho sản phẩm mới)
    - "realistic_strict"  : loại thêm cả categorical dùng trực tiếp trong luật
                             gán nhãn — dùng để kiểm chứng mức leakage còn lại

Chạy: python -m src.modeling.train_models
"""

import sys
import os
import json
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import config
from src.modeling.feature_selector import FEATURE_SETS

MODEL_REGISTRY = {
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "Naive Bayes": lambda: GaussianNB(),
}

# Thứ tự train các bộ feature. "realistic_strict" chỉ dùng để đối chiếu/kiểm
# chứng leakage, KHÔNG phải kết quả chính của báo cáo (kết quả chính vẫn là
# "realistic"), nên đặt cuối cùng.
FEATURE_SET_ORDER = ("realistic", "full", "realistic_strict")


def _load_train_test(train_file: str, test_file: str):
    with open(train_file, "r", encoding="utf-8") as f:
        df_train = pd.DataFrame(json.load(f))
    with open(test_file, "r", encoding="utf-8") as f:
        df_test = pd.DataFrame(json.load(f))
    return df_train, df_test


def train_all_models(train_file: str, test_file: str, feature_set_name: str = "realistic"):
    """
    Train toàn bộ model trong MODEL_REGISTRY trên 1 bộ feature cụ thể.

    Returns
    -------
    results : dict {model_name: {"model": fitted_model, "X_columns": [...], "train_time": float}}
    df_train, df_test, feature_cols
    """
    df_train, df_test = _load_train_test(train_file, test_file)

    feature_set = FEATURE_SETS[feature_set_name]
    feature_cols = feature_set["get_columns"](df_train.columns)

    print(f"\nBộ feature: {feature_set['name']} ({len(feature_cols)} feature)")
    if feature_set["warning"]:
        print(f"  CẢNH BÁO: {feature_set['warning']}")

    X_train = df_train[feature_cols].astype(float).values
    y_train = df_train["label_encoded"].values
    X_test = df_test[feature_cols].astype(float).values
    y_test = df_test["label_encoded"].values

    results = {}
    for model_name, model_factory in MODEL_REGISTRY.items():
        print(f"  Đang train: {model_name}...", end=" ", flush=True)
        model = model_factory()

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        results[model_name] = {
            "model": model,
            "feature_columns": feature_cols,
            "train_time": elapsed,
        }
        print(f"xong ({elapsed:.2f}s)")

    return results, df_train, df_test, feature_cols, X_train, y_train, X_test, y_test


def run_training(train_file: str, test_file: str, output_dir: str):
    """Train trên cả 3 bộ feature (realistic, full, realistic_strict), lưu toàn bộ model đã train."""
    print("=" * 70)
    print("BƯỚC 6a: MODELING - TRAIN MÔ HÌNH PHÂN LOẠI")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for feature_set_name in FEATURE_SET_ORDER:
        print(f"\n{'─' * 70}")
        print(f"BỘ FEATURE: {feature_set_name.upper()}")
        print("─" * 70)

        results, df_train, df_test, feature_cols, X_train, y_train, X_test, y_test = train_all_models(
            train_file, test_file, feature_set_name
        )
        all_results[feature_set_name] = {
            "results": results,
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols,
        }

        models_to_save = {name: r["model"] for name, r in results.items()}
        save_path = os.path.join(output_dir, f"trained_models_{feature_set_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"models": models_to_save, "feature_columns": feature_cols}, f)
        print(f"\nĐã lưu {len(models_to_save)} model ({feature_set_name}): {save_path}")

    return all_results


if __name__ == "__main__":
    config.ensure_all_dirs()
    run_training(
        train_file=config.ENCODED_TRAIN_FILE,
        test_file=config.ENCODED_TEST_FILE,
        output_dir=config.MODEL_DIR,
    )
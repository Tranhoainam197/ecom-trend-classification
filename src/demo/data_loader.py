"""
data_loader.py
=================
Logic dùng CHUNG cho cả 2 kiểu demo (CLI trong predict_demo.py và Web trong
web_app.py): load model đã train + dữ liệu tập test, ghép với thông tin
"thô" (giá, rating...) để hiển thị, và dự đoán 1 lần cho toàn bộ tập test.

Tách riêng ra module này để không lặp code / không lệch logic giữa 2 bản
demo (CLI và Web đều phải cho ra CÙNG MỘT kết quả dự đoán).
"""

import sys
import os
import json
import pickle
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config


def load_everything():
    """
    Trả về (records, class_names, model_name).

    records: list[dict], mỗi dict là 1 sản phẩm trong TẬP TEST kèm:
        - thông tin hiển thị (tên, giá, rating, danh mục...)
        - nhãn thực tế (actual_label)
        - nhãn mô hình dự đoán (predicted_label) + xác suất từng lớp
        - correct: bool (dự đoán có khớp nhãn thực tế không)
    """
    with open(config.BEST_MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    class_names = bundle["class_names"]
    model_name = bundle["model_name"]

    with open(config.ENCODED_TEST_FILE, "r", encoding="utf-8") as f:
        encoded_rows = json.load(f)

    with open(config.LABELED_FILE, "r", encoding="utf-8") as f:
        labeled_rows = json.load(f)

    # Tra cứu thông tin "thô" (giá, rating, số bán...) theo (platform, id) vì
    # các cột này đã bị Normalization thay bằng bản *_scaled trong
    # encoded_test.json, không còn giá trị gốc dễ đọc để hiển thị demo.
    raw_lookup = {(str(r["platform"]), str(r["id"])): r for r in labeled_rows}

    X = np.array([[float(row[c]) for c in feature_columns] for row in encoded_rows])
    probs = model.predict_proba(X)
    class_idx_to_name = {cls: class_names[cls] for cls in model.classes_}

    records = []
    for row, prob_row in zip(encoded_rows, probs):
        key = (str(row["platform"]), str(row["id"]))
        raw = raw_lookup.get(key, {})
        pred_idx = model.classes_[int(np.argmax(prob_row))]
        pred_label = class_idx_to_name[pred_idx]
        prob_map = {class_idx_to_name[cls]: float(p) for cls, p in zip(model.classes_, prob_row)}
        records.append({
            "id": str(row.get("id")),
            "platform": row.get("platform", "N/A"),
            "category": row.get("category", "N/A"),
            "brand": raw.get("brand", "N/A"),
            "product_name": row.get("product_name", "N/A"),
            "current_price": raw.get("current_price"),
            "original_price": raw.get("original_price"),
            "discount_rate": raw.get("discount_rate"),
            "rating_average": raw.get("rating_average"),
            "num_reviews": raw.get("num_reviews"),
            "quantity_sold": raw.get("quantity_sold"),
            "price_segment": raw.get("price_segment"),
            "quality_tier": raw.get("quality_tier"),
            "popularity_category": raw.get("popularity_category"),
            "discount_intensity": raw.get("discount_intensity"),
            "actual_label": row.get("label"),
            "predicted_label": pred_label,
            "probabilities": prob_map,
            "correct": bool(pred_label == row.get("label")),
        })

    return records, class_names, model_name


def load_model_comparison():
    """Đọc data/model/model_comparison.csv -> list[dict] (rỗng nếu chưa có)."""
    if not os.path.exists(config.MODEL_COMPARISON_FILE):
        return []
    with open(config.MODEL_COMPARISON_FILE, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

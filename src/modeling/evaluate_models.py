"""
evaluate_models.py
=====================
BƯỚC 6b - MODELING (Evaluation)
===================================
Đánh giá toàn bộ mô hình đã train ở train_models.py bằng các chỉ số
chuẩn của bài toán phân loại đa lớp:

    - Accuracy
    - Precision / Recall / F1-score (macro average — coi trọng đều cả
      4 nhãn, không bị lệch vì nhãn "Normal" chiếm đa số)
    - Confusion Matrix

So sánh rõ giữa 2 bộ feature (full vs realistic) để minh họa hiện tượng
label leakage, và chọn ra mô hình tốt nhất trên bộ feature REALISTIC
(vì đây là bộ feature phản ánh đúng khả năng áp dụng thực tế).

Chạy: python -m src.modeling.evaluate_models
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)

import config
from src.modeling.train_models import run_training

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _evaluate_single_model(model, X_test, y_test, class_names: list[str]) -> dict:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


def evaluate_all(all_results: dict, class_names: list[str]) -> pd.DataFrame:
    """Đánh giá toàn bộ model trên cả 2 bộ feature, trả về bảng so sánh tổng hợp."""
    rows = []
    detailed_eval = {}

    for feature_set_name, data in all_results.items():
        X_test, y_test = data["X_test"], data["y_test"]
        detailed_eval[feature_set_name] = {}

        for model_name, model_info in data["results"].items():
            eval_result = _evaluate_single_model(model_info["model"], X_test, y_test, class_names)
            detailed_eval[feature_set_name][model_name] = eval_result

            rows.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "accuracy": round(eval_result["accuracy"], 4),
                "precision_macro": round(eval_result["precision_macro"], 4),
                "recall_macro": round(eval_result["recall_macro"], 4),
                "f1_macro": round(eval_result["f1_macro"], 4),
                "train_time_sec": round(model_info["train_time"], 3),
            })

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["feature_set", "f1_macro"], ascending=[True, False]
    ).reset_index(drop=True)

    return comparison_df, detailed_eval


def plot_comparison(comparison_df: pd.DataFrame, output_dir: str):
    """Biểu đồ so sánh F1-score giữa các model, tách theo bộ feature."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, feature_set_name in zip(axes, ("realistic", "full")):
        subset = comparison_df[comparison_df["feature_set"] == feature_set_name].sort_values(
            "f1_macro", ascending=True
        )
        colors = ["#e74c3c" if feature_set_name == "full" else "#2ecc71"] * len(subset)
        ax.barh(subset["model"], subset["f1_macro"], color=colors, edgecolor="black")
        for i, v in enumerate(subset["f1_macro"]):
            ax.text(v, i, f" {v:.3f}", va="center", fontweight="bold")
        title = "Full Features (CÓ label leakage)" if feature_set_name == "full" \
            else "Realistic Features (KHÔNG leakage)"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("F1-score (macro)")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("So sánh hiệu năng mô hình: Full vs Realistic Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_model_comparison_f1.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Biểu đồ ghép: accuracy theo model, 2 nhóm cột (full vs realistic)
    pivot = comparison_df.pivot(index="model", columns="feature_set", values="accuracy")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot.index))
    width = 0.35
    ax.bar(x - width / 2, pivot["realistic"], width, label="Realistic (không leakage)", color="#2ecc71")
    ax.bar(x + width / 2, pivot["full"], width, label="Full (có leakage)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: Full vs Realistic Features theo từng mô hình", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_accuracy_full_vs_realistic.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Đã lưu biểu đồ so sánh model vào: {output_dir}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], model_name: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold")

    max_val = cm.max()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def run_evaluation(train_file: str, test_file: str, output_dir: str, viz_dir: str):
    print("=" * 70)
    print("BƯỚC 6b: MODELING - EVALUATION (ĐÁNH GIÁ MÔ HÌNH)")
    print("=" * 70)

    all_results = run_training(train_file, test_file, output_dir)

    with open(config.LABEL_ENCODER_FILE, "rb") as f:
        import pickle
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ TOÀN BỘ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 70)
    comparison_df, detailed_eval = evaluate_all(all_results, class_names)

    print("\nBảng so sánh tổng hợp:")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(config.MODEL_COMPARISON_FILE, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu bảng so sánh: {config.MODEL_COMPARISON_FILE}")

    plot_comparison(comparison_df, viz_dir)

    # Vẽ confusion matrix cho mô hình tốt nhất trên bộ REALISTIC
    realistic_df = comparison_df[comparison_df["feature_set"] == "realistic"]
    best_model_name = realistic_df.iloc[0]["model"]
    best_eval = detailed_eval["realistic"][best_model_name]
    plot_confusion_matrix(best_eval["confusion_matrix"], class_names, best_model_name, viz_dir)

    best_model_obj = all_results["realistic"]["results"][best_model_name]["model"]
    best_feature_cols = all_results["realistic"]["results"][best_model_name]["feature_columns"]
    with open(config.BEST_MODEL_FILE, "wb") as f:
        import pickle
        pickle.dump({
            "model": best_model_obj,
            "model_name": best_model_name,
            "feature_columns": best_feature_cols,
            "feature_set": "realistic",
            "class_names": class_names,
        }, f)
    print(f"\nMô hình tốt nhất (trên bộ feature thực tế): {best_model_name}")
    print(f"  Accuracy={best_eval['accuracy']:.4f} | F1-macro={best_eval['f1_macro']:.4f}")
    print(f"Đã lưu mô hình tốt nhất: {config.BEST_MODEL_FILE}")

    _save_text_report(comparison_df, detailed_eval, class_names, best_model_name, config.EVALUATION_REPORT_FILE)
    print(f"Đã lưu báo cáo đánh giá đầy đủ: {config.EVALUATION_REPORT_FILE}")

    return comparison_df, detailed_eval, best_model_name


def _save_text_report(comparison_df, detailed_eval, class_names, best_model_name, output_file):
    lines = [
        "=" * 70,
        "BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI SẢN PHẨM",
        "=" * 70,
        "",
        "GHI CHÚ VỀ LABEL LEAKAGE:",
        "Bộ feature 'full' chứa các composite score (popularity_score,",
        "engagement_score, trend_momentum, value_score, deal_quality_score)",
        "đã được dùng để TẠO RA nhãn ở bước Labeling. Vì vậy kết quả trên",
        "bộ 'full' chỉ mang tính minh họa hiện tượng label leakage, KHÔNG",
        "nên dùng để đánh giá khả năng phân loại thực sự của mô hình.",
        "",
        "Bộ feature 'realistic' đã loại bỏ các score này, phản ánh đúng",
        "tình huống thực tế: dự đoán nhãn cho 1 sản phẩm mới khi chỉ có",
        "dữ liệu thô (giá, rating, số lượng bán, review...).",
        "",
        "=" * 70,
        "BẢNG SO SÁNH TỔNG HỢP",
        "=" * 70,
        comparison_df.to_string(index=False),
        "",
        "=" * 70,
        f"CHI TIẾT MÔ HÌNH TỐT NHẤT (bộ Realistic): {best_model_name}",
        "=" * 70,
        detailed_eval["realistic"][best_model_name]["classification_report"],
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    config.ensure_all_dirs()
    run_evaluation(
        train_file=config.ENCODED_TRAIN_FILE,
        test_file=config.ENCODED_TEST_FILE,
        output_dir=config.MODEL_DIR,
        viz_dir=config.VIZ_MODEL_DIR,
    )

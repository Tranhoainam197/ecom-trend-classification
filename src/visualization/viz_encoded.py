"""
viz_encoded.py
================
Trực quan hóa dữ liệu sau bước Encoding (cuối pipeline): kiểm tra phân bố
nhãn trong train/test, số lượng feature, và tương quan giữa các feature
đã chuẩn hóa với nhãn — để xác nhận dữ liệu đã sẵn sàng cho mô hình ML.

Chạy: python -m src.visualization.viz_encoded
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import matplotlib.pyplot as plt
import config

# Lấy từ config.py (nguồn DUY NHẤT) + target columns, thay vì import list
# thiếu platform/category/brand/product_name từ normalize_features.py như
# bản cũ — bug đó khiến biểu đồ "02_feature_types.png" (Hình 7.2 trong báo
# cáo) hiển thị sai "24 categorical" thay vì đúng "20 categorical" (44 thay
# vì 40 tổng feature), không khớp với số 40 feature thực sự đưa vào Chương 8.
NON_FEATURE_COLUMNS = config.NON_FEATURE_COLUMNS + config.TARGET_COLUMNS

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def visualize_encoded_data(train_file: str, test_file: str, output_dir: str):
    print("=" * 70)
    print("TRỰC QUAN HÓA DỮ LIỆU SAU ENCODING")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    with open(train_file, "r", encoding="utf-8") as f:
        df_train = pd.DataFrame(json.load(f))
    with open(test_file, "r", encoding="utf-8") as f:
        df_test = pd.DataFrame(json.load(f))
    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

    feature_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLUMNS]
    print(f"Tổng input feature: {len(feature_cols)}")

    # 1. Phân bố nhãn train vs test
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_train["label"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Phân bố nhãn - Train set", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    df_test["label"].value_counts().plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("Phân bố nhãn - Test set", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_label_train_vs_test.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Phân loại feature: scaled numerical vs one-hot categorical
    scaled_cols = [c for c in feature_cols if c.endswith("_scaled")]
    onehot_cols = [c for c in feature_cols if c not in scaled_cols]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Numerical (scaled)", "Categorical (one-hot)"], [len(scaled_cols), len(onehot_cols)],
          color=["#3498db", "#e67e22"])
    ax.set_ylabel("Số lượng cột")
    ax.set_title("Cơ cấu Feature đầu vào mô hình", fontweight="bold")
    for i, v in enumerate([len(scaled_cols), len(onehot_cols)]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_feature_types.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Phân bố vài scaled feature quan trọng
    key_scaled = [c for c in ("popularity_score_scaled", "engagement_score_scaled",
                          "trend_momentum_scaled",
                          "value_score_scaled", "deal_quality_score_scaled") if c in df_train.columns]
    if key_scaled:
        fig, axes = plt.subplots(1, len(key_scaled), figsize=(5 * len(key_scaled), 4))
        if len(key_scaled) == 1:
            axes = [axes]
        for ax, col in zip(axes, key_scaled):
            ax.hist(df_train[col], bins=40, color="slateblue", edgecolor="black", alpha=0.7)
            ax.set_title(col.replace("_scaled", "").replace("_", " ").title(), fontweight="bold")
            ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_scaled_features_distribution.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # 4. Correlation heatmap giữa các scaled feature numerical (giúp giải
    # thích vì sao model đơn giản như Decision Tree vẫn đạt hiệu năng cao:
    # nếu feature tương quan mạnh với nhau, model phức tạp không có lợi thế).
    if scaled_cols:
        corr = df_train[scaled_cols].corr()
        labels = [c.replace("_scaled", "") for c in scaled_cols]
        fig, ax = plt.subplots(figsize=(max(8, len(scaled_cols) * 0.5), max(7, len(scaled_cols) * 0.45)))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Ma trận tương quan các Numerical Feature (sau Normalization)", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_correlation_heatmap.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Đã lưu biểu đồ vào: {output_dir}")

    # Báo cáo chất lượng encoding
    report_lines = [
        "=" * 70, "BÁO CÁO CHẤT LƯỢNG ENCODING", "=" * 70, "",
        f"Train records: {len(df_train):,}",
        f"Test records : {len(df_test):,}",
        f"Tổng input feature: {len(feature_cols)}",
        f"  - Numerical (scaled)  : {len(scaled_cols)}",
        f"  - Categorical (one-hot): {len(onehot_cols)}",
        "",
        "Phân bố nhãn (Train):",
    ]
    for label, count in df_train["label"].value_counts().items():
        report_lines.append(f"  {label}: {count:,} ({count / len(df_train) * 100:.1f}%)")
    report_lines.append("\nPhân bố nhãn (Test):")
    for label, count in df_test["label"].value_counts().items():
        report_lines.append(f"  {label}: {count:,} ({count / len(df_test) * 100:.1f}%)")

    with open(os.path.join(output_dir, "encoding_quality_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("Đã lưu báo cáo: encoding_quality_report.txt")


if __name__ == "__main__":
    config.ensure_all_dirs()
    visualize_encoded_data(
        train_file=config.ENCODED_TRAIN_FILE,
        test_file=config.ENCODED_TEST_FILE,
        output_dir=config.VIZ_ENCODED_DIR,
    )
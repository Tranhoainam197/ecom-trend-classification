"""
viz_label.py
==============
Trực quan hóa phân bố nhãn (label) sau bước Labeling: tỷ lệ Hot Trend,
Best Seller, Best Deal, Normal — và mối quan hệ giữa các feature chính
theo từng nhãn.

Chạy: python -m src.visualization.viz_label
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import matplotlib.pyplot as plt

import config

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LABEL_ORDER = ["Hot Trend", "Best Seller", "Best Deal", "Normal"]
LABEL_COLORS = {"Hot Trend": "#e74c3c", "Best Seller": "#f1c40f", "Best Deal": "#2ecc71", "Normal": "#95a5a6"}


def visualize_labeled_data(input_file: str, output_dir: str):
    print("=" * 70)
    print("TRỰC QUAN HÓA DỮ LIỆU SAU LABELING")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Đã đọc {len(df):,} records")

    # 1. Phân bố nhãn
    label_counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    colors = [LABEL_COLORS[label] for label in label_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[0].set_title("Phân bố nhãn sản phẩm", fontweight="bold")
    axes[1].bar(label_counts.index, label_counts.values, color=colors, edgecolor="black")
    axes[1].set_ylabel("Số lượng sản phẩm")
    axes[1].set_title("Số lượng theo từng nhãn", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(label_counts.values):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_label_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. So sánh chỉ số chính theo nhãn
    key_metrics = ["popularity_score", "engagement_score", "value_score", "deal_quality_score"]
    available_metrics = [m for m in key_metrics if m in df.columns]
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available_metrics):
        means = df.groupby("label")[metric].mean().reindex(LABEL_ORDER, fill_value=0)
        ax.bar(means.index, means.values, color=[LABEL_COLORS[l] for l in means.index])
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_metrics_by_label.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Scatter: trend_momentum vs engagement_score theo nhãn
    if "trend_momentum" in df.columns and "engagement_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in LABEL_ORDER:
            subset = df[df["label"] == label]
            ax.scatter(subset["engagement_score"], subset["trend_momentum"],
                      label=label, color=LABEL_COLORS[label], alpha=0.5, s=15)
        ax.set_xlabel("Engagement Score")
        ax.set_ylabel("Trend Momentum")
        ax.set_title("Trend Momentum vs Engagement Score theo nhãn", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_scatter_trend_engagement.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Đã lưu biểu đồ vào: {output_dir}")


if __name__ == "__main__":
    config.ensure_all_dirs()
    visualize_labeled_data(input_file=config.LABELED_FILE, output_dir=config.VIZ_LABEL_DIR)
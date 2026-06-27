"""
viz_feature.py
=================
Trực quan hóa các đặc trưng (feature) đã được tạo ra ở bước Transformation
(feature_engineering.py): popularity_category, discount_intensity,
quality_tier, price_segment, và các composite score (popularity_score,
engagement_score, value_score, deal_quality_score).

Chạy: python -m src.visualization.viz_feature
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _plot_categorical(df: pd.DataFrame, feature: str, title: str, output_path: str, order=None):
    if feature not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df[feature].value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_numerical(df: pd.DataFrame, feature: str, title: str, output_path: str, color="royalblue"):
    if feature not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    df[feature].hist(bins=50, ax=ax, color=color, edgecolor="black", alpha=0.7)
    ax.axvline(df[feature].mean(), color="red", linestyle="--", label=f"Mean: {df[feature].mean():.1f}")
    ax.axvline(df[feature].median(), color="green", linestyle="--", label=f"Median: {df[feature].median():.1f}")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_features(df: pd.DataFrame, output_dir: str):
    """Trực quan hóa phân bố các feature chính đã tạo ra ở bước Transformation."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    _plot_categorical(df, "popularity_category", "Popularity Category",
                      os.path.join(output_dir, "01_popularity_category.png"),
                      order=["Viral", "Hot", "Trending", "Normal", "Low"])
    _plot_categorical(df, "discount_intensity", "Discount Intensity",
                      os.path.join(output_dir, "02_discount_intensity.png"),
                      order=["Heavy", "Aggressive", "Moderate", "Mild", "No Discount"])
    _plot_categorical(df, "quality_tier", "Quality Tier",
                      os.path.join(output_dir, "03_quality_tier.png"),
                      order=["Premium", "High", "Good", "Average", "Low"])
    _plot_categorical(df, "price_segment", "Price Segment",
                      os.path.join(output_dir, "04_price_segment.png"),
                      order=["Budget", "Economy", "Mid-Range", "Premium"])
    _plot_numerical(df, "popularity_score", "Popularity Score (Best Seller)",
                    os.path.join(output_dir, "05_popularity_score.png"), "green")
    _plot_numerical(df, "engagement_score", "Engagement Score (Hot Trend)",
                    os.path.join(output_dir, "06_engagement_score.png"), "red")
    _plot_numerical(df, "value_score", "Value Score (Best Deal)",
                    os.path.join(output_dir, "07_value_score.png"), "orange")
    _plot_numerical(df, "deal_quality_score", "Deal Quality Score (Best Deal)",
                    os.path.join(output_dir, "08_deal_quality_score.png"), "purple")
    _plot_numerical(df, "trend_momentum", "Trend Momentum (Hot Trend)",
                    os.path.join(output_dir, "09_trend_momentum.png"), "crimson")
    _plot_numerical(df, "sales_velocity", "Sales Velocity (sản phẩm bán/ngày)",
                    os.path.join(output_dir, "10_sales_velocity.png"), "teal")
    _plot_numerical(df, "review_velocity", "Review Velocity (review/ngày)",
                    os.path.join(output_dir, "11_review_velocity.png"), "darkorange")
    _plot_product_age(df, os.path.join(output_dir, "12_product_age.png"))

    print(f"  Đã lưu biểu đồ feature vào: {output_dir}")


def _plot_product_age(df: pd.DataFrame, output_path: str):
    """
    Biểu đồ product_age — MINH CHỨNG hạn chế dữ liệu: nếu dữ liệu chỉ crawl
    trong rất ít ngày, product_age sẽ đồng nhất 1 giá trị duy nhất
    ("Brand New"), không có biến thiên để phân nhóm tuổi sản phẩm.
    """
    if "product_age" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df["product_age"].value_counts()
    counts.plot(kind="bar", ax=ax, color="slategray")
    ax.set_title("Phân bố Product Age", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    if len(counts) == 1:
        ax.text(
            0.5, 0.5,
            f"Toàn bộ dữ liệu chỉ có 1 giá trị duy nhất: '{counts.index[0]}'\n"
            f"(do dữ liệu chỉ crawl trong rất ít ngày, không đủ\n"
            f"biến thiên thời gian để phân nhóm tuổi sản phẩm)",
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
            color="darkred", style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="darkred"),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    config.ensure_all_dirs()
    with open(config.FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    visualize_features(df, config.VIZ_FEATURE_DIR)
"""
viz_clean.py
==============
Trực quan hóa dữ liệu SAU khi Cleaning, để đánh giá chất lượng dữ liệu
đã được làm sạch (phân bố giá, rating, discount, category, brand...).

Chạy: python -m src.visualization.viz_clean
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


def visualize_clean_data(input_file: str, output_dir: str):
    print("=" * 70)
    print("TRỰC QUAN HÓA DỮ LIỆU SAU CLEANING")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Đã đọc {len(df):,} records, {len(df.columns)} cột")

    # 1. Platform
    platform_counts = df["platform"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(platform_counts.values, labels=platform_counts.index, autopct="%1.1f%%",
               colors=["#FF6B6B", "#4ECDC4", "#45B7D1"], startangle=90)
    axes[0].set_title("Phân bố theo Platform (sau Cleaning)", fontweight="bold")
    axes[1].bar(platform_counts.index, platform_counts.values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("Số lượng theo Platform", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_platform.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Giá
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["current_price"].hist(bins=100, ax=axes[0], color="#4ecdc4", edgecolor="black")
    axes[0].set_title("Phân bố giá sau Cleaning", fontweight="bold")
    axes[0].set_xlabel("Giá (VNĐ)")
    price_by_platform = [df[df["platform"] == p]["current_price"] for p in df["platform"].unique()]
    axes[1].boxplot(price_by_platform, tick_labels=list(df["platform"].unique()), showfliers=False)
    axes[1].set_title("Giá theo Platform (Boxplot)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_price.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Rating & Review
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["rating_average"].hist(bins=30, ax=axes[0], color="gold", edgecolor="black")
    axes[0].set_title("Phân bố Rating", fontweight="bold")
    axes[0].set_xlabel("Rating")
    df["num_reviews"].clip(upper=df["num_reviews"].quantile(0.95)).hist(
        bins=50, ax=axes[1], color="coral", edgecolor="black")
    axes[1].set_title("Phân bố số Review (đã cắt percentile 95%)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_rating_review.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Discount
    fig, ax = plt.subplots(figsize=(8, 5))
    df["discount_rate"].hist(bins=40, ax=ax, color="mediumpurple", edgecolor="black")
    ax.set_title("Phân bố Discount Rate (%)", fontweight="bold")
    ax.set_xlabel("Discount (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_discount.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 5. Category & Brand
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_cat = df["category"].value_counts().head(10)
    axes[0].barh(top_cat.index[::-1], top_cat.values[::-1], color="teal")
    axes[0].set_title("Top 10 Category", fontweight="bold")
    top_brand = df[df["brand"] != "No Brand"]["brand"].value_counts().head(10)
    axes[1].barh(top_brand.index[::-1], top_brand.values[::-1], color="indianred")
    axes[1].set_title("Top 10 Brand (loại trừ 'No Brand')", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_category_brand.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 6. Correlation matrix
    numeric_cols = ["current_price", "discount_rate", "rating_average", "num_reviews", "quantity_sold"]
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[numeric_cols].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Ma trận tương quan", fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_correlation_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 7. Seller location (minh chứng cột seller_location đã được Cleaning fill missing value)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_location = df["seller_location"].value_counts().head(10)
    axes[0].barh(top_location.index[::-1], top_location.values[::-1], color="darkcyan")
    axes[0].set_title("Top 10 khu vực người bán (seller_location)", fontweight="bold")
    axes[0].set_xlabel("Số lượng sản phẩm")

    unknown_pct = (df["seller_location"] == "Unknown Location").mean() * 100
    location_status = pd.Series({
        "Có vị trí xác định": 100 - unknown_pct,
        "Unknown Location\n(đã fill ở bước Cleaning)": unknown_pct,
    })
    axes[1].pie(location_status.values, labels=location_status.index, autopct="%1.1f%%",
               colors=["#3498db", "#e74c3c"], startangle=90)
    axes[1].set_title("Tỷ lệ seller_location đã fill missing value", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_seller_location.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu 7 biểu đồ vào: {output_dir}")


if __name__ == "__main__":
    config.ensure_all_dirs()
    visualize_clean_data(input_file=config.CLEAN_FILE, output_dir=config.VIZ_CLEAN_DIR)
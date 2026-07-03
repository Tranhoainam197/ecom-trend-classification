"""
viz_raw.py
============
Trực quan hóa dữ liệu THÔ (trước khi Cleaning), giúp phát hiện vấn đề
chất lượng dữ liệu cần xử lý: giá trị NULL, phân bố theo platform/category,
giá, rating, discount...

Chạy: python -m src.visualization.viz_raw
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _clean_price_for_viz(price_str):
    """Parse giá thô CHỈ để vẽ biểu đồ (không phải bước Cleaning chính thức)."""
    if pd.isna(price_str):
        return None
    try:
        cleaned = str(price_str).replace("₫", "").replace(".", "").replace(",", "").strip()
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _add_bar_labels(ax, bars, fmt="{:,.0f}"):
    """Ghi số liệu lên đầu mỗi cột của bar chart (trục dọc)."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            fmt.format(height),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )


def _add_barh_labels(ax, bars, fmt="{:,.0f}"):
    """Ghi số liệu ở cuối mỗi cột của bar chart ngang (barh)."""
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width, bar.get_y() + bar.get_height() / 2,
            f" {fmt.format(width)}",
            ha="left", va="center", fontsize=8, fontweight="bold",
        )


def visualize_raw_data(input_file: str, output_dir: str):
    print("=" * 70)
    print("TRỰC QUAN HÓA DỮ LIỆU THÔ (trước Cleaning)")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["price_numeric"] = df["price"].apply(_clean_price_for_viz)
    print(f"Đã đọc {len(df):,} records, {len(df.columns)} cột")

    # 1. NULL values
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#ff6b6b" if v > 50 else "#ffd93d" if v > 20 else "#6bcf7f" for v in null_pct]
    bars = ax.barh(null_pct.index, null_pct.values, color=colors)
    _add_barh_labels(ax, bars, fmt="{:.1f}%")
    ax.set_xlabel("Phần trăm giá trị NULL (%)")
    ax.set_title("Tỷ lệ giá trị NULL theo từng trường dữ liệu thô", fontweight="bold")
    ax.set_xlim(0, max(null_pct.max() * 1.15, 10))
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_null_values.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Phân bố platform
    platform_counts = df["platform"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(platform_counts.values, labels=platform_counts.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Phân bố theo Platform", fontweight="bold")
    bars = axes[1].bar(platform_counts.index, platform_counts.values,
                        color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    _add_bar_labels(axes[1], bars)
    axes[1].set_ylabel("Số lượng sản phẩm")
    axes[1].set_title("Số lượng sản phẩm theo Platform", fontweight="bold")
    axes[1].set_ylim(0, platform_counts.max() * 1.15)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_platform_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Top category
    top_categories = df["category_name"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top_categories.index[::-1], top_categories.values[::-1], color="steelblue")
    _add_barh_labels(ax, bars)
    ax.set_xlabel("Số lượng sản phẩm")
    ax.set_title("Top 15 danh mục sản phẩm phổ biến nhất", fontweight="bold")
    ax.set_xlim(0, top_categories.max() * 1.15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_category_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Phân bố giá
    prices = df["price_numeric"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(prices, bins=100, color="#4ecdc4", edgecolor="black")
    axes[0].set_title("Phân bố giá (toàn bộ)", fontweight="bold")
    axes[0].set_xlabel("Giá (VNĐ)")
    axes[1].hist(prices[prices < 20_000_000], bins=100, color="#ff6b6b", edgecolor="black")
    axes[1].set_title("Phân bố giá (< 20 triệu VNĐ)", fontweight="bold")
    axes[1].set_xlabel("Giá (VNĐ)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_price_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nĐã lưu 4 biểu đồ vào: {output_dir}")

    # Báo cáo chất lượng dữ liệu
    _save_quality_report(df, null_pct, output_dir)


def _save_quality_report(df: pd.DataFrame, null_pct: pd.Series, output_dir: str):
    lines = ["=" * 70, "BÁO CÁO CHẤT LƯỢNG DỮ LIỆU THÔ", "=" * 70, ""]
    lines.append(f"Tổng số records: {len(df):,}")
    lines.append(f"Tổng số cột: {len(df.columns)}")
    lines.append(f"Platforms: {', '.join(df['platform'].unique())}")
    lines.append("")
    lines.append("Top cột có tỷ lệ NULL cao nhất:")
    for col, pct in null_pct.head(10).items():
        lines.append(f"  - {col}: {pct:.1f}%")

    duplicate_count = df.duplicated(subset=["id", "platform"], keep=False).sum()
    lines.append(f"\nSố record có (id, platform) trùng lặp: {duplicate_count:,}")

    with open(os.path.join(output_dir, "data_quality_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Đã lưu báo cáo chất lượng dữ liệu: data_quality_report.txt")


if __name__ == "__main__":
    config.ensure_all_dirs()
    visualize_raw_data(input_file=config.RAW_MERGED_FILE, output_dir=config.VIZ_RAW_DIR)
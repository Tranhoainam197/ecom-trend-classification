"""
outlier_handler.py
====================
Xử lý các giá trị ngoại lệ (outlier) trong dữ liệu sản phẩm đã được
chuẩn hóa schema. Thuộc bước 1 - CLEANING.

Mỗi hàm handle_* xử lý outlier cho một trường dữ liệu riêng, dựa trên
percentile và logic nghiệp vụ (vd: số review không thể nhiều hơn số đã bán).
Toàn bộ thay đổi được ghi log để phục vụ báo cáo và trực quan hóa so sánh.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class OutlierHandler:
    """Phát hiện và xử lý outlier theo từng trường, có log lại số liệu trước/sau."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_before_outlier_removal = df.copy()
        self.change_log: list[dict] = []

    def _log(self, step: str, before: int, after: int, detail: str = ""):
        removed = before - after
        rate = (removed / before * 100) if before else 0.0
        self.change_log.append({
            "step": step,
            "records_before": before,
            "records_after": after,
            "records_removed": removed,
            "removal_rate_pct": round(rate, 2),
            "detail": detail,
        })
        print(f"  [{step}] {before:,} -> {after:,} (loại {removed:,}, {rate:.2f}%) {detail}")

    def handle_price(self, upper_percentile: float = 99.5, min_valid_price: float = 1000) -> "OutlierHandler":
        """Loại giá quá thấp (< min_valid_price, khả năng lỗi) và quá cao (trên percentile)."""
        before = len(self.df)
        upper_bound = self.df["current_price"].quantile(upper_percentile / 100)

        self.df = self.df[
            (self.df["current_price"] >= min_valid_price) &
            (self.df["current_price"] <= upper_bound)
        ]
        self._log(
            "Outlier giá", before, len(self.df),
            f"giữ [{min_valid_price:,.0f}, {upper_bound:,.0f}] VNĐ (percentile {upper_percentile}%)",
        )
        return self

    def handle_quantity_sold(self, upper_percentile: float = 99.0) -> "OutlierHandler":
        """Loại số lượng bán bất thường cao (trên percentile)."""
        has_value = self.df["quantity_sold"].notna()
        if not has_value.any():
            return self

        before = len(self.df)
        upper_bound = self.df.loc[has_value, "quantity_sold"].quantile(upper_percentile / 100)
        outlier_mask = has_value & (self.df["quantity_sold"] > upper_bound)

        self.df = self.df[~outlier_mask]
        self._log(
            "Outlier số lượng bán", before, len(self.df),
            f"loại quantity_sold > {upper_bound:,.0f} (percentile {upper_percentile}%)",
        )
        return self

    def handle_discount(self, max_discount: float = 80) -> "OutlierHandler":
        """Loại discount bất thường cao (thường do lỗi hiển thị hoặc mánh marketing)."""
        has_value = self.df["discount_rate"].notna()
        if not has_value.any():
            return self

        before = len(self.df)
        outlier_mask = has_value & (self.df["discount_rate"] > max_discount)

        self.df = self.df[~outlier_mask]
        self._log("Outlier discount", before, len(self.df), f"loại discount_rate > {max_discount}%")
        return self

    def handle_rating(self) -> "OutlierHandler":
        """Loại rating = 0 (thường là sản phẩm chưa có đánh giá thực sự, không phải rating thật)."""
        has_value = self.df["rating_average"].notna()
        if not has_value.any():
            return self

        before = len(self.df)
        zero_rating_mask = has_value & (self.df["rating_average"] == 0)

        self.df = self.df[~zero_rating_mask]
        self._log("Outlier rating", before, len(self.df), "loại rating_average = 0")
        return self

    def handle_num_reviews(
        self,
        upper_percentile: float = 99.7,
        max_review_per_sold_ratio: float = 1.0,
        strategy: str = "drop",
    ) -> "OutlierHandler":
        """
        Xử lý outlier số lượng review theo logic nghiệp vụ:
        review không hợp lý nếu vượt percentile HOẶC vượt quantity_sold * ratio
        (vì số người review không thể nhiều hơn số người đã mua).
        """
        mask_sold = (
            self.df["num_reviews"].notna() &
            self.df["quantity_sold"].notna() &
            (self.df["quantity_sold"] > 0)
        )
        if not mask_sold.any():
            return self

        before = len(self.df)
        upper_bound = self.df.loc[mask_sold, "num_reviews"].quantile(upper_percentile / 100)
        max_allowed = self.df["quantity_sold"] * max_review_per_sold_ratio

        outlier_mask = mask_sold & (
            (self.df["num_reviews"] > upper_bound) | (self.df["num_reviews"] > max_allowed)
        )

        if not outlier_mask.any():
            return self

        if strategy == "drop":
            self.df = self.df[~outlier_mask]
            self._log(
                "Outlier num_reviews", before, len(self.df),
                f"drop review > min(percentile {upper_percentile}%, sold×{max_review_per_sold_ratio})",
            )
        else:
            cap_value = np.minimum(upper_bound, max_allowed)
            self.df.loc[outlier_mask, "num_reviews"] = cap_value[outlier_mask].astype("int64")
            self._log("Outlier num_reviews", before, len(self.df), "cap về ngưỡng hợp lệ (không drop)")

        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df

    def save_comparison_report(self, output_dir: str):
        """Lưu biểu đồ so sánh trước/sau xử lý outlier + bảng tóm tắt CSV."""
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].hist(self.df_before_outlier_removal["current_price"], bins=100,
                        alpha=0.5, label="Trước xử lý", color="red")
        axes[0, 0].hist(self.df["current_price"], bins=100,
                        alpha=0.5, label="Sau xử lý", color="green")
        axes[0, 0].set(xlabel="Giá (VNĐ)", ylabel="Số lượng", title="So sánh phân bố giá")
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].boxplot(
            [self.df_before_outlier_removal["current_price"].dropna(), self.df["current_price"].dropna()],
            tick_labels=["Trước xử lý", "Sau xử lý"],
        )
        axes[0, 1].set(ylabel="Giá (VNĐ)", title="Boxplot so sánh giá")
        axes[0, 1].grid(alpha=0.3)

        before_disc = self.df_before_outlier_removal["discount_rate"].dropna()
        after_disc = self.df["discount_rate"].dropna()
        if len(before_disc) and len(after_disc):
            axes[1, 0].hist(before_disc, bins=50, alpha=0.5, label="Trước xử lý", color="red")
            axes[1, 0].hist(after_disc, bins=50, alpha=0.5, label="Sau xử lý", color="green")
            axes[1, 0].set(xlabel="Discount (%)", ylabel="Số lượng", title="So sánh phân bố discount")
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        top_cats_before = self.df_before_outlier_removal["category"].value_counts().head(10)
        top_cats_after = self.df["category"].value_counts()
        x = np.arange(len(top_cats_before))
        width = 0.35
        axes[1, 1].bar(x - width / 2, top_cats_before.values, width, label="Trước xử lý", color="red", alpha=0.8)
        axes[1, 1].bar(x + width / 2, [top_cats_after.get(c, 0) for c in top_cats_before.index],
                       width, label="Sau xử lý", color="green", alpha=0.8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(
            [c[:20] + "..." if len(c) > 20 else c for c in top_cats_before.index], rotation=45, ha="right"
        )
        axes[1, 1].set(ylabel="Số lượng", title="So sánh Top 10 Category")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "outlier_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

        pd.DataFrame(self.change_log).to_csv(
            os.path.join(output_dir, "outlier_handling_summary.csv"), index=False, encoding="utf-8-sig"
        )

        print(f"  Đã lưu báo cáo outlier vào: {output_dir}")
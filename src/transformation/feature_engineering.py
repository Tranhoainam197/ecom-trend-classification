"""
feature_engineering.py
========================
BƯỚC 3a - TRANSFORMATION (Feature Engineering)
================================================
Tạo các đặc trưng (feature) mới từ dữ liệu đã được Integration, phục vụ
việc phân loại sản phẩm thành 4 nhóm: Hot Trend, Best Seller, Best Deal,
Normal (bước labeling.py kế tiếp).

Nhóm feature được tạo:
    1. Time features      : days_active, product_age
    2. Velocity features  : sales_velocity, review_velocity (và bản chuẩn hóa 0-100)
    3. Price features      : original_price, absolute_saving, price_per_rating
    4. Composite scores    : popularity_score, engagement_score, trend_momentum,
                              value_score, deal_quality_score
    5. Categorical features: popularity_category, price_segment, quality_tier,
                              discount_intensity
    6. Category context     : category_popularity_rank, category_price_percentile

LƯU Ý QUAN TRỌNG VỀ HẠN CHẾ DỮ LIỆU:
Nếu dữ liệu chỉ crawl trong rất ít ngày (<=5 ngày khác nhau), trường
`product_age` sẽ bị cố định "Brand New" cho toàn bộ dataset (vì không đủ
biến thiên thời gian để phân nhóm tuổi sản phẩm có ý nghĩa). Khi đó
`trend_momentum` chỉ còn là `engagement_score x hệ_số_cố_định`, không mang
thêm thông tin mới — đây là hạn chế của DỮ LIỆU, không phải lỗi logic, và
cần được ghi rõ trong báo cáo đồ án.

Chạy: python -m src.transformation.feature_engineering
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

import config
from src.visualization.viz_feature import visualize_features

REQUIRED_INPUT_COLUMNS = [
    "current_price", "discount_rate", "rating_average",
    "num_reviews", "quantity_sold", "crawl_date",
]


class FeatureEngineer:
    """Sinh các đặc trưng phục vụ phân vùng sản phẩm Hot Trend / Best Seller / Best Deal."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc cho feature engineering: {missing}")

    def run(self) -> pd.DataFrame:
        print("[1/6] Tính time features (days_active, product_age)...")
        self._add_time_features()

        print("[2/6] Tính velocity features (sales_velocity, review_velocity)...")
        self._add_velocity_features()

        print("[3/6] Tính price features (original_price, absolute_saving, price_per_rating)...")
        self._add_price_features()

        print("[4/6] Tính composite scores (popularity, engagement, trend, value, deal_quality)...")
        self._add_composite_scores()

        print("[5/6] Tính categorical features (price_segment, quality_tier, ...)...")
        self._add_categorical_features()

        print("[6/6] Tính category context features (rank/percentile theo category)...")
        self._add_category_context()

        self._select_final_columns()
        return self.df

    # ------------------------------------------------------------------ #
    # 1. TIME FEATURES
    # ------------------------------------------------------------------ #
    def _add_time_features(self):
        self.df["crawl_date"] = pd.to_datetime(self.df["crawl_date"])
        max_date = self.df["crawl_date"].max()
        self.df["days_active"] = (max_date - self.df["crawl_date"]).dt.days + 1

        unique_days = self.df["crawl_date"].nunique()
        if unique_days <= 5:
            print(f"  Cảnh báo: chỉ có {unique_days} ngày crawl khác nhau -> "
                  f"product_age cố định 'Brand New' (không đủ biến thiên thời gian)")
            self.df["product_age"] = "Brand New"
        else:
            self.df["product_age"] = pd.cut(
                self.df["days_active"],
                bins=[0, 7, 30, 90, 365, float("inf")],
                labels=["Brand New", "New", "Recent", "Established", "Mature"],
                include_lowest=True,
            )

    # ------------------------------------------------------------------ #
    # 2. VELOCITY FEATURES
    # ------------------------------------------------------------------ #
    def _add_velocity_features(self):
        self.df["sales_velocity"] = (self.df["quantity_sold"] / self.df["days_active"]).round(2)
        max_sales_vel = self.df["sales_velocity"].quantile(0.98)
        self.df["sales_velocity_normalized"] = (
            (self.df["sales_velocity"] / max_sales_vel) * 100
        ).clip(0, 100).round(2)

        self.df["review_velocity"] = (self.df["num_reviews"] / self.df["days_active"]).round(2)
        max_review_vel = self.df["review_velocity"].quantile(0.98)
        self.df["review_velocity_normalized"] = (
            (self.df["review_velocity"] / max_review_vel) * 100
        ).clip(0, 100).round(2)

    # ------------------------------------------------------------------ #
    # 3. PRICE FEATURES
    # ------------------------------------------------------------------ #
    def _add_price_features(self):
        self.df["original_price"] = np.where(
            self.df["discount_rate"] > 0,
            self.df["current_price"] / (1 - self.df["discount_rate"] / 100),
            self.df["current_price"],
        ).round(2)

        self.df["absolute_saving"] = (self.df["original_price"] - self.df["current_price"]).round(2)

        self.df["price_per_rating"] = np.where(
            self.df["rating_average"] > 0,
            self.df["current_price"] / self.df["rating_average"],
            np.nan,
        ).round(2)

    # ------------------------------------------------------------------ #
    # 4. COMPOSITE SCORES
    # ------------------------------------------------------------------ #
    def _add_composite_scores(self):
        self._add_popularity_score()
        self._add_engagement_score()
        self._add_trend_momentum()
        self._add_discount_intensity_score()
        self._add_value_score()
        self._add_deal_quality_score()

    def _add_popularity_score(self):
        """popularity_score (0-100): quantity_sold 50% + reviews 30% + rating 20%. Dùng cho Best Seller."""
        max_sold = self.df["quantity_sold"].quantile(0.98)
        sold_score = (self.df["quantity_sold"] / max_sold * 100).clip(0, 100)

        max_reviews = self.df["num_reviews"].quantile(0.98)
        review_score = (self.df["num_reviews"] / max_reviews * 100).clip(0, 100)

        rating_score = (self.df["rating_average"] / 5.0 * 100).fillna(0)

        self.df["popularity_score"] = (
            sold_score * 0.50 + review_score * 0.30 + rating_score * 0.20
        ).round(2)

    def _add_engagement_score(self):
        """engagement_score (0-100): review_velocity 40% + sales_velocity 40% + rating 20%. Dùng cho Hot Trend."""
        rating_score = (self.df["rating_average"] / 5.0 * 100).fillna(0)
        self.df["engagement_score"] = (
            self.df["review_velocity_normalized"] * 0.40 +
            self.df["sales_velocity_normalized"] * 0.40 +
            rating_score * 0.20
        ).round(2)

    def _add_trend_momentum(self):
        """trend_momentum: engagement_score nhân hệ số theo độ mới của sản phẩm. Dùng cho Hot Trend."""
        age_factor_map = {
            "Brand New": 1.5, "New": 1.3, "Recent": 1.0, "Established": 0.7, "Mature": 0.5,
        }
        age_factor = self.df["product_age"].map(age_factor_map).astype(float)

        if age_factor.nunique() == 1:
            print(f"  Hạn chế dữ liệu: product_age đồng nhất -> age_factor cố định "
                  f"({age_factor.iloc[0]}), trend_momentum = engagement_score x {age_factor.iloc[0]}")

        self.df["trend_momentum"] = (self.df["engagement_score"] * age_factor).clip(0, 150).round(2)

    def _add_discount_intensity_score(self):
        """discount_intensity (nhãn) + discount_score (0-100, chuẩn hóa theo percentile)."""
        def categorize(rate):
            if pd.isna(rate) or rate < 5:
                return "No Discount"
            if rate < 15:
                return "Mild"
            if rate < 30:
                return "Moderate"
            if rate < 50:
                return "Aggressive"
            return "Heavy"

        self.df["discount_intensity"] = self.df["discount_rate"].apply(categorize)

        max_discount = self.df["discount_rate"].quantile(0.98)
        self.df["discount_score"] = (
            (self.df["discount_rate"] / max_discount) * 100
        ).clip(0, 100).round(2)

    def _relative_rating_score(self) -> pd.Series:
        """
        Rating tương đối so với median trong cùng category (z-score), thay vì
        raw_rating/5. Lý do: rating trung bình thực tế thường rất cao (~4.7-4.8)
        nên raw/5 cho gần như mọi sản phẩm điểm ~95, mất khả năng phân biệt.
        Dùng z-score theo category để chỉ "khen" sản phẩm thực sự tốt hơn mặt
        bằng chung, tránh thiên lệch (bias) khi gán nhãn Best Deal.
        """
        if "category" not in self.df.columns:
            return (self.df["rating_average"] / 5.0 * 100).fillna(0)

        cat_median = self.df.groupby("category")["rating_average"].transform("median")
        cat_std = self.df.groupby("category")["rating_average"].transform("std").clip(lower=0.01)
        z = (self.df["rating_average"] - cat_median) / cat_std
        return ((z * 20) + 50).clip(0, 100).fillna(50)

    def _add_value_score(self):
        """
        value_score (0-100): discount 45% + rating tương đối 20% + giá cạnh tranh
        trong category 35%. Dùng cho Best Deal.
        """
        discount_component = self.df["discount_score"]
        rating_component = self._relative_rating_score()

        if "category" in self.df.columns:
            price_percentile_in_cat = self.df.groupby("category")["current_price"].rank(pct=True)
        else:
            price_percentile_in_cat = self.df["current_price"].rank(pct=True)
        price_competitiveness = ((1 - price_percentile_in_cat) * 100).clip(0, 100)

        self.df["value_score"] = (
            discount_component * 0.45 + rating_component * 0.20 + price_competitiveness * 0.35
        ).round(2)

    def _add_deal_quality_score(self):
        """
        deal_quality_score (0-100): tiết kiệm tuyệt đối 45% + rating tương đối 20%
        + độ tin cậy review 20% + discount gate bonus 15%.
        HARD GATE: nếu discount_rate < 15% thì deal_quality_score = 0, vì sản phẩm
        không giảm giá đáng kể thì không thể coi là "Best Deal" dù các yếu tố khác cao.
        """
        max_saving = self.df["absolute_saving"].quantile(0.98)
        saving_score = (self.df["absolute_saving"] / max_saving * 100).clip(0, 100)

        rating_score = self._relative_rating_score()

        max_reviews = self.df["num_reviews"].quantile(0.98)
        credibility_score = (self.df["num_reviews"] / max_reviews * 100).clip(0, 100)

        discount_gate = np.select(
            [self.df["discount_rate"] >= 30, self.df["discount_rate"] >= 15],
            [100, 50],
            default=0,
        )

        self.df["deal_quality_score"] = (
            saving_score * 0.45 + rating_score * 0.20 + credibility_score * 0.20 + discount_gate * 0.15
        ).round(2)

        self.df.loc[self.df["discount_rate"] < 15, "deal_quality_score"] = 0.0

    # ------------------------------------------------------------------ #
    # 5. CATEGORICAL FEATURES (hỗ trợ)
    # ------------------------------------------------------------------ #
    def _add_categorical_features(self):
        self._add_popularity_category()
        self._add_price_segment()
        self._add_quality_tier()

    def _add_popularity_category(self):
        """Phân loại độ phổ biến dựa trên percentile của popularity_score."""
        p90, p75, p50, p25 = (self.df["popularity_score"].quantile(q) for q in (0.90, 0.75, 0.50, 0.25))

        def categorize(score):
            if score >= p90:
                return "Viral"
            if score >= p75:
                return "Hot"
            if score >= p50:
                return "Trending"
            if score >= p25:
                return "Normal"
            return "Low"

        self.df["popularity_category"] = self.df["popularity_score"].apply(categorize)

    def _add_price_segment(self):
        """Phân khúc giá dựa trên quartile của current_price."""
        q1, q2, q3 = (self.df["current_price"].quantile(q) for q in (0.25, 0.50, 0.75))

        def categorize(price):
            if price <= q1:
                return "Budget"
            if price <= q2:
                return "Economy"
            if price <= q3:
                return "Mid-Range"
            return "Premium"

        self.df["price_segment"] = self.df["current_price"].apply(categorize)

    def _add_quality_tier(self):
        """Phân tầng chất lượng dựa trên rating và số lượng review."""
        median_reviews = self.df["num_reviews"].median()

        def categorize(row):
            rating, reviews = row["rating_average"], row["num_reviews"]
            if rating >= 4.5 and reviews >= median_reviews:
                return "Premium"
            if rating >= 4.0:
                return "High"
            if rating >= 3.5:
                return "Good"
            if rating >= 3.0:
                return "Average"
            return "Low"

        self.df["quality_tier"] = self.df.apply(categorize, axis=1)

    # ------------------------------------------------------------------ #
    # 6. CATEGORY CONTEXT
    # ------------------------------------------------------------------ #
    def _add_category_context(self):
        if "category" not in self.df.columns:
            self.df["category_popularity_rank"] = 50.0
            self.df["category_price_percentile"] = 50.0
            return

        category_total_sold = self.df.groupby("category")["quantity_sold"].sum()
        category_rank = category_total_sold.rank(pct=True) * 100
        self.df["category_popularity_rank"] = self.df["category"].map(category_rank).round(2)

        self.df["category_price_percentile"] = (
            self.df.groupby("category")["current_price"].rank(pct=True) * 100
        ).round(2)

    # ------------------------------------------------------------------ #
    # FINAL SELECTION
    # ------------------------------------------------------------------ #
    def _select_final_columns(self):
        metadata = ["id", "crawl_date", "platform"]
        context = [c for c in ("category", "brand", "product_name") if c in self.df.columns]

        raw_numerical = [
            "current_price", "original_price", "absolute_saving",
            "discount_rate", "rating_average", "num_reviews", "quantity_sold", "days_active",
        ]
        engineered_numerical = [
            "sales_velocity", "sales_velocity_normalized",
            "review_velocity", "review_velocity_normalized",
            "popularity_score", "engagement_score", "trend_momentum",
            "discount_score", "value_score", "deal_quality_score",
            "category_popularity_rank", "category_price_percentile",
        ]
        categorical = [
            "popularity_category", "price_segment", "quality_tier",
            "discount_intensity", "product_age",
        ]

        final_columns = metadata + context + raw_numerical + engineered_numerical + categorical
        final_columns = [c for c in final_columns if c in self.df.columns]
        self.df = self.df[final_columns]


def run_feature_engineering(input_file: str, output_file: str, visualize: bool = True) -> pd.DataFrame:
    print("=" * 70)
    print("BƯỚC 3a: TRANSFORMATION - FEATURE ENGINEERING")
    print("=" * 70)

    print(f"\nĐọc dữ liệu đã integration: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Đã đọc {len(df):,} records, {len(df.columns)} cột\n")

    df_features = FeatureEngineer(df).run()
    print(f"\nHoàn thành. Tổng cộng {len(df_features.columns)} cột sau feature engineering")

    for col in ("crawl_date",):
        if col in df_features.columns and pd.api.types.is_datetime64_any_dtype(df_features[col]):
            df_features[col] = df_features[col].astype(str)
    for col in df_features.select_dtypes(include=["category"]).columns:
        df_features[col] = df_features[col].astype(str)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(df_features.to_dict("records"), f, ensure_ascii=False, indent=2)
    print(f"Đã lưu: {output_file}")

    if visualize:
        visualize_features(df_features, config.VIZ_FEATURE_DIR)

    return df_features


if __name__ == "__main__":
    config.ensure_all_dirs()
    run_feature_engineering(input_file=config.INTEGRATED_FILE, output_file=config.FEATURES_FILE)
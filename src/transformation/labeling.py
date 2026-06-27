"""
labeling.py
=============
BƯỚC 3b - TRANSFORMATION (Labeling)
=====================================
Gán nhãn phân loại sản phẩm thành 4 nhóm dựa trên các feature đã tạo ở
feature_engineering.py:

    Hot Trend   : sản phẩm đang viral, tăng trưởng nhanh
    Best Seller : sản phẩm bán chạy, ổn định
    Best Deal   : sản phẩm có ưu đãi/giá trị tốt
    Normal      : sản phẩm bình thường (mặc định)

Chiến lược: HYBRID (rule-based seeds + ML model)
    1. Tính ngưỡng (threshold) từ percentile của từng feature
    2. Gán "seed label" cho các record thỏa quy tắc NGẶT (high precision) —
       đây là các trường hợp rõ ràng, ít gây tranh cãi
    3. Train RandomForest/LogisticRegression từ các seed label
    4. Dùng model dự đoán cho phần dữ liệu còn lại (chưa có seed), chỉ nhận
       dự đoán khi độ tin cậy (probability) đủ cao, còn lại gán Normal
    5. Áp giới hạn (cap) tỷ lệ tối đa cho nhóm Best Deal để tránh lệch nhãn
    6. Nếu việc train model thất bại (thiếu sklearn, không đủ seed...),
       fallback về rule-based đầy đủ cho toàn bộ dataset

Lý do chọn hybrid thay vì rule-based thuần: rule cứng dễ tạo ra biên giới
"gãy" (discontinuity) giữa các nhóm ở đúng ngưỡng percentile. Model học
được ranh giới mượt hơn từ chính các trường hợp rõ ràng nhất.

Chạy: python -m src.transformation.labeling
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

import config

REQUIRED_FEATURES = {
    "hot_trend": ["trend_momentum", "engagement_score", "product_age"],
    "best_seller": ["popularity_score", "sales_velocity_normalized", "quality_tier"],
    "best_deal": ["deal_quality_score", "value_score", "discount_intensity"],
}

NUMERIC_MODEL_FEATURES = [
    "popularity_score", "engagement_score", "trend_momentum",
    "value_score", "deal_quality_score",
    "sales_velocity", "sales_velocity_normalized",
    "review_velocity", "review_velocity_normalized",
    "discount_score", "category_popularity_rank", "category_price_percentile",
    "quantity_sold", "rating_average", "num_reviews",
    "current_price", "discount_rate", "absolute_saving", "days_active",
]
CATEGORICAL_MODEL_FEATURES = [
    "popularity_category", "discount_intensity", "quality_tier", "price_segment", "product_age",
]

MAX_BEST_DEAL_RATIO = 0.30  # Best Deal tối đa 30% tổng dataset, tránh gán nhãn quá rộng


class ProductLabeler:
    """Gán nhãn phân vùng sản phẩm theo chiến lược hybrid (rule seeds + ML)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.thresholds: dict = {}
        self._warn_missing_features()

    def _warn_missing_features(self):
        for group, features in REQUIRED_FEATURES.items():
            missing = [f for f in features if f not in self.df.columns]
            if missing:
                print(f"  Cảnh báo: thiếu feature {missing} cho nhóm '{group}' -> "
                      f"labeling nhóm này có thể kém chính xác")

    def label(
        self,
        use_model: bool = True,
        prob_threshold: float = 0.70,
        min_seed_per_class: int = 50,
        model_type: str = "random_forest",
    ) -> pd.DataFrame:
        print("[1/4] Tính ngưỡng (threshold) từ percentile...")
        self._calculate_thresholds()

        print("[2/4] Gán seed labels (quy tắc chặt, độ chính xác cao)...")
        self._assign_seed_labels()

        success = False
        if use_model:
            print("[3/4] Train model từ seeds & dự đoán cho phần còn lại...")
            try:
                success = self._train_and_predict(prob_threshold, min_seed_per_class, model_type)
            except ImportError as e:
                print(f"  {e}")

        if not success:
            print("[3/4] Fallback: gán nhãn rule-based đầy đủ cho toàn bộ dataset...")
            self._assign_labels_rule_based()
            self.df["label_source"] = "rule_full"

        print("[4/4] Thống kê kết quả labeling...")
        self._print_label_distribution()
        return self.df

    # ------------------------------------------------------------------ #
    # THRESHOLDS
    # ------------------------------------------------------------------ #
    def _calculate_thresholds(self):
        def q(col, *percentiles):
            if col not in self.df.columns:
                return {f"p{int(p * 100)}": 0 for p in percentiles}
            return {f"p{int(p * 100)}": self.df[col].quantile(p) for p in percentiles}

        self.thresholds = {
            "hot_trend": {
                **{f"trend_momentum_{k}": v for k, v in q("trend_momentum", 0.70, 0.75, 0.80).items()},
                **{f"engagement_score_{k}": v for k, v in q("engagement_score", 0.65, 0.70).items()},
                "new_age_categories": ["Brand New", "New", "Recent"],
            },
            "best_seller": {
                **{f"popularity_score_{k}": v for k, v in q("popularity_score", 0.55, 0.60, 0.65, 0.70).items()},
                **{f"sales_velocity_{k}": v for k, v in q("sales_velocity_normalized", 0.65, 0.70).items()},
                "high_quality_tiers": ["Premium", "High", "Good", "Average"],
            },
            "best_deal": {
                **{f"deal_quality_score_{k}": v for k, v in q("deal_quality_score", 0.70, 0.75, 0.80, 0.85).items()},
                **{f"value_score_{k}": v for k, v in q("value_score", 0.65, 0.70, 0.75).items()},
                "strong_discount_levels": ["Aggressive", "Heavy", "Moderate"],
            },
        }

    # ------------------------------------------------------------------ #
    # SEED LABELS (high precision rules)
    # ------------------------------------------------------------------ #
    def _assign_seed_labels(self):
        ht, bs, bd = self.thresholds["hot_trend"], self.thresholds["best_seller"], self.thresholds["best_deal"]

        def seed_rule(row):
            trend_momentum = row.get("trend_momentum", np.nan)
            product_age = row.get("product_age")
            popularity_score = row.get("popularity_score", np.nan)
            sales_velocity = row.get("sales_velocity_normalized", np.nan)
            quality_tier = row.get("quality_tier")
            deal_quality_score = row.get("deal_quality_score", np.nan)
            value_score = row.get("value_score", np.nan)
            discount_intensity = row.get("discount_intensity")

            # Hot Trend: rất cao momentum, hoặc cao momentum + sản phẩm mới
            if pd.notna(trend_momentum):
                if trend_momentum >= ht["trend_momentum_p80"]:
                    return "Hot Trend"
                if trend_momentum >= ht["trend_momentum_p75"] and product_age in ["Brand New", "New"]:
                    return "Hot Trend"

            # Best Seller: phổ biến cao + chất lượng tốt, theo nhiều tier
            if pd.notna(popularity_score):
                if popularity_score >= bs["popularity_score_p70"] and quality_tier in ["Premium", "High"]:
                    return "Best Seller"
                if (popularity_score >= bs["popularity_score_p65"] and
                        sales_velocity >= bs["sales_velocity_p65"] and
                        quality_tier in bs["high_quality_tiers"]):
                    return "Best Seller"
                if popularity_score >= bs["popularity_score_p65"]:
                    return "Best Seller"

            # Best Deal: deal_quality cao, có gate discount
            if pd.notna(deal_quality_score):
                if deal_quality_score >= bd["deal_quality_score_p85"]:
                    return "Best Deal"
                if (deal_quality_score >= bd["deal_quality_score_p80"] and
                        discount_intensity in ["Aggressive", "Heavy", "Moderate"]):
                    return "Best Deal"
                if deal_quality_score >= bd["deal_quality_score_p75"] and value_score >= bd["value_score_p70"]:
                    return "Best Deal"

            return np.nan

        self.df["seed_label"] = self.df.apply(seed_rule, axis=1)
        self.df["label_source"] = np.where(self.df["seed_label"].notna(), "rule_seed", "unlabeled")

        counts = self.df["seed_label"].value_counts(dropna=True)
        print("  Phân bố seed:")
        for label, count in counts.items():
            print(f"    {label}: {count:,} ({count / len(self.df) * 100:.1f}%)")

    # ------------------------------------------------------------------ #
    # FULL RULE-BASED (fallback)
    # ------------------------------------------------------------------ #
    def _assign_labels_rule_based(self):
        ht, bs, bd = self.thresholds["hot_trend"], self.thresholds["best_seller"], self.thresholds["best_deal"]

        def categorize(row):
            trend_momentum = row.get("trend_momentum", 0)
            engagement_score = row.get("engagement_score", 0)
            product_age = row.get("product_age", "")
            popularity_score = row.get("popularity_score", 0)
            quality_tier = row.get("quality_tier", "")
            deal_quality_score = row.get("deal_quality_score", 0)
            value_score = row.get("value_score", 0)
            discount_intensity = row.get("discount_intensity", "")

            # Priority 1: Hot Trend
            if trend_momentum >= ht["trend_momentum_p80"]:
                return "Hot Trend"
            if trend_momentum >= ht["trend_momentum_p75"] and product_age in ht["new_age_categories"]:
                return "Hot Trend"
            if engagement_score >= ht["engagement_score_p70"] and product_age in ["Brand New", "New"]:
                return "Hot Trend"
            if engagement_score >= ht["engagement_score_p70"] + 4:
                return "Hot Trend"

            # Priority 2: Best Seller
            if popularity_score >= bs["popularity_score_p70"] and quality_tier in ["Premium", "High"]:
                return "Best Seller"
            if popularity_score >= bs["popularity_score_p65"] and quality_tier in bs["high_quality_tiers"]:
                return "Best Seller"
            if popularity_score >= bs["popularity_score_p65"]:
                return "Best Seller"

            # Priority 3: Best Deal (hard gate discount)
            if discount_intensity not in ["Aggressive", "Heavy", "Moderate"]:
                return "Normal"
            if deal_quality_score >= bd["deal_quality_score_p85"]:
                return "Best Deal"
            if deal_quality_score >= bd["deal_quality_score_p80"] and value_score >= bd["value_score_p75"]:
                return "Best Deal"
            if deal_quality_score >= bd["deal_quality_score_p75"] and value_score >= bd["value_score_p70"]:
                return "Best Deal"
            if deal_quality_score >= bd["deal_quality_score_p70"] and value_score >= bd["value_score_p65"]:
                return "Best Deal"

            return "Normal"

        self.df["label"] = self.df.apply(categorize, axis=1)

    # ------------------------------------------------------------------ #
    # ML MODEL (predict phần unlabeled)
    # ------------------------------------------------------------------ #
    def _build_feature_matrix(self, df: pd.DataFrame, reference_columns=None) -> pd.DataFrame:
        numeric_cols = [c for c in NUMERIC_MODEL_FEATURES if c in self.df.columns]
        cat_cols = [c for c in CATEGORICAL_MODEL_FEATURES if c in self.df.columns]

        X_num = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_cat = df[cat_cols].astype("object").fillna("Unknown")
        X_cat = pd.get_dummies(X_cat, columns=cat_cols, dummy_na=False) if cat_cols else X_cat

        X = pd.concat([X_num, X_cat], axis=1)
        if reference_columns is not None:
            X = X.reindex(columns=reference_columns, fill_value=0.0)
        return X

    def _train_and_predict(self, prob_threshold: float, min_seed_per_class: int, model_type: str) -> bool:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError(
                "Cần scikit-learn cho labeling hybrid. Cài đặt: pip install scikit-learn "
                "(hoặc dùng --no-model để chạy thuần rule-based)"
            )

        seed_df = self.df[self.df["seed_label"].notna()].copy()
        if seed_df.empty:
            print("  Không có seed samples -> fallback rule-based")
            return False

        class_counts = seed_df["seed_label"].value_counts()
        valid_classes = [c for c, n in class_counts.items() if n >= min_seed_per_class]
        if len(valid_classes) < 2:
            print(f"  Không đủ seed cho >=2 class (cần >={min_seed_per_class} mỗi class) -> fallback rule-based")
            return False

        seed_df = seed_df[seed_df["seed_label"].isin(valid_classes)]
        X = self._build_feature_matrix(seed_df)
        y = seed_df["seed_label"].astype(str)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if model_type in ("logreg", "logistic", "logistic_regression"):
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = RandomForestClassifier(
                n_estimators=400, max_depth=15, min_samples_split=10, min_samples_leaf=5,
                random_state=42, class_weight="balanced_subsample", n_jobs=-1,
            )

        model.fit(X_train, y_train)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        print(f"  Model: {type(model).__name__} | Seeds: {len(seed_df):,} | Val accuracy: {val_accuracy:.3f}")

        unlabeled_idx = self.df[self.df["seed_label"].isna()].index
        if unlabeled_idx.empty:
            self.df["label"] = self.df["seed_label"]
            self.df["label_source"] = "rule_seed"
            return True

        X_all = self._build_feature_matrix(self.df, reference_columns=X.columns)
        prob = model.predict_proba(X_all.loc[unlabeled_idx])
        classes = list(model.classes_)
        best_prob = prob.max(axis=1)
        pred_labels = np.array([classes[i] for i in prob.argmax(axis=1)], dtype=object)
        pred_labels[best_prob < prob_threshold] = "Normal"

        pred_labels = self._apply_best_deal_cap(pred_labels, best_prob)

        self.df["label"] = self.df["seed_label"]
        self.df.loc[unlabeled_idx, "label"] = pred_labels
        self.df["label"] = self.df["label"].fillna("Normal")
        self.df.loc[unlabeled_idx, "label_source"] = "model"
        self.df.loc[self.df["seed_label"].notna(), "label_source"] = "rule_seed"

        return True

    def _apply_best_deal_cap(self, pred_labels: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        Guardrail chống bias: giới hạn Best Deal tối đa MAX_BEST_DEAL_RATIO của
        toàn dataset. Nếu model dự đoán vượt quota, các dự đoán có độ tin cậy
        thấp nhất trong số Best Deal bị chuyển về Normal trước.
        """
        max_count = int(len(self.df) * MAX_BEST_DEAL_RATIO)
        seed_best_deal_count = (self.df["seed_label"] == "Best Deal").sum()
        remaining_quota = max(0, max_count - seed_best_deal_count)

        bd_mask = pred_labels == "Best Deal"
        bd_count = bd_mask.sum()

        if bd_count > remaining_quota:
            bd_positions = np.where(bd_mask)[0]
            bd_confidences = confidences[bd_positions]
            excess = bd_count - remaining_quota
            lowest_confidence_positions = bd_positions[np.argsort(bd_confidences)[:excess]]
            pred_labels[lowest_confidence_positions] = "Normal"
            print(f"  Áp cap Best Deal: {bd_count} -> {remaining_quota} "
                  f"(chuyển {excess} record có độ tin cậy thấp nhất về Normal)")

        return pred_labels

    # ------------------------------------------------------------------ #
    def _print_label_distribution(self):
        counts = self.df["label"].value_counts()
        print(f"  Tổng: {len(self.df):,} records")
        for label, count in counts.items():
            print(f"    {label}: {count:,} ({count / len(self.df) * 100:.1f}%)")
        if "label_source" in self.df.columns:
            print("  Nguồn gốc nhãn:")
            for source, count in self.df["label_source"].value_counts().items():
                print(f"    {source}: {count:,} ({count / len(self.df) * 100:.1f}%)")


def run_labeling(
    input_file: str,
    output_file: str,
    use_model: bool = True,
    prob_threshold: float = 0.70,
    min_seed_per_class: int = 50,
    model_type: str = "random_forest",
) -> pd.DataFrame:
    print("=" * 70)
    print("BƯỚC 3b: TRANSFORMATION - LABELING")
    print("   Hot Trend | Best Seller | Best Deal | Normal")
    print("=" * 70)

    print(f"\nĐọc dữ liệu đã engineer feature: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Đã đọc {len(df):,} records\n")

    labeler = ProductLabeler(df)
    df_labeled = labeler.label(
        use_model=use_model, prob_threshold=prob_threshold,
        min_seed_per_class=min_seed_per_class, model_type=model_type,
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(df_labeled.to_dict("records"), f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {output_file}")

    return df_labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Labeling (Hybrid: rule seeds + ML)")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.70)
    parser.add_argument("--min-seed-per-class", type=int, default=50)
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "logistic_regression"])
    parser.add_argument("--no-model", action="store_true", help="Chạy thuần rule-based, không dùng ML")
    args = parser.parse_args()

    config.ensure_all_dirs()
    input_file = args.input or config.FEATURES_FILE
    output_file = args.output or config.LABELED_FILE

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Không tìm thấy file input: {input_file}\n"
            "Hãy chạy feature_engineering.py trước."
        )

    run_labeling(
        input_file=input_file, output_file=output_file,
        use_model=not args.no_model, prob_threshold=args.prob_threshold,
        min_seed_per_class=args.min_seed_per_class, model_type=args.model,
    )
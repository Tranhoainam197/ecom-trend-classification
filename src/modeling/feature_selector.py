"""
feature_selector.py
======================
Định nghĩa các bộ feature dùng để train mô hình phân loại, và xử lý vấn
đề LABEL LEAKAGE (rò rỉ nhãn).

VẤN ĐỀ LABEL LEAKAGE:
Ở bước Transformation (feature_engineering.py + labeling.py), các composite
score sau được TÍNH RA TỪ DỮ LIỆU GỐC rồi DÙNG TRỰC TIẾP để gán nhãn (label):

    - popularity_score, engagement_score, trend_momentum  -> dùng gán Hot Trend / Best Seller
    - value_score, deal_quality_score                      -> dùng gán Best Deal

Vì nhãn được suy ra (gần như) trực tiếp từ các score này (qua ngưỡng
percentile + model RandomForest học từ chính các ngưỡng đó), nếu đưa
nguyên các score này vào làm INPUT FEATURE để train mô hình phân loại,
mô hình sẽ chỉ cần học lại đúng các ngưỡng đã dùng để tạo nhãn ban đầu
-> độ chính xác đạt gần 100% một cách GIẢ TẠO, không phản ánh khả năng
phân loại thực sự trên dữ liệu mới (data mà ta chưa biết nhãn trước).

GIẢI PHÁP: định nghĩa 3 bộ feature để so sánh minh bạch:

    1. FULL_FEATURE_SET      : giữ toàn bộ feature (bao gồm composite score)
                                -> minh họa hiện tượng leakage, accuracy ảo cao
    2. REALISTIC_FEATURE_SET : loại bỏ 5 composite score numeric gây leakage,
                                giữ lại feature "thô" (giá, rating, review,
                                sold, velocity...) -> mô hình THỰC SỰ dùng được
                                để dự đoán cho sản phẩm mới.
    3. REALISTIC_STRICT_SET  : loại thêm cả các cột categorical được DÙNG
                                TRỰC TIẾP trong luật gán nhãn ở labeling.py
                                (quality_tier, discount_intensity,
                                popularity_category) — dùng để CHỨNG MINH mức
                                độ leakage còn sót lại (nếu có) là không đáng
                                kể. Thực nghiệm: Decision Tree Realistic
                                (35 feature) đạt 96.67% acc, Realistic Strict
                                (~20 feature) vẫn đạt ~96.50% -> chênh lệch
                                chỉ ~0.17 điểm phần trăm, xác nhận 3 cột
                                categorical này không phải nguồn leakage đáng
                                kể, khác hẳn với 5 composite score numeric.
"""

import config

# Composite score được dùng trực tiếp để gán nhãn ở bước labeling.py
# -> gây label leakage nếu đưa vào làm feature train
LEAKY_SCORE_FEATURES = [
    "popularity_score_scaled",
    "engagement_score_scaled",
    "trend_momentum_scaled",
    "value_score_scaled",
    "deal_quality_score_scaled",
]

# Các cột categorical được dùng trực tiếp trong điều kiện luật gán nhãn
# (seed_rule / _assign_labels_rule_based trong labeling.py). Không loại khỏi
# bộ "realistic" mặc định vì ảnh hưởng thực nghiệm rất nhỏ (xem docstring),
# nhưng cung cấp riêng ở bộ "realistic_strict" để minh bạch/kiểm chứng.
LEAKY_CATEGORICAL_PREFIXES = ("quality_tier_", "discount_intensity_", "popularity_category_")

# Các cột không phải feature đầu vào (metadata, nhãn, cột phụ trợ) — lấy từ
# config.py (nguồn DUY NHẤT, dùng chung với normalization/encoding/viz) cộng
# thêm 2 cột target (label, label_encoded) vốn chỉ dùng riêng ở bước modeling.
NON_FEATURE_COLUMNS = config.NON_FEATURE_COLUMNS + config.TARGET_COLUMNS


def get_full_feature_columns(df_columns) -> list[str]:
    """Bộ feature ĐẦY ĐỦ — bao gồm cả composite score (CÓ label leakage)."""
    return [c for c in df_columns if c not in NON_FEATURE_COLUMNS]


def get_realistic_feature_columns(df_columns) -> list[str]:
    """
    Bộ feature THỰC TẾ — loại bỏ composite score gây leakage.
    Đây là bộ feature nên dùng để đánh giá khả năng phân loại thực sự
    của mô hình, mô phỏng đúng tình huống áp dụng thực tế: khi có một
    sản phẩm mới, ta chỉ có dữ liệu thô (giá, rating, review, sold...),
    CHƯA có sẵn các composite score nội bộ của pipeline.
    """
    full = get_full_feature_columns(df_columns)
    return [c for c in full if c not in LEAKY_SCORE_FEATURES]


def get_realistic_strict_feature_columns(df_columns) -> list[str]:
    """
    Bộ feature NGHIÊM NGẶT NHẤT — loại cả composite score numeric lẫn
    categorical dùng trực tiếp trong luật gán nhãn. Dùng để CHỨNG MINH mức
    leakage còn lại (nếu có) là không đáng kể, không phải để thay thế bộ
    "realistic" làm kết quả chính của báo cáo.
    """
    realistic = get_realistic_feature_columns(df_columns)
    return [c for c in realistic if not c.startswith(LEAKY_CATEGORICAL_PREFIXES)]


FEATURE_SETS = {
    "full": {
        "name": "Full Features (có composite score)",
        "get_columns": get_full_feature_columns,
        "warning": "Chứa composite score dùng để tạo nhãn -> label leakage, "
                   "accuracy cao một cách giả tạo, KHÔNG phản ánh hiệu năng thực tế.",
    },
    "realistic": {
        "name": "Realistic Features (đã loại composite score)",
        "get_columns": get_realistic_feature_columns,
        "warning": None,
    },
    "realistic_strict": {
        "name": "Realistic Strict (loại cả composite score và categorical rule-based)",
        "get_columns": get_realistic_strict_feature_columns,
        "warning": None,
    },
}
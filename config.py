"""
config.py
=========
Cấu hình đường dẫn tập trung cho toàn bộ project.

Mọi module trong src/ import từ đây để lấy đường dẫn, thay vì mỗi file
tự tính `os.path.dirname(os.path.dirname(...))` rời rạc như trước.
Giúp project portable hơn (chạy được trên Windows/macOS/Linux) vì chỉ
phụ thuộc vị trí của chính file config.py này, không hardcode path tuyệt đối.
"""

import os

# =============================================================================
# ROOT DIRECTORY
# =============================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# DATA DIRECTORIES — theo đúng 5 bước: Cleaning -> Integration -> Transformation
# -> Normalization -> Encoding
# =============================================================================
DATA_DIR = os.path.join(ROOT_DIR, "data")

# --- Bước 0: Collection (thu thập dữ liệu thô) ---
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_SHOPEE_DIR = os.path.join(RAW_DIR, "shopee", "categories")
RAW_LAZADA_DIR = os.path.join(RAW_DIR, "lazada", "categories")
RAW_TIKI_DIR = os.path.join(RAW_DIR, "tiki", "categories")
RAW_MERGED_FILE = os.path.join(RAW_DIR, "merged_raw_data.json")

# --- Bước 1: Cleaning ---
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
CLEAN_FILE = os.path.join(CLEAN_DIR, "cleaned_data.json")
OUTLIER_REPORT_DIR = os.path.join(CLEAN_DIR, "outlier_report")

# --- Bước 2: Integration ---
INTEGRATION_DIR = os.path.join(DATA_DIR, "integrated")
INTEGRATED_FILE = os.path.join(INTEGRATION_DIR, "integrated_data.json")

# --- Bước 3: Transformation (feature engineering + labeling) ---
TRANSFORM_DIR = os.path.join(DATA_DIR, "transformation")
FEATURES_FILE = os.path.join(TRANSFORM_DIR, "engineered_features.json")
LABELED_FILE = os.path.join(TRANSFORM_DIR, "labeled_data.json")

# --- Bước 4: Normalization (scaling) ---
NORMALIZATION_DIR = os.path.join(DATA_DIR, "normalization")
NORMALIZED_TRAIN_FILE = os.path.join(NORMALIZATION_DIR, "normalized_train.json")
NORMALIZED_TEST_FILE = os.path.join(NORMALIZATION_DIR, "normalized_test.json")
SCALER_FILE = os.path.join(NORMALIZATION_DIR, "scaler.pkl")

# --- Bước 5: Encoding ---
ENCODING_DIR = os.path.join(DATA_DIR, "encoding")
ENCODED_TRAIN_FILE = os.path.join(ENCODING_DIR, "encoded_train.json")
ENCODED_TEST_FILE = os.path.join(ENCODING_DIR, "encoded_test.json")
LABEL_ENCODER_FILE = os.path.join(ENCODING_DIR, "label_encoder.pkl")
FEATURE_MAPPING_FILE = os.path.join(ENCODING_DIR, "feature_mapping.json")

# --- Visualizations ---
VIZ_DIR = os.path.join(DATA_DIR, "visualizations")
VIZ_RAW_DIR = os.path.join(VIZ_DIR, "01_raw")
VIZ_CLEAN_DIR = os.path.join(VIZ_DIR, "02_clean")
VIZ_FEATURE_DIR = os.path.join(VIZ_DIR, "03_feature")
VIZ_LABEL_DIR = os.path.join(VIZ_DIR, "04_label")
VIZ_ENCODED_DIR = os.path.join(VIZ_DIR, "05_encoded")
VIZ_MODEL_DIR = os.path.join(VIZ_DIR, "06_model")

# --- Bước 6: Modeling & Evaluation ---
MODEL_DIR = os.path.join(DATA_DIR, "model")
MODEL_COMPARISON_FILE = os.path.join(MODEL_DIR, "model_comparison.csv")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
EVALUATION_REPORT_FILE = os.path.join(MODEL_DIR, "evaluation_report.txt")

# =============================================================================
# CRAWL SETTINGS
# =============================================================================
MAX_PAGES = 20
SLEEP_MIN = 10
SLEEP_MAX = 20

# Thư mục lưu Chrome profile khi crawl Shopee (giữ session đăng nhập).
# Dùng đường dẫn tương đối trong ROOT_DIR để chạy nhất quán trên Windows.
CHROME_PROFILE_DIR = os.path.join(ROOT_DIR, ".chrome_profile")


def ensure_all_dirs():
    """Tạo toàn bộ thư mục dữ liệu cần thiết nếu chưa tồn tại."""
    dirs = [
        RAW_SHOPEE_DIR, RAW_LAZADA_DIR, RAW_TIKI_DIR,
        CLEAN_DIR, OUTLIER_REPORT_DIR,
        INTEGRATION_DIR,
        TRANSFORM_DIR,
        NORMALIZATION_DIR,
        ENCODING_DIR,
        VIZ_RAW_DIR, VIZ_CLEAN_DIR, VIZ_FEATURE_DIR, VIZ_LABEL_DIR, VIZ_ENCODED_DIR, VIZ_MODEL_DIR,
        MODEL_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
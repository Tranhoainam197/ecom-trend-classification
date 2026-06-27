"""
run_pipeline.py
=================
Chạy toàn bộ pipeline xử lý dữ liệu theo đúng 5 bước:

    Cleaning -> Integration -> Transformation -> Normalization -> Encoding

Script này KHÔNG bao gồm bước Collection (crawl dữ liệu) vì đó là quá
trình thu thập dữ liệu thô ban đầu, chỉ cần chạy 1 lần và tốn nhiều thời
gian (rate-limit, captcha...). Trước khi chạy script này, đảm bảo đã có
file data/raw/merged_raw_data.json (chạy crawler + merge_sources.py).

Chạy: python run_pipeline.py
Chạy không kèm visualize (nhanh hơn): python run_pipeline.py --no-viz
"""

import argparse
import time

import config
from src.cleaning.clean_data import clean_data
from src.integration.integrate_data import integrate_data
from src.transformation.feature_engineering import run_feature_engineering
from src.transformation.labeling import run_labeling
from src.normalization.normalize_features import normalize_features
from src.encoding.encode_data import encode_data
from src.modeling.evaluate_models import run_evaluation
from src.visualization.viz_raw import visualize_raw_data
from src.visualization.viz_clean import visualize_clean_data
from src.visualization.viz_label import visualize_labeled_data
from src.visualization.viz_encoded import visualize_encoded_data


def main(visualize: bool = True, use_model: bool = True):
    config.ensure_all_dirs()
    start = time.time()

    print("\n" + "#" * 70)
    print("# PIPELINE KHAI THÁC DỮ LIỆU THƯƠNG MẠI ĐIỆN TỬ")
    print("# Cleaning -> Integration -> Transformation -> Normalization -> Encoding")
    print("#" * 70 + "\n")

    if visualize:
        visualize_raw_data(config.RAW_MERGED_FILE, config.VIZ_RAW_DIR)
        print()

    clean_data(config.RAW_MERGED_FILE, config.CLEAN_FILE)
    print()
    if visualize:
        visualize_clean_data(config.CLEAN_FILE, config.VIZ_CLEAN_DIR)
        print()

    integrate_data(config.CLEAN_FILE, config.INTEGRATED_FILE)
    print()

    run_feature_engineering(config.INTEGRATED_FILE, config.FEATURES_FILE, visualize=visualize)
    print()

    run_labeling(config.FEATURES_FILE, config.LABELED_FILE, use_model=use_model)
    print()
    if visualize:
        visualize_labeled_data(config.LABELED_FILE, config.VIZ_LABEL_DIR)
        print()

    normalize_features(config.LABELED_FILE, test_size=0.2)
    print()

    encode_data(config.NORMALIZED_TRAIN_FILE, config.NORMALIZED_TEST_FILE)
    print()
    if visualize:
        visualize_encoded_data(config.ENCODED_TRAIN_FILE, config.ENCODED_TEST_FILE, config.VIZ_ENCODED_DIR)
        print()

    run_evaluation(
        train_file=config.ENCODED_TRAIN_FILE,
        test_file=config.ENCODED_TEST_FILE,
        output_dir=config.MODEL_DIR,
        viz_dir=config.VIZ_MODEL_DIR,
    )

    elapsed = time.time() - start
    print("\n" + "#" * 70)
    print(f"# HOÀN THÀNH TOÀN BỘ PIPELINE trong {elapsed:.1f} giây")
    print("#" * 70)
    print(f"\nDữ liệu sẵn sàng cho mô hình tại:")
    print(f"  - {config.ENCODED_TRAIN_FILE}")
    print(f"  - {config.ENCODED_TEST_FILE}")
    print(f"\nMô hình tốt nhất đã lưu tại:")
    print(f"  - {config.BEST_MODEL_FILE}")
    print(f"  - {config.MODEL_COMPARISON_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline xử lý dữ liệu")
    parser.add_argument("--no-viz", action="store_true", help="Bỏ qua việc tạo biểu đồ trực quan hóa")
    parser.add_argument("--no-model", action="store_true", help="Labeling thuần rule-based, không dùng ML")
    args = parser.parse_args()

    main(visualize=not args.no_viz, use_model=not args.no_model)
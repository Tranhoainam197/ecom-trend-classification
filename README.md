# Phân tích Xu hướng và Phân loại Sản phẩm trên các Sàn Thương mại Điện tử

Đồ án môn **Khai thác Dữ liệu** — xây dựng pipeline xử lý dữ liệu sản phẩm
thu thập từ 3 sàn TMĐT (Shopee, Lazada, Tiki) để phân tích xu hướng và
phân loại sản phẩm thành 4 nhóm: **Hot Trend**, **Best Seller**,
**Best Deal**, **Normal**.

## Quy trình xử lý dữ liệu

```
Collection → Cleaning → Integration → Transformation → Normalization → Encoding
```

| Bước | Mục đích | Module |
|------|----------|--------|
| 0. Collection | Thu thập dữ liệu thô từ Shopee/Lazada/Tiki | `src/collection/` |
| 1. Cleaning | Chuẩn hóa schema, trích xuất số liệu, xử lý missing/outlier, loại trùng lặp | `src/cleaning/` |
| 2. Integration | Hợp nhất 3 nguồn, kiểm tra toàn vẹn schema & định danh duy nhất | `src/integration/` |
| 3. Transformation | Feature engineering (31 đặc trưng) + gán nhãn (rule-based + ML) | `src/transformation/` |
| 4. Normalization | Chuẩn hóa thang đo (StandardScaler), fit-on-train-only | `src/normalization/` |
| 5. Encoding | One-Hot Encoding categorical + Label Encoding target | `src/encoding/` |

## Cài đặt

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cách chạy

### Chạy toàn bộ pipeline (sau khi đã có dữ liệu thô)

```powershell
python run_pipeline.py
```

Tuỳ chọn:
```powershell
python run_pipeline.py --no-viz      # bỏ qua việc tạo biểu đồ (chạy nhanh hơn)
python run_pipeline.py --no-model    # labeling thuần rule-based, không train ML
```

### Chạy từng bước riêng lẻ

```powershell
# Bước 0: Thu thập dữ liệu (chạy 1 lần, tốn thời gian — bỏ qua nếu đã có data/raw/)
python -m src.collection.crawl_shopee
python -m src.collection.crawl_lazada
python -m src.collection.crawl_tiki
python -m src.collection.merge_sources

# Bước 1: Cleaning
python -m src.cleaning.clean_data

# Bước 2: Integration
python -m src.integration.integrate_data

# Bước 3: Transformation
python -m src.transformation.feature_engineering
python -m src.transformation.labeling

# Bước 4: Normalization
python -m src.normalization.normalize_features

# Bước 5: Encoding
python -m src.encoding.encode_data

# Trực quan hóa (tuỳ chọn, chạy độc lập từng bước)
python -m src.visualization.viz_raw
python -m src.visualization.viz_clean
python -m src.visualization.viz_label
python -m src.visualization.viz_encoded
```

## Cấu trúc project

```
ecom-trend-classification/
├── config.py                       # Đường dẫn tập trung cho toàn project
├── run_pipeline.py                 # Chạy toàn bộ pipeline 1 lệnh
├── requirements.txt
├── src/
│   ├── collection/                 # Bước 0: thu thập dữ liệu (Selenium/API)
│   ├── cleaning/                   # Bước 1: làm sạch dữ liệu
│   │   ├── normalizer.py           #   chuẩn hóa schema 3 platform
│   │   ├── value_extractor.py      #   parse giá/discount/sold text -> số
│   │   ├── outlier_handler.py      #   xử lý outlier theo percentile + nghiệp vụ
│   │   └── clean_data.py           #   orchestrator
│   ├── integration/                # Bước 2: hợp nhất & kiểm tra toàn vẹn
│   ├── transformation/             # Bước 3: feature engineering + labeling
│   ├── normalization/              # Bước 4: chuẩn hóa thang đo (StandardScaler)
│   ├── encoding/                   # Bước 5: one-hot + label encoding
│   └── visualization/              # Biểu đồ cho từng bước
└── data/
    ├── raw/                        # Output Collection
    ├── clean/                      # Output Cleaning
    ├── integrated/                 # Output Integration
    ├── transformation/             # Output Transformation
    ├── normalization/              # Output Normalization
    ├── encoding/                   # Output Encoding (sẵn sàng cho mô hình ML)
    └── visualizations/              # Biểu đồ từng bước
```

## Các đặc trưng (feature) chính được tạo ở bước Transformation

- **Velocity**: `sales_velocity`, `review_velocity` (và bản chuẩn hóa 0–100)
- **Composite scores**:
  - `popularity_score` (Best Seller) = sold 50% + reviews 30% + rating 20%
  - `engagement_score` (Hot Trend) = review_velocity 40% + sales_velocity 40% + rating 20%
  - `trend_momentum` (Hot Trend) = engagement_score × hệ số độ mới sản phẩm
  - `value_score` (Best Deal) = discount 45% + rating tương đối 20% + giá cạnh tranh 35%
  - `deal_quality_score` (Best Deal) = tiết kiệm tuyệt đối 45% + rating tương đối 20% + độ tin cậy review 20% + discount gate 15%
- **Categorical**: `popularity_category`, `price_segment`, `quality_tier`, `discount_intensity`, `product_age`


## Công nghệ sử dụng

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Selenium, DrissionPage
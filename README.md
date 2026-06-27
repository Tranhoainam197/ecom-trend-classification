# Phân tích Xu hướng và Phân loại Sản phẩm trên các Sàn Thương mại Điện tử

Đồ án môn **Khai thác Dữ liệu** — xây dựng pipeline xử lý dữ liệu sản phẩm
thu thập từ 3 sàn TMĐT (Shopee, Lazada, Tiki) để phân tích xu hướng và
phân loại sản phẩm thành 4 nhóm: **Hot Trend**, **Best Seller**,
**Best Deal**, **Normal**.

## Quy trình xử lý dữ liệu

```
Collection → Cleaning → Integration → Transformation → Normalization → Encoding → Modeling
```

| Bước | Mục đích | Module |
|------|----------|--------|
| 0. Collection | Thu thập dữ liệu thô từ Shopee/Lazada/Tiki | `src/collection/` |
| 1. Cleaning | Chuẩn hóa schema, trích xuất số liệu, xử lý missing/outlier, loại trùng lặp | `src/cleaning/` |
| 2. Integration | Hợp nhất 3 nguồn, kiểm tra toàn vẹn schema & định danh duy nhất | `src/integration/` |
| 3. Transformation | Feature engineering (31 đặc trưng) + gán nhãn (rule-based + ML) | `src/transformation/` |
| 4. Normalization | Chuẩn hóa thang đo (StandardScaler), fit-on-train-only | `src/normalization/` |
| 5. Encoding | One-Hot Encoding categorical + Label Encoding target | `src/encoding/` |
| 6. Modeling | Train + đánh giá 5 mô hình phân loại (Decision Tree, Random Forest, Logistic Regression, KNN, Naive Bayes) | `src/modeling/` |

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

# Bước 6: Modeling & Evaluation (train + đánh giá 5 mô hình, chọn mô hình tốt nhất)
python -m src.modeling.evaluate_models

# Trực quan hóa (tuỳ chọn, chạy độc lập từng bước)
python -m src.visualization.viz_raw
python -m src.visualization.viz_clean
python -m src.visualization.viz_feature
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
│   ├── modeling/                   # Bước 6: train + đánh giá mô hình phân loại
│   │   ├── feature_selector.py     #   định nghĩa bộ feature full/realistic (chống leakage)
│   │   ├── train_models.py         #   train 5 thuật toán phân loại
│   │   └── evaluate_models.py      #   đánh giá, so sánh, chọn mô hình tốt nhất
│   └── visualization/              # Biểu đồ cho từng bước
└── data/
    ├── raw/                        # Output Collection
    ├── clean/                      # Output Cleaning
    ├── integrated/                 # Output Integration
    ├── transformation/             # Output Transformation
    ├── normalization/              # Output Normalization
    ├── encoding/                   # Output Encoding (sẵn sàng cho mô hình ML)
    ├── model/                      # Output Modeling (model đã train + báo cáo đánh giá)
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

## Lưu ý quan trọng về hạn chế dữ liệu

Nếu dữ liệu chỉ được crawl trong rất ít ngày khác nhau, đặc trưng
`product_age` sẽ không có đủ biến thiên thời gian để phân nhóm và bị cố
định "Brand New" cho toàn bộ dataset. Khi đó `trend_momentum` chỉ là
`engagement_score × hằng số`, không mang thêm thông tin mới. Đây là hạn
chế của **dữ liệu đầu vào** (crawl trong thời gian ngắn), không phải lỗi
logic xử lý — cần nêu rõ trong báo cáo đồ án và khuyến nghị crawl lại
nhiều lần trong nhiều ngày để có dữ liệu xu hướng thời gian thực sự.

## Chống Data Leakage

Pipeline tuân thủ nguyên tắc: **mọi phép biến đổi học thống kê từ dữ liệu
(StandardScaler) phải fit SAU khi chia train/test, và chỉ fit trên tập
train.** Vì vậy thứ tự bắt buộc là Normalization fit-on-train-only được
thực hiện trước Encoding trong cùng 1 lần chia train/test, đảm bảo tập
test không "rò" thông tin vào quá trình huấn luyện.

## Modeling & Evaluation (Bước 6)

5 thuật toán phân loại được train và so sánh: **Decision Tree, Random Forest,
Logistic Regression, KNN, Naive Bayes**.

### Vấn đề Label Leakage và cách xử lý

Các composite score (`popularity_score`, `engagement_score`, `trend_momentum`,
`value_score`, `deal_quality_score`) được **dùng để tạo ra nhãn** ở bước
Labeling. Nếu đưa nguyên các score này vào làm feature train mô hình, độ
chính xác sẽ cao một cách giả tạo (~99%) vì mô hình chỉ học lại đúng ngưỡng
đã dùng để gán nhãn ban đầu — không phản ánh khả năng phân loại thực sự.

Vì vậy `src/modeling/` train và so sánh **2 bộ feature**:

| Bộ feature | Gồm gì | Ý nghĩa |
|---|---|---|
| `full` | Tất cả 40 feature, có cả 5 composite score | Minh họa hiện tượng leakage, accuracy ảo cao |
| `realistic` | 35 feature, đã loại 5 composite score | Mô hình dùng được thực tế cho sản phẩm mới |

Mô hình tốt nhất được chọn và lưu (`data/model/best_model.pkl`) luôn dựa trên
bộ **`realistic`**, vì đây mới là kịch bản áp dụng thực tế: khi có một sản
phẩm mới, ta chỉ có dữ liệu thô (giá, rating, số lượng bán, review...), chưa
có sẵn composite score được tính theo công thức nội bộ của pipeline.

Output: `data/model/model_comparison.csv` (bảng so sánh accuracy/precision/
recall/F1 của cả 10 tổ hợp model × bộ feature), `evaluation_report.txt` (báo
cáo đầy đủ + classification report), và biểu đồ so sánh trong
`data/visualizations/06_model/`.

## Biểu đồ trực quan hóa

Tổng cộng **33 biểu đồ** trải đều qua 6 bước, đủ để minh họa cho báo cáo:

| Bước | Số biểu đồ | Nội dung chính |
|---|---|---|
| 01_raw | 4 | NULL values, platform, category, giá |
| 02_clean | 7 | platform, giá, rating/review, discount, category/brand, correlation, **seller_location** |
| 03_feature | 12 | popularity/discount/quality/price category, 4 composite score, trend_momentum, 2 velocity, **product_age** (minh chứng hạn chế dữ liệu) |
| 04_label | 3 | phân bố nhãn, so sánh metric theo nhãn, scatter trend vs engagement |
| 05_encoded | 4 | nhãn train/test, cơ cấu feature, phân bố scaled feature, **correlation heatmap** |
| 06_model | 3 | so sánh F1 full vs realistic, so sánh accuracy, confusion matrix |

Riêng biểu đồ **`12_product_age.png`** quan trọng để minh chứng trực quan
cho phần "Hạn chế dữ liệu" trong báo cáo — biểu đồ tự nhận diện và in chú
giải ngay trên hình nếu `product_age` chỉ có 1 giá trị duy nhất.

## Công nghệ sử dụng

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Selenium, DrissionPage
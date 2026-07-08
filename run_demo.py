"""
run_demo.py
=============
Demo trực quan (giao diện terminal) cho mô hình phân loại đã train.

Dùng để DEMO khi thuyết trình đồ án: chọn ngẫu nhiên / tìm kiếm / duyệt
theo danh mục các sản phẩm trong TẬP TEST (mô hình chưa từng thấy khi
train), xem mô hình dự đoán nhãn gì, so sánh với nhãn thực tế, và xem
bảng so sánh hiệu năng giữa các mô hình.

YÊU CẦU: đã chạy xong `python run_pipeline.py` ít nhất 1 lần (để có sẵn
data/model/best_model.pkl, data/encoding/encoded_test.json,
data/transformation/labeled_data.json).

Chạy: python run_demo.py
"""

from src.demo.predict_demo import run_demo_app

if __name__ == "__main__":
    run_demo_app()

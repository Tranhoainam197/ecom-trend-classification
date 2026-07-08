"""
web_app.py
============
DEMO WEB — Phân loại xu hướng sản phẩm TMĐT
===============================================
Backend Flask nhỏ gọn phục vụ demo trực quan qua trình duyệt. Toàn bộ dữ
liệu (sản phẩm tập test + dự đoán + so sánh model) được tính 1 lần lúc
khởi động và trả về qua 1 API endpoint duy nhất (/api/data); mọi thao tác
lọc/tìm kiếm/duyệt danh mục sau đó xử lý ở phía trình duyệt (JavaScript)
để phản hồi tức thì, không cần gọi lại server.

Cùng nguyên tắc trung thực như bản demo CLI (predict_demo.py):
  - Chỉ hiển thị/dự đoán trên TẬP TEST (dữ liệu mô hình chưa từng thấy).
  - Dùng chung logic load dữ liệu với bản CLI (xem data_loader.py) để đảm
    bảo 2 bản demo luôn cho ra cùng kết quả.

Chạy: python run_web_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, jsonify, render_template

import config
from src.demo.data_loader import load_everything, load_model_comparison

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Cache trong bộ nhớ, tính 1 lần lúc app khởi động (xem create_app()).
_CACHE = {}


def create_app():
    app = Flask(__name__, template_folder=_TEMPLATE_DIR, static_folder=_STATIC_DIR)

    records, class_names, model_name = load_everything()
    comparison = load_model_comparison()
    n_correct = sum(1 for r in records if r["correct"])

    _CACHE["records"] = records
    _CACHE["class_names"] = class_names
    _CACHE["model_name"] = model_name
    _CACHE["comparison"] = comparison
    _CACHE["accuracy"] = round(100 * n_correct / len(records), 2) if records else 0

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/data")
    def api_data():
        return jsonify({
            "records": _CACHE["records"],
            "class_names": _CACHE["class_names"],
            "model_name": _CACHE["model_name"],
            "comparison": _CACHE["comparison"],
            "accuracy": _CACHE["accuracy"],
            "n_test": len(_CACHE["records"]),
        })

    return app


if __name__ == "__main__":
    config.ensure_all_dirs()
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)

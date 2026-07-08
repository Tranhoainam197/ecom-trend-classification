"""
run_web_demo.py
==================
Demo trực quan qua TRÌNH DUYỆT (web) cho mô hình phân loại đã train.

Dùng khi thuyết trình đồ án và muốn có giao diện web thật (thay vì terminal):
chọn ngẫu nhiên / tìm kiếm / duyệt danh mục / xem ca dự đoán sai các sản
phẩm trong TẬP TEST, so sánh với nhãn thực tế, và xem bảng so sánh hiệu
năng giữa các mô hình — tất cả trên 1 trang web chạy hoàn toàn ở máy local
(không cần deploy, không cần Internet để hoạt động, chỉ cần trình duyệt).

YÊU CẦU: đã chạy xong `python run_pipeline.py` ít nhất 1 lần (để có sẵn
data/model/best_model.pkl, data/encoding/encoded_test.json,
data/transformation/labeled_data.json).

Chạy: python run_web_demo.py
Sau đó mở trình duyệt tại: http://127.0.0.1:5000
(script sẽ tự mở trình duyệt giúp bạn sau ~1 giây)
"""

import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def _open_browser_later():
    threading.Timer(1.0, lambda: webbrowser.open(URL)).start()


def main():
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print()
    console.print(Panel.fit(
        f"[bold white]E-COMMERCE TREND CLASSIFIER — WEB DEMO[/]\n\n"
        f"[grey70]Đang khởi động server tại[/] [bold cyan]{URL}[/]\n"
        f"[grey70]Trình duyệt sẽ tự mở sau giây lát. Nhấn[/] [bold]Ctrl+C[/] [grey70]để dừng server.[/]",
        border_style="green",
    ))
    console.print()

    import config
    config.ensure_all_dirs()
    from src.demo.web_app import create_app

    try:
        app = create_app()
    except FileNotFoundError as e:
        console.print(f"[bold red]Thiếu file dữ liệu: {e}[/]")
        console.print("[yellow]Hãy chạy 'python run_pipeline.py' trước để tạo model và dữ liệu test.[/]")
        return

    _open_browser_later()
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()

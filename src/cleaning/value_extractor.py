"""
value_extractor.py
====================
Các hàm trích xuất / ép kiểu giá trị thô (string) sang dạng số sạch.
Thuộc bước 1 - CLEANING.

Dữ liệu thô từ 3 sàn TMĐT có format không đồng nhất, ví dụ:
    - Giá: "499.000 ₫", 499000, "499000"
    - Discount: "17% Off", 17, "17"
    - Số lượng bán: "1.2k đã bán", "1.2K Sold", 1200, "1.234.567" (nghìn)
"""

import re
import pandas as pd


def extract_price(value) -> float | None:
    """Trích xuất giá trị số từ chuỗi giá. Ví dụ: '499.000 ₫' -> 499000.0"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    digits_only = re.sub(r"[^\d]", "", str(value))
    return float(digits_only) if digits_only else None


def extract_discount_rate(value) -> float | None:
    """Trích xuất tỷ lệ giảm giá (%) từ chuỗi. Ví dụ: '17% Off' -> 17.0"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else None


def extract_quantity_sold(text) -> int | None:
    """
    Trích xuất số lượng bán dạng số từ text hiển thị.
    Ví dụ: '1.2K Sold' -> 1200, '3M' -> 3000000, '1.234.567' -> 1234567.

    Lưu ý xử lý dấu chấm 2 nghĩa khác nhau:
    - Có hậu tố K/M/B (vd "1.8K")  -> dấu chấm là NGĂN CÁCH THẬP PHÂN.
    - Không có hậu tố, có >=2 dấu chấm (vd "1.234.567") -> dấu chấm là
      NGĂN CÁCH HÀNG NGHÌN kiểu Việt Nam, phải bỏ hết trước khi ép kiểu số,
      nếu không float() sẽ ném ValueError và làm crash toàn bộ pipeline.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    if not isinstance(text, str):
        return None

    text = text.upper().strip()
    match = re.search(r"([\d.]+)\s*([KMB]?)", text)
    if not match:
        return None

    number_part, unit = match.group(1), match.group(2)

    if number_part.count(".") >= 2 and not unit:
        number_part = number_part.replace(".", "")

    try:
        value = float(number_part)
    except ValueError:
        return None

    multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(unit, 1)
    return int(value * multiplier)


def safe_to_numeric(value) -> float | None:
    """Ép kiểu an toàn sang số: number -> number, string số -> number, còn lại -> None."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
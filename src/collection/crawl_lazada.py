"""
crawl_lazada.py
================
Crawl dữ liệu sản phẩm từ Lazada thông qua API ajax nội bộ (?ajax=true),
dùng cookie lấy từ Selenium để xác thực request.

Chạy: python -m src.collection.crawl_lazada
"""

import sys
import os
import time
import random
import re
import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
from src.collection.base_crawler import crawl_all_categories, get_fresh_cookies
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LAZADA_CATEGORIES = [
    {"name": "Điện thoại di động", "path": "dien-thoai-di-dong"},
    {"name": "Máy tính bảng", "path": "may-tinh-bang"},
    {"name": "Laptop", "path": "laptop"},
    {"name": "Pin sạc dự phòng", "path": "pin-sac-du-phong"},
    {"name": "Tai nghe không dây", "path": "shop-wireless-earbuds"},
    {"name": "Máy ảnh máy quay phim", "path": "may-anh-may-quay-phim"},
    {"name": "Tủ lạnh", "path": "tu-lanh"},
    {"name": "Máy giặt", "path": "may-giat"},
    {"name": "Máy lạnh", "path": "may-lanh"},
    {"name": "Áo phông & Áo ba lỗ", "path": "shop-t-shirts-&-tanks"},
    {"name": "Quần jeans", "path": "shop-men-jeans"},
    {"name": "Dưỡng da & Serum", "path": "duong-da-va-serum"},
    {"name": "Son thỏi", "path": "son-thoi"},
    {"name": "Bách hóa online", "path": "bach-hoa-online"},
    {"name": "Phụ kiện làm thơm phòng", "path": "do-dung-lam-thom-phong"},
    {"name": "Giường", "path": "giuong"},
    {"name": "Bóng đá", "path": "bong-da"},
    {"name": "Máy chạy bộ", "path": "may-chay-bo"},
    {"name": "Bikini", "path": "bikini-2"},
    {"name": "Búp bê cho bé", "path": "bup-be-cho-be"},
    {"name": "Xe máy", "path": "xe-may"},
]


def _get_cookies_for_path(path: str) -> dict:
    return get_fresh_cookies(url=f"https://www.lazada.vn/{path}", headless=False, scroll=True, wait_time=15)


def _extract_sold_value(sold_text: str):
    """Trích xuất số lượng bán dạng số từ text hiển thị (vd '1.2k đã bán' -> 1200)."""
    if not sold_text:
        return None
    numbers = re.findall(r"\d+", sold_text.replace(",", ""))
    if not numbers:
        return None
    value = int(numbers[0])
    if "k" in sold_text.lower():
        value *= 1000
    return value


def crawl_category_lazada(cat: dict, cookies, max_pages: int, retries: int = 2) -> list:
    """Crawl toàn bộ sản phẩm của một danh mục Lazada qua API ajax phân trang."""
    path = cat["path"]
    name = cat["name"]
    base_url = f"https://www.lazada.vn/{path}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9",
        "Referer": base_url,
        "X-Requested-With": "XMLHttpRequest",
    }

    if not cookies:
        cookies = _get_cookies_for_path(path)
    if not cookies:
        logging.error(f"Không lấy được cookie cho {name} -> bỏ qua danh mục")
        return []

    products: list = []
    page_sleep_min, page_sleep_max = config.SLEEP_MIN / 3, config.SLEEP_MAX / 3

    page_num = 1
    while page_num <= max_pages:
        attempt = 0
        success = False

        while attempt < retries and not success:
            try:
                resp = requests.get(
                    base_url,
                    headers=headers,
                    cookies=cookies,
                    params={"ajax": "true", "page": page_num},
                    timeout=20,
                )

                if resp.status_code != 200:
                    logging.warning(f"{name} trang {page_num}: HTTP {resp.status_code} -> làm mới cookie")
                    cookies = _get_cookies_for_path(path)
                    attempt += 1
                    time.sleep(random.uniform(config.SLEEP_MIN, config.SLEEP_MAX))
                    continue

                items = resp.json().get("mods", {}).get("listItems", [])
                if not items:
                    logging.info(f"{name}: hết dữ liệu ở trang {page_num}")
                    return products

                for item in items:
                    sold_text = (item.get("itemSoldCntShow") or "").strip()
                    products.append({
                        "crawl_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "platform": "Lazada",
                        "category_name": name,
                        "id": item.get("itemId"),
                        "name": item.get("name"),
                        "price": item.get("priceShow") or item.get("price"),
                        "original_price": item.get("originalPriceShow") or item.get("originalPrice"),
                        "discount_rate": item.get("discount"),
                        "rating_average": item.get("ratingScore"),
                        "review_count": item.get("review"),
                        "quantity_sold_value": _extract_sold_value(sold_text),
                        "quantity_sold_text": sold_text,
                        "brand": item.get("brandName"),
                        "location": item.get("location"),
                        "seller_name": item.get("sellerName"),
                        "url": ("https:" + item["itemUrl"]) if item.get("itemUrl") else None,
                    })

                logging.info(f"{name} - trang {page_num}: {len(items)} sản phẩm")
                success = True
                time.sleep(random.uniform(page_sleep_min, page_sleep_max))

            except Exception as e:
                attempt += 1
                logging.error(f"{name} trang {page_num} (lần {attempt}): {e}")
                time.sleep(random.uniform(config.SLEEP_MIN, config.SLEEP_MAX))

        page_num += 1

    return products


if __name__ == "__main__":
    config.ensure_all_dirs()
    crawl_all_categories(
        platform_name="Lazada",
        categories=LAZADA_CATEGORIES,
        crawl_category_func=crawl_category_lazada,
        category_output_dir=config.RAW_LAZADA_DIR,
        get_cookies_func=None,
        max_pages=config.MAX_PAGES,
        retries=2,
        sleep_min=config.SLEEP_MIN,
        sleep_max=config.SLEEP_MAX,
    )
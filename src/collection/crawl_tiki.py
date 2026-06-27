"""
crawl_tiki.py
=============
Crawl dữ liệu sản phẩm từ Tiki qua API "listings" công khai.

Chạy: python -m src.collection.crawl_tiki
"""

import sys
import os
import time
import random
import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
from src.collection.base_crawler import crawl_all_categories, get_fresh_cookies
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TIKI_CATEGORIES = [
    {"name": "Nhà Sách Tiki", "urlKey": "nha-sach-tiki", "category": "8322"},
    {"name": "Nhà Cửa - Đời sống", "urlKey": "nha-cua-doi-song", "category": "1883"},
    {"name": "Điện Thoại - Máy Tính Bảng", "urlKey": "dien-thoai-may-tinh-bang", "category": "1789"},
    {"name": "Đồ Chơi - Mẹ & Bé", "urlKey": "do-choi-me-be", "category": "2549"},
    {"name": "Thiết Bị Số - Phụ Kiện Số", "urlKey": "thiet-bi-so-phu-kien-so", "category": "1815"},
    {"name": "Điện Gia Dụng", "urlKey": "dien-gia-dung", "category": "20824"},
    {"name": "Làm Đẹp - Sức Khỏe", "urlKey": "lam-dep-suc-khoe", "category": "1520"},
    {"name": "Ô Tô - Xe Máy - Xe Đạp", "urlKey": "o-to-xe-may-xe-dap", "category": "21346"},
    {"name": "Thời Trang Nữ", "urlKey": "thoi-trang-nu", "category": "931"},
    {"name": "Bách Hóa Online", "urlKey": "bach-hoa-online", "category": "4384"},
    {"name": "Thể Thao - Dã Ngoại", "urlKey": "the-thao-da-ngoai", "category": "1975"},
    {"name": "Thời Trang Nam", "urlKey": "thoi-trang-nam", "category": "915"},
    {"name": "Laptop - Máy Vi Tính - Linh Kiện", "urlKey": "laptop-may-vi-tinh-linh-kien", "category": "1846"},
    {"name": "Giày Dép Nam", "urlKey": "giay-dep-nam", "category": "1686"},
    {"name": "Điện Tử - Điện Lạnh", "urlKey": "dien-tu-dien-lanh", "category": "4221"},
    {"name": "Giày Dép Nữ", "urlKey": "giay-dep-nu", "category": "1703"},
    {"name": "Máy Ảnh - Máy Quay Phim", "urlKey": "may-anh", "category": "1801"},
    {"name": "Phụ kiện thời trang", "urlKey": "phu-kien-thoi-trang", "category": "27498"},
    {"name": "Đồng hồ và Trang sức", "urlKey": "dong-ho-va-trang-suc", "category": "8371"},
    {"name": "Balo và Vali", "urlKey": "balo-va-vali", "category": "6000"},
    {"name": "Túi thời trang nữ", "urlKey": "tui-thoi-trang-nu", "category": "976"},
    {"name": "Túi thời trang nam", "urlKey": "tui-thoi-trang-nam", "category": "27616"},
    {"name": "Chăm sóc nhà cửa", "urlKey": "cham-soc-nha-cua", "category": "15078"},
]


def _get_cookies_tiki() -> dict:
    return get_fresh_cookies(url="https://tiki.vn", headless=True, scroll=False, wait_time=10)


def crawl_category_tiki(cat: dict, cookies, max_pages: int, retries: int = 2) -> list:
    """Crawl toàn bộ sản phẩm của một danh mục Tiki qua API listings phân trang."""
    base_url = "https://tiki.vn/api/personalish/v1/blocks/listings"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://tiki.vn/{cat['urlKey']}/c{cat['category']}",
        "X-Requested-With": "XMLHttpRequest",
    }

    products: list = []
    params = {"limit": 48, "sort": "top_seller", "urlKey": cat["urlKey"], "category": cat["category"], "page": 1}
    page_sleep_min, page_sleep_max = config.SLEEP_MIN / 3, config.SLEEP_MAX / 3

    for page_num in range(1, max_pages + 1):
        params["page"] = page_num
        attempt = 0
        success = False

        while attempt < retries and not success:
            try:
                resp = requests.get(base_url, headers=headers, cookies=cookies, params=params, timeout=20)
                if resp.status_code != 200:
                    logging.warning(f"{cat['name']} trang {page_num}: HTTP {resp.status_code}")
                    break

                data = resp.json()
                items = data.get("listings") or data.get("data") or []
                if not isinstance(items, list) or not items:
                    logging.info(f"{cat['name']}: hết dữ liệu ở trang {page_num}")
                    return products

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    quantity_sold = item.get("quantity_sold") or {}
                    seller = item.get("seller") or {}
                    impression_info = item.get("visible_impression_info") or {}
                    amplitude = impression_info.get("amplitude") or {}

                    url_path = item.get("url_path", "")
                    if url_path and not url_path.startswith("/"):
                        url_path = "/" + url_path

                    products.append({
                        "crawl_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "platform": "Tiki",
                        "category_name": cat["name"],
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "price": item.get("price"),
                        "original_price": item.get("original_price"),
                        "discount_rate": item.get("discount_rate"),
                        "rating_average": item.get("rating_average"),
                        "review_count": item.get("review_count"),
                        "quantity_sold_value": quantity_sold.get("value"),
                        "quantity_sold_text": quantity_sold.get("text"),
                        "brand": item.get("brand_name"),
                        "location": amplitude.get("origin"),
                        "seller_name": seller.get("name"),
                        # FIX: bug gốc thiếu f-string nên url luôn là literal "https://tiki.vn{url_path}"
                        "url": f"https://tiki.vn{url_path}" if url_path else None,
                    })

                logging.info(f"{cat['name']} - trang {page_num}: {len(items)} sản phẩm")
                success = True
                time.sleep(random.uniform(page_sleep_min, page_sleep_max))

            except Exception as e:
                attempt += 1
                logging.error(f"{cat['name']} trang {page_num} (lần {attempt}): {e}")
                time.sleep(random.uniform(config.SLEEP_MIN, config.SLEEP_MAX))

        if not success:
            break

    return products


if __name__ == "__main__":
    config.ensure_all_dirs()
    crawl_all_categories(
        platform_name="Tiki",
        categories=TIKI_CATEGORIES,
        crawl_category_func=crawl_category_tiki,
        category_output_dir=config.RAW_TIKI_DIR,
        get_cookies_func=_get_cookies_tiki,
        max_pages=config.MAX_PAGES,
        retries=2,
        sleep_min=config.SLEEP_MIN,
        sleep_max=config.SLEEP_MAX,
    )
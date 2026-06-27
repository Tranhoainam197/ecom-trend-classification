"""
crawl_shopee.py
================
Crawl dữ liệu sản phẩm từ Shopee bằng DrissionPage (đọc network response
trực tiếp từ trình duyệt, không cần gọi API thủ công vì Shopee chặn khá gắt).

Yêu cầu: đăng nhập tay vào Shopee khi trình duyệt mở lên (giảm tỷ lệ bị captcha).

Chạy: python -m src.collection.crawl_shopee
"""

import sys
import os
import time
import random
import math
import logging
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.collection.base_crawler import crawl_all_categories
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SHOPEE_CATEGORIES = [
    {"name": "Điện thoại", "url": "https://shopee.vn/search?keyword=Điện%20thoại"},
    {"name": "Laptop", "url": "https://shopee.vn/search?keyword=Laptop"},
    {"name": "Thời trang nữ", "url": "https://shopee.vn/search?keyword=Thời%20trang%20nữ"},
    {"name": "Thời trang nam", "url": "https://shopee.vn/search?keyword=Thời%20trang%20nam"},
    {"name": "Giày dép", "url": "https://shopee.vn/search?keyword=Giày%20dép"},
    {"name": "Túi xách", "url": "https://shopee.vn/search?keyword=Túi%20xách"},
    {"name": "Đồng hồ", "url": "https://shopee.vn/search?keyword=Đồng%20hồ"},
    {"name": "Trang sức", "url": "https://shopee.vn/search?keyword=Trang%20sức"},
    {"name": "Mỹ phẩm", "url": "https://shopee.vn/search?keyword=Mỹ%20phẩm"},
    {"name": "Chăm sóc da", "url": "https://shopee.vn/search?keyword=Chăm%20sóc%20da"},
    {"name": "Máy ảnh", "url": "https://shopee.vn/search?keyword=Máy%20ảnh"},
    {"name": "Máy tính bảng", "url": "https://shopee.vn/search?keyword=Máy%20tính%20bảng"},
    {"name": "Headphone", "url": "https://shopee.vn/search?keyword=Headphone"},
    {"name": "Loa", "url": "https://shopee.vn/search?keyword=Loa"},
    {"name": "Phụ kiện điện thoại", "url": "https://shopee.vn/search?keyword=Phụ%20kiện%20điện%20thoại"},
    {"name": "Sách", "url": "https://shopee.vn/search?keyword=Sách"},
]

_GLOBAL_PAGE = None


def get_browser():
    """Lấy (hoặc tạo mới) instance trình duyệt Chrome dùng chung cho cả quá trình crawl."""
    global _GLOBAL_PAGE
    if _GLOBAL_PAGE:
        return _GLOBAL_PAGE

    from DrissionPage import ChromiumPage, ChromiumOptions

    co = ChromiumOptions()
    co.set_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    co.set_argument("--disable-blink-features=AutomationControlled")
    co.set_argument("--no-sandbox")
    co.set_argument("--disable-dev-shm-usage")
    co.set_argument("--start-maximized")
    co.set_user_data_path(config.CHROME_PROFILE_DIR)

    _GLOBAL_PAGE = ChromiumPage(addr_or_opts=co)

    print("\n⚠ QUAN TRỌNG - thực hiện trước khi crawl tiếp tục:")
    print("   1. Đăng nhập tài khoản Shopee (giúp giảm tỷ lệ bị chặn/captcha)")
    print("   2. Giải captcha nếu có")
    print("   3. Chương trình tự tiếp tục sau 20 giây...")
    _GLOBAL_PAGE.get("https://shopee.vn")
    time.sleep(20)

    return _GLOBAL_PAGE


def close_browser():
    """Đóng trình duyệt sau khi crawl xong."""
    global _GLOBAL_PAGE
    if _GLOBAL_PAGE:
        try:
            _GLOBAL_PAGE.quit()
        except Exception as e:
            logging.error(f"Lỗi khi đóng Chrome: {e}")
        _GLOBAL_PAGE = None


def _smart_delay(kind: str = "normal"):
    """Delay ngẫu nhiên mô phỏng hành vi người dùng thật."""
    ranges = {"quick": (1.5, 3), "normal": (3, 7), "careful": (5, 12)}
    lo, hi = ranges.get(kind, (3, 7))
    time.sleep(random.uniform(lo, hi))


def crawl_category_shopee(cat: dict, cookies=None, max_pages: int = 5, retries: int = 2) -> list:
    """Crawl toàn bộ sản phẩm của một danh mục Shopee bằng cách bắt response API nội bộ."""
    page = get_browser()
    name = cat["name"]
    products: list = []

    logging.info(f"Bắt đầu crawl Shopee: {name}")

    try:
        if page.url != "https://shopee.vn/":
            page.get("https://shopee.vn")
        time.sleep(2)

        page.listen.start("search_items")
        page.get(cat["url"])
        time.sleep(5)
    except Exception as e:
        logging.error(f"Lỗi truy cập danh mục {name}: {e}")
        return []

    for page_num in range(1, max_pages + 1):
        if page.ele("text:Trang không khả dụng", timeout=1) or page.ele("text:Traffic Error", timeout=1):
            logging.warning("Bị chặn bởi Shopee. Hãy giải captcha thủ công.")
            input("Nhấn Enter sau khi xử lý xong để tiếp tục...")
            page.listen.start("search_items")
            page.refresh()
            time.sleep(5)

        page.scroll.to_bottom()
        _smart_delay("quick")

        found_in_page = 0
        try:
            for packet in page.listen.steps(timeout=8):
                body = packet.response.body  # type: ignore
                if not isinstance(body, dict) or "items" not in body:
                    continue

                for item in body["items"]:
                    basic = item.get("item_basic", item)
                    itemid = basic.get("itemid")
                    shopid = basic.get("shopid")

                    raw_rating = basic.get("item_rating", {}).get("rating_star", 0)
                    try:
                        rating_star = math.floor(float(raw_rating) * 10) / 10 if raw_rating is not None else 0.0
                    except (TypeError, ValueError):
                        rating_star = 0.0

                    sold_info = basic.get("item_card_display_sold_count", {})

                    product = {
                        "crawl_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "platform": "Shopee",
                        "category_name": name,
                        "id": str(itemid),
                        "name": basic.get("name", "N/A"),
                        "price": (basic.get("price", 0) or 0) / 100000,
                        "original_price": (basic.get("price_before_discount", 0) or 0) / 100000,
                        "discount_rate": basic.get("discount", ""),
                        "rating_average": rating_star,
                        "review_count": basic.get("cmt_count"),
                        "quantity_sold_value": basic.get("historical_sold", 0),
                        "quantity_sold_text": sold_info.get("display_sold_count_text", ""),
                        "brand": str(basic.get("brand")),
                        "location": basic.get("shop_location", "N/A"),
                        "seller_name": basic.get("shop_name"),
                        "url": f"https://shopee.vn/product/{shopid}/{itemid}",
                    }

                    if not any(p["id"] == product["id"] for p in products):
                        products.append(product)
                        found_in_page += 1
        except Exception as e:
            logging.warning(f"Lỗi đọc gói tin network: {e}")

        logging.info(f"  Trang {page_num}: +{found_in_page} sản phẩm mới (tổng {len(products)})")

        if page_num < max_pages:
            btn_next = page.ele(".shopee-icon-button--right:not(.shopee-button-disabled)", timeout=2)
            if not btn_next:
                logging.info("Hết trang hoặc không tìm thấy nút Next.")
                break
            btn_next.click()
            _smart_delay("careful")

    return products


if __name__ == "__main__":
    config.ensure_all_dirs()
    try:
        crawl_all_categories(
            platform_name="Shopee",
            categories=SHOPEE_CATEGORIES,
            crawl_category_func=crawl_category_shopee,
            category_output_dir=config.RAW_SHOPEE_DIR,
            get_cookies_func=None,
            max_pages=config.MAX_PAGES,
            retries=2,
            sleep_min=config.SLEEP_MIN,
            sleep_max=config.SLEEP_MAX,
        )
    finally:
        close_browser()
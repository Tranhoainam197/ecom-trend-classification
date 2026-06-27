"""
base_crawler.py
================
Các hàm dùng chung cho việc crawl dữ liệu từ nhiều sàn TMĐT (Shopee, Lazada, Tiki).

Đây thuộc bước 0 - COLLECTION DATA của quy trình khai thác dữ liệu:
    Collection -> Cleaning -> Integration -> Transformation -> Normalization -> Encoding
"""

import os
import time
import random
import logging
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_fresh_cookies(url: str, headless: bool = True, scroll: bool = False, wait_time: int = 10) -> dict:
    """
    Mở trình duyệt Chrome (Selenium) để lấy cookie hợp lệ từ một URL.
    Dùng cho các sàn cần cookie để gọi API (Tiki, Lazada).

    Parameters
    ----------
    url : str
        URL cần truy cập để lấy cookie.
    headless : bool
        Chạy chế độ ẩn trình duyệt hay không.
    scroll : bool
        Có scroll trang xuống để trigger lazy-load không.
    wait_time : int
        Thời gian (giây) chờ trang tải xong.

    Returns
    -------
    dict
        Dictionary cookie {name: value}, rỗng nếu lỗi.
    """
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(wait_time)

        if scroll:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(5)

        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
        logging.info(f"Lấy cookie thành công: {len(cookies)} cookie từ {url}")
        return cookies
    except Exception as e:
        logging.error(f"Lỗi lấy cookie từ {url}: {e}")
        return {}
    finally:
        driver.quit()


def crawl_all_categories(
    *,
    platform_name: str,
    categories: list,
    crawl_category_func,
    category_output_dir: str,
    get_cookies_func=None,
    max_pages: int = 20,
    retries: int = 2,
    sleep_min: int = 8,
    sleep_max: int = 15,
):
    """
    Vòng lặp crawl tất cả danh mục của một platform, lưu mỗi danh mục
    thành 1 file JSON riêng trong category_output_dir.

    Mỗi platform (Shopee/Lazada/Tiki) chỉ cần viết hàm crawl_category_func
    riêng cho logic lấy dữ liệu, hàm này chỉ điều phối việc lặp + lưu file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(category_output_dir, exist_ok=True)

    cookies = get_cookies_func() if get_cookies_func else None

    logging.info(f"Bắt đầu crawl {platform_name} ({len(categories)} danh mục)")

    total_products = 0
    for cat in categories:
        try:
            products = crawl_category_func(cat, cookies=cookies, max_pages=max_pages, retries=retries)
        except Exception as e:
            logging.error(f"Lỗi khi crawl danh mục {cat}: {e}")
            products = []

        if products:
            total_products += len(products)
            cat_name_safe = cat.get("name", "unknown").replace("/", "_").replace(" ", "_")
            out_path = os.path.join(
                category_output_dir,
                f"{platform_name.lower()}_{cat_name_safe}_{timestamp}.json",
            )
            pd.DataFrame(products).to_json(out_path, orient="records", force_ascii=False, indent=2)
            logging.info(f"{cat.get('name')}: lưu {len(products)} sản phẩm -> {out_path}")
        else:
            logging.warning(f"{cat.get('name')}: không lấy được sản phẩm nào")

        time.sleep(random.uniform(sleep_min, sleep_max))

    logging.info(f"Hoàn thành crawl {platform_name}: tổng {total_products:,} sản phẩm")
    return total_products
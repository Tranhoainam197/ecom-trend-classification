"""
predict_demo.py
==================
DEMO TRỰC QUAN — Phân loại xu hướng sản phẩm TMĐT
=====================================================
Ứng dụng CLI (giao diện terminal, dùng thư viện `rich`) để demo trực quan
mô hình phân loại đã train, phục vụ thuyết trình đồ án.

Nguyên tắc quan trọng của demo này (để đảm bảo TRUNG THỰC khi trình bày):
  - Chỉ dự đoán trên các sản phẩm thuộc TẬP TEST (data/encoding/encoded_test.json)
    — tức là dữ liệu mô hình CHƯA từng thấy trong lúc train. Không có việc
    "diễn" kết quả bằng dữ liệu đã học thuộc.
  - Toàn bộ feature đưa vào model được lấy nguyên từ kết quả bước
    Normalization + Encoding của chính pipeline (không tự chế thêm), vì bộ
    feature "realistic" có nhiều feature ngữ cảnh (percentile theo category,
    velocity theo thời gian crawl...) không thể tính đúng nếu nhập tay rời
    rạc từng trường mà không có toàn bộ tập dữ liệu category tương ứng.

Chạy: python run_demo.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich import box

import config
from src.demo.data_loader import load_everything as _load_everything

console = Console()

# ----------------------------------------------------------------------
# Giao diện: icon + màu theo từng nhãn, để hiển thị nhất quán xuyên suốt
# ----------------------------------------------------------------------
LABEL_STYLE = {
    "Hot Trend":   {"icon": "🔥", "color": "bold red",     "bar": "red3"},
    "Best Seller": {"icon": "🏆", "color": "bold yellow",  "bar": "gold3"},
    "Best Deal":   {"icon": "💰", "color": "bold green",   "bar": "green3"},
    "Normal":      {"icon": "📦", "color": "grey62",       "bar": "grey50"},
}


def _style(label: str) -> dict:
    return LABEL_STYLE.get(label, {"icon": "❔", "color": "white", "bar": "white"})


# ----------------------------------------------------------------------
# Load model + dữ liệu test, dự đoán 1 lần cho toàn bộ tập test (nhanh),
# lưu vào bộ nhớ để mọi thao tác trong menu (random/tìm kiếm/lọc) đều tức thì.
# ----------------------------------------------------------------------
def load_everything():
    with console.status("[bold cyan]Đang tải mô hình và dữ liệu test...", spinner="dots"):
        return _load_everything()


# ----------------------------------------------------------------------
# Các thành phần hiển thị (rich)
# ----------------------------------------------------------------------
def print_banner(model_name: str, n_test: int):
    title = Text()
    title.append("  E-COMMERCE TREND CLASSIFIER  ", style="bold white on dark_green")
    subtitle = Text(f"Demo dự đoán trực tiếp • Mô hình: {model_name} • {n_test:,} sản phẩm trong tập TEST",
                     style="italic grey70")
    console.print()
    console.print(Align.center(title))
    console.print(Align.center(subtitle))
    console.print(Rule(style="green"))


def _fmt_money(v):
    if v is None:
        return "N/A"
    return f"{v:,.0f} đ"


def print_product_card(rec: dict):
    info = Table(box=None, show_header=False, padding=(0, 1))
    info.add_column(style="bold grey70", justify="right")
    info.add_column(style="white")

    info.add_row("Nền tảng:", rec["platform"])
    info.add_row("Danh mục:", rec["category"])
    info.add_row("Thương hiệu:", str(rec["brand"]))
    info.add_row("Giá bán:", f"{_fmt_money(rec['current_price'])}  "
                              f"[strike grey50]{_fmt_money(rec['original_price'])}[/]  "
                              f"[bold red](-{rec['discount_rate']:.0f}%)[/]" if rec["discount_rate"] else _fmt_money(rec["current_price"]))
    info.add_row("Đánh giá:", f"⭐ {rec['rating_average']:.1f}  ({rec['num_reviews']:,} lượt review)"
                              if rec["rating_average"] is not None else "N/A")
    info.add_row("Đã bán:", f"{rec['quantity_sold']:,.0f}" if rec["quantity_sold"] is not None else "N/A")
    info.add_row("Phân khúc giá:", str(rec["price_segment"]))
    info.add_row("Chất lượng:", str(rec["quality_tier"]))
    info.add_row("Mức giảm giá:", str(rec["discount_intensity"]))

    name = rec["product_name"]
    name = name if len(name) <= 90 else name[:87] + "..."
    console.print(Panel(info, title=f"[bold]{name}[/]", title_align="left",
                         border_style="grey42", box=box.ROUNDED))


def _prob_bar(label: str, prob: float, width: int = 30) -> Text:
    filled = int(round(prob * width))
    style = _style(label)["bar"]
    bar = Text()
    bar.append(f"{label:<12}", style="grey70")
    bar.append("█" * filled, style=style)
    bar.append("░" * (width - filled), style="grey15")
    bar.append(f"  {prob * 100:5.1f}%", style="bold white")
    return bar


def print_prediction(rec: dict):
    pred, actual = rec["predicted_label"], rec["actual_label"]
    pred_style = _style(pred)

    header = Text()
    header.append(f"{pred_style['icon']}  DỰ ĐOÁN: ", style="bold white")
    header.append(pred, style=pred_style["color"])

    result_line = Text()
    if rec["correct"]:
        result_line.append("✅ ĐÚNG", style="bold green")
        result_line.append(f"  — nhãn thực tế cũng là ", style="grey70")
        result_line.append(actual, style=_style(actual)["color"])
    else:
        result_line.append("❌ SAI", style="bold red")
        result_line.append(f"  — nhãn thực tế là ", style="grey70")
        result_line.append(actual, style=_style(actual)["color"])

    body = Table.grid(padding=(0, 0))
    body.add_row(header)
    body.add_row(result_line)
    body.add_row(Text(""))
    for label, prob in sorted(rec["probabilities"].items(), key=lambda x: -x[1]):
        body.add_row(_prob_bar(label, prob))

    border = "green" if rec["correct"] else "red"
    console.print(Panel(body, title="Kết quả mô hình", title_align="left",
                         border_style=border, box=box.HEAVY))


# ----------------------------------------------------------------------
# Các màn hình / chức năng của menu
# ----------------------------------------------------------------------
def screen_random(records):
    rec = random.choice(records)
    console.print()
    print_product_card(rec)
    print_prediction(rec)


def screen_search(records):
    keyword = Prompt.ask("\n[bold cyan]Nhập từ khóa tên sản phẩm[/]").strip().lower()
    if not keyword:
        return
    matches = [r for r in records if keyword in r["product_name"].lower()]
    if not matches:
        console.print("[red]Không tìm thấy sản phẩm nào khớp trong tập test.[/]")
        return
    _pick_from_list(matches[:20])


def screen_browse_category(records):
    categories = sorted(set(r["category"] for r in records))
    table = Table(title="Danh mục sản phẩm (trong tập test)", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Danh mục")
    table.add_column("Số sản phẩm", justify="right")
    for i, cat in enumerate(categories, 1):
        count = sum(1 for r in records if r["category"] == cat)
        table.add_row(str(i), cat, str(count))
    console.print(table)

    idx = IntPrompt.ask("\n[bold cyan]Chọn số thứ tự danh mục[/]", default=0)
    if not (1 <= idx <= len(categories)):
        return
    chosen = categories[idx - 1]
    matches = [r for r in records if r["category"] == chosen]
    _pick_from_list(matches[:30])


def screen_wrong_predictions(records):
    wrong = [r for r in records if not r["correct"]]
    console.print(f"\n[bold red]Có {len(wrong)} / {len(records)} sản phẩm mô hình dự đoán SAI trên tập test "
                  f"(accuracy thực tế ≈ {100 * (1 - len(wrong) / len(records)):.1f}%)[/]\n")
    if not wrong:
        return
    _pick_from_list(wrong[:30])


def _pick_from_list(matches):
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Tên sản phẩm")
    table.add_column("Nền tảng")
    table.add_column("Thực tế")
    table.add_column("Dự đoán")
    table.add_column("Kết quả", justify="center")

    for i, r in enumerate(matches, 1):
        name = r["product_name"]
        name = name if len(name) <= 55 else name[:52] + "..."
        mark = "[green]✅[/]" if r["correct"] else "[red]❌[/]"
        table.add_row(str(i), name, r["platform"],
                      f"[{_style(r['actual_label'])['color']}]{r['actual_label']}[/]",
                      f"[{_style(r['predicted_label'])['color']}]{r['predicted_label']}[/]",
                      mark)
    console.print(table)

    idx = IntPrompt.ask("\n[bold cyan]Chọn số thứ tự để xem chi tiết (0 để bỏ qua)[/]", default=0)
    if 1 <= idx <= len(matches):
        console.print()
        print_product_card(matches[idx - 1])
        print_prediction(matches[idx - 1])


def screen_model_stats():
    from src.demo.data_loader import load_model_comparison
    rows = load_model_comparison()
    if not rows:
        console.print("[red]Chưa có file model_comparison.csv — hãy chạy run_pipeline.py trước.[/]")
        return

    table = Table(title="So sánh hiệu năng các mô hình (trên tập TEST)", box=box.ROUNDED)
    table.add_column("Bộ feature")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1-macro", justify="right")

    for row in rows:
        fset = row["feature_set"]
        style = "grey50" if fset == "full" else ("bold green" if fset == "realistic" else "yellow")
        note = " ⚠ leakage" if fset == "full" else (" ✓ dùng thực tế" if fset == "realistic" else "")
        table.add_row(f"[{style}]{fset}{note}[/]", row["model"], f"{float(row['accuracy']) * 100:.2f}%",
                      f"{float(row['f1_macro']) * 100:.2f}%")

    console.print(table)
    console.print(
        "\n[grey70]Ghi chú:[/] bộ [grey50]full[/] chứa các composite score dùng để TẠO ra nhãn "
        "(popularity_score, engagement_score...) → accuracy cao [bold]giả tạo[/] do label leakage.\n"
        "Bộ [bold green]realistic[/] đã loại các score này → phản ánh đúng khả năng dự đoán "
        "cho sản phẩm mới, thực sự chưa biết nhãn trước."
    )


def print_menu():
    console.print()
    console.print(Rule(style="grey42"))
    menu = Table.grid(padding=(0, 2))
    menu.add_column()
    menu.add_column()
    menu.add_row("[bold cyan]1.[/] 🎲 Sản phẩm ngẫu nhiên",     "[bold cyan]4.[/] ❌ Xem các ca dự đoán SAI")
    menu.add_row("[bold cyan]2.[/] 🔍 Tìm theo tên sản phẩm",   "[bold cyan]5.[/] 📊 So sánh hiệu năng model")
    menu.add_row("[bold cyan]3.[/] 📂 Duyệt theo danh mục",      "[bold cyan]0.[/] 🚪 Thoát")
    console.print(menu)


def run_demo_app():
    console.print("\n[bold cyan]Đang khởi động demo...[/]")
    try:
        records, class_names, model_name = load_everything()
    except FileNotFoundError as e:
        console.print(f"[bold red]Thiếu file dữ liệu: {e}[/]")
        console.print("[yellow]Hãy chạy 'python run_pipeline.py' trước để tạo model và dữ liệu test.[/]")
        return

    print_banner(model_name, len(records))

    actions = {
        "1": screen_random,
        "2": screen_search,
        "3": screen_browse_category,
        "4": screen_wrong_predictions,
    }

    while True:
        print_menu()
        choice = Prompt.ask("[bold cyan]Chọn chức năng[/]", choices=["0", "1", "2", "3", "4", "5"], default="1")
        if choice == "0":
            console.print("\n[bold green]Tạm biệt! 👋[/]\n")
            break
        elif choice == "5":
            screen_model_stats()
        else:
            actions[choice](records)


if __name__ == "__main__":
    run_demo_app()

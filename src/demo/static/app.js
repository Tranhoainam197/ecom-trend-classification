/* ==========================================================================
   State
   ========================================================================== */
const state = {
  records: [],
  classNames: [],
  modelName: "",
  comparison: [],
  accuracy: 0,
  mode: "random",
  currentList: [],
  selectedId: null,
  activeCategory: null,
};

const LABEL_META = {
  "Hot Trend": { icon: "🔥", color: "var(--c-hot)" },
  "Best Seller": { icon: "🏆", color: "var(--c-seller)" },
  "Best Deal": { icon: "💰", color: "var(--c-deal)" },
  "Normal": { icon: "📦", color: "var(--c-normal)" },
};

function labelMeta(label) {
  return LABEL_META[label] || { icon: "❔", color: "var(--text-muted)" };
}

/* ==========================================================================
   Formatting helpers
   ========================================================================== */
const moneyFmt = new Intl.NumberFormat("vi-VN");
function fmtMoney(v) {
  if (v === null || v === undefined) return "N/A";
  return moneyFmt.format(Math.round(v)) + " đ";
}
function fmtPct(v, digits = 1) {
  return (v * 100).toFixed(digits) + "%";
}
function fmtNum(v) {
  if (v === null || v === undefined) return "N/A";
  return moneyFmt.format(Math.round(v));
}
function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

/* ==========================================================================
   Boot
   ========================================================================== */
async function boot() {
  const res = await fetch("/api/data");
  const data = await res.json();
  state.records = data.records;
  state.classNames = data.class_names;
  state.modelName = data.model_name;
  state.comparison = data.comparison;
  state.accuracy = data.accuracy;

  document.getElementById("model-name-badge").textContent = state.modelName;
  document.getElementById("accuracy-badge").textContent = state.accuracy.toFixed(2) + "%";
  document.getElementById("footer-count").textContent = data.n_test.toLocaleString("vi-VN");

  document.querySelectorAll(".nav__item").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  setMode("random");
}

/* ==========================================================================
   Mode switching
   ========================================================================== */
function setMode(mode) {
  state.mode = mode;
  state.selectedId = null;
  document.querySelectorAll(".nav__item").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.mode === mode);
  });
  renderSidebarContext(mode);
  renderMain(mode);
}

function renderSidebarContext(mode) {
  const el = document.getElementById("sidebar-context");
  if (mode === "search") {
    el.innerHTML = `
      <input class="search-input" id="search-input" type="text"
             placeholder="Nhập tên sản phẩm…" autocomplete="off">
      <p class="context-note" style="margin-top:10px;">Tìm trong ${state.records.length.toLocaleString("vi-VN")} sản phẩm tập test.</p>
    `;
    const input = document.getElementById("search-input");
    input.addEventListener("input", () => renderSearchResults(input.value));
    input.focus();
  } else if (mode === "category") {
    const counts = {};
    state.records.forEach((r) => { counts[r.category] = (counts[r.category] || 0) + 1; });
    const cats = Object.keys(counts).sort();
    el.innerHTML = `
      <p class="context-note" style="margin-bottom:8px;">${cats.length} danh mục</p>
      <div class="cat-list">
        ${cats.map((c) => `
          <button class="cat-item" data-cat="${escapeAttr(c)}">
            <span>${escapeHtml(c)}</span>
            <span class="cat-item__count">${counts[c]}</span>
          </button>`).join("")}
      </div>
    `;
    el.querySelectorAll(".cat-item").forEach((btn) => {
      btn.addEventListener("click", () => selectCategory(btn.dataset.cat, el));
    });
  } else if (mode === "wrong") {
    const wrong = state.records.filter((r) => !r.correct);
    const acc = (100 * (1 - wrong.length / state.records.length)).toFixed(1);
    el.innerHTML = `
      <p class="context-note">
        <strong>${wrong.length}</strong> / ${state.records.length} sản phẩm bị dự đoán sai
        trên tập test (accuracy thực tế ≈ ${acc}%).
      </p>`;
  } else if (mode === "compare") {
    el.innerHTML = `<p class="context-note">So sánh 15 tổ hợp model × bộ feature, đo trên cùng tập test.</p>`;
  } else {
    el.innerHTML = `<p class="context-note">Chọn ngẫu nhiên 1 sản phẩm trong tập test để xem mô hình dự đoán.</p>`;
  }
}

function selectCategory(cat, sidebarEl) {
  state.activeCategory = cat;
  sidebarEl.querySelectorAll(".cat-item").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.cat === cat);
  });
  const list = state.records.filter((r) => r.category === cat);
  renderSplitView(list, `Danh mục: ${cat}`);
}

/* ==========================================================================
   Main area rendering per mode
   ========================================================================== */
function renderMain(mode) {
  const main = document.getElementById("main");
  if (mode === "random") {
    main.innerHTML = `
      <button class="reroll-btn" id="reroll-btn">🎲 Chọn sản phẩm ngẫu nhiên khác</button>
      <div id="random-detail"></div>
    `;
    document.getElementById("reroll-btn").addEventListener("click", showRandom);
    showRandom();
  } else if (mode === "search") {
    main.innerHTML = `<div class="empty-state">Nhập từ khóa ở thanh bên trái để tìm sản phẩm.</div>`;
  } else if (mode === "category") {
    main.innerHTML = `<div class="empty-state">Chọn 1 danh mục ở thanh bên trái.</div>`;
  } else if (mode === "wrong") {
    const wrong = state.records.filter((r) => !r.correct);
    renderSplitView(wrong, "Các ca dự đoán sai");
  } else if (mode === "compare") {
    renderCompare(main);
  }
}

function showRandom() {
  const rec = state.records[Math.floor(Math.random() * state.records.length)];
  const container = document.getElementById("random-detail");
  container.innerHTML = buildDetailHTML(rec);
  animateBars(container);
}

function renderSearchResults(query) {
  const main = document.getElementById("main");
  const q = query.trim().toLowerCase();
  if (!q) {
    main.innerHTML = `<div class="empty-state">Nhập từ khóa ở thanh bên trái để tìm sản phẩm.</div>`;
    return;
  }
  const matches = state.records.filter((r) => r.product_name.toLowerCase().includes(q)).slice(0, 60);
  renderSplitView(matches, `Kết quả cho "${query}"`, main);
}

/* ---- Split layout: results list (left) + detail panel (right) ---------- */
function renderSplitView(list, title, mainEl) {
  const main = mainEl || document.getElementById("main");
  state.currentList = list;

  if (list.length === 0) {
    main.innerHTML = `<div class="empty-state">Không tìm thấy sản phẩm nào phù hợp.</div>`;
    return;
  }

  main.innerHTML = `
    <div class="main--split">
      <div class="panel results">
        <p class="results__header">${escapeHtml(title)} — ${list.length.toLocaleString("vi-VN")} kết quả</p>
        <div id="results-list"></div>
      </div>
      <div class="panel detail-panel" id="detail-panel">
        <div class="empty-state">Chọn 1 sản phẩm bên trái để xem chi tiết dự đoán.</div>
      </div>
    </div>
  `;

  const listEl = document.getElementById("results-list");
  listEl.innerHTML = list.map((r, i) => buildResultRowHTML(r, i)).join("");
  listEl.querySelectorAll(".result-row").forEach((row) => {
    row.addEventListener("click", () => {
      const idx = parseInt(row.dataset.idx, 10);
      selectFromList(list, idx);
    });
  });

  // Chọn sẵn kết quả đầu tiên để demo mượt hơn (không cần bấm thêm 1 lần).
  selectFromList(list, 0);
}

function selectFromList(list, idx) {
  const rec = list[idx];
  state.selectedId = rec.id + "|" + rec.platform;

  document.querySelectorAll(".result-row").forEach((row) => {
    row.classList.toggle("is-selected", parseInt(row.dataset.idx, 10) === idx);
  });

  const panel = document.getElementById("detail-panel");
  panel.innerHTML = buildDetailHTML(rec);
  animateBars(panel);
}

function buildResultRowHTML(r, idx) {
  const verdict = r.correct ? `<span style="color:var(--ok)">✅</span>` : `<span style="color:var(--bad)">❌</span>`;
  return `
    <button class="result-row" data-idx="${idx}">
      <div>
        <div class="result-row__name">${escapeHtml(r.product_name)}</div>
        <div class="result-row__meta">${escapeHtml(r.platform)} · ${escapeHtml(r.actual_label)} → ${escapeHtml(r.predicted_label)}</div>
      </div>
      <div class="result-row__verdict">${verdict}</div>
    </button>
  `;
}

/* ---- Detail card: product info + prediction bars ------------------------ */
function buildDetailHTML(r) {
  const predMeta = labelMeta(r.predicted_label);
  const stampClass = r.correct ? "" : "is-wrong";
  const stampText = r.correct ? "✓ Đúng" : "✗ Sai";

  const discountHtml = r.discount_rate
    ? `<span class="strike">${fmtMoney(r.original_price)}</span><span class="discount-chip">-${Math.round(r.discount_rate)}%</span>`
    : "";

  const sortedProbs = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);

  return `
    <div class="detail">
      <div class="stamp ${stampClass}">${stampText}</div>
      <h2 class="product-name">${escapeHtml(r.product_name)}</h2>

      <div class="tags">
        <span class="tag">${escapeHtml(r.platform)}</span>
        <span class="tag">${escapeHtml(r.category)}</span>
        <span class="tag">${escapeHtml(String(r.brand))}</span>
      </div>

      <div class="product-grid">
        <div class="product-field">
          <span class="product-field__label">Giá bán</span>
          <span class="product-field__value mono">${fmtMoney(r.current_price)} ${discountHtml}</span>
        </div>
        <div class="product-field">
          <span class="product-field__label">Đánh giá</span>
          <span class="product-field__value mono">${r.rating_average != null ? "⭐ " + r.rating_average.toFixed(1) : "N/A"} · ${fmtNum(r.num_reviews)} review</span>
        </div>
        <div class="product-field">
          <span class="product-field__label">Đã bán</span>
          <span class="product-field__value mono">${fmtNum(r.quantity_sold)}</span>
        </div>
        <div class="product-field">
          <span class="product-field__label">Phân khúc / chất lượng</span>
          <span class="product-field__value">${escapeHtml(String(r.price_segment))} · ${escapeHtml(String(r.quality_tier))}</span>
        </div>
      </div>

      <div class="divider"></div>

      <div class="predict-header">
        <span class="predict-title">Kết quả mô hình</span>
      </div>
      <div class="predict-result" style="color:${predMeta.color}">
        <span>${predMeta.icon}</span><span>${escapeHtml(r.predicted_label)}</span>
      </div>
      <p class="actual-line">Nhãn thực tế: <strong>${escapeHtml(r.actual_label)}</strong></p>

      <div class="bars">
        ${sortedProbs.map(([label, p]) => `
          <div class="bar-row">
            <div class="bar-row__label">${labelMeta(label).icon} ${escapeHtml(label)}</div>
            <div class="bar-track"><div class="bar-fill" data-target="${(p * 100).toFixed(1)}"
                 style="background:${labelMeta(label).color}"></div></div>
            <div class="bar-row__pct">${fmtPct(p)}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function animateBars(container) {
  const fills = container.querySelectorAll(".bar-fill");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fills.forEach((el) => { el.style.width = el.dataset.target + "%"; });
    });
  });
}

/* ---- Compare mode --------------------------------------------------------- */
function renderCompare(main) {
  const rows = state.comparison;
  if (!rows || rows.length === 0) {
    main.innerHTML = `<div class="empty-state">Chưa có model_comparison.csv — hãy chạy run_pipeline.py trước.</div>`;
    return;
  }

  const bySet = {};
  rows.forEach((r) => {
    const fs = r.feature_set;
    const acc = parseFloat(r.accuracy);
    if (!bySet[fs] || acc > bySet[fs].accuracy) bySet[fs] = { ...r, accuracy: acc };
  });
  const full = bySet["full"];
  const realistic = bySet["realistic"];
  const gap = full && realistic ? (full.accuracy - realistic.accuracy) * 100 : null;

  const chipClass = { full: "fset-chip--full", realistic: "fset-chip--realistic", realistic_strict: "fset-chip--realistic_strict" };
  const chipLabel = { full: "full", realistic: "realistic", realistic_strict: "strict" };

  main.innerHTML = `
    <p class="section-title">So sánh hiệu năng — Full (có leakage) vs Realistic (không leakage)</p>
    <div class="hero-stat">
      <div class="hero-card hero-card--leak">
        <div class="hero-card__label">Full features</div>
        <div class="hero-card__value">${full ? (full.accuracy * 100).toFixed(2) : "—"}%</div>
        <div class="hero-card__sub">${full ? full.model : ""} · chứa composite score → leakage</div>
      </div>
      <div class="hero-card hero-card--real">
        <div class="hero-card__label">Realistic features</div>
        <div class="hero-card__value">${realistic ? (realistic.accuracy * 100).toFixed(2) : "—"}%</div>
        <div class="hero-card__sub">${realistic ? realistic.model : ""} · dùng cho ứng dụng thực tế</div>
      </div>
      <div class="hero-card hero-card--gap">
        <div class="hero-card__label">Chênh lệch</div>
        <div class="hero-card__value">${gap != null ? gap.toFixed(2) : "—"} đ.%</div>
        <div class="hero-card__sub">càng lớn → leakage càng rõ</div>
      </div>
    </div>

    <div class="panel">
      <table class="compare-table">
        <thead>
          <tr>
            <th>Bộ feature</th><th>Model</th>
            <th class="num">Accuracy</th><th class="num">Precision</th>
            <th class="num">Recall</th><th class="num">F1-macro</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((r) => `
            <tr>
              <td><span class="fset-chip ${chipClass[r.feature_set] || ""}">${chipLabel[r.feature_set] || r.feature_set}</span></td>
              <td>${escapeHtml(r.model)}</td>
              <td class="num">${(parseFloat(r.accuracy) * 100).toFixed(2)}%</td>
              <td class="num">${(parseFloat(r.precision_macro) * 100).toFixed(2)}%</td>
              <td class="num">${(parseFloat(r.recall_macro) * 100).toFixed(2)}%</td>
              <td class="num">${(parseFloat(r.f1_macro) * 100).toFixed(2)}%</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

/* ==========================================================================
   Utils
   ========================================================================== */
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
function escapeAttr(s) {
  return escapeHtml(s).replaceAll('"', "&quot;");
}

boot();

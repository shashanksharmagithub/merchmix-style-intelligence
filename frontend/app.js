// The API and frontend are served by the same FastAPI deployment.
const API_BASE = "";

const state = {
  styles: [],
  mode: "forecast",
  selectedSeason: null,
  selectedStyleId: null,
};

const styleList = document.querySelector("#styleList");
const detail = document.querySelector("#detail");
const statusBox = document.querySelector("#status");
const generatedAt = document.querySelector("#generatedAt");
const refreshButton = document.querySelector("#refreshButton");
const listTitle = document.querySelector("#listTitle");
const modeTabs = document.querySelectorAll(".modeTab");

function showStatus(message) {
  statusBox.hidden = !message;
  statusBox.textContent = message || "";
}

function imageTag(src, className, alt) {
  if (src) {
    return `<img class="${className}" src="${API_BASE}${src}" alt="${alt}" />`;
  }
  return `<div class="${className}" role="img" aria-label="${alt}"></div>`;
}

function formatScore(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function updateTabs() {
  modeTabs.forEach((tab) => {
    const active =
      (state.mode === "forecast" && tab.dataset.mode === "forecast") ||
      (state.mode === "season" && tab.dataset.season === state.selectedSeason);
    tab.classList.toggle("active", active);
  });
}

function renderList() {
  updateTabs();
  styleList.innerHTML = state.styles
    .map((style) => {
      const active = style.style_id === state.selectedStyleId ? " active" : "";
      const metric = state.mode === "forecast" ? formatScore(style.prediction_score) : formatNumber(style.season_units);
      const metricLabel = state.mode === "forecast" ? "" : " units";
      const performanceLine =
        state.mode === "forecast"
          ? `History: ${formatNumber(style.sales_history)} units · Recent: ${formatNumber(style.recent_sales)}`
          : `${style.season_label} sales: ${formatNumber(style.season_units)} units · Customers: ${formatNumber(style.season_customers)}`;
      return `
        <button class="styleCard${active}" data-style-id="${style.style_id}">
          ${imageTag(style.image_url, "thumb", style.name || style.style_id)}
          <span>
            <span class="rankLine">
              <span>#${style.rank} ${style.name || "Unnamed style"}</span>
              <span class="score">${metric}${metricLabel}</span>
            </span>
            <span class="meta">${style.category || "Category unavailable"} · ${style.colour || "Colour unavailable"}</span>
            <span class="meta">${performanceLine}</span>
          </span>
        </button>
      `;
    })
    .join("");

  styleList.querySelectorAll(".styleCard").forEach((button) => {
    button.addEventListener("click", () => selectStyle(button.dataset.styleId));
  });
}

function renderDetail(style) {
  if (!style) {
    detail.innerHTML = `
      <div class="emptyState">
        <h2>Select a style</h2>
        <p>The recommendation details will appear here after predictions are loaded.</p>
      </div>
    `;
    return;
  }

  const attrs = style.attributes || {};
  const hist = style.historical_performance || {};
  const reasons = style.model_explanation || [];
  const isSeason = state.mode === "season";
  const scorePill = isSeason
    ? `${formatNumber(style.season_units)} units`
    : `Rank score ${formatScore(style.prediction_score)}`;
  const stats = isSeason
    ? [
        [formatNumber(style.season_units), `${style.season_label} units`],
        [formatNumber(style.season_customers), "Season customers"],
        [formatNumber(style.active_days), "Active selling days"],
        [`${Math.round(Number(style.season_sales_share || 0) * 1000) / 10}%`, "Season sales share"],
      ]
    : [
        [formatNumber(style.predicted_units), "Predicted units"],
        [formatNumber(style.sales_history), "Historical units"],
        [formatNumber(style.recent_sales), "Recent units"],
        [formatNumber(hist.active_days), "Active selling days"],
      ];
  detail.innerHTML = `
    <div class="detailHero">
      ${imageTag(style.image_url, "heroImage", style.name || style.style_id)}
      <div>
        <div class="detailTitle">
          <div>
            <p class="eyebrow">Style ${style.style_id}</p>
            <h2>${style.name || "Unnamed style"}</h2>
          </div>
          <span class="pill">${scorePill}</span>
        </div>
        <p>${style.description || "No catalogue description available."}</p>
        <div class="stats">
          ${stats.map(([value, label]) => `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`).join("")}
        </div>
        <p class="meta">${
          isSeason
            ? `Seasonal range observed: ${style.first_sale} to ${style.last_sale}`
            : `Feature window: ${(hist.feature_window || []).join(" to ")} · Prediction window: ${(hist.prediction_window || []).join(" to ")}`
        }</p>
      </div>
    </div>
    <div class="reasoning">
      <h3>${isSeason ? "Seasonal concept direction" : "Model-based selection signals"}</h3>
      ${
        isSeason
          ? `<p>${style.seasonal_concept_direction || "No seasonal concept direction available."}</p>`
          : `<ul>${reasons.map((reason) => `<li>${reason}</li>`).join("")}</ul><p class="meta">${style.explanation_method || ""}</p>`
      }
    </div>
    <div class="attributes">
      <h3>Product attributes</h3>
      <div class="attrGrid">
        <div class="attr"><span>Category</span>${style.category || "-"}</div>
        <div class="attr"><span>Garment group</span>${style.garment_group || "-"}</div>
        <div class="attr"><span>Product group</span>${attrs.product_group || "-"}</div>
        <div class="attr"><span>Appearance</span>${attrs.graphical_appearance || "-"}</div>
        <div class="attr"><span>Department</span>${attrs.department || "-"}</div>
        <div class="attr"><span>Section</span>${attrs.section || "-"}</div>
      </div>
    </div>
  `;
}

async function selectStyle(styleId) {
  state.selectedStyleId = styleId;
  renderList();
  if (state.mode === "season") {
    renderDetail(state.styles.find((style) => String(style.style_id) === String(styleId)));
    return;
  }
  try {
    const response = await fetch(`${API_BASE}/styles/${styleId}`);
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const style = await response.json();
    renderDetail(style);
  } catch (error) {
    showStatus(`Could not load style details: ${error.message}`);
  }
}

async function loadTopStyles() {
  showStatus("");
  state.mode = "forecast";
  state.selectedSeason = null;
  listTitle.textContent = "Top Predicted Styles";
  try {
    const response = await fetch(`${API_BASE}/styles/top`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `API returned ${response.status}`);
    }
    const payload = await response.json();
    state.styles = payload.styles || [];
    state.selectedStyleId = state.styles[0]?.style_id || null;
    generatedAt.textContent = payload.generated_at ? new Date(payload.generated_at).toLocaleString() : "";
    renderList();
    if (state.selectedStyleId) await selectStyle(state.selectedStyleId);
  } catch (error) {
    state.styles = [];
    renderList();
    renderDetail(null);
    showStatus(`${error.message} Run the pipeline after placing the H&M data under data/raw/.`);
  }
}

async function loadSeasonStyles(season) {
  showStatus("");
  state.mode = "season";
  state.selectedSeason = season;
  try {
    const response = await fetch(`${API_BASE}/styles/season/${season}`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `API returned ${response.status}`);
    }
    const payload = await response.json();
    state.styles = payload.styles || [];
    state.selectedStyleId = state.styles[0]?.style_id || null;
    generatedAt.textContent = payload.definition || "";
    listTitle.textContent = `${payload.label} Historical Winners`;
    renderList();
    renderDetail(state.styles[0]);
  } catch (error) {
    state.styles = [];
    renderList();
    renderDetail(null);
    showStatus(`${error.message} Run the pipeline to generate seasonal outputs.`);
  }
}

refreshButton.addEventListener("click", () => {
  if (state.mode === "season" && state.selectedSeason) {
    loadSeasonStyles(state.selectedSeason);
  } else {
    loadTopStyles();
  }
});

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    if (tab.dataset.mode === "forecast") {
      loadTopStyles();
    } else {
      loadSeasonStyles(tab.dataset.season);
    }
  });
});

loadTopStyles();

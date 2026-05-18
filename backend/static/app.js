const SAMPLE = {
  Gender: "female",
  Pregnancies: 6,
  Glucose: 148,
  BloodPressure: 72,
  SkinThickness: 35,
  Insulin: 0,
  BMI: 33.6,
  DiabetesPedigreeFunction: 0.627,
  Age: 50,
};

const NUMERIC_FIELDS = [
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age",
];

const form = document.getElementById("predict-form");
const resultPanel = document.getElementById("result-panel");
const errorMsg = document.getElementById("error-msg");
const submitBtn = document.getElementById("submit-btn");
const genderSelect = document.getElementById("Gender");
const pregnanciesInput = document.getElementById("Pregnancies");
const pregnanciesField = document.getElementById("pregnancies-field");
const chartPlaceholder = document.getElementById("chart-placeholder");

let datasetChart = null;
let featuresChart = null;
let riskChart = null;

function initNav() {
  const toggle = document.getElementById("nav-toggle");
  const links = document.getElementById("nav-links");
  if (!toggle || !links) return;

  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });

  links.querySelectorAll("a").forEach((a) => {
    a.addEventListener("click", () => {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });
  });
}

function initCharts() {
  if (typeof Chart === "undefined") return;

  const datasetCtx = document.getElementById("dataset-chart");
  if (datasetCtx) {
    datasetChart = new Chart(datasetCtx, {
      type: "doughnut",
      data: {
        labels: ["No diabetes", "Diabetes"],
        datasets: [
          {
            data: [65, 35],
            backgroundColor: ["#34d399", "#f87171"],
            borderWidth: 0,
          },
        ],
      },
      options: chartOptions(false),
    });
  }

  const featuresCtx = document.getElementById("features-chart");
  if (featuresCtx) {
    featuresChart = new Chart(featuresCtx, {
      type: "bar",
      data: {
        labels: [
          "Glucose",
          "BMI",
          "Age",
          "Insulin",
          "BP",
          "Pedigree",
          "Pregnancies",
          "Skin",
          "Gender",
        ],
        datasets: [
          {
            label: "Importance",
            data: [28, 18, 14, 12, 10, 8, 5, 3, 2],
            backgroundColor: "#0284c7",
            borderRadius: 6,
          },
        ],
      },
      options: {
        ...chartOptions(true),
        indexAxis: "y",
        scales: {
          x: {
            max: 35,
            grid: { color: "#e2e8f0" },
            ticks: { font: { size: 11 } },
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 11 } },
          },
        },
      },
    });
  }

  const riskCtx = document.getElementById("risk-chart");
  if (riskCtx) {
    riskChart = new Chart(riskCtx, {
      type: "doughnut",
      data: {
        labels: ["No diabetes", "Diabetes risk"],
        datasets: [
          {
            data: [50, 50],
            backgroundColor: ["#34d399", "#f87171"],
            borderWidth: 0,
          },
        ],
      },
      options: chartOptions(false),
    });
  }
}

function chartOptions(showLegend) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: showLegend,
        position: "bottom",
        labels: { boxWidth: 12, padding: 14, font: { size: 11 } },
      },
    },
  };
}

function updateRiskChart(noPct, yesPct) {
  if (!riskChart) return;
  chartPlaceholder.classList.add("hidden");
  riskChart.data.datasets[0].data = [noPct, yesPct];
  riskChart.update();
}

function formatCount(n) {
  return Number(n || 0).toLocaleString();
}

function updatePredictionCounters(count, animate) {
  const ids = [
    "hero-predictions",
    "meta-predictions",
    "monitor-predictions",
    "nav-predictions",
  ];
  const text = formatCount(count);
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = text;
      if (animate) {
        el.classList.remove("count-bump");
        void el.offsetWidth;
        el.classList.add("count-bump");
      }
    }
  });
}

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    const acc = data.accuracy || "74%";
    document.getElementById("meta-accuracy").textContent = acc;
    if (data.dataset_rows) {
      document.getElementById("meta-rows").textContent = data.dataset_rows;
    }
    const heroAcc = document.getElementById("hero-accuracy");
    if (heroAcc) heroAcc.textContent = acc;
    updatePredictionCounters(data.prediction_count, false);
    if (data.brand) {
      document.title = data.brand + " | Diabetes Risk Intelligence";
    }
  } catch {
    /* optional */
  }
}

function syncPregnanciesForGender() {
  const isMale = genderSelect.value === "male";
  pregnanciesInput.disabled = isMale;
  pregnanciesField.classList.toggle("field--disabled", isMale);
  if (isMale) pregnanciesInput.value = 0;
}

genderSelect.addEventListener("change", syncPregnanciesForGender);

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("visible");
  resultPanel.classList.remove("visible");
}

function hideError() {
  errorMsg.classList.remove("visible");
}

function formatGender(g) {
  if (g === "male") return "Male";
  if (g === "female") return "Female";
  return "Other";
}

function showResult(data) {
  hideError();
  const isHigh = data.prediction === 1;
  const icon = document.getElementById("result-icon");
  icon.textContent = isHigh ? "!" : "\u2713";
  icon.className = "result-icon " + (isHigh ? "high" : "low");

  document.getElementById("result-title").textContent = data.label;
  document.getElementById("result-sub").textContent =
    formatGender(data.gender || "") +
    " | Risk " +
    data.risk_percent +
    "%";

  const risk = data.risk_percent;
  document.getElementById("risk-pct").textContent = risk + "%";
  const fill = document.getElementById("risk-fill");
  fill.style.width = risk + "%";
  fill.style.background = isHigh
    ? "linear-gradient(90deg, #f87171, #dc2626)"
    : "linear-gradient(90deg, #34d399, #059669)";

  const noPct = data.probabilities.no_diabetes;
  const yesPct = data.probabilities.diabetes;
  document.getElementById("prob-no").textContent = noPct + "%";
  document.getElementById("prob-yes").textContent = yesPct + "%";

  updateRiskChart(noPct, yesPct);
  if (typeof data.prediction_count === "number") {
    updatePredictionCounters(data.prediction_count, true);
  }
  resultPanel.classList.add("visible");

  document.getElementById("insights")?.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  const gender = genderSelect.value;
  if (!gender) {
    showError("Please select a gender.");
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict risk";
    genderSelect.focus();
    return;
  }

  syncPregnanciesForGender();
  const body = { Gender: gender };

  for (const name of NUMERIC_FIELDS) {
    const el = document.getElementById(name);
    const val = el.value.trim();
    if (val === "" || Number.isNaN(Number(val))) {
      showError("Please fill in all fields with valid numbers.");
      submitBtn.disabled = false;
      submitBtn.textContent = "Predict risk";
      el.focus();
      return;
    }
    body[name] =
      name === "Pregnancies" || name === "Age"
        ? parseInt(val, 10)
        : parseFloat(val);
  }

  if (gender === "male") body.Pregnancies = 0;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg).join(" ")
        : detail || "Prediction failed.";
      showError(msg);
      return;
    }
    showResult(data);
  } catch {
    showError("Could not reach the server. Start the app with uvicorn.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict risk";
  }
});

document.getElementById("sample-btn").addEventListener("click", () => {
  genderSelect.value = SAMPLE.Gender;
  for (const [k, v] of Object.entries(SAMPLE)) {
    if (k !== "Gender") document.getElementById(k).value = v;
  }
  syncPregnanciesForGender();
  hideError();
});

form.addEventListener("reset", () => {
  resultPanel.classList.remove("visible");
  hideError();
  chartPlaceholder?.classList.remove("hidden");
  if (riskChart) {
    riskChart.data.datasets[0].data = [50, 50];
    riskChart.update();
  }
  setTimeout(syncPregnanciesForGender, 0);
});

initNav();
initCharts();
syncPregnanciesForGender();
loadModelInfo();

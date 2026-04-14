// ============================================================
// NAVIGATION
// ============================================================
const navItems = document.querySelectorAll(".nav-item");
const sections = document.querySelectorAll(".page-section");
const topbarTitle = document.getElementById("topbarTitle");

const sectionTitles = {
  dashboard: "Vue d'ensemble",
  predict:   "Prédiction du churn",
  analytics: "Analytics",
  about:     "À propos"
};

navItems.forEach(item => {
  item.addEventListener("click", () => {
    const target = item.dataset.section;

    // Mettre à jour nav
    navItems.forEach(n => n.classList.remove("active"));
    item.classList.add("active");

    // Mettre à jour sections
    sections.forEach(s => s.classList.remove("active"));
    document.getElementById("section-" + target).classList.add("active");

    // Mettre à jour titre topbar
    topbarTitle.textContent = sectionTitles[target] || "";

    // Fermer sidebar mobile si ouvert
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("sidebarOverlay").classList.remove("open");
  });
});

// ============================================================
// SIDEBAR MOBILE
// ============================================================
document.getElementById("sidebarToggle").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("sidebarOverlay").classList.toggle("open");
});
document.getElementById("sidebarOverlay").addEventListener("click", () => {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("sidebarOverlay").classList.remove("open");
});

// ============================================================
// THEME TOGGLE
// ============================================================
const themeToggle = document.getElementById("themeToggle");
const themeIcon   = document.getElementById("themeIcon");
const themeLabel  = document.getElementById("themeLabel");

themeToggle.addEventListener("click", () => {
  const html = document.documentElement;
  const isLight = html.getAttribute("data-theme") === "light";
  html.setAttribute("data-theme", isLight ? "dark" : "light");
  themeIcon.className  = isLight ? "fa-solid fa-moon"  : "fa-solid fa-sun";
  themeLabel.textContent = isLight ? "Mode sombre" : "Mode clair";
});

// ============================================================
// CHARTS (Dashboard + Analytics)
// ============================================================
Chart.defaults.color = "#8b93a8";
Chart.defaults.borderColor = "rgba(255,255,255,0.07)";

// -- Pie chart risque (Dashboard)
new Chart(document.getElementById("riskPieChart"), {
  type: "doughnut",
  data: {
    labels: ["Faible", "Modéré", "Élevé"],
    datasets: [{
      data: [58, 24, 18],
      backgroundColor: ["#22d3a0", "#f59e0b", "#f43f5e"],
      borderWidth: 0,
      hoverOffset: 6
    }]
  },
  options: {
    cutout: "65%",
    plugins: { legend: { display: false } },
    responsive: true,
    maintainAspectRatio: false
  }
});

// -- Feature importance (Analytics)
new Chart(document.getElementById("featureImportanceChart"), {
  type: "bar",
  data: {
    labels: ["Recency", "MonetaryPerDay", "CustomerTenureDays",
             "AvgBasketValue", "TenureRatio", "Frequency",
             "TotalTransactions", "UniqueInvoices", "FirstPurchaseDaysAgo", "Age"],
    datasets: [{
      label: "Importance",
      data: [0.221, 0.083, 0.056, 0.048, 0.029, 0.026, 0.025, 0.022, 0.021, 0.018],
      backgroundColor: "#4f7cff",
      borderRadius: 4
    }]
  },
  options: {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: "rgba(255,255,255,0.05)" } },
      y: { grid: { display: false } }
    }
  }
});

// -- Churn distribution donut (Analytics)
new Chart(document.getElementById("churnDistChart"), {
  type: "doughnut",
  data: {
    labels: ["Fidèles (0)", "Churned (1)"],
    datasets: [{
      data: [81.7, 18.3],
      backgroundColor: ["#22d3a0", "#f43f5e"],
      borderWidth: 0,
      hoverOffset: 6
    }]
  },
  options: {
    cutout: "65%",
    plugins: { legend: { display: false } },
    responsive: true,
    maintainAspectRatio: false
  }
});

async function loadMetrics() {
    try {
        const res = await fetch("/metrics");
        const data = await res.json();

        document.getElementById("rocAuc").textContent = data.roc_auc.toFixed(3);
        document.getElementById("recall").textContent = Math.round(data.recall * 100) + "%";
        document.getElementById("accuracy").textContent = Math.round(data.accuracy * 100) + "%";

    } catch (err) {
        console.error("Metrics load error:", err);
    }
}

document.addEventListener("DOMContentLoaded", loadMetrics);

// ============================================================
// FORMULAIRE DE PRÉDICTION
// ============================================================
document.getElementById("predictForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  const btn     = document.getElementById("predictBtn");
  const btnText = document.getElementById("btnText");
  const spinner = document.getElementById("btnSpinner");

  btn.disabled = true;
  btnText.style.display = "none";
  spinner.style.display = "block";

  const data = {
    Recency:            parseFloat(document.getElementById("Recency").value),
    Frequency:          parseFloat(document.getElementById("Frequency").value),
    MonetaryTotal:      parseFloat(document.getElementById("MonetaryTotal").value),
    AvgBasketValue:     parseFloat(document.getElementById("AvgBasketValue").value),
    CustomerTenureDays: parseFloat(document.getElementById("CustomerTenureDays").value)
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!response.ok) throw new Error("Erreur serveur " + response.status);
    const result = await response.json();

    if (result.error) throw new Error(result.error);

    const prob  = result.probability;
    const level = prob < 35 ? "low" : prob < 65 ? "medium" : "high";
    const labelText = { low: "Risque faible", medium: "Risque modéré", high: "Risque élevé" }[level];
    const verdicts  = {
      low:    "Ce client est fidèle. Maintenez l'engagement avec des offres de fidélité.",
      medium: "Risque modéré. Envisagez une campagne de réactivation ciblée.",
      high:   "Risque élevé ! Contactez ce client immédiatement avec une offre personnalisée."
    };

    // Cacher l'idle, montrer le contenu
    document.getElementById("resultIdle").style.display    = "none";
    document.getElementById("resultContent").style.display = "block";

    // Badge
    const badge = document.getElementById("riskBadge");
    badge.className = "result-label-badge " + level;
    document.getElementById("riskLabel").textContent = labelText;

    // Probabilité
    const probDisplay = document.getElementById("probDisplay");
    probDisplay.className   = "result-probability " + level;
    probDisplay.textContent = prob + "%";

    // Barre de progression
    const fill = document.getElementById("progressFill");
    fill.className  = "progress-fill " + level;
    fill.style.width = "0%";
    setTimeout(() => { fill.style.width = prob + "%"; }, 50);

    // Verdict
    const verdict = document.getElementById("resultVerdict");
    verdict.className   = "result-verdict " + level;
    verdict.textContent = verdicts[level];

    document.getElementById("resultPanel").classList.add("has-result");

  } catch (err) {
    alert("Erreur : " + err.message + "\n\nVérifiez que Flask tourne bien (python app.py).");
  } finally {
    btn.disabled = false;
    btnText.style.display = "flex";
    spinner.style.display = "none";
  }
});
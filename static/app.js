// Frontend Javascript logic for predictive route optimization dashboard

let map;
let baselinePathLayer;
let optimizedPathLayer;
let markersGroup;
let comparisonChart;

// Available warehouse details
let locations = {};

document.addEventListener("DOMContentLoaded", async () => {
    initMap();
    await checkModelStatus();
    await fetchLocations();
    initChart();
    
    // Wire up event listeners
    document.getElementById("btn-generate-data").addEventListener("click", generateData);
    document.getElementById("btn-train-model").addEventListener("click", trainModel);
    document.getElementById("btn-optimize").addEventListener("click", runOptimization);
    
    // Auto-select start hub changes to update options
    document.getElementById("start-hub").addEventListener("change", populateStopsSelector);
});

// 1. Initialize Leaflet Map
function initMap() {
    // Center at SF Bay Area
    map = L.map('map').setView([37.76, -122.25], 10);
    
    // Use CartoDB Dark Matter tiles for modern dark glassmorphic styling
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    markersGroup = L.layerGroup().addTo(map);
    baselinePathLayer = L.featureGroup().addTo(map);
    optimizedPathLayer = L.featureGroup().addTo(map);
}

// 2. Check ML engine and data status
async function checkModelStatus() {
    try {
        const res = await fetch("/api/model/status");
        const status = await res.json();
        
        const dot = document.getElementById("status-dot");
        const text = document.getElementById("status-text");
        
        if (status.model_trained) {
            dot.className = "status-dot active";
            text.textContent = "ML Model Ready";
        } else if (status.dataset_generated) {
            dot.className = "status-dot";
            dot.style.backgroundColor = "var(--accent-amber)";
            text.textContent = "Data Ready (Train Model)";
        } else {
            dot.className = "status-dot";
            text.textContent = "No Data (Generate Data)";
        }
    } catch (e) {
        console.error("Error checking model status", e);
    }
}

// 3. Fetch location options
async function fetchLocations() {
    try {
        const res = await fetch("/api/locations");
        locations = await res.json();
        populateStopsSelector();
    } catch (e) {
        console.error("Error fetching locations", e);
    }
}

// Populate stops selector based on selected hub (to exclude it)
function populateStopsSelector() {
    const startHub = document.getElementById("start-hub").value;
    const selector = document.getElementById("stops-selector");
    selector.innerHTML = "";
    
    Object.keys(locations).forEach(loc => {
        if (loc === startHub) return;
        
        const div = document.createElement("div");
        div.className = "stop-item";
        
        // Use readable name
        const displayName = loc.replace(/_/g, ' ');
        
        div.innerHTML = `
            <input type="checkbox" id="chk-${loc}" value="${loc}" checked>
            <label for="chk-${loc}">${displayName}</label>
        `;
        selector.appendChild(div);
    });
}

// 4. Generate Operational Data
async function generateData() {
    toggleSpinner("btn-generate-data", "spinner-data", true);
    try {
        const res = await fetch("/api/data/generate", { method: "POST" });
        const result = await res.json();
        alert(result.message);
        await checkModelStatus();
    } catch (e) {
        alert("Error generating data: " + e.message);
    } finally {
        toggleSpinner("btn-generate-data", "spinner-data", false);
    }
}

// 5. Train Model
async function trainModel() {
    toggleSpinner("btn-train-model", "spinner-train", true);
    try {
        const res = await fetch("/api/model/train", { method: "POST" });
        const result = await res.json();
        if (result.status === "success") {
            const m = result.metrics;
            alert(`${result.message}\nMAE: ${m.mae.toFixed(2)} mins\nRMSE: ${m.rmse.toFixed(2)} mins\nR2: ${m.r2.toFixed(3)}`);
        } else {
            alert("Training failed: " + result.detail);
        }
        await checkModelStatus();
    } catch (e) {
        alert("Error training model: " + e.message);
    } finally {
        toggleSpinner("btn-train-model", "spinner-train", false);
    }
}

// Helper to toggle buttons spinners
function toggleSpinner(btnId, spinnerId, loading) {
    const btn = document.getElementById(btnId);
    const spinner = document.getElementById(spinnerId);
    const textSpan = btn.querySelector(".btn-text");
    
    if (loading) {
        btn.disabled = true;
        spinner.classList.remove("hidden");
        if (textSpan) textSpan.style.opacity = "0.6";
    } else {
        btn.disabled = false;
        spinner.classList.add("hidden");
        if (textSpan) textSpan.style.opacity = "1";
    }
}

// 6. Run Route Optimization API Call
async function runOptimization() {
    const startHub = document.getElementById("start-hub").value;
    
    // Gather selected checkboxes
    const checkboxes = document.querySelectorAll("#stops-selector input[type=checkbox]:checked");
    const stops = [startHub];
    checkboxes.forEach(chk => {
        stops.push(chk.value);
    });
    
    if (stops.length < 2) {
        alert("Please select at least one delivery stop to visit.");
        return;
    }
    
    const simParams = {
        weather: document.getElementById("sim-weather").value,
        traffic_density: document.getElementById("sim-traffic").value,
        vehicle_type: document.getElementById("sim-vehicle").value,
        package_weight_kg: parseFloat(document.getElementById("sim-weight").value) || 30.0,
        driver_experience_years: parseFloat(document.getElementById("sim-driver").value) || 5.0,
        hour: new Date().getHours(),
        day_of_week: new Date().getDay(),
        month: new Date().getMonth() + 1
    };
    
    toggleSpinner("btn-optimize", "spinner-optimize", true);
    
    try {
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                stops: stops,
                start_hub: startHub,
                sim_params: simParams
            })
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Optimization failed.");
        }
        
        const data = await response.json();
        updateDashboard(data);
    } catch (e) {
        alert(e.message);
    } finally {
        toggleSpinner("btn-optimize", "spinner-optimize", false);
    }
}

// 7. Update Frontend Dashboard Components
function updateDashboard(data) {
    const kpis = data.kpis;
    
    // Update KPI panels
    document.getElementById("kpi-efficiency").textContent = `${kpis.efficiency_score.toFixed(1)}%`;
    document.getElementById("kpi-cost-savings").textContent = `$${kpis.cost_saved_usd.toFixed(2)}`;
    document.getElementById("kpi-fuel-savings").textContent = `${kpis.fuel_saved_liters.toFixed(1)}L fuel saved`;
    
    document.getElementById("kpi-risk-level").textContent = kpis.risk_level;
    const riskColor = kpis.risk_level === "LOW" ? "var(--accent-green)" : (kpis.risk_level === "MEDIUM" ? "var(--accent-amber)" : "var(--accent-red)");
    document.getElementById("kpi-risk-level").style.color = riskColor;
    document.getElementById("kpi-risk-ratio").textContent = `Delay Ratio: ${(kpis.risk_ratio * 100).toFixed(1)}%`;
    
    document.getElementById("kpi-time-saved").textContent = `-${kpis.time_saved_mins.toFixed(1)}m`;
    document.getElementById("kpi-percent-saved").textContent = `${kpis.percent_time_saved.toFixed(1)}% duration reduced`;
    
    // Update comparison sequences
    const b = data.baseline_route;
    const o = data.optimized_route;
    
    document.getElementById("baseline-seq-list").innerHTML = b.sequence.map(s => s.replace(/_/g, ' ')).join(" &rarr; ");
    document.getElementById("baseline-duration").textContent = `${b.baseline_duration_mins.toFixed(1)} mins`;
    document.getElementById("baseline-delay").textContent = `+${b.predicted_delay_mins.toFixed(1)} mins delay`;
    
    document.getElementById("optimized-seq-list").innerHTML = o.sequence.map(s => s.replace(/_/g, ' ')).join(" &rarr; ");
    document.getElementById("optimized-duration").textContent = `${o.baseline_duration_mins.toFixed(1)} mins`;
    document.getElementById("optimized-delay").textContent = `+${o.predicted_delay_mins.toFixed(1)} mins delay`;
    
    // Clear old map layers
    markersGroup.clearLayers();
    baselinePathLayer.clearLayers();
    optimizedPathLayer.clearLayers();
    
    // Draw Baseline Route (Dashed silver/grey line)
    const baseCoords = b.coords;
    L.polyline(baseCoords, {
        color: '#8e9bb0',
        weight: 3,
        dashArray: '5, 8',
        opacity: 0.7
    }).addTo(baselinePathLayer);
    
    // Draw Optimized Route (Solid Green line)
    const optCoords = o.coords;
    L.polyline(optCoords, {
        color: 'var(--accent-green)',
        weight: 4,
        opacity: 0.95
    }).addTo(optimizedPathLayer);
    
    // Plot markers and indices
    o.sequence.forEach((locName, idx) => {
        const coord = locations[locName];
        if (!coord) return;
        
        let label = idx === 0 ? "Start" : (idx === o.sequence.length - 1 ? "End" : idx);
        
        const customIcon = L.divIcon({
            html: `<div style="background-color: ${idx === 0 ? 'var(--accent-blue)' : 'var(--accent-green)'}; 
                               color: #080c14; 
                               width: 24px; 
                               height: 24px; 
                               border-radius: 50%; 
                               display: flex; 
                               justify-content: center; 
                               align-items: center; 
                               font-size: 0.75rem; 
                               font-weight: 700; 
                               border: 2px solid #ffffff; 
                               box-shadow: 0 0 10px rgba(0,0,0,0.5);">${label}</div>`,
            className: 'custom-div-icon',
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
        
        L.marker([coord.lat, coord.lon], { icon: customIcon })
            .bindPopup(`<b>${locName.replace(/_/g, ' ')}</b><br>Stop sequence: ${idx}`)
            .addTo(markersGroup);
    });
    
    // Fit map bounds to show the full route
    const allRoutes = L.featureGroup([baselinePathLayer, optimizedPathLayer]);
    map.fitBounds(allRoutes.getBounds(), { padding: [40, 40] });
    
    // Update comparison chart
    updateChartData(b, o);
}

// 8. Setup and Update Chart.js component
function initChart() {
    const ctx = document.getElementById('comparison-chart').getContext('2d');
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Baseline Route', 'Optimized Route (GA)'],
            datasets: [
                {
                    label: 'Base Travel Duration (mins)',
                    data: [0, 0],
                    backgroundColor: 'rgba(0, 198, 255, 0.4)',
                    borderColor: 'var(--accent-blue)',
                    borderWidth: 1.5
                },
                {
                    label: 'Predicted Delays (mins)',
                    data: [0, 0],
                    backgroundColor: 'rgba(255, 170, 0, 0.4)',
                    borderColor: 'var(--accent-amber)',
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8e9bb0', font: { family: 'Outfit' } }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8e9bb0' }
                },
                y: {
                    stacked: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8e9bb0' },
                    title: {
                        display: true,
                        text: 'Minutes',
                        color: '#8e9bb0'
                    }
                }
            }
        }
    });
}

function updateChartData(baseline, optimized) {
    comparisonChart.data.datasets[0].data = [
        baseline.baseline_duration_mins,
        optimized.baseline_duration_mins
    ];
    comparisonChart.data.datasets[1].data = [
        baseline.predicted_delay_mins,
        optimized.predicted_delay_mins
    ];
    comparisonChart.update();
}

// ══════════════════════════════════════════════════════════════════════════════
// SHAP Feature Importance — added feature
// ══════════════════════════════════════════════════════════════════════════════

let shapChart = null;

// Human-readable labels and colour hints per feature
const FEATURE_META = {
    weather:                 { label: "Weather conditions",       color: "rgba(0,198,255,0.75)",   desc: "Impact of rain, snow, and storms on delay time" },
    traffic_density:         { label: "Traffic density",          color: "rgba(255,100,100,0.75)", desc: "Congestion level from low to gridlock" },
    baseline_duration_mins:  { label: "Route base duration",      color: "rgba(0,255,135,0.75)",   desc: "Longer routes accumulate more delay variance" },
    package_weight_kg:       { label: "Cargo weight (kg)",        color: "rgba(255,170,0,0.75)",   desc: "Heavier loads slow vehicles and extend stops" },
    driver_experience_years: { label: "Driver experience (yrs)",  color: "rgba(180,120,255,0.75)", desc: "Experienced drivers reduce delay significantly" },
    hour:                    { label: "Departure hour",           color: "rgba(0,210,180,0.75)",   desc: "Rush-hour departures correlate with higher delays" },
    day_of_week:             { label: "Day of week",              color: "rgba(255,200,60,0.75)",  desc: "Weekday vs weekend traffic patterns" },
    month:                   { label: "Month of year",            color: "rgba(100,180,255,0.75)", desc: "Seasonal weather patterns affect delivery times" },
    vehicle_type:            { label: "Vehicle type",             color: "rgba(255,140,60,0.75)",  desc: "Trucks vs vans vs EVs have different delay profiles" },
};

function getMeta(feature) {
    return FEATURE_META[feature] || {
        label: feature.replace(/_/g, " "),
        color: "rgba(120,140,180,0.75)",
        desc: ""
    };
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("btn-load-shap").addEventListener("click", loadShapImportance);
});

async function loadShapImportance() {
    toggleSpinner("btn-load-shap", "spinner-shap", true);
    try {
        const res = await fetch("/api/model/shap");
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || "Could not load SHAP data. Train the model first.");
            return;
        }
        const data = await res.json();
        renderShapChart(data);
    } catch (e) {
        alert("Error loading feature importance: " + e.message);
    } finally {
        toggleSpinner("btn-load-shap", "spinner-shap", false);
    }
}

function renderShapChart(data) {
    const { features, importances, method, model } = data;

    // Show method badge
    const badge = document.getElementById("shap-method-badge");
    badge.textContent = `${method} · ${model}`;
    badge.style.display = "inline-block";

    // Hide empty state, show chart
    document.getElementById("shap-empty").style.display = "none";
    document.getElementById("shap-chart-wrapper").style.display = "block";

    // Compute bar heights dynamically: at least 44px per bar
    const barHeight = 44;
    const container = document.getElementById("shap-canvas-container");
    container.style.height = (features.length * barHeight + 60) + "px";

    const colors     = features.map(f => getMeta(f).color);
    const labels     = features.map(f => getMeta(f).label);
    const maxVal     = Math.max(...importances);

    // Destroy old chart if it exists
    if (shapChart) {
        shapChart.destroy();
        shapChart = null;
    }

    const ctx = document.getElementById("shap-chart").getContext("2d");
    shapChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: method === "SHAP" ? "Mean |SHAP| value (delay mins)" : "Feature importance score",
                data: importances,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace("0.75", "1")),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const val = ctx.parsed.x;
                            const pct = ((val / importances.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                            const suffix = method === "SHAP" ? " mins avg impact" : " importance score";
                            return ` ${val.toFixed(4)}${suffix} (${pct}% of total)`;
                        }
                    },
                    backgroundColor: "#0d1520",
                    titleColor: "#f5f6f8",
                    bodyColor: "#8e9bb0",
                    borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1,
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255,255,255,0.05)" },
                    ticks: {
                        color: "#8e9bb0",
                        callback: v => method === "SHAP" ? v.toFixed(3) : v.toFixed(4)
                    },
                    title: {
                        display: true,
                        text: method === "SHAP" ? "Mean |SHAP| value (minutes of delay)" : "Relative importance",
                        color: "#8e9bb0",
                        font: { size: 11 }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f5f6f8", font: { size: 12 } }
                }
            }
        }
    });

    // Render insight cards (top 4 features)
    const insightsEl = document.getElementById("shap-insights");
    insightsEl.innerHTML = "";
    const total = importances.reduce((a, b) => a + b, 0);
    const top4  = features.slice(0, 4);

    top4.forEach((feat, i) => {
        const meta = getMeta(feat);
        const pct  = ((importances[i] / total) * 100).toFixed(1);
        const card = document.createElement("div");
        card.className = "shap-insight-card";
        card.innerHTML = `
            <div class="shap-insight-feature" style="color:${meta.color}">${meta.label}</div>
            <div class="shap-insight-value" style="color:${meta.color}">${pct}%<span style="font-size:0.7rem;color:var(--text-secondary);font-weight:400;"> of variance</span></div>
            <div class="shap-insight-desc">${meta.desc}</div>
        `;
        insightsEl.appendChild(card);
    });
}

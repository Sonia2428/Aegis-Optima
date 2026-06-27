import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from data_generator import generate_messy_data
from ml_engine import train_and_evaluate, get_shap_importance
from optimizer import RouteOptimizer, LOCATIONS, get_baseline_duration

app = FastAPI(title="Predictive Supply Chain & Route Optimization API")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

optimizer = RouteOptimizer()

class SimulationParams(BaseModel):
    weather: str = "sunny"
    traffic_density: str = "low"
    vehicle_type: str = "van"
    package_weight_kg: float = 30.0
    driver_experience_years: float = 5.0
    hour: int = 12
    day_of_week: int = 2
    month: int = 6

class OptimizeRequest(BaseModel):
    stops: List[str]
    start_hub: str
    sim_params: SimulationParams

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/data/generate")
def generate_data_endpoint(records: int = 5000):
    try:
        generate_messy_data(num_records=records)
        return {"status": "success", "message": f"Generated operational dataset with {records} records."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/train")
def train_model_endpoint():
    try:
        metrics = train_and_evaluate()
        optimizer.reload_model()
        return {
            "status": "success",
            "message": "Model trained successfully.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/status")
def model_status():
    model_exists = os.path.exists("models/route_predictor.joblib")
    data_exists = os.path.exists("data/supply_chain_data.csv")
    shap_exists = os.path.exists("models/shap_importance.joblib")
    return {
        "model_trained": model_exists,
        "dataset_generated": data_exists,
        "shap_available": shap_exists
    }

@app.get("/api/locations")
def get_locations():
    return LOCATIONS

# ── NEW: SHAP feature importance endpoint ────────────────────────────────────
@app.get("/api/model/shap")
def get_shap_endpoint():
    """
    Returns pre-computed SHAP (or fallback feature importance) values.
    Train the model first — importance is computed and cached during training.
    """
    data = get_shap_importance()
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="SHAP importance not available. Train the model first."
        )
    return data

@app.post("/api/optimize")
def optimize_route_endpoint(req: OptimizeRequest):
    for stop in req.stops:
        if stop not in LOCATIONS:
            raise HTTPException(status_code=400, detail=f"Location '{stop}' is invalid.")
    if req.start_hub not in LOCATIONS:
        raise HTTPException(status_code=400, detail=f"Start Hub '{req.start_hub}' is invalid.")

    sim_dict = req.sim_params.model_dump()

    baseline_sequence = [req.start_hub] + [s for s in req.stops if s != req.start_hub] + [req.start_hub]
    baseline_base, baseline_delay = optimizer.calculate_route_cost(baseline_sequence, sim_dict)
    baseline_total = baseline_base + baseline_delay

    opt_sequence, opt_base, opt_delay = optimizer.optimize_genetic_algorithm(
        stops=req.stops,
        start_hub=req.start_hub,
        sim_params=sim_dict
    )
    opt_total = opt_base + opt_delay

    time_saved_mins = max(0.0, baseline_total - opt_total)
    percent_time_saved = (time_saved_mins / baseline_total * 100.0) if baseline_total > 0 else 0.0

    baseline_cost = baseline_total * (0.15 * 1.20 + 0.40)
    opt_cost = opt_total * (0.15 * 1.20 + 0.40)
    cost_saved = max(0.0, baseline_cost - opt_cost)
    fuel_saved_liters = time_saved_mins * 0.15

    efficiency_score = min(100.0, (baseline_total / opt_total * 100.0) if opt_total > 0 else 100.0)

    delay_risk_score = opt_delay / opt_base if opt_base > 0 else 0.0
    risk_level = "LOW"
    if delay_risk_score > 0.4:
        risk_level = "CRITICAL"
    elif delay_risk_score > 0.2:
        risk_level = "HIGH"
    elif delay_risk_score > 0.05:
        risk_level = "MEDIUM"

    return {
        "baseline_route": {
            "sequence": baseline_sequence,
            "coords": [[LOCATIONS[loc]["lat"], LOCATIONS[loc]["lon"]] for loc in baseline_sequence],
            "baseline_duration_mins": float(baseline_base),
            "predicted_delay_mins": float(baseline_delay),
            "total_duration_mins": float(baseline_total)
        },
        "optimized_route": {
            "sequence": opt_sequence,
            "coords": [[LOCATIONS[loc]["lat"], LOCATIONS[loc]["lon"]] for loc in opt_sequence],
            "baseline_duration_mins": float(opt_base),
            "predicted_delay_mins": float(opt_delay),
            "total_duration_mins": float(opt_total)
        },
        "kpis": {
            "efficiency_score": float(efficiency_score),
            "time_saved_mins": float(time_saved_mins),
            "percent_time_saved": float(percent_time_saved),
            "fuel_saved_liters": float(fuel_saved_liters),
            "cost_saved_usd": float(cost_saved),
            "risk_level": risk_level,
            "risk_ratio": float(delay_risk_score)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

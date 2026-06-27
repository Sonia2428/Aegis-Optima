# Aegis Optima: Predictive Supply Chain & Route Optimization Dashboard

Aegis Optima is an end-to-end predictive logistics and route optimization dashboard. This system mimics a high-stakes, Kaggle-style data optimization competition by utilizing synthetic messy operational data, training machine learning regression models to predict delivery delays, and employing a tournament-selection Genetic Algorithm to resolve Traveling Salesperson Problem (TSP) bottlenecks in real-time.

---

## 🛠️ Project Architecture

The application is modularized into dedicated backend, machine learning, and frontend components:

1. **`data_generator.py` (Messy Operational Data Generator)**
   - Generates simulated historical delivery routes containing missing values (`NaN`), raw unformatted timestamps, and outlier variables (e.g. driver experience values of `-1` and `999` years) to simulate Kaggle data cleaning conditions.
   - Correlates travel times with dynamic variables: weather severity, traffic density, cargo weight, driver experience, and vehicle types (van, truck, electric).

2. **`ml_engine.py` (Machine Learning Predictor)**
   - Automatically cleans extreme outlier driver values and normalizes raw timestamps.
   - Uses a Scikit-Learn `Pipeline` combined with a `ColumnTransformer` to impute missing features, encode categorical columns, scale numerical inputs, and train an **XGBoost Regressor**.
   - Validates modeling capacity utilizing a 5-Fold Cross Validation strategy (Mean MAE: **~36.05 mins**, $R^2$: **~0.7955**).

3. **`optimizer.py` (Genetic Algorithm Engine)**
   - Resolves route sequencing (TSP) utilizing tournament selection, Ordered Crossover (OX1), swap mutation, and elitism.
   - **Performance Optimization**: Employs an elite batch prediction mechanism. All unique location edges are predicted in a single batch inference at start, reducing computational lookup latency during the GA generations to under **0.01 seconds**.

4. **`main.py` (FastAPI Core Server)**
   - Exposes REST endpoints to generate data, trigger model retraining, and solve path optimization.
   - Serves the frontend assets and performs KPI calculations (e.g., fuel consumption and USD financial savings).

5. **`static/` (Glassmorphic Interface)**
   - **`index.html`**: Premium layout containing real-time simulation controls, KPI metric widgets, route sequence comparisons, and canvas charts.
   - **`style.css`**: Modern dark-mode layout with blur backdrops, neon accent borders, and state transitions.
   - **`app.js`**: Integrates Leaflet.js to map paths and handles API triggers.

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla Async/Fetch UI, dynamic charts)
* **Backend Framework:** FastAPI (Python)
* **Machine Learning & Explainability:** Scikit-Learn / XGBoost, SHAP (SHapley Additive exPlanations)
* **Route Optimization:** Custom Heuristic / Genetic Algorithm Pipeline

---

## 🚀 Getting Started

### 📋 Prerequisites

Install Python 3.10+ and the required packages:

```bash
pip install scikit-learn joblib fastapi uvicorn xgboost pandas numpy scipy requests
```

### 🏃 Running the Application

1. **Initialize the Dataset and Model**:
   ```bash
   python data_generator.py
   python ml_engine.py
   ```

2. **Start the FastAPI Server**:
   ```bash
   python main.py
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to:
   **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
---
### Screenshot
<img width="1850" height="888" alt="Screenshot 2026-06-27 123915" src="https://github.com/user-attachments/assets/aad943e0-5de9-4e7a-bceb-09b18f69afad" />
<img width="1197" height="622" alt="Screenshot 2026-06-27 124132" src="https://github.com/user-attachments/assets/6385b9e8-4d66-414c-96a2-83c84a495fae" />
<img width="1822" height="872" alt="Screenshot 2026-06-27 124149" src="https://github.com/user-attachments/assets/26bf5406-a50b-425e-bf14-55ee48195472" />
<img width="1796" height="895" alt="Screenshot 2026-06-27 171436" src="https://github.com/user-attachments/assets/664fb43b-22a7-44d2-bf5d-85f53c0b0885" />


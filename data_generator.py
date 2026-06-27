import pandas as pd
import numpy as np
import os
import random

def generate_messy_data(num_records=5000, output_path="data/supply_chain_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)
    
    # 1. Base route identifiers
    route_ids = [f"ROUTE_{str(i).zfill(5)}" for i in range(1, num_records + 1)]
    
    # 2. Origins and Destinations
    cities = ["Warehouse_A", "Warehouse_B", "Warehouse_C", "Distribution_Center_North", "Distribution_Center_South", "Metro_Hub_East", "Metro_Hub_West"]
    origins = [random.choice(cities) for _ in range(num_records)]
    destinations = []
    for org in origins:
        dest = random.choice([c for c in cities if c != org])
        destinations.append(dest)
        
    # 3. Categorical features with messy entries and NaNs
    weather_choices = ["sunny", "rainy", "snowy", "stormy"]
    weather = [random.choice(weather_choices) if random.random() > 0.08 else np.nan for _ in range(num_records)]
    
    traffic_choices = ["low", "medium", "high", "jam"]
    traffic = [random.choice(traffic_choices) if random.random() > 0.05 else np.nan for _ in range(num_records)]
    
    vehicle_choices = ["van", "truck", "electric"]
    vehicle = [random.choice(vehicle_choices) for _ in range(num_records)]
    
    # 4. Numerical features with messiness, missing values, and outliers
    package_weight = np.random.uniform(5.0, 120.0, num_records)
    # Add NaN to weights
    package_weight = [w if random.random() > 0.04 else np.nan for w in package_weight]
    
    # Driver experience with extreme value outliers (e.g., 999 or -1)
    driver_exp = np.random.randint(1, 25, num_records).astype(float)
    for i in range(num_records):
        rand = random.random()
        if rand < 0.02:
            driver_exp[i] = 999.0
        elif rand < 0.04:
            driver_exp[i] = -1.0
        elif rand < 0.08:
            driver_exp[i] = np.nan
            
    # 5. Baseline durations (standard clear travel time in minutes)
    baseline_duration = np.random.uniform(30.0, 300.0, num_records)
    
    # 6. Actual durations calculated with logical correlations (plus noise and target missingness)
    actual_duration = []
    for i in range(num_records):
        base = baseline_duration[i]
        
        # Weather delay factor
        w_factor = 1.0
        w = weather[i]
        if w == "rainy": w_factor = 1.2
        elif w == "snowy": w_factor = 1.5
        elif w == "stormy": w_factor = 1.8
        
        # Traffic delay factor
        t_factor = 1.0
        t = traffic[i]
        if t == "medium": t_factor = 1.15
        elif t == "high": t_factor = 1.4
        elif t == "jam": t_factor = 1.9
        
        # Vehicle weight factor (heavier load slow down slightly)
        weight = package_weight[i]
        weight_factor = 1.0 + (0.0 if np.isnan(weight) else (weight / 500.0))
        
        # Driver experience factor (more experienced drivers save time)
        exp = driver_exp[i]
        exp_factor = 1.0
        if not np.isnan(exp) and 0 <= exp <= 40:
            exp_factor = 1.0 - (exp * 0.008) # up to ~16% speedup
            
        # Vehicle speed coefficient
        veh = vehicle[i]
        veh_factor = 1.0
        if veh == "electric": veh_factor = 0.95 # highly efficient route mapping
        elif veh == "truck": veh_factor = 1.1 # slower speed limit
        
        # Cumulative actual duration
        act = base * w_factor * t_factor * weight_factor * exp_factor * veh_factor
        # Add random gaussian noise
        act += np.random.normal(0.0, base * 0.05)
        
        # Ensure it's not below baseline threshold
        act = max(act, base * 0.7)
        actual_duration.append(act)
        
    # Introduce NaNs in actual duration (predictive target) to mimic messy data
    actual_duration = [d if random.random() > 0.02 else np.nan for d in actual_duration]
    
    # 7. Unstructured messy timestamps
    # Create raw timestamps with format variations
    timestamp_fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y/%m/%d %H:%M:%S"
    ]
    timestamps = []
    base_time = pd.Timestamp("2026-01-01 08:00:00")
    for i in range(num_records):
        delta = pd.Timedelta(days=random.randint(0, 150), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        dt = base_time + delta
        fmt = random.choice(timestamp_fmts)
        timestamps.append(dt.strftime(fmt))
        
    # Combine into DataFrame
    df = pd.DataFrame({
        "route_id": route_ids,
        "timestamp": timestamps,
        "origin": origins,
        "destination": destinations,
        "weather": weather,
        "traffic_density": traffic,
        "vehicle_type": vehicle,
        "package_weight_kg": package_weight,
        "driver_experience_years": driver_exp,
        "baseline_duration_mins": baseline_duration,
        "actual_duration_mins": actual_duration
    })
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {num_records} records at {output_path}")

if __name__ == "__main__":
    generate_messy_data()

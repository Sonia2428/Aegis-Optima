import numpy as np
import pandas as pd
import random
import os
import joblib

# Coordinates for nodes
LOCATIONS = {
    "Warehouse_A": {"lat": 37.7749, "lon": -122.4194},
    "Warehouse_B": {"lat": 37.8044, "lon": -122.2712},
    "Warehouse_C": {"lat": 37.6879, "lon": -122.4702},
    "Distribution_Center_North": {"lat": 37.9101, "lon": -122.3478},
    "Distribution_Center_South": {"lat": 37.5485, "lon": -121.9886},
    "Metro_Hub_East": {"lat": 37.6974, "lon": -122.0808},
    "Metro_Hub_West": {"lat": 37.7564, "lon": -122.4862}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def get_baseline_duration(loc1, loc2):
    coords1 = LOCATIONS[loc1]
    coords2 = LOCATIONS[loc2]
    dist = haversine_distance(coords1["lat"], coords1["lon"], coords2["lat"], coords2["lon"])
    # 40 km/h is ~1.5 mins per km + 10 mins loading/unloading overhead
    return 10.0 + dist * 1.5

class RouteOptimizer:
    def __init__(self, model_path="models/route_predictor.joblib"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
    def reload_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def precompute_edge_delays(self, stops, sim_params):
        """
        Pre-computes and caches delays for all possible edges in batch to avoid calling
        model.predict() inside the GA loop.
        """
        cache = {}
        edges_to_predict = []
        
        # Determine all unique directed edges
        for loc1 in stops:
            for loc2 in stops:
                if loc1 == loc2:
                    cache[(loc1, loc2)] = 0.0
                    continue
                
                baseline = get_baseline_duration(loc1, loc2)
                
                # If no model is trained, use heuristic
                if self.model is None:
                    w_factor = 0.0
                    w = sim_params.get("weather")
                    if w == "rainy": w_factor = 0.15
                    elif w == "snowy": w_factor = 0.35
                    elif w == "stormy": w_factor = 0.6
                    
                    t_factor = 0.0
                    t = sim_params.get("traffic_density")
                    if t == "medium": t_factor = 0.1
                    elif t == "high": t_factor = 0.3
                    elif t == "jam": t_factor = 0.7
                    
                    cache[(loc1, loc2)] = baseline * (w_factor + t_factor)
                else:
                    edges_to_predict.append({
                        'loc_from': loc1,
                        'loc_to': loc2,
                        'baseline_duration_mins': baseline,
                        'weather': sim_params.get('weather', 'sunny'),
                        'traffic_density': sim_params.get('traffic_density', 'low'),
                        'vehicle_type': sim_params.get('vehicle_type', 'van'),
                        'package_weight_kg': float(sim_params.get('package_weight_kg', 30.0)),
                        'driver_experience_years': float(sim_params.get('driver_experience_years', 5.0)),
                        'hour': int(sim_params.get('hour', 12)),
                        'day_of_week': int(sim_params.get('day_of_week', 2)),
                        'month': int(sim_params.get('month', 6))
                    })
                    
        # Predict all edges in a single batch
        if edges_to_predict:
            df_predict = pd.DataFrame(edges_to_predict)
            # Remove helper columns before feeding to model
            features = [
                'weather', 'traffic_density', 'vehicle_type', 
                'package_weight_kg', 'driver_experience_years', 
                'baseline_duration_mins', 'hour', 'day_of_week', 'month'
            ]
            predictions = self.model.predict(df_predict[features])
            
            for idx, row in df_predict.iterrows():
                loc_from = row['loc_from']
                loc_to = row['loc_to']
                pred_delay = predictions[idx]
                # Cap minimum delay
                cache[(loc_from, loc_to)] = max(pred_delay, -0.2 * row['baseline_duration_mins'])
                
        return cache

    def calculate_route_cost(self, route_sequence, cached_delays_or_params):
        """
        Calculates total travel duration (baseline + predicted delays) for a route sequence using cache or params.
        """
        if isinstance(cached_delays_or_params, dict) and "weather" in cached_delays_or_params:
            cached_delays = self.precompute_edge_delays(list(set(route_sequence)), cached_delays_or_params)
        else:
            cached_delays = cached_delays_or_params
            
        total_baseline = 0.0
        total_delay = 0.0
        
        for i in range(len(route_sequence) - 1):
            loc_from = route_sequence[i]
            loc_to = route_sequence[i+1]
            base = get_baseline_duration(loc_from, loc_to)
            delay = cached_delays.get((loc_from, loc_to), 0.0)
            
            total_baseline += base
            total_delay += delay
            
        return total_baseline, total_delay


    def optimize_genetic_algorithm(self, stops, start_hub, sim_params, pop_size=50, generations=100, mutation_rate=0.15):
        """
        Finds the optimal path visiting all stops starting and ending at start_hub using cached edge delays.
        """
        # Ensure start_hub is in stops for cache generation
        all_unique_stops = list(set(stops + [start_hub]))
        
        # Precompute edge delays
        cached_delays = self.precompute_edge_delays(all_unique_stops, sim_params)
        
        # Exclude start_hub from the list of stops to permute
        other_stops = [s for s in stops if s != start_hub]
        if not other_stops:
            # Just route from start to start
            base = get_baseline_duration(start_hub, start_hub)
            delay = cached_delays.get((start_hub, start_hub), 0.0)
            return [start_hub, start_hub], base, delay
            
        # Create initial population
        population = []
        for _ in range(pop_size):
            individual = list(other_stops)
            random.shuffle(individual)
            population.append(individual)
            
        def get_fitness(ind):
            full_route = [start_hub] + ind + [start_hub]
            base, delay = self.calculate_route_cost(full_route, cached_delays)
            return base + delay, base, delay
            
        best_ind = None
        best_cost = float('inf')
        best_base = 0.0
        best_delay = 0.0
        
        for gen in range(generations):
            evaluated = []
            for ind in population:
                total, base, delay = get_fitness(ind)
                evaluated.append((ind, total, base, delay))
                
            evaluated.sort(key=lambda x: x[1])
            
            if evaluated[0][1] < best_cost:
                best_cost = evaluated[0][1]
                best_ind = list(evaluated[0][0])
                best_base = evaluated[0][2]
                best_delay = evaluated[0][3]
                
            new_population = [evaluated[0][0]] # Elitism
            
            while len(new_population) < pop_size:
                t1 = random.sample(evaluated, min(len(evaluated), 3))
                t1.sort(key=lambda x: x[1])
                parent1 = t1[0][0]
                
                t2 = random.sample(evaluated, min(len(evaluated), 3))
                t2.sort(key=lambda x: x[1])
                parent2 = t2[0][0]
                
                size = len(parent1)
                if size >= 2:
                    a, b = sorted(random.sample(range(size), 2))
                    child = [None] * size
                    child[a:b] = parent1[a:b]
                    
                    p2_idx = 0
                    for c_idx in range(size):
                        if child[c_idx] is None:
                            while p2_idx < size and parent2[p2_idx] in child:
                                p2_idx += 1
                            if p2_idx < size:
                                child[c_idx] = parent2[p2_idx]
                            else:
                                # Fallback
                                for item in parent2:
                                    if item not in child:
                                        child[c_idx] = item
                                        break
                else:
                    child = list(parent1)
                    
                # Mutation
                if size >= 2 and random.random() < mutation_rate:
                    m1, m2 = random.sample(range(size), 2)
                    child[m1], child[m2] = child[m2], child[m1]
                    
                new_population.append(child)
                
            population = new_population
            
        optimized_route = [start_hub] + best_ind + [start_hub]
        return optimized_route, best_base, best_delay

if __name__ == "__main__":
    # Test routing optimization
    stops = ["Warehouse_A", "Warehouse_B", "Warehouse_C", "Distribution_Center_North", "Metro_Hub_West"]
    opt = RouteOptimizer()
    sim_params = {"weather": "stormy", "traffic_density": "jam", "vehicle_type": "van"}
    
    # Generate temporary cache just for testing
    cached_delays = opt.precompute_edge_delays(stops, sim_params)
    route, base, delay = opt.optimize_genetic_algorithm(stops, "Warehouse_A", sim_params)
    print("Optimized route sequence:", route)
    print(f"Baseline travel time: {base:.2f} mins, Delay predicted: {delay:.2f} mins, Total: {base + delay:.2f} mins")

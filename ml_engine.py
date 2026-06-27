import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def clean_driver_experience(df):
    df = df.copy()
    if 'driver_experience_years' in df.columns:
        df.loc[(df['driver_experience_years'] < 0) | (df['driver_experience_years'] > 50), 'driver_experience_years'] = np.nan
    return df

def parse_dates(df):
    df = df.copy()
    hours, days_of_week, months = [], [], []
    for val in df['timestamp']:
        parsed_dt = None
        for fmt in ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M:%S"]:
            try:
                parsed_dt = datetime.strptime(str(val).strip(), fmt)
                break
            except ValueError:
                continue
        if parsed_dt:
            hours.append(parsed_dt.hour)
            days_of_week.append(parsed_dt.weekday())
            months.append(parsed_dt.month)
        else:
            hours.append(12)
            days_of_week.append(2)
            months.append(6)
    df['hour'] = hours
    df['day_of_week'] = days_of_week
    df['month'] = months
    return df

def train_and_evaluate(data_path="data/supply_chain_data.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}.")
        return

    print("Loading operational dataset...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['actual_duration_mins'])
    df['delay_mins'] = df['actual_duration_mins'] - df['baseline_duration_mins']
    df = clean_driver_experience(df)
    df = parse_dates(df)

    features = [
        'weather', 'traffic_density', 'vehicle_type',
        'package_weight_kg', 'driver_experience_years',
        'baseline_duration_mins', 'hour', 'day_of_week', 'month'
    ]
    X = df[features]
    y = df['delay_mins']

    categorical_features = ['weather', 'traffic_density', 'vehicle_type']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_features = ['package_weight_kg', 'driver_experience_years',
                          'baseline_duration_mins', 'hour', 'day_of_week', 'month']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    try:
        from xgboost import XGBRegressor
        regressor = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42)
        model_name = "XGBoost"
        print("Using XGBoost Regressor.")
    except ImportError:
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_name = "RandomForest"
        print("Using RandomForest Regressor (fallback).")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    print("Running 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2_scores.append(r2_score(y_val, y_pred))

    print("Training final model on full dataset...")
    pipeline.fit(X, y)

    joblib.dump(pipeline, os.path.join(model_dir, "route_predictor.joblib"))
    print("Model saved.")

    # ── SHAP feature importance ──────────────────────────────────────────────
    shap_data = compute_shap_importance(pipeline, X, categorical_features, numerical_features, model_name)
    if shap_data:
        joblib.dump(shap_data, os.path.join(model_dir, "shap_importance.joblib"))
        print("SHAP importance saved.")

    return {
        "mae": float(np.mean(mae_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "r2": float(np.mean(r2_scores)),
        "model_name": model_name,
        "shap_available": shap_data is not None
    }


def compute_shap_importance(pipeline, X, categorical_features, numerical_features, model_name):
    """
    Compute mean |SHAP| values per original feature and return a serialisable dict.
    Falls back to built-in feature_importances_ if shap is not installed.
    """
    try:
        import shap

        preprocessor = pipeline.named_steps['preprocessor']
        regressor    = pipeline.named_steps['regressor']

        # Transform a sample (up to 500 rows) for speed
        X_sample = X.sample(min(500, len(X)), random_state=42)
        X_transformed = preprocessor.transform(X_sample)

        # Build correct feature names after OHE
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
        all_feature_names = numerical_features + cat_feature_names

        # SHAP explainer – TreeExplainer works for RF and XGBoost
        explainer   = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)
        mean_abs    = np.abs(shap_values).mean(axis=0)

        # Collapse OHE columns back to their original categorical feature
        importance_map = {}
        for fname, val in zip(all_feature_names, mean_abs):
            # OHE names look like "weather_sunny" → original = "weather"
            original = next((c for c in categorical_features if fname.startswith(c + "_")), fname)
            importance_map[original] = importance_map.get(original, 0.0) + float(val)

        # Sort descending
        sorted_items = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)

        return {
            "method": "SHAP",
            "model": model_name,
            "features": [k for k, _ in sorted_items],
            "importances": [round(v, 4) for _, v in sorted_items]
        }

    except ImportError:
        # Graceful fallback: use built-in feature importances
        print("shap not installed – using built-in feature importances as fallback.")
        return _fallback_importance(pipeline, X, categorical_features, numerical_features, model_name)
    except Exception as e:
        print(f"SHAP computation failed ({e}) – using fallback.")
        return _fallback_importance(pipeline, X, categorical_features, numerical_features, model_name)


def _fallback_importance(pipeline, X, categorical_features, numerical_features, model_name):
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        regressor    = pipeline.named_steps['regressor']

        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
        all_feature_names = numerical_features + cat_feature_names

        importances = regressor.feature_importances_
        importance_map = {}
        for fname, val in zip(all_feature_names, importances):
            original = next((c for c in categorical_features if fname.startswith(c + "_")), fname)
            importance_map[original] = importance_map.get(original, 0.0) + float(val)

        sorted_items = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
        return {
            "method": "Feature Importance",
            "model": model_name,
            "features": [k for k, _ in sorted_items],
            "importances": [round(v, 4) for _, v in sorted_items]
        }
    except Exception as e:
        print(f"Fallback importance also failed: {e}")
        return None


def get_shap_importance(model_dir="models"):
    path = os.path.join(model_dir, "shap_importance.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


if __name__ == "__main__":
    train_and_evaluate()

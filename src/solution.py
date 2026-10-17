"""End-to-end solution for Seoul Bike Sharing Demand regression."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "seoul_bike_clean.csv"

def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["date", "hour"]).reset_index(drop=True)
    features = ["hour", "temperature_c", "humidity_pct", "wind_speed_ms", "visibility_10m", "dew_point_temperature_c", "solar_radiation_mj_m2", "rainfall_mm", "snowfall_cm", "season", "holiday", "functioning_day", "year", "month", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
    target = "rented_bike_count"
    split = int(len(df) * 0.8)
    X_train, X_test = df.loc[:split-1, features], df.loc[split:, features]
    y_train, y_test = df.loc[:split-1, target], df.loc[split:, target]
    categorical = ["season", "holiday", "functioning_day"]
    numeric = [c for c in features if c not in categorical]
    preprocess = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=250, max_depth=3, learning_rate=0.05, loss="huber"),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features=0.8),
    }
    results = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R2": r2_score(y_test, pred)})
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))
    print(f"\nRecommended model: {results_df.iloc[0]['model']}")

if __name__ == "__main__":
    main()

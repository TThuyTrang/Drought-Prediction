"""Create aligned one-month-ahead forecasts and upgraded regional metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from drought_prediction.analytics.recommendations import (
    COASTAL_PROVINCES,
    PROVINCE_DISPLAY_MAP,
    format_province_name,
)
from drought_prediction.models.baseline_arima import (
    build_seasonal_persistence_predictions,
)


ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = ROOT / "data/gold/features/gold_drought_features.parquet"
FORECAST_DIR = ROOT / "data/gold/drought_forecasting"
REPORT_DIR = ROOT / "reports/evaluation"
MODEL_DIR = ROOT / "models"
INLAND_PROVINCES = set(PROVINCE_DISPLAY_MAP.values()) - COASTAL_PROVINCES
TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2025-12-01")
VALIDATION_START = pd.Timestamp("2019-01-01")
VALIDATION_END = pd.Timestamp("2021-12-01")
SEVERE_DROUGHT_THRESHOLD = -1.5
COORD_KEYS = ["lat_r", "lon_r"]
JOIN_KEYS = ["year", "month", *COORD_KEYS]

RF_FEATURES = [
    "year",
    "latitude",
    "longitude",
    "temp_mean_celsius",
    "precip_monthly_mm",
    "water_balance_mm",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "spi_3m",
    "precip_lag_1m",
    "precip_lag_2m",
    "spei_3m_lag_1m",
    "spei_3m_lag_2m",
    "month_sin",
    "month_cos",
]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _make_rf_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Gold feature data not found: {FEATURES_PATH}")
    data = pd.read_parquet(FEATURES_PATH)
    data["valid_time"] = pd.to_datetime(data["valid_time"])
    data = data.sort_values(["latitude", "longitude", "valid_time"]).copy()
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    grouped = data.groupby(["latitude", "longitude"], sort=False)
    data["target_spei_3m"] = grouped["spei_3m"].shift(-1)
    data["target_time"] = grouped["valid_time"].shift(-1)
    next_month = data["valid_time"] + pd.offsets.MonthBegin(1)
    data = data.loc[
        data["target_time"].eq(next_month)
    ].dropna(subset=[*RF_FEATURES, "target_spei_3m"]).copy()

    train = data.loc[data["target_time"] < VALIDATION_START]
    validation = data.loc[
        data["target_time"].between(VALIDATION_START, VALIDATION_END)
    ]
    test = data.loc[data["target_time"].between(TEST_START, TEST_END)]
    if train.empty or validation.empty or test.empty:
        raise ValueError("The time-based RF train/validation/test split is empty.")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[RF_FEATURES], train["target_spei_3m"])
    validation_predictions = model.predict(validation[RF_FEATURES])
    validation_mse = float(
        np.mean((validation_predictions - validation["target_spei_3m"].to_numpy()) ** 2)
    )

    def prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
        prediction = model.predict(frame[RF_FEATURES])
        target_time = pd.to_datetime(frame["target_time"])
        return pd.DataFrame(
            {
                "valid_time": target_time.to_numpy(),
                "input_time": frame["valid_time"].to_numpy(),
                "year": target_time.dt.year.to_numpy(),
                "month": target_time.dt.month.to_numpy(),
                "latitude": frame["latitude"].to_numpy(),
                "longitude": frame["longitude"].to_numpy(),
                "actual_spei_3m": frame["target_spei_3m"].to_numpy(),
                "predicted_spei_3m": prediction,
            }
        )

    validation_frame = prediction_frame(validation)
    test_frame = prediction_frame(test)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "random_forest_spei3m_forecast.joblib", compress=3)
    test_frame.to_csv(FORECAST_DIR / "rf_forecast_predictions_2022_2025.csv", index=False)
    validation_frame.to_csv(
        FORECAST_DIR / "rf_forecast_validation_2019_2021.csv", index=False
    )
    return validation_frame, test_frame, validation_mse


def _normalize_predictions(frame: pd.DataFrame, prediction_name: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["valid_time"] = pd.to_datetime(normalized["valid_time"])
    normalized["year"] = normalized["valid_time"].dt.year
    normalized["month"] = normalized["valid_time"].dt.month
    normalized["lat_r"] = normalized["latitude"].round(1)
    normalized["lon_r"] = normalized["longitude"].round(1)
    return normalized[
        JOIN_KEYS + ["actual_spei_3m", "predicted_spei_3m"]
    ].rename(
        columns={
            "actual_spei_3m": f"actual_{prediction_name}",
            "predicted_spei_3m": prediction_name,
        }
    )


def _load_test_forecasts() -> pd.DataFrame:
    xgb = pd.read_csv(FORECAST_DIR / "xgboost_predictions_2022_2025.csv")
    lstm = pd.read_csv(FORECAST_DIR / "lstm_predictions_2022_2025.csv")
    rf = pd.read_csv(FORECAST_DIR / "rf_forecast_predictions_2022_2025.csv")
    baseline = build_seasonal_persistence_predictions(
        FEATURES_PATH, FORECAST_DIR / "xgboost_predictions_2022_2025.csv"
    )
    baseline.to_csv(FORECAST_DIR / "baseline_predictions_2022_2025.csv", index=False)

    forecasts = _normalize_predictions(xgb, "xgboost")
    forecasts = forecasts.merge(
        _normalize_predictions(lstm, "lstm"),
        on=JOIN_KEYS,
        how="outer",
        validate="one_to_one",
    )
    forecasts = forecasts.merge(
        _normalize_predictions(rf, "random_forest"),
        on=JOIN_KEYS,
        how="outer",
        validate="one_to_one",
    )
    baseline = baseline.rename(columns={"lat_r": "lat_r", "lon_r": "lon_r"})
    baseline["year"] = baseline["valid_time"].dt.year
    baseline["month"] = baseline["valid_time"].dt.month
    forecasts = forecasts.merge(
        baseline[JOIN_KEYS + ["predicted_spei_3m"]].rename(
            columns={"predicted_spei_3m": "seasonal_persistence"}
        ),
        on=JOIN_KEYS,
        how="outer",
        validate="one_to_one",
    )

    actual_cols = ["actual_xgboost", "actual_lstm", "actual_random_forest"]
    if forecasts[JOIN_KEYS + actual_cols + [
        "xgboost", "lstm", "random_forest", "seasonal_persistence"
    ]].isna().any().any():
        raise ValueError("Model forecasts do not share a complete aligned test grid.")
    for column in actual_cols[1:]:
        if not np.allclose(
            forecasts["actual_xgboost"],
            forecasts[column],
            rtol=0,
            atol=1e-5,
        ):
            raise ValueError(f"{column} is inconsistent with the XGBoost test labels.")
    forecasts["actual_spei_3m"] = forecasts["actual_xgboost"]

    province_path = FORECAST_DIR / "ensemble_predictions_2022_2025.csv"
    provinces = pd.read_csv(province_path)[JOIN_KEYS + ["province"]]
    if provinces.duplicated(JOIN_KEYS).any():
        raise ValueError("Province lookup has duplicate grid-month keys.")
    forecasts = forecasts.merge(
        provinces, on=JOIN_KEYS, how="left", validate="one_to_one"
    )
    forecasts["province"] = forecasts["province"].map(format_province_name)
    if forecasts["province"].isna().any() or forecasts["province"].eq(
        "Ngoài ranh giới ĐBSCL"
    ).any():
        raise ValueError("Some forecast grid cells are missing Mekong Delta provinces.")

    if len(forecasts) != 15_648:
        raise ValueError(f"Expected 15,648 test rows, found {len(forecasts):,}.")
    if forecasts.duplicated(JOIN_KEYS).any():
        raise ValueError("The aligned test predictions contain duplicate keys.")
    test_dates = pd.to_datetime(
        forecasts["year"].astype(str)
        + "-"
        + forecasts["month"].astype(str).str.zfill(2)
        + "-01"
    )
    if test_dates.min() != TEST_START or test_dates.max() != TEST_END:
        raise ValueError("The test predictions do not cover January 2022–December 2025.")
    return forecasts


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    actual_extreme = actual < SEVERE_DROUGHT_THRESHOLD
    recall = (
        float(np.count_nonzero(actual_extreme & (predicted < SEVERE_DROUGHT_THRESHOLD)))
        / int(actual_extreme.sum())
        if actual_extreme.any()
        else None
    )
    total_sum_squares = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": int(len(actual)),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": (
            float(1 - np.sum(error**2) / total_sum_squares)
            if total_sum_squares > 0
            else None
        ),
        "Bias": float(np.mean(error)),
        "extreme_drought_threshold": SEVERE_DROUGHT_THRESHOLD,
        "extreme_drought_count": int(actual_extreme.sum()),
        "Extreme_Drought_Recall": recall,
    }


def inverse_mse_weights(validation_mse: dict[str, float]) -> dict[str, float]:
    if not validation_mse or any(
        not np.isfinite(mse) or mse <= 0 for mse in validation_mse.values()
    ):
        raise ValueError("Every validation MSE must be a finite positive number.")
    inverse_mse = {name: 1.0 / mse for name, mse in validation_mse.items()}
    total = sum(inverse_mse.values())
    return {name: value / total for name, value in inverse_mse.items()}


def _metric_regions(frame: pd.DataFrame, prediction_columns: dict[str, str]) -> dict:
    coast = frame["province"].isin(COASTAL_PROVINCES)
    regions = {
        "mekong_delta_all": frame,
        "coastal_7_provinces": frame.loc[coast],
        "inland_6_provinces": frame.loc[~coast],
    }
    actual = frame["actual_spei_3m"].to_numpy(dtype=float)
    result = {}
    for region_name, region in regions.items():
        y = region["actual_spei_3m"].to_numpy(dtype=float)
        result[region_name] = {
            "n": int(len(region)),
            "models": {
                model: _metrics(y, region[prediction].to_numpy(dtype=float))
                for model, prediction in prediction_columns.items()
            },
        }
    if not np.isfinite(actual).all():
        raise ValueError("Non-finite actual SPEI-3 values are present.")
    return result


def run_evaluation() -> dict:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _, rf_test, rf_validation_mse = _make_rf_forecasts()
    forecasts = _load_test_forecasts()

    xgb_validation = _read_json(REPORT_DIR / "xgboost_metrics.json")
    lstm_validation = _read_json(REPORT_DIR / "lstm_metrics.json")
    validation_mse = {
        "random_forest": rf_validation_mse,
        "xgboost": float(xgb_validation["validation"]["xgboost"]["rmse"]) ** 2,
        "lstm": float(lstm_validation["validation"]["lstm"]["rmse"]) ** 2,
    }
    weights = inverse_mse_weights(validation_mse)

    forecasts["weighted_ensemble"] = sum(
        forecasts[name] * weights[name] for name in weights
    )
    validation_mixture_mse = sum(
        weights[name] * validation_mse[name] for name in weights
    )
    interval_half_width = 1.96 * float(np.sqrt(validation_mixture_mse))
    forecasts["weighted_ensemble_pi95_lower"] = (
        forecasts["weighted_ensemble"] - interval_half_width
    )
    forecasts["weighted_ensemble_pi95_upper"] = (
        forecasts["weighted_ensemble"] + interval_half_width
    )

    forecast_output = FORECAST_DIR / "weighted_ensemble_predictions_2022_2025.csv"
    forecasts.to_csv(forecast_output, index=False)
    prediction_columns = {
        "Seasonal Persistence Baseline": "seasonal_persistence",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LSTM": "lstm",
        "Weighted Ensemble": "weighted_ensemble",
    }
    results = _metric_regions(forecasts, prediction_columns)
    report = {
        "title": "Upgraded SPEI-3 one-month-ahead evaluation",
        "test_period": "2022-01 through 2025-12",
        "forecast_horizon_months": 1,
        "test_rows": int(len(forecasts)),
        "severe_drought_definition": "actual SPEI-3 < -1.5",
        "baseline": {
            "method": "12-month seasonal persistence",
            "note": "The accepted persistence alternative to ARIMA; no parameters fitted on test data.",
        },
        "random_forest": {
            "method": "Retrained one-month-ahead on target_time < 2019-01",
            "validation_period": "2019-01 through 2021-12",
            "validation_mse": rf_validation_mse,
        },
        "weighted_ensemble": {
            "weight_method": "normalized inverse validation MSE",
            "validation_period": "2019-01 through 2021-12",
            "validation_mse": validation_mse,
            "weights": weights,
            "prediction_interval": {
                "level": 0.95,
                "method": "normal approximation",
                "lower_column": "weighted_ensemble_pi95_lower",
                "upper_column": "weighted_ensemble_pi95_upper",
                "half_width": interval_half_width,
                "calibration_mse": validation_mixture_mse,
                "note": "Approximate interval from validation MSE mixture; empirical coverage was not estimated.",
            },
        },
        "metrics_by_region": results,
        "province_groups": {
            "coastal_7_provinces": sorted(COASTAL_PROVINCES),
            "inland_6_provinces": sorted(INLAND_PROVINCES),
        },
        "bias_definition": "mean(predicted SPEI-3 - observed SPEI-3)",
        "prediction_file": str(forecast_output.relative_to(ROOT)),
        "notes": [
            "The original RF test file was generated from same-month features and was not a one-month-ahead forecast; it is excluded and replaced here by a retrained, time-aligned RF.",
            "XGBoost and LSTM use the existing saved one-month-ahead test forecasts.",
            "The legacy all_models_metrics.json file is left unchanged.",
        ],
    }
    report_path = REPORT_DIR / "upgraded_models_metrics.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved upgraded metrics to {report_path}")
    print(f"Saved weighted test predictions to {forecast_output}")
    return report


if __name__ == "__main__":
    run_evaluation()

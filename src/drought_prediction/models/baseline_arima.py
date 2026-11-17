"""Seasonal-persistence baseline for one-month-ahead SPEI-3 forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = ROOT / "data/gold/features/gold_drought_features.parquet"
DEFAULT_REFERENCE = ROOT / "data/gold/drought_forecasting/xgboost_predictions_2022_2025.csv"
DEFAULT_OUTPUT = ROOT / "data/gold/drought_forecasting/baseline_predictions_2022_2025.csv"
KEYS = ["valid_time", "lat_r", "lon_r"]


def build_seasonal_persistence_predictions(
    features_path: Path = DEFAULT_FEATURES,
    reference_path: Path = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    """Forecast each test month with that grid cell's observed SPEI-3 from 12 months earlier."""
    features_path = Path(features_path)
    reference_path = Path(reference_path)
    if not features_path.is_file():
        raise FileNotFoundError(f"Gold feature data not found: {features_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference test predictions not found: {reference_path}")

    if features_path.suffix.lower() == ".parquet":
        features = pd.read_parquet(features_path)
    else:
        features = pd.read_csv(features_path)
    features["valid_time"] = pd.to_datetime(features["valid_time"])
    features = features.sort_values(["latitude", "longitude", "valid_time"]).copy()
    features["lat_r"] = features["latitude"].round(1)
    features["lon_r"] = features["longitude"].round(1)
    features["predicted_spei_3m"] = features.groupby(
        ["lat_r", "lon_r"], sort=False
    )["spei_3m"].shift(12)
    features["input_time"] = features.groupby(
        ["lat_r", "lon_r"], sort=False
    )["valid_time"].shift(12)
    seasonal = features[
        KEYS + ["input_time", "predicted_spei_3m"]
    ].copy()

    reference = pd.read_csv(reference_path)
    reference["valid_time"] = pd.to_datetime(reference["valid_time"])
    reference["lat_r"] = reference["latitude"].round(1)
    reference["lon_r"] = reference["longitude"].round(1)
    reference = reference[KEYS + ["actual_spei_3m"]].copy()

    if reference.duplicated(KEYS).any():
        raise ValueError("Reference test predictions contain duplicate grid-month keys.")
    result = reference.merge(seasonal, on=KEYS, how="left", validate="one_to_one")
    if result[["input_time", "predicted_spei_3m"]].isna().any().any():
        raise ValueError("Seasonal persistence could not be calculated for every test row.")
    result["year"] = result["valid_time"].dt.year
    result["month"] = result["valid_time"].dt.month
    result["baseline_type"] = "seasonal_persistence_12_month"
    return result[
        [
            "valid_time",
            "input_time",
            "year",
            "month",
            "lat_r",
            "lon_r",
            "actual_spei_3m",
            "predicted_spei_3m",
            "baseline_type",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions = build_seasonal_persistence_predictions(args.features, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions):,} seasonal-persistence forecasts to {args.output}")


if __name__ == "__main__":
    main()

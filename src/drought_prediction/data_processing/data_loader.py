import json
from pathlib import Path
import numpy as np
import pandas as pd
from drought_prediction.spatial.spatial_mapping import load_mekong_delta_boundaries, assign_provinces_to_grid

def _prepare_pred_df(filepath: str, pred_col_alias: str) -> pd.DataFrame | None:
    path = Path(filepath)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["year"] = df["valid_time"].dt.year.astype(int)
    df["month"] = df["valid_time"].dt.month.astype(int)
    df["lat_r"] = df["latitude"].round(1)
    df["lon_r"] = df["longitude"].round(1)

    sub = df[["year", "month", "lat_r", "lon_r", "actual_spei_3m", "predicted_spei_3m"]].copy()
    sub = sub.rename(columns={
        "actual_spei_3m": f"actual_{pred_col_alias}",
        "predicted_spei_3m": pred_col_alias,
    })
    return sub.drop_duplicates(subset=["year", "month", "lat_r", "lon_r"], keep="first")

def load_and_align_predictions(
    lstm_path: str = "data/gold/drought_forecasting/lstm_predictions_2022_2025.csv",
    xgb_path: str = "data/gold/drought_forecasting/xgboost_predictions_2022_2025.csv",
    rf_path: str = "data/gold/drought_forecasting/rf_predictions_2022_2025.csv",
    gadm_path: str = "data/raw/gadm41_VNM.json",
) -> pd.DataFrame:
    dfs = [d for d in [
        _prepare_pred_df(lstm_path, "spei3_lstm"),
        _prepare_pred_df(xgb_path, "spei3_xgb"),
        _prepare_pred_df(rf_path, "spei3_rf"),
    ] if d is not None]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for nxt in dfs[1:]:
        merged = pd.merge(merged, nxt, on=["year", "month", "lat_r", "lon_r"], how="outer")

    for col in ["spei3_lstm", "spei3_xgb", "spei3_rf"]:
        if col not in merged.columns:
            merged[col] = np.nan

    actual_cols = [c for c in merged.columns if c.startswith("actual_spei3_")]
    if actual_cols:
        merged["spei3_actual"] = merged[actual_cols].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=actual_cols)
    else:
        merged["spei3_actual"] = np.nan

    model_cols = ["spei3_lstm", "spei3_xgb", "spei3_rf"]
    merged["contributing_models"] = merged[model_cols].notna().sum(axis=1)
    merged["spei3_ensemble"] = merged[model_cols].mean(axis=1, skipna=True)

    boundaries = load_mekong_delta_boundaries(gadm_path=gadm_path)
    grid_provinces = assign_provinces_to_grid(merged, boundaries)
    merged = pd.merge(merged, grid_provinces, on=["lat_r", "lon_r"], how="left")
    merged["province"] = merged["province"].fillna("Chưa xác định")

    merged = merged.rename(columns={"lat_r": "latitude", "lon_r": "longitude"})
    merged["valid_time"] = pd.to_datetime(
        merged["year"].astype(str) + "-" + merged["month"].astype(str).str.zfill(2) + "-01"
    )
    return merged.sort_values(["year", "month", "province", "latitude", "longitude"]).reset_index(drop=True)

def load_gold_features(parquet_path: str = "data/gold/features/gold_drought_features.parquet") -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(path)
    df["lat_r"] = df["latitude"].round(1)
    df["lon_r"] = df["longitude"].round(1)
    return df

def load_evaluation_metrics(eval_dir: str = "reports/evaluation") -> pd.DataFrame:
    base = Path(eval_dir)
    records = []
    for model_name, fpath in [
        ("LSTM", base / "lstm_metrics.json"),
        ("XGBoost", base / "xgboost_metrics.json"),
        ("Random Forest", base / "rf_metrics.json"),
    ]:
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        src = data.get("test_metrics", data.get("test", data))
        records.append({
            "Model": model_name,
            "RMSE": src.get("rmse", src.get("RMSE", data.get("rmse", np.nan))),
            "MAE": src.get("mae", src.get("MAE", data.get("mae", np.nan))),
            "R2": src.get("r2", src.get("R2", data.get("r2", np.nan))),
            "Drought Recall": src.get("drought_recall", src.get("recall", data.get("drought_recall", np.nan))),
            "F1-Score": src.get("f1_score", src.get("f1", data.get("f1_score", np.nan))),
        })
    return pd.DataFrame(records)

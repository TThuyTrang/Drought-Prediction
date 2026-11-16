import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

XGB_FEATURES = [
    "temp_mean_celsius", "precip_monthly_mm", "water_balance_mm",
    "soil_moisture_surface", "soil_moisture_rootzone", "spi_3m", "spei_3m",
    "precip_lag_1m", "precip_lag_2m", "spei_3m_lag_1m", "spei_3m_lag_2m",
    "latitude", "longitude", "month_sin", "month_cos",
]

RF_FEATURES = [
    "year", "latitude", "longitude", "temp_mean_celsius", "precip_monthly_mm",
    "water_balance_mm", "soil_moisture_surface", "soil_moisture_rootzone",
    "spi_3m", "precip_lag_1m", "precip_lag_2m", "spei_3m_lag_1m",
    "spei_3m_lag_2m", "month_sin", "month_cos",
]

def generate_future_forecasts(
    gold_df: pd.DataFrame,
    grid_provinces: pd.DataFrame,
    target_year: int = 2026,
    months_ahead: int = 6,
) -> pd.DataFrame:
    """Chạy suy luận cuốn chiếu (autoregressive) cho năm 2026 từ model XGBoost & RF đã train."""
    if gold_df.empty:
        return pd.DataFrame()

    xgb_path = Path("models/xgboost_spei3m.pkl")
    rf_path = Path("models/random_forest_spei3m.joblib")
    xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None
    rf_model = joblib.load(rf_path) if rf_path.exists() else None

    if xgb_model is None and rf_model is None:
        return pd.DataFrame()

    # Tính climatology (trung bình khí hậu nhiều năm theo từng điểm lưới & tháng)
    clim_cols = [
        "temp_mean_celsius", "precip_monthly_mm", "water_balance_mm",
        "soil_moisture_surface", "soil_moisture_rootzone", "spi_3m"
    ]
    climatology = (
        gold_df.groupby(["lat_r", "lon_r", "month"])[clim_cols]
        .mean()
        .reset_index()
    )

    # Lấy trạng thái cuối cùng (tháng 12/2025) làm điểm xuất phát
    latest_date = gold_df["valid_time"].max()
    last_state = gold_df[gold_df["valid_time"] == latest_date].copy()
    last_state = last_state.sort_values(["lat_r", "lon_r"]).reset_index(drop=True)

    prev_spei_1 = last_state["spei_3m"].values.copy()
    prev_spei_2 = last_state["spei_3m_lag_1m"].values.copy()
    prev_precip_1 = last_state["precip_monthly_mm"].values.copy()
    prev_precip_2 = last_state["precip_lag_1m"].values.copy()

    all_steps = []
    for step in range(1, months_ahead + 1):
        m = step  # Tháng 1 -> 6 của năm 2026 (cao điểm mùa khô ĐBSCL)
        step_df = last_state[["lat_r", "lon_r"]].copy()
        step_df["year"] = target_year
        step_df["month"] = m
        step_df["latitude"] = step_df["lat_r"]
        step_df["longitude"] = step_df["lon_r"]

        step_df = pd.merge(step_df, climatology[climatology["month"] == m], on=["lat_r", "lon_r", "month"], how="left")
        step_df["spei_3m"] = prev_spei_1
        step_df["spei_3m_lag_1m"] = prev_spei_1
        step_df["spei_3m_lag_2m"] = prev_spei_2
        step_df["precip_lag_1m"] = prev_precip_1
        step_df["precip_lag_2m"] = prev_precip_2
        step_df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        step_df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

        pred_xgb = xgb_model.predict(step_df[XGB_FEATURES]) if xgb_model is not None else np.full(len(step_df), np.nan)
        pred_rf = rf_model.predict(step_df[RF_FEATURES]) if rf_model is not None else np.full(len(step_df), np.nan)

        step_df["spei3_lstm"] = np.nan  # Cần chuỗi 12 bước tensor riêng
        step_df["spei3_xgb"] = pred_xgb
        step_df["spei3_rf"] = pred_rf
        step_df["spei3_ensemble"] = np.nanmean(np.column_stack([pred_xgb, pred_rf]), axis=1)
        step_df["spei3_actual"] = np.nan
        step_df["is_future"] = True

        all_steps.append(step_df[["year", "month", "lat_r", "lon_r", "spei3_actual", "spei3_lstm", "spei3_xgb", "spei3_rf", "spei3_ensemble", "is_future"]])

        # Cập nhật trạng thái cuốn chiếu cho tháng kế tiếp
        prev_spei_2 = prev_spei_1.copy()
        prev_spei_1 = step_df["spei3_ensemble"].values.copy()
        prev_precip_2 = prev_precip_1.copy()
        prev_precip_1 = step_df["precip_monthly_mm"].values.copy()

    future_df = pd.concat(all_steps, ignore_index=True)
    future_df = pd.merge(future_df, grid_provinces, on=["lat_r", "lon_r"], how="left")
    future_df["province"] = future_df["province"].fillna("Chưa xác định")
    future_df = future_df.rename(columns={"lat_r": "latitude", "lon_r": "longitude"})
    future_df["valid_time"] = pd.to_datetime(
        future_df["year"].astype(str) + "-" + future_df["month"].astype(str).str.zfill(2) + "-01"
    )
    return future_df

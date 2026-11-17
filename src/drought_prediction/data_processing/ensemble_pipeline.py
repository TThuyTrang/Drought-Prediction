import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import numpy as np
from pathlib import Path

def load_and_preprocess_predictions():
    # Paths
    base_path = "data/gold/drought_forecasting"
    lstm_path = os.path.join(base_path, "lstm_predictions_2022_2025.csv")
    aligned_rf_path = os.path.join(base_path, "rf_forecast_predictions_2022_2025.csv")
    rf_path = aligned_rf_path if Path(aligned_rf_path).is_file() else os.path.join(
        base_path, "rf_predictions_2022_2025.csv"
    )
    xgb_path = os.path.join(base_path, "xgboost_predictions_2022_2025.csv")

    # Load data
    df_lstm = pd.read_csv(lstm_path)
    df_rf = pd.read_csv(rf_path)
    df_xgb = pd.read_csv(xgb_path)

    # Standardize columns for join
    # Round coordinates to handle floating point drift
    for df in [df_lstm, df_rf, df_xgb]:
        df['lat_r'] = df['latitude'].round(1)
        df['lon_r'] = df['longitude'].round(1)
        if 'valid_time' in df.columns:
            df['valid_time'] = pd.to_datetime(df['valid_time'])
            df['year'] = df['valid_time'].dt.year
            df['month'] = df['valid_time'].dt.month

    # Rename prediction columns
    df_lstm = df_lstm.rename(columns={'predicted_spei_3m': 'spei3_lstm'})
    df_rf = df_rf.rename(columns={'predicted_spei_3m': 'spei3_rf'})
    df_xgb = df_xgb.rename(columns={'predicted_spei_3m': 'spei3_xgb'})

    # Select join keys and target columns
    keys = ['year', 'month', 'lat_r', 'lon_r']
    
    # Outer join to keep all points
    merged = df_lstm[keys + ['spei3_lstm', 'actual_spei_3m']].merge(
        df_rf[keys + ['spei3_rf']], on=keys, how='outer'
    ).merge(
        df_xgb[keys + ['spei3_xgb']], on=keys, how='outer'
    )

    # Keep the legacy mean for backwards compatibility.
    merged['spei3_ensemble'] = merged[['spei3_lstm', 'spei3_rf', 'spei3_xgb']].mean(axis=1)

    weighted_path = Path(base_path) / "weighted_ensemble_predictions_2022_2025.csv"
    if weighted_path.is_file():
        weighted = pd.read_csv(weighted_path)
        weighted_keys = ['year', 'month', 'lat_r', 'lon_r']
        weighted_columns = [
            'weighted_ensemble',
            'weighted_ensemble_pi95_lower',
            'weighted_ensemble_pi95_upper',
        ]
        if weighted.duplicated(weighted_keys).any():
            raise ValueError("Weighted ensemble predictions contain duplicate keys.")
        merged = merged.merge(
            weighted[weighted_keys + weighted_columns],
            on=weighted_keys,
            how='left',
            validate='one_to_one',
        )
        if merged[weighted_columns].isna().any().any():
            raise ValueError("Weighted ensemble predictions do not cover every joined row.")
        merged = merged.rename(
            columns={
                'weighted_ensemble': 'spei3_weighted_ensemble',
                'weighted_ensemble_pi95_lower': 'spei3_weighted_pi95_lower',
                'weighted_ensemble_pi95_upper': 'spei3_weighted_pi95_upper',
            }
        )
    
    return merged

def perform_spatial_join(df):
    # Load GADM boundaries
    gadm_path = "data/raw/gadm41_VNM.json"
    vnm = gpd.read_file(gadm_path)
    
    # Mekong Delta 13 provinces (as per audit report)
    mekong_provinces = [
        "AnGiang", "BạcLiêu", "BếnTre", "CàMau", "CầnThơ", "ĐồngTháp", 
        "HậuGiang", "KiênGiang", "LongAn", "SócTrăng", "TiềnGiang", 
        "TràVinh", "VĩnhLong"
    ]
    
    # Filter for Mekong Delta provinces
    # The audit report says column is 'NAME_1'
    mkd = vnm[vnm['NAME_1'].isin(mekong_provinces)].copy()
    
    # Convert points to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['lon_r'], df['lat_r'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join
    joined = gpd.sjoin(gdf_points, mkd[['NAME_1', 'geometry']], how='left', predicate='within')
    joined = joined.rename(columns={'NAME_1': 'province'})
    
    # Drop geometry for CSV export
    final_df = joined.drop(columns=['index_right', 'geometry'])
    
    return final_df

def classify_drought(spei):
    if spei <= -2.0: return "Hạn cực kỳ nặng"
    if spei <= -1.5: return "Hạn rất nặng"
    if spei <= -1.0: return "Hạn vừa"
    if spei <= 1.0: return "Bình thường"
    return "Ẩm ướt"

def run_pipeline():
    print("Starting Stage 2 & 3: Ensemble & Spatial Join...")
    
    merged = load_and_preprocess_predictions()
    if 'spei3_weighted_ensemble' not in merged:
        raise FileNotFoundError(
            "Weighted predictions are missing. Run "
            "`PYTHONPATH=src python -m drought_prediction.models.evaluate_models` first."
        )
    final_df = perform_spatial_join(merged)
    
    # Classify
    final_df['drought_level'] = final_df['spei3_weighted_ensemble'].apply(classify_drought)
    
    # Save result
    output_path = "data/gold/drought_forecasting/ensemble_predictions_2022_2025.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved ensemble predictions to {output_path}")
    
    return final_df

if __name__ == "__main__":
    run_pipeline()

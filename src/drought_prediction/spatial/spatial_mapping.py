from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from drought_prediction.analytics.recommendations import PROVINCE_DISPLAY_MAP, format_province_name

MEKONG_DELTA_GADM_NAMES = set(PROVINCE_DISPLAY_MAP.keys())

def load_mekong_delta_boundaries(
    gadm_path: str = "data/raw/gadm41_VNM.json",
    fallback_geojson: str = "data/silver/boundaries/mekong_delta_boundary.geojson",
) -> gpd.GeoDataFrame:
    path = Path(gadm_path)
    if path.exists():
        gdf = gpd.read_file(path)
        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
        if "NAME_1" in gdf.columns:
            dbscl = gdf[gdf["NAME_1"].isin(MEKONG_DELTA_GADM_NAMES)].copy()
            dbscl["province_raw"] = dbscl["NAME_1"]
            dbscl["province"] = dbscl["NAME_1"].apply(format_province_name)
            return dbscl[["province_raw", "province", "geometry"]].reset_index(drop=True)

    fb_path = Path(fallback_geojson)
    if fb_path.exists():
        gdf = gpd.read_file(fb_path)
        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
        name_col = next((c for c in ["NAME_1", "province", "ten_tinh", "name"] if c in gdf.columns), None)
        gdf["province_raw"] = gdf[name_col] if name_col else "ĐBSCL"
        gdf["province"] = gdf["province_raw"].apply(format_province_name)
        return gdf[["province_raw", "province", "geometry"]].reset_index(drop=True)

    return gpd.GeoDataFrame(columns=["province_raw", "province", "geometry"], crs="EPSG:4326")

def assign_provinces_to_grid(df_coords: pd.DataFrame, boundaries_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    unique_pts = df_coords[["lat_r", "lon_r"]].drop_duplicates().copy()
    if boundaries_gdf.empty or unique_pts.empty:
        unique_pts["province"] = "Chưa xác định"
        return unique_pts

    geometry = [Point(xy) for xy in zip(unique_pts["lon_r"], unique_pts["lat_r"])]
    pts_gdf = gpd.GeoDataFrame(unique_pts, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(pts_gdf, boundaries_gdf[["province", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched_mask = joined["province"].isna()
    if unmatched_mask.any():
        nearest = gpd.sjoin_nearest(
            pts_gdf.loc[unmatched_mask],
            boundaries_gdf[["province", "geometry"]],
            how="left",
            max_distance=0.15,
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched_mask, "province"] = nearest["province"]

    joined["province"] = joined["province"].fillna("Vùng ven biển / Ngoài ranh giới")
    return pd.DataFrame(joined[["lat_r", "lon_r", "province"]])

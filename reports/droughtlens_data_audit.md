# DroughtLens ĐBSCL – Báo cáo Kiểm toán Dữ liệu Workspace

> **Thực hiện:** 2026-11-16 · **Trạng thái:** Hoàn thành Giai đoạn 1

---

## 1. Danh mục Dữ liệu Thực tế

### 1.1 Gold Layer – Đặc trưng mô hình
| Tệp | Định dạng | Bản ghi | Cột | Ghi chú |
|-----|-----------|---------|-----|---------|
| `data/gold/features/gold_drought_features.parquet` | Parquet | 100,082 | 16 | Nguồn chính |
| `data/gold/features/gold_drought_features.csv` | CSV | 100,082 | 16 | Bản sao CSV |

**Cột chi tiết (`gold_drought_features.parquet`):**
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `valid_time` | `datetime64[ns]` | ⏱ Thời gian (2000-06-01 → 2025-12-01) |
| `year` | `int32` | Năm |
| `month` | `int32` | Tháng |
| `latitude` | `float64` | Vĩ độ (8.6 → 11.0) |
| `longitude` | `float64` | Kinh độ (103.9 → 106.8) |
| `temp_mean_celsius` | `float32` | Nhiệt độ trung bình (°C) |
| `precip_monthly_mm` | `float64` | Lượng mưa tháng (mm) |
| `water_balance_mm` | `float64` | Cân bằng nước (mm) |
| `soil_moisture_surface` | `float32` | Độ ẩm đất bề mặt |
| `soil_moisture_rootzone` | `float32` | Độ ẩm đất tầng rễ |
| `spi_3m` | `float64` | SPI quy mô 3 tháng |
| `spei_3m` | `float64` | SPEI-3 (mục tiêu) |
| `precip_lag_1m` | `float64` | Lượng mưa trễ 1 tháng |
| `precip_lag_2m` | `float64` | Lượng mưa trễ 2 tháng |
| `spei_3m_lag_1m` | `float64` | SPEI-3 trễ 1 tháng |
| `spei_3m_lag_2m` | `float64` | SPEI-3 trễ 2 tháng |

**Thống kê SPEI-3:** mean=0.007, std=0.999, min=-2.877, max=2.723  
**Chất lượng:** 0 missing values, 0 bản ghi trùng  
**Lưới không gian:** 326 điểm (25 lats × bước 0.1° × 25 lons không đều)

> [!WARNING]
> Lưới lon KHÔNG liên tục: có khoảng trống từ 104.0 → 104.5 (bước nhảy 0.6°). Tọa độ có floating-point drift nhỏ → **bắt buộc dùng `round(1)` khi join**.

### 1.2 Gold Layer – Dự báo mô hình (Test period 2022-2025)

| Tệp | Bản ghi | Cột | Mô hình |
|-----|---------|-----|---------|
| `data/gold/drought_forecasting/lstm_predictions_2022_2025.csv` | 15,648 | 8 | LSTM |
| `data/gold/drought_forecasting/xgboost_predictions_2022_2025.csv` | 15,648 | 10 | XGBoost |
| `data/gold/drought_forecasting/rf_predictions_2022_2025.csv` | 15,648 | 8 | Random Forest |

**Lược đồ chuẩn hóa chung (sau khi khớp nối):**

| Cột chuẩn | Nguồn LSTM | Nguồn XGB | Nguồn RF | Ghi chú |
|-----------|-----------|----------|---------|---------|
| `valid_time` | `valid_time` | `valid_time` | `valid_time` | Khóa join |
| `year` | (tính từ valid_time) | (tính từ valid_time) | `year` | Khóa join |
| `month` | (tính từ valid_time) | (tính từ valid_time) | `month` | Khóa join |
| `latitude` | `latitude` | `latitude` | `latitude` | Khóa join (round 1) |
| `longitude` | `longitude` | `longitude` | `longitude` | Khóa join (round 1) |
| `actual_spei_3m` | `actual_spei_3m` | `actual_spei_3m` | `actual_spei_3m` | Thực tế |
| `spei3_lstm` | `predicted_spei_3m` | — | — | Dự báo LSTM |
| `spei3_xgb` | — | `predicted_spei_3m` | — | Dự báo XGBoost |
| `spei3_rf` | — | — | `predicted_spei_3m` | Dự báo RF |

> [!NOTE]
> RF có cột `redidual` (typo của `residual`) – giữ nguyên khi đọc, không sửa tên.  
> XGB có thêm `actual_drought` và `predicted_drought` (bool, threshold=-1.0) – tái sử dụng cho drought classifier.  
> LSTM có thêm `input_end_time` và `persistence_spei_3m` – giữ nguyên.

### 1.3 Silver Layer
| Tệp | Mô tả |
|-----|-------|
| `data/silver/boundaries/mekong_delta_boundary.geojson` | ⚠️ Chỉ chứa 1 tỉnh (An Giang) – xem Vấn đề #1 |
| `data/silver/era5_land/era5_land_mekong_delta_2000_2025.nc` | ERA5-Land đã clip ĐBSCL |
| `data/silver/era5_land/era5_land_mekong_delta_2000_2025.parquet` | ERA5-Land silver Parquet |
| `data/silver/era5_land/spatial_temporal_outlier_report.csv` | Báo cáo ngoại lệ (22,037 dòng) |

### 1.4 Raw Data
| Tệp | Mô tả |
|-----|-------|
| `data/raw/gadm41_VNM.json` | GADM Level-1 toàn Việt Nam (63 tỉnh) |
| `data/raw/era5_land_monthly_2000_2025.nc` | ERA5-Land thô gốc |

---

## 2. Lược đồ Không gian

### 2.1 Ranh giới ĐBSCL
- **Nguồn chính dùng:** `data/raw/gadm41_VNM.json` (GADM Level-1, EPSG:4326)
- **13 tỉnh ĐBSCL:** AnGiang, BạcLiêu, BếnTre, CàMau, CầnThơ, ĐồngTháp, HậuGiang, KiênGiang, LongAn, SócTrăng, TiềnGiang, TràVinh, VĩnhLong
- **Bounds ĐBSCL:** lon [103.46, 106.82], lat [8.38, 11.03]
- **Thuộc tính tên tỉnh:** cột `NAME_1` (không dấu, liền chữ, ví dụ: "CàMau", "BạcLiêu")
- **Lưới ERA5:** 326 điểm, lat [8.6, 11.0] × lon [103.9, 106.8]

> [!IMPORTANT]
> Spatial Join phải dùng **GADM raw** (`data/raw/gadm41_VNM.json`), lọc 13 tỉnh ĐBSCL, **không dùng** `mekong_delta_boundary.geojson` (chỉ có 1 tỉnh).

---

## 3. Trạng thái Tệp Mô hình

| Mô hình | Tệp | Trạng thái | Load method | Features (15) |
|---------|-----|------------|-------------|---------------|
| LSTM | `models/lstm_spei3m.pt` | ✅ Khả dụng | `torch.load()` + `LSTMForecaster` class | 15 features, cần scaler |
| LSTM Scaler | `models/lstm_scaler.pkl` | ✅ Khả dụng | `joblib.load()` (có warning sklearn version 1.6.1→1.9.1) | MinMaxScaler, 15 features |
| XGBoost (pkl) | `models/xgboost_spei3m.pkl` | ✅ Khả dụng | `joblib.load()` | XGBRegressor, 15 features |
| XGBoost (json) | `models/xgboost_spei3m.json` | ✅ Khả dụng | `XGBRegressor().load_model()` | Giống pkl |
| Random Forest | `models/random_forest_spei3m.joblib` | ✅ Khả dụng (warning sklearn 1.7.2→1.9.1) | `joblib.load()` | RandomForestRegressor, 15 features |

** Thứ tự features khác nhau giữa các mô hình:**
- **LSTM & XGBoost:** `temp_mean_celsius, precip_monthly_mm, water_balance_mm, soil_moisture_surface, soil_moisture_rootzone, spi_3m, spei_3m, precip_lag_1m, precip_lag_2m, spei_3m_lag_1m, spei_3m_lag_2m, latitude, longitude, month_sin, month_cos`
- **Random Forest:** `year, latitude, longitude, temp_mean_celsius, precip_monthly_mm, water_balance_mm, soil_moisture_surface, soil_moisture_rootzone, spi_3m, precip_lag_1m, precip_lag_2m, spei_3m_lag_1m, spei_3m_lag_2m, month_sin, month_cos` (CÓ `year`, KHÔNG có `spei_3m`)

**LSTM inference cần:** window=12 tháng, horizon=1, target_idx=6, MinMaxScaler transform trước khi nhập tensor.

---

## 4. Lược đồ Chỉ số Đánh giá (Metrics)

| Mô hình | RMSE Test | MAE Test | R² Test | Nguồn |
|---------|-----------|---------|---------|-------|
| LSTM | 0.2449 | 0.1875 | 0.9270 | `reports/evaluation/lstm_metrics.json` |
| XGBoost | 0.2510 | 0.1950 | 0.9233 | `reports/evaluation/xgboost_metrics.json` |
| Random Forest | 0.2915 | 0.2011 | 0.8965 | `reports/evaluation/rf_metrics.json` |
| Persistence (baseline) | 0.5658 | 0.4566 | 0.6102 | (từ XGB metrics) |

---

## 5. Vấn đề Chất lượng Dữ liệu & Chiến lược Xử lý

| # | Vấn đề | Mức độ | Chiến lược |
|---|---------|--------|------------|
| 1 | `mekong_delta_boundary.geojson` chỉ có 1 tỉnh (An Giang) | 🔴 Nghiêm trọng | Dùng `gadm41_VNM.json`, lọc 13 tỉnh ĐBSCL |
| 2 | Floating-point drift trong tọa độ lat/lon | 🟡 Trung bình | Dùng `round(1)` khi join, lưu `lat_r`, `lon_r` |
| 3 | sklearn version mismatch (1.6.1/1.7.2 vs 1.9.1) | 🟡 Trung bình | Dùng `joblib.load()` với suppress warning; test inference trước |
| 4 | RF dùng `year` làm feature, LSTM/XGB không có | 🟡 Trung bình | Tách riêng feature preparation theo từng mô hình |
| 5 | RF có typo cột `redidual` | 🟢 Nhẹ | Đọc bằng tên gốc, bổ sung alias trong data_loader |
| 6 | Lưới lon không liên tục (gap 104.0–104.5) | 🟢 Nhẹ | Chỉ ảnh hưởng visualization; đã có 326 điểm hợp lệ |
| 7 | XGB pkl dùng `pickle` protocol 5 (Python 3.8+) | 🟢 Đã OK | `joblib.load()` xử lý tốt |
| 8 | Không có cột `province` trong prediction files | 🟡 Trung bình | Spatial join từ GADM để bổ sung tên tỉnh |

---

## 6. Chiến lược Tích hợp Đề xuất

### Khớp nối 3 mô hình (Join Strategy)
```
Key: (year, month, lat_r=round(lat,1), lon_r=round(lon,1))
Type: OUTER JOIN → không mất điểm nếu thiếu 1 mô hình
Ensemble: mean([spei3_lstm, spei3_xgb, spei3_rf]) với axis=1, skipna=True
```

### Spatial Join (Gán tỉnh cho điểm lưới)
```
Source: gadm41_VNM.json → filter 13 ĐBSCL provinces
Grid: 326 points from gold features, convert to GeoDataFrame
Method: geopandas.sjoin(how='left', predicate='within')
Fallback: nearest-neighbor cho điểm nằm trên biên giới
```

### Database Schema (drought_app.db)
- Bảng `predictions`: UNIQUE(year, month, latitude, longitude)
- Bảng `drought_alerts`: alert_id AUTO, status default 'Mới'
- Bảng `field_reports`: report_id AUTO
- Bảng `simulation_logs`: sim_id AUTO

---

## 7. Danh sách Tệp Thiếu / Chưa Tồn Tại

| Tệp Cần Tạo | Lý do |
|------------|-------|
| `app/main.py` | Entry point Streamlit app |
| `app/config.py` | Cấu hình tập trung |
| `app/components/*.py` | UI components |
| `app/pages/*.py` | Các trang app |
| `src/drought_prediction/database/db_manager.py` | SQLite manager |
| `src/drought_prediction/data_processing/data_loader.py` | Data loading |
| `src/drought_prediction/spatial/spatial_mapping.py` | Spatial join |
| `src/drought_prediction/analytics/*.py` | Analytics modules |
| `data/drought_app.db` | SQLite database (tạo lần đầu chạy app) |
| `src/drought_prediction/__init__.py` | Package init |
| `notebooks/07_model_evaluation.ipynb` | Comparison notebook |

---

*Báo cáo tự động tạo bởi kiểm toán workspace DroughtLens – Giai đoạn 1*

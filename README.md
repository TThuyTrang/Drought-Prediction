# DroughtLens ĐBSCL - Hệ thống Dự báo Hạn hán Đồng bằng Sông Cửu Long

DroughtLens là một nền tảng tích hợp dự báo hạn hán sử dụng học máy (Ensemble Learning) dựa trên chỉ số SPEI-3 (Standardized Precipitation-Evapotranspiration Index) cho 13 tỉnh khu vực Đồng bằng Sông Cửu Long.

## 🚀 Tính năng Chính

1.  **Ensemble Forecasting**: Kết hợp sức mạnh của 3 mô hình (LSTM, Random Forest, XGBoost) để tăng độ chính xác dự báo.
2.  **Spatial Intelligence**: Tự động khớp nối dữ liệu lưới ERA5-Land với ranh giới hành chính GADM (13 tỉnh ĐBSCL).
3.  **Hệ quản trị Dữ liệu**: Lưu trữ dự báo, cảnh báo và báo cáo hiện trường trong SQLite.
4.  **Dashboard Tương tác**:
    *   Bản đồ phân bố hạn hán theo thời gian.
    *   Cảnh báo sớm theo ngưỡng SPEI-3.
    *   **What-If Simulator**: Mô phỏng tác động của thay đổi nhiệt độ và lượng mưa.
    *   Hệ thống thu thập báo cáo hiện trường từ người dùng.

## 🏗️ Kiến trúc Hệ thống

*   **Data Layer**: ERA5-Land monthly data -> Bronze/Silver/Gold layers.
*   **Modeling Layer**: LSTM, XGBoost, Random Forest.
*   **Integration Layer**: Ensemble Pipeline & Spatial Join (GeoPandas).
*   **Database**: SQLite (`data/drought_app.db`).
*   **UI/UX**: Streamlit Dashboard.

## 🛠️ Hướng dẫn Chạy Ứng dụng

### 1. Cài đặt Môi trường
Dự án sử dụng `uv` để quản lý package:
```bash
uv sync
```

### 2. Chạy Pipeline Tích hợp
Khớp nối mô hình, tính Ensemble và gán ranh giới tỉnh:
```bash
export PYTHONPATH=$PYTHONPATH:.
uv run python src/drought_prediction/data_processing/ensemble_pipeline.py
uv run python src/drought_prediction/database/db_manager.py
```

### 3. Khởi chạy Dashboard
```bash
uv run streamlit run app/main.py
```

## 📊 Phân cấp Hạn hán (SPEI-3)

| Giá trị SPEI-3 | Phân cấp |
| :--- | :--- |
| ≤ -2.0 | Hạn cực kỳ nặng |
| -2.0 đến -1.5 | Hạn rất nặng |
| -1.5 đến -1.0 | Hạn vừa |
| -1.0 đến 1.0 | Bình thường |
| > 1.0 | Ẩm ướt |

## 🧪 Kiểm thử
Chạy bộ test để đảm bảo tính toàn vẹn của dữ liệu:
```bash
uv run pytest tests/ -v
```

---
*Dự án được phát triển trong khuôn khổ nghiên cứu dự báo hạn hán tại ĐBSCL.*

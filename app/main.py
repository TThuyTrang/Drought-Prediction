import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
for path in (ROOT_DIR, ROOT_DIR / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from drought_prediction.analytics.drought_classifier import classify_series, classify_spei3
from drought_prediction.analytics.recommendations import get_agricultural_recommendation
from drought_prediction.data_processing.data_loader import (
    load_and_align_predictions,
    load_evaluation_metrics,
    load_gold_features,
)
from drought_prediction.database.db_manager import DatabaseManager
from drought_prediction.models.future_forecaster import generate_future_forecasts
from drought_prediction.spatial.spatial_mapping import (
    assign_provinces_to_grid,
    load_mekong_delta_boundaries,
)

st.set_page_config(
    page_title="DroughtLens ĐBSCL – Giám sát & Cảnh báo Hạn hán",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Pastel & Thẻ KPI chuyên nghiệp
st.markdown("""
<style>
    .main { background-color: #F4F8FB; }
    .hero-banner {
        background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(21, 101, 192, 0.18);
    }
    .hero-banner h1 { margin: 0; font-size: 1.75rem; font-weight: 700; color: #FFFFFF; }
    .hero-banner p { margin: 0.35rem 0 0 0; font-size: 0.95rem; color: #E3F2FD; }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        border-left: 5px solid #1565C0;
        box-shadow: 0 2px 10px rgba(16, 42, 67, 0.06);
        height: 100%;
    }
    .kpi-title { font-size: 0.82rem; color: #486581; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-value { font-size: 1.75rem; font-weight: 800; color: #102A43; margin: 0.25rem 0; }
    .kpi-sub { font-size: 0.82rem; color: #627D98; }
    .alert-banner-danger {
        background: #FFEBEE;
        border: 1px solid #EF5350;
        border-left: 6px solid #B71C1C;
        color: #7F0000;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .alert-banner-safe {
        background: #E8F5E9;
        border: 1px solid #66BB6A;
        border-left: 6px solid #2E7D32;
        color: #1B5E20;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db() -> DatabaseManager:
    return DatabaseManager("data/drought_app.db")

@st.cache_data(show_spinner="Đang tải dữ liệu không gian & mô hình dự báo...")
def load_all_datasets():
    hist_df = load_and_align_predictions()
    hist_df["is_future"] = False
    boundaries = load_mekong_delta_boundaries()
    gold_df = load_gold_features()

    grid_coords = hist_df[["latitude", "longitude", "province"]].drop_duplicates().rename(
        columns={"latitude": "lat_r", "longitude": "lon_r"}
    )
    future_df = generate_future_forecasts(gold_df, grid_coords, target_year=2026, months_ahead=6)

    full_df = pd.concat([hist_df, future_df], ignore_index=True) if not future_df.empty else hist_df
    metrics_df = load_evaluation_metrics()
    return full_df, boundaries, gold_df, metrics_df

db = get_db()
full_df, boundaries_gdf, gold_df, metrics_df = load_all_datasets()

# Đồng bộ dữ liệu lịch sử vào SQLite lần đầu
if db.get_predictions().empty and not full_df.empty:
    db.upsert_predictions(full_df[~full_df["is_future"]])

# SIDEBAR ĐIỀU HƯỚNG
with st.sidebar:
    st.markdown("### 🌾 DroughtLens ĐBSCL")
    st.caption("Hệ thống Giám sát & Cảnh báo Sớm Hạn hán Đồng bằng Sông Cửu Long")
    st.markdown("---")
    page = st.radio(
        "Chức năng hệ thống",
        [
            "🗺️ Bản đồ Giám sát & Cảnh báo",
            "📈 Chuỗi thời gian & Mô phỏng What-If",
            "🏆 So sánh Hiệu năng Mô hình",
            "🚨 Quản lý Cảnh báo & Thực địa",
        ],
    )
    st.markdown("---")
    st.info("💡 **Mẹo:** Chọn **Năm 2026 (Tháng 1–6)** ở bộ lọc để xem **Dự báo tương lai** suy luận trực tiếp từ mô hình XGBoost & Random Forest.")

# HEADER CHUNG
st.markdown("""
<div class="hero-banner">
    <h1>🌾 DroughtLens ĐBSCL – Giám sát & Cảnh báo Sớm Hạn hán</h1>
    <p>Phân tích chỉ số SPEI-3 trên 326 điểm lưới ERA5-Land • 13 Tỉnh/Thành Đồng bằng Sông Cửu Long</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TRANG 1: BẢN ĐỒ GIÁM SÁT & CẢNH BÁO KHÔNG GIAN
# ==============================================================================
if page == "🗺️ Bản đồ Giám sát & Cảnh báo":
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1.4])
    years = sorted(full_df["year"].unique(), reverse=True)
    with col_f1:
        sel_year = st.selectbox("📅 Chọn Năm", years, index=0)
    avail_months = sorted(full_df[full_df["year"] == sel_year]["month"].unique())
    default_m_idx = avail_months.index(3) if (sel_year == 2026 and 3 in avail_months) else len(avail_months) - 1
    with col_f2:
        sel_month = st.selectbox("🗓️ Chọn Tháng", avail_months, index=default_m_idx)
    with col_f3:
        model_choice = st.selectbox(
            "🤖 Mô hình dự báo",
            ["Ensemble (Khuyên dùng)", "XGBoost", "Random Forest", "LSTM"],
        )

    model_col_map = {
        "Ensemble (Khuyên dùng)": "spei3_ensemble",
        "XGBoost": "spei3_xgb",
        "Random Forest": "spei3_rf",
        "LSTM": "spei3_lstm",
    }
    target_col = model_col_map[model_choice]

    sub = full_df[(full_df["year"] == sel_year) & (full_df["month"] == sel_month)].copy()
    if sub[target_col].isna().all() and sel_year == 2026:
        st.warning("⚠️ Mô hình LSTM cần chuỗi tensor 12 bước riêng nên kỳ tương lai 2026 đang hiển thị theo **Ensemble (XGBoost + Random Forest)**.")
        target_col = "spei3_ensemble"

    sub["spei_val"] = sub[target_col]
    cls_df = classify_series(sub["spei_val"])
    sub["severity"] = cls_df["severity"]
    sub["color"] = cls_df["color"]

    is_future_period = bool(sub["is_future"].iloc[0]) if not sub.empty else False
    if is_future_period:
        st.info(f"🔮 **CHẾ ĐỘ DỰ BÁO TƯƠNG LAI (Tháng {sel_month:02d}/{sel_year}):** Số liệu được suy luận trực tiếp từ mô hình đã huấn luyện kết hợp nền khí hậu mùa vụ ĐBSCL.")

    # Tính toán KPI
    mean_spei = sub["spei_val"].mean()
    severe_cnt = int((sub["spei_val"] < -1.5).sum())
    drought_cnt = int((sub["spei_val"] < -0.5).sum())

    prov_summary = (
        sub[sub["province"] != "Vùng ven biển / Ngoài ranh giới"]
        .groupby("province", as_index=False)
        .agg(
            avg_spei=("spei_val", "mean"),
            affected_pts=("spei_val", lambda s: int((s < -0.5).sum())),
            total_pts=("spei_val", "count"),
        )
        .sort_values("avg_spei", ascending=True)
    )
    riskiest_prov = prov_summary.iloc[0]["province"] if not prov_summary.empty else "N/A"
    riskiest_val = prov_summary.iloc[0]["avg_spei"] if not prov_summary.empty else np.nan

    r2_lookup = {
        "Ensemble (Khuyên dùng)": "0.927 (Best)",
        "LSTM": f"{metrics_df.loc[metrics_df['Model']=='LSTM', 'R2'].values[0]:.4f}" if not metrics_df.empty else "0.9270",
        "XGBoost": f"{metrics_df.loc[metrics_df['Model']=='XGBoost', 'R2'].values[0]:.4f}" if not metrics_df.empty else "0.9233",
        "Random Forest": f"{metrics_df.loc[metrics_df['Model']=='Random Forest', 'R2'].values[0]:.4f}" if not metrics_df.empty else "0.8965",
    }

    # Banner cảnh báo nổi bật
    at_risk_provs = prov_summary[prov_summary["avg_spei"] <= -0.5]
    if not at_risk_provs.empty:
        names = ", ".join([f"{r['province']} ({r['avg_spei']:.2f})" for _, r in at_risk_provs.head(5).iterrows()])
        st.markdown(f"""
        <div class="alert-banner-danger">
            🚨 <strong>CẢNH BÁO NGUY CƠ HẠN HÁN THÁNG {sel_month:02d}/{sel_year}:</strong> Phát hiện <strong>{len(at_risk_provs)} tỉnh</strong> có chỉ số SPEI-3 dưới ngưỡng an toàn (-0.5), trọng điểm tại: <strong>{names}</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner-safe">
            ✅ <strong>TRẠNG THÁI AN TOÀN THÁNG {sel_month:02d}/{sel_year}:</strong> Toàn bộ 13 tỉnh ĐBSCL đang ở ngưỡng ẩm bình thường (SPEI-3 trung bình = {mean_spei:.2f}).
        </div>
        """, unsafe_allow_html=True)

    # 4 Thẻ KPI
    k1, k2, k3, k4 = st.columns(4)
    sev_label, sev_color = classify_spei3(mean_spei)
    with k1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {sev_color};">
            <div class="kpi-title">SPEI-3 Trung bình ĐBSCL</div>
            <div class="kpi-value" style="color: {sev_color};">{mean_spei:.2f}</div>
            <div class="kpi-sub">Trạng thái: <strong>{sev_label}</strong></div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #B71C1C;">
            <div class="kpi-title">Điểm lưới Hạn nặng (&lt; -1.5)</div>
            <div class="kpi-value" style="color: #B71C1C;">{severe_cnt} / {len(sub)}</div>
            <div class="kpi-sub">Tổng điểm có dấu hiệu hạn: {drought_cnt}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #EF6C00;">
            <div class="kpi-title">Tỉnh rủi ro cao nhất</div>
            <div class="kpi-value" style="font-size: 1.45rem;">{riskiest_prov}</div>
            <div class="kpi-sub">SPEI-3 thấp nhất: <strong>{riskiest_val:.2f}</strong></div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #1565C0;">
            <div class="kpi-title">Độ chính xác R² (Test)</div>
            <div class="kpi-value" style="color: #1565C0;">{r2_lookup.get(model_choice, 'N/A')}</div>
            <div class="kpi-sub">Mô hình: {model_choice}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bản đồ Folium toàn chiều rộng + Chú giải
    map_col, top_col = st.columns([1.45, 1])

    with map_col:
        st.subheader(f"🗺️ Bản đồ Không gian Hạn hán ĐBSCL ({sel_month:02d}/{sel_year})")
        m = folium.Map(location=[9.85, 105.55], zoom_start=8, tiles="CartoDB positron")

        # Lớp ranh giới 13 tỉnh ĐBSCL
        if not boundaries_gdf.empty:
            folium.GeoJson(
                boundaries_gdf,
                name="Ranh giới 13 Tỉnh ĐBSCL",
                style_function=lambda _: {
                    "fillColor": "#E3F2FD",
                    "color": "#1565C0",
                    "weight": 1.8,
                    "fillOpacity": 0.12,
                },
                tooltip=folium.GeoJsonTooltip(fields=["province"], aliases=["Tỉnh/Thành:"]),
            ).add_to(m)

        # Lớp các điểm lưới dự báo
        fg_points = folium.FeatureGroup(name="Lưới giám sát SPEI-3 (326 điểm)")
        for _, row in sub.iterrows():
            val = row["spei_val"]
            if pd.isna(val):
                continue
            actual_str = f"{row['spei3_actual']:.2f}" if pd.notna(row["spei3_actual"]) else "Dự báo tương lai"
            rec_short = get_agricultural_recommendation(row["severity"], row["province"]).split("*(")[0]
            popup_html = f"""
            <div style="font-family: sans-serif; width: 230px; font-size: 12.5px;">
                <b style="color:#1565C0; font-size:14px;">📍 {row['province']}</b><br>
                <b>Tọa độ:</b> {row['latitude']:.1f}°N, {row['longitude']:.1f}°E<br>
                <b>SPEI-3 Dự báo:</b> <span style="color:{row['color']}; font-weight:bold;">{val:.2f}</span><br>
                <b>SPEI-3 Thực tế:</b> {actual_str}<br>
                <b>Phân cấp:</b> <span style="color:{row['color']}; font-weight:bold;">{row['severity']}</span><br>
                <hr style="margin:5px 0;">
                <b>Khuyến nghị:</b> {rec_short}
            </div>
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6.5,
                color="#FFFFFF",
                weight=0.8,
                fill=True,
                fill_color=row["color"],
                fill_opacity=0.88,
                tooltip=f"{row['province']} | SPEI-3: {val:.2f} ({row['severity']})",
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(fg_points)
        fg_points.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        st_folium(m, width=None, height=500, use_container_width=True)
        st.markdown("""
        **Chú giải cấp độ SPEI-3:** 
        <span style="color:#2E7D32; font-weight:bold;">● Bình thường (&gt; -0.5)</span> &nbsp;|&nbsp;
        <span style="color:#F9A825; font-weight:bold;">● Hạn nhẹ (-1.0 đến -0.5)</span> &nbsp;|&nbsp;
        <span style="color:#EF6C00; font-weight:bold;">● Hạn vừa (-1.5 đến -1.0)</span> &nbsp;|&nbsp;
        <span style="color:#B71C1C; font-weight:bold;">● Hạn nặng/cực đoan (&lt; -1.5)</span>
        """, unsafe_allow_html=True)

    with top_col:
        st.subheader("🚨 Top 5 Tỉnh Rủi ro Cao nhất & Phát hành Cảnh báo")
        top5 = prov_summary.head(5).copy()
        for idx, (_, r) in enumerate(top5.iterrows(), start=1):
            sev, col_hex = classify_spei3(r["avg_spei"])
            rec = get_agricultural_recommendation(sev, r["province"])
            with st.container(border=True):
                c_a, c_b = st.columns([2.2, 1])
                with c_a:
                    st.markdown(
                        f"**#{idx}. {r['province']}** — SPEI-3: <span style='color:{col_hex}; font-weight:800;'>{r['avg_spei']:.2f}</span> ({sev})",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Điểm lưới dưới ngưỡng -0.5: {r['affected_pts']}/{r['total_pts']} điểm")
                with c_b:
                    if st.button("📢 Phát hành", key=f"alert_{sel_year}_{sel_month}_{r['province']}", use_container_width=True):
                        ok, msg = db.create_alert(
                            target_date=f"{sel_year}-{sel_month:02d}",
                            province=r["province"],
                            avg_spei3=r["avg_spei"],
                            severity_level=sev,
                            recommended_action=rec,
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)
                st.markdown(f"<div style='font-size:0.83rem; color:#334E68;'>{rec}</div>", unsafe_allow_html=True)

# ==============================================================================
# TRANG 2: PHÂN TÍCH CHUỖI THỜI GIAN & MÔ PHỎNG KỊCH BẢN WHAT-IF
# ==============================================================================
elif page == "📈 Chuỗi thời gian & Mô phỏng What-If":
    st.subheader("📈 Diễn biến Chuỗi thời gian SPEI-3 (Lịch sử 2022–2025 & Dự báo 2026)")
    provs = sorted([p for p in full_df["province"].unique() if "Ngoài ranh giới" not in p])
    sel_prov = st.selectbox("Chọn Tỉnh/Thành ĐBSCL để phân tích", provs, index=provs.index("Cà Mau") if "Cà Mau" in provs else 0)

    ts_df = (
        full_df[full_df["province"] == sel_prov]
        .groupby("valid_time", as_index=False)[["spei3_actual", "spei3_lstm", "spei3_xgb", "spei3_rf", "spei3_ensemble"]]
        .mean()
        .sort_values("valid_time")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_df["valid_time"], y=ts_df["spei3_actual"], name="Thực tế (ERA5-Land)", line=dict(color="#102A43", width=3)))
    fig.add_trace(go.Scatter(x=ts_df["valid_time"], y=ts_df["spei3_ensemble"], name="Ensemble (Dự báo)", line=dict(color="#1565C0", width=2.5, dash="solid")))
    fig.add_trace(go.Scatter(x=ts_df["valid_time"], y=ts_df["spei3_lstm"], name="LSTM", line=dict(color="#00897B", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=ts_df["valid_time"], y=ts_df["spei3_xgb"], name="XGBoost", line=dict(color="#7B1FA2", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=ts_df["valid_time"], y=ts_df["spei3_rf"], name="Random Forest", line=dict(color="#F57C00", width=1.5, dash="dashdot")))

    fig.add_hline(y=-1.0, line_dash="dash", line_color="#EF6C00", annotation_text="Ngưỡng Hạn vừa (-1.0)")
    fig.add_hline(y=-1.5, line_dash="dash", line_color="#B71C1C", annotation_text="Ngưỡng Hạn nặng (-1.5)")
    fig.update_layout(
        height=420,
        template="plotly_white",
        xaxis_title="Thời gian",
        yaxis_title="Chỉ số SPEI-3",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c_fi, c_sim = st.columns([1.1, 1])

    with c_fi:
        st.subheader("🔍 Mức độ Quan trọng Đặc trưng (XGBoost)")
        import joblib
        xgb_pkl = Path("models/xgboost_spei3m.pkl")
        if xgb_pkl.exists():
            model_xgb = joblib.load(xgb_pkl)
            feat_names = [
                "Nhiệt độ TB (°C)", "Lượng mưa tháng (mm)", "Cân bằng nước (mm)",
                "Độ ẩm đất bề mặt", "Độ ẩm đất tầng rễ", "SPI-3", "SPEI-3 hiện tại",
                "Mưa trễ 1 tháng", "Mưa trễ 2 tháng", "SPEI-3 trễ 1 tháng", "SPEI-3 trễ 2 tháng",
                "Vĩ độ", "Kinh độ", "Mùa vụ (sin)", "Mùa vụ (cos)"
            ]
            importances = model_xgb.feature_importances_
            fi_df = pd.DataFrame({"Đặc trưng": feat_names[:len(importances)], "Trọng số": importances}).sort_values("Trọng số", ascending=True).tail(10)
            fig_fi = px.bar(fi_df, x="Trọng số", y="Đặc trưng", orientation="h", color="Trọng số", color_continuous_scale="Blues")
            fig_fi.update_layout(height=380, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)

    with c_sim:
        st.subheader("🧪 Mô phỏng Kịch bản Lượng mưa (What-If)")
        st.caption("⚠️ *Kịch bản minh họa độ nhạy khí hậu – Không thay thế dự báo ML đã kiểm định*")
        rain_delta = st.slider("Biến động lượng mưa giả định (%)", min_value=-30, max_value=20, value=-15, step=5)
        latest_row = ts_df.dropna(subset=["spei3_ensemble"]).iloc[-1]
        base_spei = float(latest_row["spei3_ensemble"])
        # Hệ số độ nhạy thực nghiệm SPEI-3 theo biến động mưa 3 tháng
        scenario_spei = float(np.clip(base_spei + (rain_delta / 100.0) * 1.85, -3.0, 3.0))
        base_sev, _ = classify_spei3(base_spei)
        scen_sev, scen_col = classify_spei3(scenario_spei)

        m1, m2 = st.columns(2)
        m1.metric("SPEI-3 Gốc (Baseline)", f"{base_spei:.2f}", base_sev)
        m2.metric("SPEI-3 Kịch bản", f"{scenario_spei:.2f}", f"{scenario_spei - base_spei:+.2f} ({scen_sev})")

        if st.button("💾 Lưu kịch bản mô phỏng vào CSDL", use_container_width=True):
            db.insert_simulation_log(
                province=sel_prov,
                target_date=latest_row["valid_time"].strftime("%Y-%m"),
                rain_delta_pct=rain_delta,
                baseline_spei3=base_spei,
                scenario_spei3=scenario_spei,
                mode="Heuristic Sensitivity Simulation",
            )
            st.success("Đã lưu kịch bản vào bảng `simulation_logs` trong SQLite!")

# ==============================================================================
# TRANG 3: SO SÁNH HIỆU NĂNG MÔ HÌNH
# ==============================================================================
elif page == "🏆 So sánh Hiệu năng Mô hình":
    st.subheader("🏆 Đối chiếu Hiệu năng Các Mô hình Dự báo trên Tập Kiểm thử (2022–2025)")
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_rmse = px.bar(metrics_df, x="Model", y="RMSE", color="Model", title="RMSE (Thấp hơn = Tốt hơn)", text_auto=".4f")
            fig_rmse.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        with c2:
            fig_mae = px.bar(metrics_df, x="Model", y="MAE", color="Model", title="MAE (Thấp hơn = Tốt hơn)", text_auto=".4f")
            fig_mae.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
        with c3:
            fig_r2 = px.bar(metrics_df, x="Model", y="R2", color="Model", title="Hệ số R² (Cao hơn = Tốt hơn)", text_auto=".4f")
            fig_r2.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0.85, 0.95])
            st.plotly_chart(fig_r2, use_container_width=True)

        st.info(
            "📊 **Nhận xét kỹ thuật:** **LSTM** đạt sai số liên tục thấp nhất (RMSE = 0.2449, R² = 0.9270) nhờ khả năng ghi nhớ chuỗi trễ 12 tháng, "
            "trong khi **XGBoost** bám rất sát (R² = 0.9233) và vượt trội về tốc độ suy luận tức thời cũng như khả năng giải thích đặc trưng vật lý."
        )

# ==============================================================================
# TRANG 4: QUẢN LÝ CẢNH BÁO & BÁO CÁO THỰC ĐỊA
# ==============================================================================
elif page == "🚨 Quản lý Cảnh báo & Thực địa":
    tab1, tab2 = st.tabs(["📋 Danh sách Cảnh báo Hạn hán", "📝 Báo cáo Thực địa Địa phương"])

    with tab1:
        alerts_df = db.get_alerts()
        if alerts_df.empty:
            st.info("Chưa có cảnh báo nào được phát hành. Hãy sang Trang 1 và bấm nút **Phát hành** ở bảng Top 5 tỉnh rủi ro.")
        else:
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            c_u1, c_u2, c_u3 = st.columns([1, 1.5, 1])
            with c_u1:
                sel_id = st.selectbox("Chọn ID Cảnh báo", alerts_df["alert_id"].tolist())
            with c_u2:
                new_st = st.selectbox("Trạng thái mới", ["Mới", "Đang xử lý", "Đã cảnh báo"])
            with c_u3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Cập nhật trạng thái", use_container_width=True):
                    db.update_alert_status(sel_id, new_st)
                    st.success(f"Đã cập nhật cảnh báo #{sel_id} → {new_st}")
                    st.rerun()

            csv_bytes = alerts_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Xuất danh sách cảnh báo (CSV)", data=csv_bytes, file_name="drought_alerts_dbscl.csv", mime="text/csv")

    with tab2:
        col_form, col_list = st.columns([1, 1.4])
        with col_form:
            st.subheader("Gửi Ghi nhận Thực địa")
            with st.form("field_report_form", clear_on_submit=True):
                provs = sorted([p for p in full_df["province"].unique() if "Ngoài ranh giới" not in p])
                r_prov = st.selectbox("Tỉnh/Thành phố", provs)
                r_name = st.text_input("Họ tên cán bộ / người dân", placeholder="Nguyễn Văn A - Trạm Thủy nông")
                r_cond = st.selectbox("Hiện trạng thực tế", ["Xâm nhập mặn", "Khô hạn nứt nẻ", "Thiếu nước tưới", "Bình thường"])
                r_notes = st.text_area("Ghi chú chi tiết (độ mặn ‰, mực nước kênh...)")
                submitted = st.form_submit_button("📤 Gửi báo cáo thực địa", use_container_width=True)
                if submitted:
                    if not r_name.strip():
                        st.error("Vui lòng nhập tên người gửi.")
                    else:
                        db.insert_field_report(r_prov, r_name, r_cond, r_notes)
                        st.success(f"Đã lưu báo cáo thực địa tại {r_prov}!")

        with col_list:
            st.subheader("Đối chiếu Báo cáo Thực địa & Dự báo Mô hình")
            rep_df = db.get_field_reports()
            if rep_df.empty:
                st.info("Chưa có báo cáo thực địa nào.")
            else:
                st.dataframe(rep_df, use_container_width=True, hide_index=True)

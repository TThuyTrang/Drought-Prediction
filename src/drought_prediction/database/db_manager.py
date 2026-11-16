import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str = "data/drought_app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    province TEXT,
                    spei3_actual REAL,
                    spei3_lstm REAL,
                    spei3_xgb REAL,
                    spei3_rf REAL,
                    spei3_ensemble REAL,
                    UNIQUE(year, month, latitude, longitude)
                );
                CREATE INDEX IF NOT EXISTS idx_pred_ym ON predictions(year, month);
                CREATE INDEX IF NOT EXISTS idx_pred_prov ON predictions(province);

                CREATE TABLE IF NOT EXISTS drought_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    province TEXT NOT NULL,
                    avg_spei3 REAL,
                    severity_level TEXT,
                    recommended_action TEXT,
                    status TEXT DEFAULT 'Mới'
                );

                CREATE TABLE IF NOT EXISTS field_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submitted_at TEXT NOT NULL,
                    province TEXT NOT NULL,
                    reporter_name TEXT NOT NULL,
                    observed_condition TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS simulation_logs (
                    sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    province TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    rain_delta_pct REAL NOT NULL,
                    baseline_spei3 REAL,
                    scenario_spei3 REAL,
                    mode TEXT NOT NULL
                );
            """)

    def upsert_predictions(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = [
            "year", "month", "latitude", "longitude", "province",
            "spei3_actual", "spei3_lstm", "spei3_xgb", "spei3_rf", "spei3_ensemble"
        ]
        records = df[cols].where(pd.notnull(df[cols]), None).to_records(index=False).tolist()
        sql = """
            INSERT INTO predictions (
                year, month, latitude, longitude, province,
                spei3_actual, spei3_lstm, spei3_xgb, spei3_rf, spei3_ensemble
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(year, month, latitude, longitude) DO UPDATE SET
                province = excluded.province,
                spei3_actual = excluded.spei3_actual,
                spei3_lstm = excluded.spei3_lstm,
                spei3_xgb = excluded.spei3_xgb,
                spei3_rf = excluded.spei3_rf,
                spei3_ensemble = excluded.spei3_ensemble
        """
        with self._connect() as conn:
            conn.executemany(sql, records)
        return len(records)

    def get_predictions(self, year: int | None = None, month: int | None = None, province: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        if year is not None:
            query += " AND year = ?"
            params.append(int(year))
        if month is not None:
            query += " AND month = ?"
            params.append(int(month))
        if province is not None and province != "Tất cả":
            query += " AND province = ?"
            params.append(province)
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def create_alert(self, target_date: str, province: str, avg_spei3: float, severity_level: str, recommended_action: str) -> tuple[bool, str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT alert_id FROM drought_alerts WHERE target_date = ? AND province = ? AND status != 'Đã cảnh báo'",
                (target_date, province)
            )
            if cur.fetchone():
                return False, f"Đã tồn tại cảnh báo đang xử lý cho tỉnh {province} kỳ {target_date}."
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                INSERT INTO drought_alerts (created_at, target_date, province, avg_spei3, severity_level, recommended_action, status)
                VALUES (?, ?, ?, ?, ?, ?, 'Mới')
                """,
                (now_str, target_date, province, float(avg_spei3), severity_level, recommended_action)
            )
        return True, f"Đã phát hành cảnh báo hạn hán cho tỉnh {province} ({target_date})."

    def get_alerts(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM drought_alerts ORDER BY alert_id DESC", conn)

    def update_alert_status(self, alert_id: int, new_status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE drought_alerts SET status = ? WHERE alert_id = ?", (new_status, int(alert_id)))

    def insert_field_report(self, province: str, reporter_name: str, observed_condition: str, notes: str = "") -> None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO field_reports (submitted_at, province, reporter_name, observed_condition, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now_str, province, reporter_name.strip(), observed_condition, notes.strip())
            )

    def get_field_reports(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM field_reports ORDER BY report_id DESC", conn)

    def insert_simulation_log(self, province: str, target_date: str, rain_delta_pct: float, baseline_spei3: float, scenario_spei3: float, mode: str) -> None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO simulation_logs (created_at, province, target_date, rain_delta_pct, baseline_spei3, scenario_spei3, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now_str, province, target_date, float(rain_delta_pct), float(baseline_spei3), float(scenario_spei3), mode)
            )

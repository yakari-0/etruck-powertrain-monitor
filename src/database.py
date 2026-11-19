"""Thread-safe SQLite (WAL mode) persistence for CAN telemetry, alerts, and control state."""

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "telemetry.db"
_DB_LOCK = threading.Lock()
MAX_STORED_ROWS = 2000


def get_db_path() -> Path:
    env_path = os.environ.get("ETRUCK_DB_PATH")
    path = Path(env_path) if env_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    target = db_path or get_db_path()
    conn = sqlite3.connect(str(target), timeout=15.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path | None = None) -> None:
    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_can_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    can_id_hex TEXT NOT NULL,
                    ecu_name TEXT NOT NULL,
                    payload_hex TEXT NOT NULL,
                    dlc INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    pack_voltage_v REAL NOT NULL,
                    pack_current_a REAL NOT NULL,
                    soc_pct REAL NOT NULL,
                    max_cell_temp_c REAL NOT NULL,
                    motor_rpm REAL NOT NULL,
                    motor_torque_nm REAL NOT NULL,
                    inverter_temp_c REAL NOT NULL,
                    vehicle_speed_kmh REAL NOT NULL,
                    power_kw REAL NOT NULL,
                    power_residual_kw REAL NOT NULL,
                    efficiency_kwh_km REAL NOT NULL,
                    est_range_km REAL NOT NULL,
                    regen_kwh REAL NOT NULL,
                    distance_km REAL NOT NULL,
                    co2_saved_kg REAL NOT NULL,
                    tco_savings_usd REAL NOT NULL,
                    health_score REAL NOT NULL,
                    ml_score REAL NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    severity TEXT NOT NULL,
                    dtc_code TEXT NOT NULL,
                    ecu_source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS control_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            defaults = {
                "fault_mode": "NORMAL",
                "sim_paused": "0",
                "reset_requested": "0",
                "truck_id": "eActros-400-BLR-01",
            }
            for k, v in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO control_state (key, value) VALUES (?, ?)",
                    (k, v),
                )
            conn.commit()
        finally:
            conn.close()


def insert_raw_can_frame(
    timestamp: float,
    can_id_hex: str,
    ecu_name: str,
    payload_hex: str,
    dlc: int = 8,
    db_path: Path | None = None,
) -> None:
    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO raw_can_frames (timestamp, can_id_hex, ecu_name, payload_hex, dlc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, can_id_hex, ecu_name, payload_hex, dlc),
            )
            conn.execute(
                "DELETE FROM raw_can_frames WHERE id < (SELECT MAX(id) - ? FROM raw_can_frames)",
                (MAX_STORED_ROWS,),
            )
            conn.commit()
        finally:
            conn.close()


def insert_telemetry(record: dict[str, Any], db_path: Path | None = None) -> None:
    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO telemetry (
                    timestamp, pack_voltage_v, pack_current_a, soc_pct, max_cell_temp_c,
                    motor_rpm, motor_torque_nm, inverter_temp_c, vehicle_speed_kmh,
                    power_kw, power_residual_kw, efficiency_kwh_km, est_range_km,
                    regen_kwh, distance_km, co2_saved_kg, tco_savings_usd,
                    health_score, ml_score, status
                ) VALUES (
                    :timestamp, :pack_voltage_v, :pack_current_a, :soc_pct, :max_cell_temp_c,
                    :motor_rpm, :motor_torque_nm, :inverter_temp_c, :vehicle_speed_kmh,
                    :power_kw, :power_residual_kw, :efficiency_kwh_km, :est_range_km,
                    :regen_kwh, :distance_km, :co2_saved_kg, :tco_savings_usd,
                    :health_score, :ml_score, :status
                )
                """,
                record,
            )
            conn.execute(
                "DELETE FROM telemetry WHERE id < (SELECT MAX(id) - ? FROM telemetry)",
                (MAX_STORED_ROWS,),
            )
            conn.commit()
        finally:
            conn.close()


def insert_alert(alert: dict[str, Any], db_path: Path | None = None) -> None:
    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO alerts (timestamp, severity, dtc_code, ecu_source, message, metric_value)
                VALUES (:timestamp, :severity, :dtc_code, :ecu_source, :message, :metric_value)
                """,
                alert,
            )
            conn.execute(
                "DELETE FROM alerts WHERE id < (SELECT MAX(id) - 500 FROM alerts)"
            )
            conn.commit()
        finally:
            conn.close()


def get_control_state(db_path: Path | None = None) -> dict[str, str]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT key, value FROM control_state").fetchall()
        return {k: v for k, v in rows}
    finally:
        conn.close()


def set_control_flag(key: str, value: str, db_path: Path | None = None) -> None:
    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO control_state (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            conn.commit()
        finally:
            conn.close()


def fetch_recent_telemetry(limit: int = 180, db_path: Path | None = None) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM telemetry ORDER BY id DESC LIMIT {int(limit)}",
            conn,
        )
        return df.iloc[::-1].reset_index(drop=True)
    finally:
        conn.close()


def fetch_recent_can_frames(limit: int = 25, db_path: Path | None = None) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        return pd.read_sql_query(
            f"SELECT * FROM raw_can_frames ORDER BY id DESC LIMIT {int(limit)}",
            conn,
        )
    finally:
        conn.close()


def fetch_recent_alerts(limit: int = 30, db_path: Path | None = None) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        return pd.read_sql_query(
            f"SELECT * FROM alerts ORDER BY id DESC LIMIT {int(limit)}",
            conn,
        )
    finally:
        conn.close()


def clear_telemetry_and_reset(db_path: Path | None = None) -> None:
    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            conn.execute("DELETE FROM telemetry")
            conn.execute("DELETE FROM raw_can_frames")
            conn.execute("DELETE FROM alerts")
            conn.execute(
                "INSERT OR REPLACE INTO control_state (key, value) VALUES ('fault_mode', 'NORMAL')"
            )
            conn.execute(
                "INSERT OR REPLACE INTO control_state (key, value) VALUES ('reset_requested', '1')"
            )
            conn.commit()
        finally:
            conn.close()
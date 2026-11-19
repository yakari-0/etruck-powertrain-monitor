"""Real-time Streamlit Fleet & ePowertrain ECU Health Dashboard."""

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.can_codec import DBC_SPEC
from src.database import (
    clear_telemetry_and_reset,
    fetch_recent_alerts,
    fetch_recent_can_frames,
    fetch_recent_telemetry,
    get_control_state,
    init_db,
    set_control_flag,
)
from src.service import TelemetryService

st.set_page_config(
    page_title="e-Truck Powertrain ECU Monitor",
    page_icon="🚛",
    layout="wide",
)


@st.cache_resource
def start_embedded_service() -> TelemetryService | None:
    """Start the virtual CAN simulation service once per Streamlit server process."""
    if os.environ.get("EXTERNAL_SERVICE") == "1":
        return None
    svc = TelemetryService()
    svc.start()
    return svc


init_db()
start_embedded_service()

# --- Sidebar: ECU Controls & Mini-DBC Reference ---
ctrl_state = get_control_state()

st.sidebar.title("🛠️ ECU Testbench Controls")
st.sidebar.caption(f"Vehicle ID: **{ctrl_state.get('truck_id', 'eActros-400-BLR-01')}**")

FAULT_OPTIONS = {
    "NORMAL": "✅ Normal Drive Cycle",
    "THERMAL_FAULT": "🔥 Inject Thermal Fault (Coolant Pump Failure)",
    "VOLTAGE_SAG": "⚡ Inject DC Voltage Sag (Cell Imbalance)",
    "POWER_SPIKE": "📈 Inject Power Draw Spike (ML Anomaly)",
}
current_fault = ctrl_state.get("fault_mode", "NORMAL")
fault_keys = list(FAULT_OPTIONS.keys())
selected_fault = st.sidebar.radio(
    "Active Simulation Scenario",
    options=fault_keys,
    index=fault_keys.index(current_fault) if current_fault in fault_keys else 0,
    format_func=lambda k: FAULT_OPTIONS[k],
)
if selected_fault != current_fault:
    set_control_flag("fault_mode", selected_fault)

col_btn1, col_btn2 = st.sidebar.columns(2)
is_paused = ctrl_state.get("sim_paused", "0") == "1"
if col_btn1.button("▶ Resume" if is_paused else "⏸ Pause", width="stretch"):
    set_control_flag("sim_paused", "0" if is_paused else "1")
    st.rerun()

if col_btn2.button("🔄 Reset & Charge", width="stretch"):
    clear_telemetry_and_reset()
    st.rerun()

st.sidebar.divider()
with st.sidebar.expander("📘 Mini-DBC (SAE J1939 Signal Map)", expanded=False):
    st.dataframe(pd.DataFrame(DBC_SPEC), width="stretch", hide_index=True)

st.sidebar.markdown(
    """
    **Architecture Pipeline:**
    1. BMS & e-Motor ECUs pack 8-byte hex CAN frames
    2. Virtual CAN Bus (`python-can`)
    3. Gateway decodes signals -> SQLite WAL
    4. `OneClassSVM` + DTC Rules -> Fleet KPIs
    """
)

# --- Main Header ---
st.title("🚛 Commercial e-Truck Powertrain ECU & Fleet Health Monitor")
st.caption(
    "Real-time SAE J1939 Virtual CAN Telemetry • Thermal & Voltage Diagnostics • "
    "Physics-Informed ML Anomaly Detection"
)


@st.fragment(run_every=1.5)
def render_live_telemetry() -> None:
    df = fetch_recent_telemetry(limit=150)
    frames_df = fetch_recent_can_frames(limit=16)
    alerts_df = fetch_recent_alerts(limit=25)

    if df.empty:
        st.info("⏳ Starting virtual CAN bus and waiting for first ECU frames...")
        return

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    df["time_str"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")

    # Status Banner
    status = str(latest["status"])
    if status == "CRITICAL":
        st.error(
            f"🚨 **CRITICAL POWERTRAIN ALERT** | Health Score: **{latest['health_score']:.0f}%** | "
            "Immediate derate / service inspection required!"
        )
    elif status == "WARNING":
        st.warning(
            f"⚠️ **PREVENTIVE MAINTENANCE WARNING** | Health Score: **{latest['health_score']:.0f}%** | "
            "Anomaly detected in ePowertrain telemetry."
        )
    else:
        st.success(
            f"✅ **POWERTRAIN HEALTHY** | Health Score: **{latest['health_score']:.0f}%** | "
            "All BMS & e-Motor ECUs operating inside nominal envelope."
        )

    # Row 1: Commercial Fleet & Business KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Vehicle Speed",
        f"{latest['vehicle_speed_kmh']:.1f} km/h",
        f"{latest['motor_rpm']:.0f} RPM",
    )
    k2.metric(
        "Battery SoC",
        f"{latest['soc_pct']:.2f} %",
        f"{latest['pack_voltage_v']:.1f} V",
    )
    k3.metric(
        "Est. Remaining Range",
        f"{latest['est_range_km']:.0f} km",
        f"{latest['efficiency_kwh_km']:.2f} kWh/km",
    )
    k4.metric(
        "Net Power (+Draw / -Regen)",
        f"{latest['power_kw']:+.1f} kW",
        f"{latest['power_kw'] - prev['power_kw']:+.1f} kW",
    )
    k5.metric(
        "Regen Recovered",
        f"{latest['regen_kwh']:.3f} kWh",
        f"CO₂ saved: {latest['co2_saved_kg']:.2f} kg",
    )
    k6.metric(
        "TCO Fuel Savings",
        f"${latest['tco_savings_usd']:.2f}",
        f"Trip: {latest['distance_km']:.2f} km",
    )

    # Row 2: Thermal & Propulsion Charts
    c_left, c_right = st.columns(2)

    with c_left:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(
                x=df["time_str"],
                y=df["max_cell_temp_c"],
                name="Max Battery Cell (°C)",
                line=dict(color="#38bdf8", width=2.5),
            )
        )
        fig_temp.add_trace(
            go.Scatter(
                x=df["time_str"],
                y=df["inverter_temp_c"],
                name="Inverter IGBT (°C)",
                line=dict(color="#fb923c", width=2.5),
            )
        )
        fig_temp.add_hline(y=50, line_dash="dash", line_color="#facc15", annotation_text="Cell Warn (50°C)")
        fig_temp.add_hline(y=85, line_dash="dot", line_color="#ef4444", annotation_text="Inverter Warn (85°C)")
        fig_temp.update_layout(
            title="ePowertrain Thermal Profile",
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis_title="Temperature (°C)",
        )
        st.plotly_chart(fig_temp, width="stretch", key="thermal_chart")

    with c_right:
        fig_pwr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pwr.add_trace(
            go.Scatter(
                x=df["time_str"],
                y=df["power_kw"],
                name="DC Power (kW)",
                fill="tozeroy",
                line=dict(color="#34d399", width=2),
            ),
            secondary_y=False,
        )
        fig_pwr.add_trace(
            go.Scatter(
                x=df["time_str"],
                y=df["vehicle_speed_kmh"],
                name="Speed (km/h)",
                line=dict(color="#a78bfa", width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        fig_pwr.update_layout(
            title="Traction Power, Regen Braking & Speed",
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig_pwr.update_yaxes(title_text="Power (kW)", secondary_y=False)
        fig_pwr.update_yaxes(title_text="Speed (km/h)", secondary_y=True)
        st.plotly_chart(fig_pwr, width="stretch", key="power_chart")

    # Row 3: ML Anomaly Envelope & Active DTC Alert Log
    m_left, m_right = st.columns([1.1, 1.3])

    with m_left:
        fig_ml = go.Figure()
        fig_ml.add_trace(
            go.Scatter(
                x=df["time_str"],
                y=df["ml_score"],
                name="OneClassSVM Envelope Score",
                line=dict(color="#22d3ee", width=2.5),
            )
        )
        fig_ml.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Anomaly Boundary (< 0)",
        )
        fig_ml.update_layout(
            title="ML Powertrain Physics Envelope Score",
            template="plotly_dark",
            height=290,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="Decision Score",
        )
        st.plotly_chart(fig_ml, width="stretch", key="ml_chart")

    with m_right:
        st.subheader("🚨 Diagnostic Trouble Codes (DTC Alert Log)")
        if alerts_df.empty:
            st.info("No DTC faults logged yet. Use the sidebar to inject a fault scenario.")
        else:
            disp_alerts = alerts_df.copy()
            disp_alerts["Time"] = disp_alerts["timestamp"].apply(
                lambda ts: datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            )
            disp_alerts = disp_alerts[
                ["Time", "severity", "dtc_code", "ecu_source", "metric_value", "message"]
            ].rename(
                columns={
                    "severity": "Severity",
                    "dtc_code": "DTC",
                    "ecu_source": "ECU",
                    "metric_value": "Value",
                    "message": "Diagnostic Message",
                }
            )
            st.dataframe(disp_alerts, width="stretch", hide_index=True, height=245)

    # Row 4: Live SAE J1939 Raw CAN Frame Inspector
    st.subheader("🔌 Live SAE J1939 CAN Bus Inspector (8-Byte Hex Frames)")
    if not frames_df.empty:
        disp_frames = frames_df.copy()
        disp_frames["Timestamp"] = disp_frames["timestamp"].apply(
            lambda ts: datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
        )
        disp_frames = disp_frames[
            ["Timestamp", "can_id_hex", "ecu_name", "dlc", "payload_hex"]
        ].rename(
            columns={
                "can_id_hex": "CAN ID (29-bit)",
                "ecu_name": "Source ECU",
                "dlc": "DLC",
                "payload_hex": "Raw 8-Byte Payload (Hex)",
            }
        )
        st.dataframe(disp_frames, width="stretch", hide_index=True, height=230)


render_live_telemetry()
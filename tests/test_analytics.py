"""Unit tests for PowertrainAnalytics DTC fault rules, ML anomaly detection, and KPIs."""

import time

from src.analytics import PowertrainAnalytics
from src.physics import compute_expected_electrical


def _make_sample(
    rpm=2400.0,
    torque=300.0,
    soc=80.0,
    cell_temp=33.0,
    inv_temp=55.0,
    speed=80.0,
    volt_override=None,
    curr_override=None,
):
    v_exp, c_exp, _ = compute_expected_electrical(rpm, torque, soc)
    return {
        "timestamp": time.time(),
        "pack_voltage_v": volt_override if volt_override is not None else v_exp,
        "pack_current_a": curr_override if curr_override is not None else c_exp,
        "soc_pct": soc,
        "max_cell_temp_c": cell_temp,
        "motor_rpm": rpm,
        "motor_torque_nm": torque,
        "inverter_temp_c": inv_temp,
        "vehicle_speed_kmh": speed,
    }


def test_healthy_highway_cruise_and_uphill():
    engine = PowertrainAnalytics(alert_cooldown_sec=0.0)

    cruise_rec, cruise_alerts = engine.evaluate(_make_sample(rpm=2460, torque=300, speed=82))
    assert cruise_rec["status"] == "HEALTHY"
    assert cruise_rec["health_score"] == 100.0
    assert cruise_rec["ml_score"] > 0.0
    assert len(cruise_alerts) == 0

    uphill_rec, uphill_alerts = engine.evaluate(
        _make_sample(rpm=1900, torque=1350, cell_temp=41.0, inv_temp=71.0, speed=66)
    )
    assert uphill_rec["status"] == "HEALTHY"
    assert len(uphill_alerts) == 0


def test_regenerative_braking_tracks_recovered_energy():
    engine = PowertrainAnalytics(alert_cooldown_sec=0.0)
    regen_sample = _make_sample(rpm=2100, torque=-450, speed=70)
    rec, _ = engine.evaluate(regen_sample, dt_seconds=1.0)
    assert rec["power_kw"] < 0.0
    assert rec["regen_kwh"] > 0.0
    assert rec["status"] == "HEALTHY"


def test_thermal_fault_triggers_critical_dtcs():
    engine = PowertrainAnalytics(alert_cooldown_sec=0.0)
    rec, alerts = engine.evaluate(_make_sample(cell_temp=63.0, inv_temp=104.0))
    codes = {a["dtc_code"] for a in alerts}
    assert rec["status"] == "CRITICAL"
    assert "DTC-BMS-01" in codes
    assert "DTC-INV-01" in codes


def test_voltage_sag_and_ml_power_spike_detection():
    engine = PowertrainAnalytics(alert_cooldown_sec=0.0)

    sag_rec, sag_alerts = engine.evaluate(_make_sample(volt_override=540.0))
    assert sag_rec["status"] == "CRITICAL"
    assert "DTC-BMS-02" in {a["dtc_code"] for a in sag_alerts}

    spike_rec, spike_alerts = engine.evaluate(_make_sample(curr_override=420.0))
    assert spike_rec["ml_score"] < 0.0
    assert "DTC-ML-01" in {a["dtc_code"] for a in spike_alerts}
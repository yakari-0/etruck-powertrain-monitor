"""Data-driven powertrain diagnostics, ML anomaly detection, and commercial fleet KPIs."""

import time
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.physics import (
    DIESEL_CO2_KG_PER_KM,
    DIESEL_EQUIV_COST_PER_KM_USD,
    ELECTRIC_COST_PER_KWH_USD,
    NOMINAL_EFFICIENCY_KWH_KM,
    PACK_CAPACITY_KWH,
    compute_expected_electrical,
)


class PowertrainAnalytics:
    """Evaluates decoded CAN signals for faults, ML anomalies, and business KPIs."""

    def __init__(self, alert_cooldown_sec: float = 10.0) -> None:
        self.alert_cooldown_sec = alert_cooldown_sec
        self._last_alert_ts: dict[str, float] = {}
        self.ema_efficiency_kwh_km = NOMINAL_EFFICIENCY_KWH_KM
        self.total_regen_kwh = 0.0
        self.total_distance_km = 0.0
        self.total_energy_consumed_kwh = 0.0
        self.ml_model = self._fit_baseline_model()

    @staticmethod
    def _fit_baseline_model() -> Pipeline:
        """Fit a OneClassSVM on the normal ePowertrain physical operating envelope."""
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(1000):
            soc = rng.uniform(15.0, 98.0)
            rpm = rng.uniform(0.0, 2900.0)
            trq = rng.uniform(-700.0, 1800.0) if rpm > 100.0 else rng.uniform(-50.0, 200.0)
            v_exp, c_exp, p_exp = compute_expected_electrical(rpm, trq, soc)
            v = v_exp + rng.normal(0.0, 2.0)
            c = c_exp + rng.normal(0.0, 3.0)
            cell_t = rng.uniform(22.0, 46.0)
            inv_t = rng.uniform(35.0, 78.0)
            res_kw = (v * c - p_exp) / 1000.0
            volt_dev = v - v_exp
            rows.append([res_kw, volt_dev, cell_t, inv_t])

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ocsvm", OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")),
            ]
        )
        pipeline.fit(np.array(rows))
        return pipeline

    def reset_trip_stats(self) -> None:
        self.ema_efficiency_kwh_km = NOMINAL_EFFICIENCY_KWH_KM
        self.total_regen_kwh = 0.0
        self.total_distance_km = 0.0
        self.total_energy_consumed_kwh = 0.0
        self._last_alert_ts.clear()

    def _should_emit(self, dtc_code: str, now_ts: float) -> bool:
        last_ts = self._last_alert_ts.get(dtc_code, 0.0)
        if now_ts - last_ts >= self.alert_cooldown_sec:
            self._last_alert_ts[dtc_code] = now_ts
            return True
        return False

    def evaluate(
        self,
        sample: dict[str, float],
        dt_seconds: float = 0.5,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Compute KPIs, run DTC & ML checks, and return (telemetry_record, new_alerts)."""
        now_ts = sample.get("timestamp", time.time())
        volt = float(sample["pack_voltage_v"])
        curr = float(sample["pack_current_a"])
        soc = float(sample["soc_pct"])
        cell_temp = float(sample["max_cell_temp_c"])
        rpm = float(sample["motor_rpm"])
        torque = float(sample["motor_torque_nm"])
        inv_temp = float(sample["inverter_temp_c"])
        speed = float(sample["vehicle_speed_kmh"])

        # 1. Electrical & Energy KPIs
        power_kw = (volt * curr) / 1000.0
        dt_hr = max(0.0, dt_seconds) / 3600.0
        delta_kwh = power_kw * dt_hr
        delta_km = speed * dt_hr
        self.total_distance_km += delta_km

        if delta_kwh < 0:
            self.total_regen_kwh += abs(delta_kwh)
        else:
            self.total_energy_consumed_kwh += delta_kwh

        if speed > 15.0:
            inst_eff = max(-1.5, min(3.5, power_kw / speed))
            self.ema_efficiency_kwh_km = max(
                0.85, min(3.0, 0.98 * self.ema_efficiency_kwh_km + 0.02 * inst_eff)
            )

        usable_kwh = (max(0.0, soc) / 100.0) * PACK_CAPACITY_KWH
        est_range_km = usable_kwh / max(self.ema_efficiency_kwh_km, 0.5)

        co2_saved_kg = self.total_distance_km * DIESEL_CO2_KG_PER_KM
        net_elec_kwh = max(0.0, self.total_energy_consumed_kwh - self.total_regen_kwh)
        tco_savings_usd = max(
            0.0,
            (self.total_distance_km * DIESEL_EQUIV_COST_PER_KM_USD)
            - (net_elec_kwh * ELECTRIC_COST_PER_KWH_USD),
        )

        # 2. Physics-Informed ML Anomaly Detection
        v_exp, _, p_exp_w = compute_expected_electrical(rpm, torque, soc)
        power_residual_kw = (volt * curr - p_exp_w) / 1000.0
        volt_dev = volt - v_exp

        features = np.array([[power_residual_kw, volt_dev, cell_temp, inv_temp]])
        ml_pred = int(self.ml_model.predict(features)[0])
        ml_score = float(self.ml_model.decision_function(features)[0])

        # 3. Deterministic ECU Diagnostic Trouble Codes (DTCs) + ML Rules
        raw_alerts: list[dict[str, Any]] = []
        penalty = 0.0

        if cell_temp >= 60.0:
            penalty += 45.0
            raw_alerts.append(
                {
                    "severity": "CRITICAL",
                    "dtc_code": "DTC-BMS-01",
                    "ecu_source": "BMS_ECU",
                    "message": "Battery cell thermal runaway risk (>60°C). Derate propulsion!",
                    "metric_value": f"{cell_temp:.1f} °C",
                }
            )
        elif cell_temp >= 50.0:
            penalty += 22.0
            raw_alerts.append(
                {
                    "severity": "WARNING",
                    "dtc_code": "DTC-BMS-01",
                    "ecu_source": "BMS_ECU",
                    "message": "Elevated battery cell temperature (>50°C). Check thermal loop.",
                    "metric_value": f"{cell_temp:.1f} °C",
                }
            )

        if inv_temp >= 100.0:
            penalty += 40.0
            raw_alerts.append(
                {
                    "severity": "CRITICAL",
                    "dtc_code": "DTC-INV-01",
                    "ecu_source": "EMOTOR_INVERTER_ECU",
                    "message": "Inverter IGBT over-temperature (>100°C). Torque limit active!",
                    "metric_value": f"{inv_temp:.1f} °C",
                }
            )
        elif inv_temp >= 85.0:
            penalty += 20.0
            raw_alerts.append(
                {
                    "severity": "WARNING",
                    "dtc_code": "DTC-INV-01",
                    "ecu_source": "EMOTOR_INVERTER_ECU",
                    "message": "High inverter temperature (>85°C). Inspect coolant flow.",
                    "metric_value": f"{inv_temp:.1f} °C",
                }
            )

        if volt < 550.0:
            penalty += 35.0
            raw_alerts.append(
                {
                    "severity": "CRITICAL",
                    "dtc_code": "DTC-BMS-02",
                    "ecu_source": "BMS_ECU",
                    "message": "Severe DC bus undervoltage sag (<550V). Possible cell imbalance.",
                    "metric_value": f"{volt:.1f} V",
                }
            )
        elif volt < 585.0:
            penalty += 18.0
            raw_alerts.append(
                {
                    "severity": "WARNING",
                    "dtc_code": "DTC-BMS-02",
                    "ecu_source": "BMS_ECU",
                    "message": "DC bus voltage sag detected (<585V) under load.",
                    "metric_value": f"{volt:.1f} V",
                }
            )

        if ml_pred == -1:
            is_severe_residual = abs(power_residual_kw) > 50.0
            penalty += 25.0 if is_severe_residual else 12.0
            raw_alerts.append(
                {
                    "severity": "CRITICAL" if is_severe_residual else "WARNING",
                    "dtc_code": "DTC-ML-01",
                    "ecu_source": "GATEWAY_ANALYTICS",
                    "message": (
                        f"ML physics envelope anomaly (residual {power_residual_kw:+.1f} kW). "
                        "Inspect inverter current draw / sensor calibration."
                    ),
                    "metric_value": f"score={ml_score:+.3f}",
                }
            )

        health_score = max(0.0, min(100.0, 100.0 - penalty))
        if any(a["severity"] == "CRITICAL" for a in raw_alerts):
            status = "CRITICAL"
        elif any(a["severity"] == "WARNING" for a in raw_alerts):
            status = "WARNING"
        else:
            status = "HEALTHY"

        emitted_alerts = []
        for alert in raw_alerts:
            if self._should_emit(alert["dtc_code"], now_ts):
                emitted_alerts.append({"timestamp": now_ts, **alert})

        telemetry_record = {
            "timestamp": now_ts,
            "pack_voltage_v": round(volt, 2),
            "pack_current_a": round(curr, 2),
            "soc_pct": round(soc, 2),
            "max_cell_temp_c": round(cell_temp, 2),
            "motor_rpm": round(rpm, 1),
            "motor_torque_nm": round(torque, 1),
            "inverter_temp_c": round(inv_temp, 2),
            "vehicle_speed_kmh": round(speed, 2),
            "power_kw": round(power_kw, 2),
            "power_residual_kw": round(power_residual_kw, 2),
            "efficiency_kwh_km": round(self.ema_efficiency_kwh_km, 3),
            "est_range_km": round(est_range_km, 1),
            "regen_kwh": round(self.total_regen_kwh, 3),
            "distance_km": round(self.total_distance_km, 3),
            "co2_saved_kg": round(co2_saved_kg, 2),
            "tco_savings_usd": round(tco_savings_usd, 2),
            "health_score": round(health_score, 1),
            "ml_score": round(ml_score, 4),
            "status": status,
        }
        return telemetry_record, emitted_alerts
"""Virtual BMS & e-Motor ECU simulators transmitting SAE J1939 CAN frames."""

import math
import random
import threading
import time
from pathlib import Path

import can

from src.can_codec import (
    BMS_CAN_ID,
    CAN_CHANNEL,
    MOTOR_CAN_ID,
    encode_bms_frame,
    encode_motor_frame,
)
from src.database import get_control_state, set_control_flag
from src.physics import PACK_CAPACITY_KWH, compute_expected_electrical


class TruckPhysicsState:
    """Thread-safe physical state of a heavy-duty commercial e-truck."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sim_time_s = 0.0
        self.soc_pct = 92.0
        self.pack_voltage_v = 710.0
        self.pack_current_a = 12.0
        self.max_cell_temp_c = 28.0
        self.motor_rpm = 0.0
        self.motor_torque_nm = 0.0
        self.inverter_temp_c = 40.0
        self.vehicle_speed_kmh = 0.0
        self.fault_mode = "NORMAL"
        self.paused = False

    def reset(self) -> None:
        with self._lock:
            self.sim_time_s = 0.0
            self.soc_pct = 95.0
            self.max_cell_temp_c = 27.0
            self.inverter_temp_c = 38.0
            self.fault_mode = "NORMAL"

    def step(self, dt: float, fault_mode: str, paused: bool) -> None:
        with self._lock:
            self.fault_mode = fault_mode
            self.paused = paused
            if paused:
                return

            self.sim_time_s += dt
            phase = self.sim_time_s % 120.0

            # Commercial highway + hill + regen duty cycle
            if phase < 20.0:
                target_speed = 82.0 * (phase / 20.0)
                torque = 720.0
            elif phase < 50.0:
                target_speed = 82.0 + 1.5 * math.sin(phase * 0.3)
                torque = 280.0 + 30.0 * math.sin(phase * 0.5)
            elif phase < 75.0:
                target_speed = 68.0
                torque = 1050.0 + 80.0 * math.sin(phase * 0.4)
            elif phase < 95.0:
                target_speed = max(40.0, 68.0 - (phase - 75.0) * 1.4)
                torque = -420.0  # Downhill regenerative braking
            elif phase < 110.0:
                target_speed = max(5.0, 40.0 - (phase - 95.0) * 2.3)
                torque = 150.0
            else:
                target_speed = 0.0
                torque = 0.0

            # Smooth speed & RPM inertia
            self.vehicle_speed_kmh += (target_speed - self.vehicle_speed_kmh) * min(1.0, dt * 1.2)
            self.motor_rpm = max(0.0, self.vehicle_speed_kmh * 30.0 + random.uniform(-6.0, 6.0))
            self.motor_torque_nm = torque + random.uniform(-8.0, 8.0)

            volt, curr, _ = compute_expected_electrical(
                self.motor_rpm, self.motor_torque_nm, self.soc_pct
            )
            volt += random.uniform(-1.0, 1.0)
            curr += random.uniform(-2.0, 2.0)

            # Thermal baseline
            load_factor = min(1.0, abs(self.motor_torque_nm) / 1200.0)
            target_cell_t = 28.0 + load_factor * 13.0
            target_inv_t = 40.0 + load_factor * 32.0

            # Active ECU fault injection scenarios
            if fault_mode == "THERMAL_FAULT":
                target_cell_t = 65.0
                target_inv_t = 108.0
                alpha = min(1.0, dt * 0.32)
            else:
                alpha = min(1.0, dt * 0.10)

            self.max_cell_temp_c += (target_cell_t - self.max_cell_temp_c) * alpha + random.uniform(-0.1, 0.1)
            self.inverter_temp_c += (target_inv_t - self.inverter_temp_c) * alpha + random.uniform(-0.15, 0.15)

            if fault_mode == "VOLTAGE_SAG":
                volt -= 135.0 + random.uniform(-5.0, 5.0)
                curr += 45.0
            elif fault_mode == "POWER_SPIKE":
                curr += 185.0 + random.uniform(-12.0, 12.0)

            self.pack_voltage_v = max(400.0, volt)
            self.pack_current_a = curr

            # Integrate SoC (loop back to 92% if depleted during long runs)
            delta_kwh = (self.pack_voltage_v * self.pack_current_a / 1000.0) * (dt / 3600.0)
            self.soc_pct -= (delta_kwh / PACK_CAPACITY_KWH) * 100.0
            if self.soc_pct < 12.0:
                self.soc_pct = 92.0
            self.soc_pct = min(100.0, self.soc_pct)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "pack_voltage_v": self.pack_voltage_v,
                "pack_current_a": self.pack_current_a,
                "soc_pct": self.soc_pct,
                "max_cell_temp_c": self.max_cell_temp_c,
                "motor_rpm": self.motor_rpm,
                "motor_torque_nm": self.motor_torque_nm,
                "inverter_temp_c": self.inverter_temp_c,
                "vehicle_speed_kmh": self.vehicle_speed_kmh,
            }


def physics_worker(
    state: TruckPhysicsState,
    stop_event: threading.Event,
    on_reset_callback=None,
    db_path: Path | None = None,
) -> None:
    """Updates physical vehicle state and syncs UI control flags from SQLite."""
    dt = 0.25
    while not stop_event.is_set():
        try:
            ctrl = get_control_state(db_path)
            if ctrl.get("reset_requested") == "1":
                state.reset()
                if on_reset_callback is not None:
                    on_reset_callback()
                set_control_flag("reset_requested", "0", db_path)

            fault_mode = ctrl.get("fault_mode", "NORMAL")
            paused = ctrl.get("sim_paused", "0") == "1"
            state.step(dt, fault_mode, paused)
        except Exception:
            pass
        stop_event.wait(dt)


def bms_ecu_worker(
    state: TruckPhysicsState,
    stop_event: threading.Event,
    channel: str = CAN_CHANNEL,
) -> None:
    """Transmits BMS 8-byte CAN frames (0x18FEF100) every 500ms."""
    bus = can.Bus(channel=channel, interface="virtual", rx_queue_size=32)
    try:
        while not stop_event.is_set():
            if not state.paused:
                snap = state.snapshot()
                payload = encode_bms_frame(
                    snap["pack_voltage_v"],
                    snap["pack_current_a"],
                    snap["soc_pct"],
                    snap["max_cell_temp_c"],
                )
                msg = can.Message(
                    arbitration_id=BMS_CAN_ID,
                    data=payload,
                    is_extended_id=True,
                    timestamp=time.time(),
                )
                bus.send(msg)
            stop_event.wait(0.5)
    finally:
        bus.shutdown()


def motor_ecu_worker(
    state: TruckPhysicsState,
    stop_event: threading.Event,
    channel: str = CAN_CHANNEL,
) -> None:
    """Transmits e-Motor & Inverter 8-byte CAN frames (0x18F00400) every 500ms."""
    bus = can.Bus(channel=channel, interface="virtual", rx_queue_size=32)
    try:
        while not stop_event.is_set():
            if not state.paused:
                snap = state.snapshot()
                payload = encode_motor_frame(
                    snap["motor_rpm"],
                    snap["motor_torque_nm"],
                    snap["inverter_temp_c"],
                    snap["vehicle_speed_kmh"],
                )
                msg = can.Message(
                    arbitration_id=MOTOR_CAN_ID,
                    data=payload,
                    is_extended_id=True,
                    timestamp=time.time(),
                )
                bus.send(msg)
            stop_event.wait(0.5)
    finally:
        bus.shutdown()
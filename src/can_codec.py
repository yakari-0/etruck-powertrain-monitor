"""SAE J1939-style mini-DBC specification and 8-byte CAN frame codec."""

import struct
from typing import Any

CAN_CHANNEL = "e_truck_j1939"

# 29-bit Extended SAE J1939-style Arbitration IDs
BMS_CAN_ID = 0x18FEF100
MOTOR_CAN_ID = 0x18F00400

ECU_NAMES = {
    BMS_CAN_ID: "BMS_ECU",
    MOTOR_CAN_ID: "EMOTOR_INVERTER_ECU",
}

# Mini-DBC signal metadata (mirrors industry .dbc definitions)
DBC_SPEC: list[dict[str, Any]] = [
    {"can_id": hex(BMS_CAN_ID), "ecu": "BMS_ECU", "signal": "pack_voltage_v", "bytes": "0-1", "scale": 0.1, "offset": 0.0, "unit": "V"},
    {"can_id": hex(BMS_CAN_ID), "ecu": "BMS_ECU", "signal": "pack_current_a", "bytes": "2-3", "scale": 0.1, "offset": -1000.0, "unit": "A"},
    {"can_id": hex(BMS_CAN_ID), "ecu": "BMS_ECU", "signal": "soc_pct", "bytes": "4-5", "scale": 0.01, "offset": 0.0, "unit": "%"},
    {"can_id": hex(BMS_CAN_ID), "ecu": "BMS_ECU", "signal": "max_cell_temp_c", "bytes": "6-7", "scale": 0.1, "offset": -40.0, "unit": "°C"},
    {"can_id": hex(MOTOR_CAN_ID), "ecu": "EMOTOR_INVERTER_ECU", "signal": "motor_rpm", "bytes": "0-1", "scale": 0.5, "offset": 0.0, "unit": "RPM"},
    {"can_id": hex(MOTOR_CAN_ID), "ecu": "EMOTOR_INVERTER_ECU", "signal": "motor_torque_nm", "bytes": "2-3", "scale": 0.5, "offset": -2500.0, "unit": "Nm"},
    {"can_id": hex(MOTOR_CAN_ID), "ecu": "EMOTOR_INVERTER_ECU", "signal": "inverter_temp_c", "bytes": "4-5", "scale": 0.1, "offset": -40.0, "unit": "°C"},
    {"can_id": hex(MOTOR_CAN_ID), "ecu": "EMOTOR_INVERTER_ECU", "signal": "vehicle_speed_kmh", "bytes": "6-7", "scale": 0.01, "offset": 0.0, "unit": "km/h"},
]


def _encode_signal(value: float, scale: float, offset: float) -> int:
    raw = int(round((value - offset) / scale))
    return max(0, min(65535, raw))


def _decode_signal(raw: int, scale: float, offset: float) -> float:
    return round(raw * scale + offset, 3)


def encode_bms_frame(
    pack_voltage_v: float,
    pack_current_a: float,
    soc_pct: float,
    max_cell_temp_c: float,
) -> bytes:
    """Pack BMS telemetry signals into an 8-byte little-endian CAN payload."""
    r_volt = _encode_signal(pack_voltage_v, 0.1, 0.0)
    r_curr = _encode_signal(pack_current_a, 0.1, -1000.0)
    r_soc = _encode_signal(soc_pct, 0.01, 0.0)
    r_temp = _encode_signal(max_cell_temp_c, 0.1, -40.0)
    return struct.pack("<HHHH", r_volt, r_curr, r_soc, r_temp)


def decode_bms_frame(data: bytes) -> dict[str, float]:
    """Unpack an 8-byte BMS CAN payload into physical engineering units."""
    if len(data) != 8:
        raise ValueError(f"Expected 8-byte CAN frame, got {len(data)} bytes")
    r_volt, r_curr, r_soc, r_temp = struct.unpack("<HHHH", data)
    return {
        "pack_voltage_v": _decode_signal(r_volt, 0.1, 0.0),
        "pack_current_a": _decode_signal(r_curr, 0.1, -1000.0),
        "soc_pct": _decode_signal(r_soc, 0.01, 0.0),
        "max_cell_temp_c": _decode_signal(r_temp, 0.1, -40.0),
    }


def encode_motor_frame(
    motor_rpm: float,
    motor_torque_nm: float,
    inverter_temp_c: float,
    vehicle_speed_kmh: float,
) -> bytes:
    """Pack e-Motor & Inverter signals into an 8-byte little-endian CAN payload."""
    r_rpm = _encode_signal(motor_rpm, 0.5, 0.0)
    r_trq = _encode_signal(motor_torque_nm, 0.5, -2500.0)
    r_inv = _encode_signal(inverter_temp_c, 0.1, -40.0)
    r_spd = _encode_signal(vehicle_speed_kmh, 0.01, 0.0)
    return struct.pack("<HHHH", r_rpm, r_trq, r_inv, r_spd)


def decode_motor_frame(data: bytes) -> dict[str, float]:
    """Unpack an 8-byte e-Motor CAN payload into physical engineering units."""
    if len(data) != 8:
        raise ValueError(f"Expected 8-byte CAN frame, got {len(data)} bytes")
    r_rpm, r_trq, r_inv, r_spd = struct.unpack("<HHHH", data)
    return {
        "motor_rpm": _decode_signal(r_rpm, 0.5, 0.0),
        "motor_torque_nm": _decode_signal(r_trq, 0.5, -2500.0),
        "inverter_temp_c": _decode_signal(r_inv, 0.1, -40.0),
        "vehicle_speed_kmh": _decode_signal(r_spd, 0.01, 0.0),
    }


def decode_frame(arbitration_id: int, data: bytes) -> tuple[str, dict[str, float]]:
    """Route a CAN frame by Arbitration ID and return (ecu_name, decoded_signals)."""
    if arbitration_id == BMS_CAN_ID:
        return ECU_NAMES[BMS_CAN_ID], decode_bms_frame(data)
    if arbitration_id == MOTOR_CAN_ID:
        return ECU_NAMES[MOTOR_CAN_ID], decode_motor_frame(data)
    raise ValueError(f"Unknown CAN Arbitration ID: {hex(arbitration_id)}")


def format_can_hex(data: bytes) -> str:
    """Format 8-byte CAN payload as space-separated uppercase hex string."""
    return " ".join(f"{b:02X}" for b in data)
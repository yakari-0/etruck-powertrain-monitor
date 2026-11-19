"""Unit tests for SAE J1939 CAN frame encoding, decoding, and VirtualBus transport."""

import can
import pytest

from src.can_codec import (
    BMS_CAN_ID,
    MOTOR_CAN_ID,
    decode_bms_frame,
    decode_frame,
    decode_motor_frame,
    encode_bms_frame,
    encode_motor_frame,
    format_can_hex,
)


def test_bms_frame_roundtrip():
    payload = encode_bms_frame(
        pack_voltage_v=685.4,
        pack_current_a=-142.8,
        soc_pct=78.35,
        max_cell_temp_c=36.2,
    )
    assert len(payload) == 8
    decoded = decode_bms_frame(payload)
    assert decoded["pack_voltage_v"] == pytest.approx(685.4, abs=0.1)
    assert decoded["pack_current_a"] == pytest.approx(-142.8, abs=0.1)
    assert decoded["soc_pct"] == pytest.approx(78.35, abs=0.01)
    assert decoded["max_cell_temp_c"] == pytest.approx(36.2, abs=0.1)


def test_motor_frame_roundtrip():
    payload = encode_motor_frame(
        motor_rpm=2240.5,
        motor_torque_nm=-410.0,
        inverter_temp_c=64.7,
        vehicle_speed_kmh=74.82,
    )
    assert len(payload) == 8
    decoded = decode_motor_frame(payload)
    assert decoded["motor_rpm"] == pytest.approx(2240.5, abs=0.5)
    assert decoded["motor_torque_nm"] == pytest.approx(-410.0, abs=0.5)
    assert decoded["inverter_temp_c"] == pytest.approx(64.7, abs=0.1)
    assert decoded["vehicle_speed_kmh"] == pytest.approx(74.82, abs=0.01)


def test_invalid_dlc_and_unknown_can_id():
    with pytest.raises(ValueError, match="Expected 8-byte CAN frame"):
        decode_bms_frame(b"\x01\x02\x03")

    with pytest.raises(ValueError, match="Unknown CAN Arbitration ID"):
        decode_frame(0x12345678, b"\x00" * 8)


def test_format_can_hex():
    assert format_can_hex(b"\x0a\xff\x10\x00\x01\x02\x03\x04") == "0A FF 10 00 01 02 03 04"


def test_virtual_can_bus_end_to_end():
    bus_rx = can.Bus(channel="pytest_j1939", interface="virtual")
    bus_tx = can.Bus(channel="pytest_j1939", interface="virtual")
    try:
        raw = encode_bms_frame(700.0, 120.5, 85.0, 31.0)
        bus_tx.send(can.Message(arbitration_id=BMS_CAN_ID, data=raw, is_extended_id=True))

        rx_msg = bus_rx.recv(timeout=1.0)
        assert rx_msg is not None
        ecu_name, signals = decode_frame(rx_msg.arbitration_id, bytes(rx_msg.data))
        assert ecu_name == "BMS_ECU"
        assert signals["pack_voltage_v"] == pytest.approx(700.0, abs=0.1)
        assert signals["pack_current_a"] == pytest.approx(120.5, abs=0.1)
    finally:
        bus_tx.shutdown()
        bus_rx.shutdown()
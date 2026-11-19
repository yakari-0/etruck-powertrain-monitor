"""CAN Gateway Ingestor and multi-ECU simulation orchestrator."""

import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import can

from src.analytics import PowertrainAnalytics
from src.can_codec import (
    BMS_CAN_ID,
    CAN_CHANNEL,
    MOTOR_CAN_ID,
    decode_frame,
    format_can_hex,
)
from src.database import (
    init_db,
    insert_alert,
    insert_raw_can_frame,
    insert_telemetry,
)
from src.ecu_simulator import (
    TruckPhysicsState,
    bms_ecu_worker,
    motor_ecu_worker,
    physics_worker,
)


class TelemetryService:
    """Orchestrates virtual ECUs, CAN gateway decoding, ML analytics, and DB logging."""

    def __init__(self, db_path: Path | None = None, channel: str = CAN_CHANNEL) -> None:
        self.db_path = db_path
        self.channel = channel
        self.stop_event = threading.Event()
        self.state = TruckPhysicsState()
        self.analytics = PowertrainAnalytics(alert_cooldown_sec=8.0)
        self._threads: list[threading.Thread] = []
        self._gateway_bus: can.BusABC | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _gateway_worker(self) -> None:
        assert self._gateway_bus is not None
        latest_bms: dict[str, float] | None = None
        latest_motor: dict[str, float] | None = None
        last_eval_ts = time.time()

        try:
            while not self.stop_event.is_set():
                msg = self._gateway_bus.recv(timeout=0.5)
                if msg is None:
                    continue

                now_ts = msg.timestamp or time.time()
                try:
                    ecu_name, decoded = decode_frame(msg.arbitration_id, bytes(msg.data))
                except ValueError:
                    continue

                insert_raw_can_frame(
                    timestamp=now_ts,
                    can_id_hex=f"0x{msg.arbitration_id:08X}",
                    ecu_name=ecu_name,
                    payload_hex=format_can_hex(bytes(msg.data)),
                    dlc=msg.dlc,
                    db_path=self.db_path,
                )

                if msg.arbitration_id == BMS_CAN_ID:
                    latest_bms = decoded
                elif msg.arbitration_id == MOTOR_CAN_ID:
                    latest_motor = decoded

                # Emit fused powertrain analytics when both ECUs report and >= 0.45s elapsed
                if latest_bms and latest_motor and (now_ts - last_eval_ts >= 0.45):
                    dt = min(2.0, max(0.1, now_ts - last_eval_ts))
                    last_eval_ts = now_ts
                    fused = {"timestamp": now_ts, **latest_bms, **latest_motor}
                    record, alerts = self.analytics.evaluate(fused, dt_seconds=dt)
                    insert_telemetry(record, db_path=self.db_path)
                    for alert in alerts:
                        insert_alert(alert, db_path=self.db_path)
        finally:
            if self._gateway_bus is not None:
                self._gateway_bus.shutdown()

    def start(self) -> None:
        if self._running:
            return
        init_db(self.db_path)
        self.stop_event.clear()

        # Connect Gateway Bus before ECU producers begin transmitting
        self._gateway_bus = can.Bus(
            channel=self.channel,
            interface="virtual",
            rx_queue_size=256,
        )

        self._threads = [
            threading.Thread(
                target=self._gateway_worker, name="CAN-Gateway", daemon=True
            ),
            threading.Thread(
                target=physics_worker,
                args=(self.state, self.stop_event, self.analytics.reset_trip_stats, self.db_path),
                name="Truck-Physics",
                daemon=True,
            ),
            threading.Thread(
                target=bms_ecu_worker,
                args=(self.state, self.stop_event, self.channel),
                name="BMS-ECU",
                daemon=True,
            ),
            threading.Thread(
                target=motor_ecu_worker,
                args=(self.state, self.stop_event, self.channel),
                name="Motor-ECU",
                daemon=True,
            ),
        ]
        for t in self._threads:
            t.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._running = False


if __name__ == "__main__":
    service = TelemetryService()
    service.start()
    print("e-Truck Virtual CAN Telemetry Service running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down Telemetry Service...")
        service.stop()
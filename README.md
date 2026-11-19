# 🚛 e-Truck Powertrain ECU Simulator & Predictive Health Monitor

An end-to-end **IoT & ePowertrain telemetry application** built for heavy-duty commercial electric vehicles. It simulates **BMS** and **e-Motor/Inverter ECUs** communicating via raw 8-byte **SAE J1939 CAN frames** over a virtual CAN bus, decodes signals in real time, runs **physics-informed ML anomaly detection (`OneClassSVM`)** + **ECU Diagnostic Trouble Codes (DTCs)**, and exposes fleet-uptime KPIs on a live dashboard.

## Architecture

```mermaid
flowchart LR
    BMS[BMS ECU Thread<br/>0x18FEF100] -->|8-byte Hex Frame| VBUS((Virtual CAN Bus<br/>python-can))
    MOT[e-Motor ECU Thread<br/>0x18F00400] -->|8-byte Hex Frame| VBUS
    VBUS --> GW[CAN Gateway & DBC Decoder]
    GW --> AN[Analytics Engine<br/>DTC Rules + OneClassSVM]
    AN --> DB[(SQLite WAL DB)]
    DB --> UI[Streamlit + Plotly<br/>Fleet Dashboard]
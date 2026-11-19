"""Shared ePowertrain physics model for a 400 kWh heavy-duty commercial e-truck."""

import math

PACK_CAPACITY_KWH = 400.0
NOMINAL_EFFICIENCY_KWH_KM = 1.28

# Commercial Fleet TCO & Sustainability benchmarks
DIESEL_CO2_KG_PER_KM = 0.85
ELECTRIC_COST_PER_KWH_USD = 0.12
DIESEL_EQUIV_COST_PER_KM_USD = 0.44


def compute_expected_electrical(
    rpm: float,
    torque_nm: float,
    soc_pct: float = 80.0,
) -> tuple[float, float, float]:
    """
    Compute expected DC bus voltage (V), current (A), and electrical power (W)
    given motor speed, shaft torque, and battery State of Charge.
    """
    omega = rpm * (2.0 * math.pi / 60.0)
    p_mech_w = torque_nm * omega
    aux_load_w = 8500.0  # HVAC, air compressor, steering pump, DC-DC

    if p_mech_w >= 0:
        p_elec_w = p_mech_w / 0.91 + aux_load_w
    else:
        # Regenerative braking recovers ~84% of mechanical braking power
        p_elec_w = p_mech_w * 0.84 + aux_load_w

    # Open-circuit voltage curve (630V empty -> 720V full)
    ocv = 630.0 + (max(0.0, min(100.0, soc_pct)) / 100.0) * 90.0
    r_internal = 0.065  # 65 mOhm pack internal resistance

    curr = p_elec_w / ocv
    volt = ocv - curr * r_internal
    curr = p_elec_w / max(volt, 100.0)
    volt = ocv - curr * r_internal
    return volt, curr, p_elec_w
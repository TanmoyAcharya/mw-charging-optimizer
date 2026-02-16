import numpy as np

def degradation_throughput(
    power_kw: np.ndarray,
    capacity_kwh: float = 600.0,
    k_throughput: float = 2e-4,
    k_crate: float = 5e-4
):
    """
    Illustrative degradation:
    - proportional to energy throughput
    - additional penalty for high C-rate (power / capacity)

    Returns:
      deg_fraction: cumulative fractional capacity loss (0..)
      deg_cost: scalar "degradation cost" (arbitrary units)
    """
    dt_h = 0.25
    energy_kwh = np.sum(power_kw) * dt_h

    # average C-rate approx
    avg_power_kw = float(np.mean(power_kw[power_kw > 1e-9])) if np.any(power_kw > 1e-9) else 0.0
    c_rate = avg_power_kw / max(capacity_kwh, 1e-9)

    deg = k_throughput * energy_kwh + k_crate * (c_rate ** 2) * energy_kwh
    return float(deg), float(deg * 1000.0)  # cost scaled for readability

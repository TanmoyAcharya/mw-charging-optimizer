import numpy as np

def transformer_thermal(
    load_kw: np.ndarray,
    rated_kw: float,
    amb_c: float = 25.0,
    tau_steps: float = 12.0,     # thermal time constant in 15-min steps (~3 hours if 12)
    rise_c_at_rated: float = 55.0,
    exponent: float = 1.6
):
    """
    Simple first-order transformer temperature model (illustrative).
    - temperature rise increases ~ (loading)^exponent
    - temperature follows with a first-order lag (time constant tau_steps)

    Returns:
      top_oil_c: array
      hot_spot_c: array (top-oil + extra rise)
    """
    T = len(load_kw)
    top_oil = np.zeros(T, dtype=float)

    def steady_state_rise(p_kw: float) -> float:
        k = max(p_kw / rated_kw, 0.0)
        return rise_c_at_rated * (k ** exponent)

    top_oil[0] = amb_c + steady_state_rise(load_kw[0])

    alpha = 1.0 / max(tau_steps, 1e-9)
    for t in range(1, T):
        ss = amb_c + steady_state_rise(load_kw[t])
        top_oil[t] = top_oil[t-1] + alpha * (ss - top_oil[t-1])

    # Hot-spot approximation: add extra rise proportional to loading
    k = np.maximum(load_kw / rated_kw, 0.0)
    hot_spot = top_oil + 20.0 * (k ** 2)  # extra rise term (illustrative)

    return top_oil, hot_spot

import numpy as np

def ieee_aging_acceleration(hot_spot_c: np.ndarray) -> np.ndarray:
    """
    IEEE-style aging acceleration factor (relative to 110°C reference hot-spot).
    """
    theta = hot_spot_c + 273.0
    return np.exp(15000.0/383.0 - 15000.0/theta)

def equivalent_aging_hours(hot_spot_c: np.ndarray, dt_hours: float = 0.25) -> float:
    """
    Equivalent aging hours over the simulated horizon.
    If dt_hours=0.25 (15 min), sum FAA*dt gives 'equivalent hours at 110°C'.
    """
    faa = ieee_aging_acceleration(hot_spot_c)
    return float(np.sum(faa) * dt_hours)

def loss_of_life_percent(hot_spot_c: np.ndarray, normal_life_hours: float = 180000.0, dt_hours: float = 0.25) -> float:
    """
    Approx loss of life % over the horizon, assuming normal insulation life ~180,000 hours (illustrative).
    """
    eah = equivalent_aging_hours(hot_spot_c, dt_hours=dt_hours)
    return float(100.0 * eah / normal_life_hours)


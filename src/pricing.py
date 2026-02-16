import numpy as np

def make_tou_prices(T: int) -> np.ndarray:
    """
    Simple time-of-use pricing ($/kWh) for T 15-minute steps (T=96 -> 24 hours).
    Cheap at night, expensive late afternoon/evening.
    """
    prices = np.full(T, 0.12, dtype=float)     # off-peak
    prices[44:64] = 0.18                       # shoulder
    prices[64:84] = 0.28                       # peak
    return prices

def schedule_price_aware(
    trucks_df,
    T: int,
    pmax_kw: float,
    base_load_kw: np.ndarray,
    site_limit_kw: float,
    prices_per_kwh: np.ndarray,
    lookahead: int = 12
) -> np.ndarray:
    """
    Greedy price-aware scheduling:
    - respects site limit headroom
    - prioritizes trucks with early deadlines and expensive future windows
    """
    n = len(trucks_df)
    power = np.zeros((T, n), dtype=float)
    remaining = trucks_df["energy_kwh"].to_numpy().copy()
    dt_h = 0.25

    for t in range(T):
        headroom = max(site_limit_kw - float(base_load_kw[t]), 0.0)

        present = []
        for i in range(n):
            a = int(trucks_df.loc[i, "arrival"])
            d = int(trucks_df.loc[i, "departure"])
            if a <= t < d and remaining[i] > 1e-9:
                present.append(i)

        def priority(i: int):
            dep = int(trucks_df.loc[i, "departure"])
            urgency = 1.0 / max(dep - t, 1)
            future_prices = prices_per_kwh[t:min(T, t + lookahead)]
            price_pressure = float(np.mean(future_prices)) if len(future_prices) else float(prices_per_kwh[t])
            # higher urgency and higher future prices => charge now
            return -(2.0 * urgency + price_pressure)

        present.sort(key=priority)

        for i in present:
            if headroom <= 1e-9:
                break
            give_kw = min(pmax_kw, headroom, remaining[i] / dt_h)
            power[t, i] = give_kw
            remaining[i] -= give_kw * dt_h
            headroom -= give_kw

    return power

def total_energy_cost(charging_kw: np.ndarray, prices_per_kwh: np.ndarray) -> float:
    """
    Cost of charging energy only.
    charging_kw is the total charging power at each time step (kW).
    """
    dt_h = 0.25
    energy_kwh = charging_kw * dt_h
    return float(np.sum(energy_kwh * prices_per_kwh))

def demand_charge_cost(total_site_load_kw: np.ndarray, demand_rate_per_kw: float = 20.0) -> float:
    """
    Monthly demand charge approximation:
    billed on the maximum kW observed (15-min average assumed).
    demand_rate_per_kw: $/kW-month (typical commercial/industrial range varies)
    """
    return float(np.max(total_site_load_kw) * demand_rate_per_kw)


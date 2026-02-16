import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Truck:
    id: int
    arrival: int      # time step index
    departure: int    # time step index
    energy_kwh: float # required energy

def simulate_trucks(n_trucks=12, T=96, seed=7) -> pd.DataFrame:
    """
    T=96 means 96*15min = 24h if we use 15-min steps.
    """
    rng = np.random.default_rng(seed)

    arrivals = rng.integers(low=10, high=T-20, size=n_trucks)
    dwell = rng.integers(low=8, high=28, size=n_trucks)  # 2h..7h (if 15-min)
    departures = np.minimum(arrivals + dwell, T-1)

    # energy needs: 200..800 kWh (truck-scale)
    energy = rng.uniform(200, 800, size=n_trucks)

    df = pd.DataFrame({
        "truck_id": np.arange(n_trucks),
        "arrival": arrivals,
        "departure": departures,
        "energy_kwh": energy
    }).sort_values("arrival").reset_index(drop=True)

    return df





DT_HOURS = 0.25

def baseline_schedule(trucks: pd.DataFrame, T: int, pmax_kw: float) -> np.ndarray:
    n = len(trucks)
    power = np.zeros((T, n), dtype=float)
    remaining_kwh = trucks["energy_kwh"].to_numpy().copy()

    for t in range(T):
        for i in range(n):
            a = int(trucks.loc[i, "arrival"])
            d = int(trucks.loc[i, "departure"])
            if a <= t < d and remaining_kwh[i] > 1e-9:
                give_kw = min(pmax_kw, remaining_kwh[i] / DT_HOURS)
                power[t, i] = give_kw
                remaining_kwh[i] -= give_kw * DT_HOURS

    return power

def peak_limited_schedule(trucks: pd.DataFrame, T: int, pmax_kw: float,
                          base_load_kw: np.ndarray, site_limit_kw: float) -> np.ndarray:
    n = len(trucks)
    power = np.zeros((T, n), dtype=float)
    remaining_kwh = trucks["energy_kwh"].to_numpy().copy()

    for t in range(T):
        headroom = max(site_limit_kw - float(base_load_kw[t]), 0.0)

        present = []
        for i in range(n):
            a = int(trucks.loc[i, "arrival"])
            d = int(trucks.loc[i, "departure"])
            if a <= t < d and remaining_kwh[i] > 1e-9:
                present.append(i)

        present.sort(key=lambda i: int(trucks.loc[i, "departure"]))

        for i in present:
            if headroom <= 1e-9:
                break
            give_kw = min(pmax_kw, headroom, remaining_kwh[i] / DT_HOURS)
            power[t, i] = give_kw
            remaining_kwh[i] -= give_kw * DT_HOURS
            headroom -= give_kw

    return power

def site_load(power: np.ndarray, base_load_kw: np.ndarray) -> np.ndarray:
    return base_load_kw + power.sum(axis=1)

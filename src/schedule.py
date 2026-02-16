import numpy as np
import pandas as pd

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

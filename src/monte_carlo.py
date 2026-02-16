import numpy as np
import pandas as pd

from src.simulate import simulate_trucks
from src.schedule import baseline_schedule, peak_limited_schedule
from src.pricing import make_tou_prices, schedule_price_aware, total_energy_cost, demand_charge_cost
from src.thermal import transformer_thermal, loss_of_life_percent

def make_base_load(T=96, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    daily = 300 + 120*np.sin(2*np.pi*(t-20)/T)
    noise = rng.normal(0, 25, size=T)
    return np.clip(daily + noise, 150, None)

def run_one(seed: int, T: int, n_trucks: int, pmax_kw: float, site_limit_kw: float, demand_rate: float):
    trucks = simulate_trucks(n_trucks=n_trucks, T=T, seed=seed)
    base = make_base_load(T=T, seed=seed + 1000)
    prices = make_tou_prices(T)

    p_base = baseline_schedule(trucks, T=T, pmax_kw=pmax_kw)
    p_peak = peak_limited_schedule(trucks, T=T, pmax_kw=pmax_kw, base_load_kw=base, site_limit_kw=site_limit_kw)
    p_price = schedule_price_aware(trucks, T=T, pmax_kw=pmax_kw, base_load_kw=base, site_limit_kw=site_limit_kw, prices_per_kwh=prices)

    ch_base = p_base.sum(axis=1)
    ch_peak = p_peak.sum(axis=1)
    ch_price = p_price.sum(axis=1)

    site_base = base + ch_base
    site_peak = base + ch_peak
    site_price = base + ch_price

    # overload probability (fraction of timesteps above limit)
    overload_base = float(np.mean(site_base > site_limit_kw))
    overload_peak = float(np.mean(site_peak > site_limit_kw))
    overload_price = float(np.mean(site_price > site_limit_kw))

    # energy cost (charging only)
    e_base = total_energy_cost(ch_base, prices)
    e_peak = total_energy_cost(ch_peak, prices)
    e_price = total_energy_cost(ch_price, prices)

    # demand charges (site peak)
    d_base = demand_charge_cost(site_base, demand_rate)
    d_peak = demand_charge_cost(site_peak, demand_rate)
    d_price = demand_charge_cost(site_price, demand_rate)

    bill_base = e_base + d_base
    bill_peak = e_peak + d_peak
    bill_price = e_price + d_price

    # thermal + IEEE loss-of-life
    _, hot_base = transformer_thermal(site_base, rated_kw=site_limit_kw)
    _, hot_peak = transformer_thermal(site_peak, rated_kw=site_limit_kw)
    _, hot_price = transformer_thermal(site_price, rated_kw=site_limit_kw)

    lol_base = loss_of_life_percent(hot_base)
    lol_peak = loss_of_life_percent(hot_peak)
    lol_price = loss_of_life_percent(hot_price)

    return {
        "seed": seed,
        "peak_kw_baseline": float(np.max(site_base)),
        "peak_kw_peak_limited": float(np.max(site_peak)),
        "peak_kw_price_aware": float(np.max(site_price)),
        "overload_frac_baseline": overload_base,
        "overload_frac_peak_limited": overload_peak,
        "overload_frac_price_aware": overload_price,
        "bill_baseline": bill_base,
        "bill_peak_limited": bill_peak,
        "bill_price_aware": bill_price,
        "loss_of_life_%_baseline": lol_base,
        "loss_of_life_%_peak_limited": lol_peak,
        "loss_of_life_%_price_aware": lol_price,
    }

def run_monte_carlo(
    n_runs: int = 100,
    seed0: int = 0,
    T: int = 96,
    n_trucks: int = 20,
    pmax_kw: float = 1000.0,
    site_limit_kw: float = 2500.0,
    demand_rate: float = 20.0
) -> pd.DataFrame:
    rows = []
    for s in range(seed0, seed0 + n_runs):
        rows.append(run_one(s, T, n_trucks, pmax_kw, site_limit_kw, demand_rate))
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame):
    def stats(col):
        x = df[col].to_numpy()
        return float(np.mean(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))

    out = {}
    for col in ["bill_baseline", "bill_peak_limited", "bill_price_aware",
                "loss_of_life_%_baseline", "loss_of_life_%_peak_limited", "loss_of_life_%_price_aware",
                "peak_kw_baseline", "peak_kw_peak_limited", "peak_kw_price_aware"]:
        out[col] = stats(col)
    return out
return {
    "seed": seed,

    # peaks + overload
    "peak_kw_baseline": float(np.max(site_base)),
    "peak_kw_peak_limited": float(np.max(site_peak)),
    "peak_kw_price_aware": float(np.max(site_price)),
    "overload_frac_baseline": overload_base,
    "overload_frac_peak_limited": overload_peak,
    "overload_frac_price_aware": overload_price,

    # energy costs (charging only)
    "energy_cost_baseline": e_base,
    "energy_cost_peak_limited": e_peak,
    "energy_cost_price_aware": e_price,

    # demand charges (based on site peak)
    "demand_charge_baseline": d_base,
    "demand_charge_peak_limited": d_peak,
    "demand_charge_price_aware": d_price,

    # totals
    "bill_baseline": bill_base,
    "bill_peak_limited": bill_peak,
    "bill_price_aware": bill_price,

    # IEEE loss-of-life
    "loss_of_life_%_baseline": lol_base,
    "loss_of_life_%_peak_limited": lol_peak,
    "loss_of_life_%_price_aware": lol_price,
}

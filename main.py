import numpy as np
from src.simulate import simulate_trucks
from src.schedule import baseline_schedule, peak_limited_schedule, site_load
from src.plot import plot_loads, summarize
from src.thermal import transformer_thermal
from src.forecasting import train_forecaster
from src.pricing import make_tou_prices, schedule_price_aware, total_energy_cost,   demand_charge_cost
from src.degradation import degradation_throughput

def make_base_load(T=96, seed=1):
    rng = np.random.default_rng(seed)
    # base load ~ 200..500 kW with daily shape
    t = np.arange(T)
    daily = 300 + 120*np.sin(2*np.pi*(t-20)/T)
    noise = rng.normal(0, 25, size=T)
    return np.clip(daily + noise, 150, None)

if __name__ == "__main__":
    T = 96                 # 24h @ 15min
    PMAX_KW = 1000.0        # 1 MW charger power
    SITE_LIMIT_KW = 2500.0  # transformer/site limit example

    trucks = simulate_trucks(n_trucks=12, T=T, seed=7)
    base = make_base_load(T=T, seed=2)

    p_baseline = baseline_schedule(trucks, T=T, pmax_kw=PMAX_KW)
    load_baseline = site_load(p_baseline, base)

    p_opt = peak_limited_schedule(trucks, T=T, pmax_kw=PMAX_KW,
                                  base_load_kw=base, site_limit_kw=SITE_LIMIT_KW)
    load_opt = site_load(p_opt, base)

    summarize(load_baseline, "Baseline")
    summarize(load_opt, "Optimized")

    plot_loads(base, load_baseline, load_opt, title="Megawatt Charging: Peak-Limited Scheduling")
peak_baseline = max(load_baseline)
peak_opt = max(load_opt)

reduction = (peak_baseline - peak_opt) / peak_baseline * 100
prices = make_tou_prices(T)

# Price-aware schedule (same constraints, but tries to use cheap energy)
p_price = schedule_price_aware(
    trucks, T=T, pmax_kw=PMAX_KW, base_load_kw=base,
    site_limit_kw=SITE_LIMIT_KW, prices_per_kwh=prices
)

charging_baseline = p_baseline.sum(axis=1)
charging_opt = p_opt.sum(axis=1)
charging_price = p_price.sum(axis=1)

cost_baseline = total_energy_cost(charging_baseline, prices)
cost_opt = total_energy_cost(charging_opt, prices)
cost_price = total_energy_cost(charging_price, prices)

print(f"Charging energy cost ($): baseline={cost_baseline:.2f}, peak-limited={cost_opt:.2f}, price-aware={cost_price:.2f}")

# Transformer thermal stress
load_total_baseline = base + charging_baseline
load_total_opt = base + charging_opt
load_total_price = base + charging_price

top_b, hot_b = transformer_thermal(load_total_baseline, rated_kw=SITE_LIMIT_KW, amb_c=25)
top_o, hot_o = transformer_thermal(load_total_opt, rated_kw=SITE_LIMIT_KW, amb_c=25)
top_p, hot_p = transformer_thermal(load_total_price, rated_kw=SITE_LIMIT_KW, amb_c=25)

print(f"Hot-spot max (C): baseline={hot_b.max():.1f}, peak-limited={hot_o.max():.1f}, price-aware={hot_p.max():.1f}")

# ML forecasting
model, mae, lags = train_forecaster(base, lags=8)
print(f"Base-load forecaster MAE (kW): {mae:.1f}")

# Battery degradation proxy (sum per truck)
deg_base = 0.0
deg_opt = 0.0
deg_price = 0.0
for i in range(p_baseline.shape[1]):
    d, _ = degradation_throughput(p_baseline[:, i])
    deg_base += d
    d, _ = degradation_throughput(p_opt[:, i])
    deg_opt += d
    d, _ = degradation_throughput(p_price[:, i])
    deg_price += d

print(f"Total degradation proxy (fractional): baseline={deg_base:.4f}, peak-limited={deg_opt:.4f}, price-aware={deg_price:.4f}")




print(f"Peak reduction: {reduction:.2f}%")



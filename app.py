import numpy as np
import streamlit as st

from src.simulate import simulate_trucks
from src.schedule import baseline_schedule, peak_limited_schedule, site_load
from src.pricing import make_tou_prices, schedule_price_aware, total_energy_cost ,    demand_charge_cost
from src.thermal import transformer_thermal
from src.degradation import degradation_throughput

st.set_page_config(page_title="Megawatt Charging Optimizer", layout="wide")

st.title("AI-Driven Grid-Constrained Megawatt Charging Optimization Platform")

with st.sidebar:
    st.header("Parameters")
    T = 96
    n_trucks = st.slider("Number of trucks", 5, 40, 12)
    PMAX_KW = st.slider("Charger max power (kW)", 250, 2000, 1000, step=50)
    SITE_LIMIT_KW = st.slider("Site limit / transformer rating (kW)", 800, 5000, 2500, step=100)
    seed = st.slider("Random seed", 0, 999, 7)

def make_base_load(T=96, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    daily = 300 + 120*np.sin(2*np.pi*(t-20)/T)
    noise = rng.normal(0, 25, size=T)
    return np.clip(daily + noise, 150, None)

trucks = simulate_trucks(n_trucks=n_trucks, T=T, seed=seed)
base = make_base_load(T=T, seed=2)
prices = make_tou_prices(T)

p_baseline = baseline_schedule(trucks, T=T, pmax_kw=float(PMAX_KW))
p_peak = peak_limited_schedule(trucks, T=T, pmax_kw=float(PMAX_KW), base_load_kw=base, site_limit_kw=float(SITE_LIMIT_KW))
p_price = schedule_price_aware(trucks, T=T, pmax_kw=float(PMAX_KW), base_load_kw=base, site_limit_kw=float(SITE_LIMIT_KW), prices_per_kwh=prices)

charging_baseline = p_baseline.sum(axis=1)
charging_peak = p_peak.sum(axis=1)
charging_price = p_price.sum(axis=1)

load_baseline = base + charging_baseline
load_peak = base + charging_peak
load_price = base + charging_price

cost_baseline = total_energy_cost(charging_baseline, prices)
cost_peak = total_energy_cost(charging_peak, prices)
cost_price = total_energy_cost(charging_price, prices)

top_b, hot_b = transformer_thermal(load_baseline, rated_kw=float(SITE_LIMIT_KW))
top_k, hot_k = transformer_thermal(load_peak, rated_kw=float(SITE_LIMIT_KW))
top_p, hot_p = transformer_thermal(load_price, rated_kw=float(SITE_LIMIT_KW))

def total_deg(power):
    s = 0.0
    for i in range(power.shape[1]):
        d, _ = degradation_throughput(power[:, i])
        s += d
    return s

deg_baseline = total_deg(p_baseline)
deg_peak = total_deg(p_peak)
deg_price = total_deg(p_price)

c1, c2, c3 = st.columns(3)
c1.metric("Charging Cost ($)", f"{cost_price:.2f}", f"{(cost_price-cost_baseline):+.2f} vs baseline")
c2.metric("Max Hot-Spot (°C)", f"{hot_p.max():.1f}", f"{(hot_p.max()-hot_b.max()):+.1f} vs baseline")
c3.metric("Degradation Proxy", f"{deg_price:.4f}", f"{(deg_price-deg_baseline):+.4f} vs baseline")

st.subheader("Load Curves")
st.line_chart({
    "Base load": base,
    "Baseline total": load_baseline,
    "Peak-limited total": load_peak,
    "Price-aware total": load_price,
    "Site limit": np.full(T, float(SITE_LIMIT_KW))
})

st.subheader("Simulated Trucks")
st.dataframe(trucks, use_container_width=True)
site_baseline = base + charging_baseline
site_peak = base + charging_peak
site_price = base + charging_price

demand_rate = 20.0  # you can make this a slider later
dc_baseline = demand_charge_cost(site_baseline, demand_rate)
dc_peak = demand_charge_cost(site_peak, demand_rate)
dc_price = demand_charge_cost(site_price, demand_rate)
bill_baseline = cost_baseline + dc_baseline
bill_peak = cost_peak + dc_peak
bill_price = cost_price + dc_price
from src.thermal import loss_of_life_percent, equivalent_aging_hours

lol_b = loss_of_life_percent(hot_b)
lol_k = loss_of_life_percent(hot_k)
lol_p = loss_of_life_percent(hot_p)

print(f"Transformer loss of life (% over day): baseline={lol_b:.4f}, peak-limited={lol_k:.4f}, price-aware={lol_p:.4f}")
demand_rate = st.slider("Demand charge rate ($/kW-month)", 0.0, 60.0, 20.0, 1.0)
from src.pricing import demand_charge_cost  # make sure it exists

dc_baseline = demand_charge_cost(load_baseline, demand_rate)
dc_peak = demand_charge_cost(load_peak, demand_rate)
dc_price = demand_charge_cost(load_price, demand_rate)

bill_baseline = cost_baseline + dc_baseline
bill_peak = cost_peak + dc_peak
bill_price = cost_price + dc_price
st.subheader("Cost Breakdown ($)")

c1, c2, c3 = st.columns(3)
c1.metric("Energy cost (baseline)", f"{cost_baseline:.2f}")
c2.metric("Demand charge (baseline)", f"{dc_baseline:.2f}")
c3.metric("Total bill (baseline)", f"{bill_baseline:.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("Energy cost (peak-limited)", f"{cost_peak:.2f}", f"{(cost_peak-cost_baseline):+.2f}")
c2.metric("Demand charge (peak-limited)", f"{dc_peak:.2f}", f"{(dc_peak-dc_baseline):+.2f}")
c3.metric("Total bill (peak-limited)", f"{bill_peak:.2f}", f"{(bill_peak-bill_baseline):+.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("Energy cost (price-aware)", f"{cost_price:.2f}", f"{(cost_price-cost_baseline):+.2f}")
c2.metric("Demand charge (price-aware)", f"{dc_price:.2f}", f"{(dc_price-dc_baseline):+.2f}")
c3.metric("Total bill (price-aware)", f"{bill_price:.2f}", f"{(bill_price-bill_baseline):+.2f}")


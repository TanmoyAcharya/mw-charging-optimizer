# Grid-Aware Megawatt Charging Optimizer ⚡🚛

A Streamlit app that simulates megawatt EV/truck charging and optimizes charging schedules under grid and transformer constraints.

## Why this matters
Megawatt charging can create extreme peaks that:
- trigger high demand charges ($/kW-month)
- stress transformers (hot-spot temperature / loss of life)
- reduce grid reliability

This project models those tradeoffs and provides an interactive dashboard.

## Features
- ✅ Peak-limited scheduling (site/transformer limit enforced)
- ✅ Price-aware scheduling (TOU pricing)
- ✅ Utility billing model: energy cost + demand charges
- ✅ Transformer thermal model + IEEE-style loss-of-life metric
- ✅ Battery degradation proxy (throughput + C-rate)
- ✅ Monte Carlo evaluation across random scenarios
- ✅ Streamlit dashboard UI

## Quickstart (Local)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m streamlit run app.py
# mw-charging-optimizer
mw-charging-optimizer

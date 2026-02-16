from src.monte_carlo import run_monte_carlo, summarize


df = run_monte_carlo(n_runs=200, n_trucks=25, pmax_kw=1500, site_limit_kw=2500, demand_rate=20)
print(df.head())

summary = summarize(df)

for k, (mean, p5, p95) in summary.items():
    if "loss_of_life" in k:
        print(f"{k}: mean={mean:.6f}, p5={p5:.6f}, p95={p95:.6f}")
    else:
        print(f"{k}: mean={mean:.3f}, p5={p5:.3f}, p95={p95:.3f}")

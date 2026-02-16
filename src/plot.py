import numpy as np
import matplotlib.pyplot as plt

def plot_loads(base, total_baseline, total_optimized, title="Site Load (kW)"):
    plt.figure()
    plt.plot(base, label="Base load")
    plt.plot(total_baseline, label="Baseline (uncontrolled)")
    plt.plot(total_optimized, label="Optimized (peak-limited)")
    plt.title(title)
    plt.xlabel("Time step (15 min)")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.show()

def summarize(load_kw: np.ndarray, name: str):
    peak = float(np.max(load_kw))
    avg = float(np.mean(load_kw))
    print(f"{name}: peak={peak:.1f} kW, avg={avg:.1f} kW")
plt.axhline(y=2500, linestyle='--', label="Site limit")

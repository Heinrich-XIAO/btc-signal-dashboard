"""Generate graphs for Round 1 results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from strategies import STRATEGIES, evaluate_signal

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)

with open("optimization_round_1.json", "r") as f:
    results = json.load(f)

# Graph 1: Bar chart of best accuracy per strategy
fig, ax = plt.subplots(figsize=(12, 7))

strategies = []
accuracies = []
trade_counts = []
colors = []

for name, res in results.items():
    if res:
        best = res[0]
        strategies.append(name.replace("_", " ").title())
        accuracies.append(best["accuracy"] * 100)
        trade_counts.append(best["n_trades"])
        colors.append("#2ecc71" if best["accuracy"] > 0.6 else "#3498db" if best["accuracy"] > 0.55 else "#e74c3c")

bars = ax.bar(strategies, accuracies, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, label="90% target")

# Add trade count labels on bars
for bar, acc, n in zip(bars, accuracies, trade_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f"{acc:.1f}%\n({n} trades)", ha="center", va="bottom", fontsize=9)

ax.set_title("Round 1: Best Accuracy per Strategy", fontsize=16, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 100)
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/round1_best_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 2: Accuracy vs Number of Trades scatter for all top results
fig, ax = plt.subplots(figsize=(12, 8))

for name, res in results.items():
    if res:
        accs = [r["accuracy"] * 100 for r in res]
        ns = [r["n_trades"] for r in res]
        ax.scatter(ns, accs, alpha=0.6, s=60, label=name.replace("_", " ").title(), edgecolors="black", linewidth=0.3)

ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7)
ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, linewidth=2)
ax.set_xscale("log")
ax.set_title("Round 1: Accuracy vs Number of Trades (Top 5 Parameter Sets per Strategy)", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Trades (log scale)")
ax.set_ylabel("Accuracy (%)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("graphs/round1_accuracy_vs_trades.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 3: Walk-forward equity curves for top 3 strategies
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

top3 = [(name, res[0]) for name, res in results.items() if res]
top3.sort(key=lambda x: x[1]["accuracy"], reverse=True)
top3 = top3[:3]

for idx, (name, best) in enumerate(top3):
    func = STRATEGIES[name]
    signal = func(df, **best["params"])
    df_test = df.copy()
    df_test["signal"] = signal
    df_test = df_test[df_test["signal"] != 0].copy()
    
    if len(df_test) == 0:
        continue
    
    # Calculate P&L (simplified: +1 for correct, -1 for wrong)
    df_test["pnl"] = np.where(
        ((df_test["signal"] == 1) & (df_test["target"] == 1)) |
        ((df_test["signal"] == -1) & (df_test["target"] == 0)),
        1, -1
    )
    df_test["equity"] = df_test["pnl"].cumsum()
    
    axes[idx].plot(df_test.index, df_test["equity"], color="#2ecc71" if best["accuracy"] > 0.6 else "#3498db", linewidth=0.8)
    axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[idx].set_title(f"{name.replace('_', ' ').title()}: {best['accuracy']*100:.1f}% ({best['n_trades']} trades)", 
                       fontsize=12, fontweight="bold")
    axes[idx].set_ylabel("Cumulative Score")

plt.tight_layout()
plt.savefig("graphs/round1_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()

print("Round 1 graphs saved!")

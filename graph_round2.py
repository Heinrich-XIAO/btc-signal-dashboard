"""Generate graphs for Round 2 results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from strategies import STRATEGIES, evaluate_signal
from extreme_strategies import EXTREME_STRATEGIES

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)

with open("optimization_round_2.json", "r") as f:
    results = json.load(f)

# Graph 1: Bar chart of best accuracy per strategy (Round 2)
fig, ax = plt.subplots(figsize=(12, 7))

strategies = []
accuracies = []
trade_counts = []
colors = []

for name, res in results.items():
    if name == "ensembles":
        continue
    if res:
        best = res[0]
        strategies.append(name.replace("_", " ").title())
        accuracies.append(best["accuracy"] * 100)
        trade_counts.append(best["n_trades"])
        if best["accuracy"] >= 0.9:
            colors.append("#2ecc71")  # green
        elif best["accuracy"] >= 0.7:
            colors.append("#3498db")  # blue
        else:
            colors.append("#e74c3c")  # red

bars = ax.bar(strategies, accuracies, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, linewidth=2, label="90% target")

for bar, acc, n in zip(bars, accuracies, trade_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f"{acc:.1f}%\n({n} trades)", ha="center", va="bottom", fontsize=9)

ax.set_title("Round 2: Best Accuracy per Strategy (Focused Optimization)", fontsize=16, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 105)
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/round2_best_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 2: Top 10 parameter sets overall
all_top = []
for name, res in results.items():
    if name == "ensembles":
        continue
    if res:
        for r in res:
            all_top.append({
                "strategy": name,
                **r
            })

all_top.sort(key=lambda x: x["accuracy"], reverse=True)
top10 = all_top[:10]

fig, ax = plt.subplots(figsize=(14, 8))
names = [f"{t['strategy'].replace('_', ' ').title()}\n({t['n_trades']} trades)" for t in top10]
accs = [t["accuracy"] * 100 for t in top10]
colors_top = ["#2ecc71" if a >= 90 else "#3498db" if a >= 70 else "#e74c3c" for a in accs]

bars = ax.barh(range(len(top10)), accs, color=colors_top, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(names)
ax.axvline(x=90, color="red", linestyle="--", linewidth=2, label="90% target")
ax.axvline(x=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
ax.invert_yaxis()

for i, (bar, acc) in enumerate(zip(bars, accs)):
    ax.text(acc + 1, i, f"{acc:.1f}%", va="center", fontsize=10)

ax.set_title("Round 2: Top 10 Parameter Sets Across All Strategies", fontsize=16, fontweight="bold")
ax.set_xlabel("Accuracy (%)")
ax.legend()
plt.tight_layout()
plt.savefig("graphs/round2_top10_parameters.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 3: Equity curves for strategies >= 80% accuracy
high_acc = [(name, res[0]) for name, res in results.items() if res and res[0]["accuracy"] >= 0.8 and name != "ensembles"]

if high_acc:
    n_plots = len(high_acc)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (name, best) in enumerate(high_acc):
        func = STRATEGIES.get(name) or EXTREME_STRATEGIES.get(name)
        signal = func(df, **best["params"])
        df_test = df.copy()
        df_test["signal"] = signal
        df_test = df_test[df_test["signal"] != 0].copy()
        
        if len(df_test) == 0:
            continue
        
        df_test["pnl"] = np.where(
            ((df_test["signal"] == 1) & (df_test["target"] == 1)) |
            ((df_test["signal"] == -1) & (df_test["target"] == 0)),
            1, -1
        )
        df_test["equity"] = df_test["pnl"].cumsum()
        
        color = "#2ecc71" if best["accuracy"] >= 0.9 else "#3498db"
        axes[idx].plot(df_test.index, df_test["equity"], color=color, linewidth=1.5)
        axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[idx].fill_between(df_test.index, 0, df_test["equity"], 
                               where=df_test["equity"] >= 0, alpha=0.3, color="green")
        axes[idx].fill_between(df_test.index, 0, df_test["equity"], 
                               where=df_test["equity"] < 0, alpha=0.3, color="red")
        axes[idx].set_title(f"{name.replace('_', ' ').title()}: {best['accuracy']*100:.1f}% ({best['n_trades']} trades)", 
                           fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Cumulative Score")
    
    plt.tight_layout()
    plt.savefig("graphs/round2_high_accuracy_equity.png", dpi=150, bbox_inches="tight")
    plt.close()

# Graph 4: Accuracy vs Trade Count scatter
fig, ax = plt.subplots(figsize=(12, 8))

for name, res in results.items():
    if name == "ensembles" or not res:
        continue
    accs = [r["accuracy"] * 100 for r in res]
    ns = [r["n_trades"] for r in res]
    ax.scatter(ns, accs, alpha=0.7, s=80, label=name.replace("_", " ").title(), 
               edgecolors="black", linewidth=0.5)

ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7)
ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, linewidth=2)
ax.set_xscale("log")
ax.set_title("Round 2: Accuracy vs Number of Trades", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Trades (log scale)")
ax.set_ylabel("Accuracy (%)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("graphs/round2_accuracy_vs_trades.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 5: Direction distribution
fig, ax = plt.subplots(figsize=(10, 6))

strategies_dir = []
up_counts = []
down_counts = []

for name, res in results.items():
    if name == "ensembles" or not res:
        continue
    best = res[0]
    strategies_dir.append(name.replace("_", " ").title())
    up_counts.append(int(best["n_up"]))
    down_counts.append(int(best["n_down"]))

x = np.arange(len(strategies_dir))
width = 0.35
ax.bar(x - width/2, up_counts, width, label="Up Predictions", color="#2ecc71", alpha=0.8)
ax.bar(x + width/2, down_counts, width, label="Down Predictions", color="#e74c3c", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(strategies_dir, rotation=45, ha="right")
ax.set_title("Round 2: Prediction Direction Distribution (Best Params)", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Predictions")
ax.legend()
plt.tight_layout()
plt.savefig("graphs/round2_direction_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("Round 2 graphs saved!")

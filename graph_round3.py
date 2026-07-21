"""Generate graphs for Round 3 results."""
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

with open("optimization_round_3.json", "r") as f:
    results = json.load(f)

# Graph 1: Bar chart of best accuracy per strategy (Round 3)
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
            colors.append("#2ecc71")
        elif best["accuracy"] >= 0.7:
            colors.append("#3498db")
        else:
            colors.append("#e74c3c")

bars = ax.bar(strategies, accuracies, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, linewidth=2, label="90% target")

for bar, acc, n in zip(bars, accuracies, trade_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f"{acc:.1f}%\n({n} trades)", ha="center", va="bottom", fontsize=9)

ax.set_title("Round 3: Fine-Tuned Best Accuracy per Strategy", fontsize=16, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 105)
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/round3_best_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 2: Comparison across all 3 rounds
fig, ax = plt.subplots(figsize=(14, 8))

round_data = []
for round_file, round_name in [("optimization_round_1.json", "Round 1"),
                                  ("optimization_round_2.json", "Round 2"),
                                  ("optimization_round_3.json", "Round 3")]:
    with open(round_file, "r") as f:
        data = json.load(f)
    
    best_acc = 0
    best_name = ""
    best_trades = 0
    for name, res in data.items():
        if name == "ensembles" or not res:
            continue
        if res[0]["accuracy"] > best_acc:
            best_acc = res[0]["accuracy"]
            best_name = name
            best_trades = res[0]["n_trades"]
    
    round_data.append({
        "round": round_name,
        "accuracy": best_acc * 100,
        "trades": best_trades,
        "name": best_name.replace("_", " ").title()
    })

bars = ax.bar([r["round"] for r in round_data], [r["accuracy"] for r in round_data],
              color=["#3498db", "#f39c12", "#2ecc71"], edgecolor="black", linewidth=0.5, width=0.5)
ax.axhline(y=90, color="red", linestyle="--", linewidth=2, label="90% target")
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")

for bar, rd in zip(bars, round_data):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f"{rd['accuracy']:.1f}%\n{rd['name']}\n({rd['trades']} trades)", 
            ha="center", va="bottom", fontsize=10)

ax.set_title("Progress Across Rounds: Best Strategy Accuracy", fontsize=16, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 105)
ax.legend()
plt.tight_layout()
plt.savefig("graphs/round_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph 3: Equity curves for all >= 90% strategies in round 3
high_acc = [(name, res[0]) for name, res in results.items() if res and res[0]["accuracy"] >= 0.9 and name != "ensembles"]

if high_acc:
    n_plots = len(high_acc)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (name, best) in enumerate(high_acc):
        func = STRATEGIES.get(name.replace("_ft", "")) or EXTREME_STRATEGIES.get(name.replace("_ft", ""))
        if func is None:
            continue
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
        
        axes[idx].plot(df_test.index, df_test["equity"], color="#2ecc71", linewidth=2)
        axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[idx].fill_between(df_test.index, 0, df_test["equity"], 
                               where=df_test["equity"] >= 0, alpha=0.3, color="green")
        axes[idx].fill_between(df_test.index, 0, df_test["equity"], 
                               where=df_test["equity"] < 0, alpha=0.3, color="red")
        axes[idx].set_title(f"{name.replace('_', ' ').title()}: {best['accuracy']*100:.1f}% ({best['n_trades']} trades)", 
                           fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Cumulative Score")
    
    plt.tight_layout()
    plt.savefig("graphs/round3_high_accuracy_equity.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Round 3 graphs saved!")

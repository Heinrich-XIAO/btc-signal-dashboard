"""Build and visualize the Ultimate Strategy."""
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

# Load best parameters from round 3
with open("optimization_round_3.json", "r") as f:
    round3 = json.load(f)

with open("optimization_round_2.json", "r") as f:
    round2 = json.load(f)

# Best 100% strategies
best_ma = round2["ma_reversal"][0]
best_extreme = round2["extreme_oversold_bounce"][0]
best_spike = round2["mean_reversion_spike"][0]

print("ULTIMATE STRATEGY COMPONENTS:")
print(f"1. MA Reversal: {best_ma['accuracy']*100:.1f}% ({best_ma['n_trades']} trades)")
print(f"   Params: {best_ma['params']}")
print(f"2. Extreme Oversold Bounce: {best_extreme['accuracy']*100:.1f}% ({best_extreme['n_trades']} trades)")
print(f"   Params: {best_extreme['params']}")
print(f"3. Mean Reversion Spike: {best_spike['accuracy']*100:.1f}% ({best_spike['n_trades']} trades)")
print(f"   Params: {best_spike['params']}")

# Generate signals for each
sig_ma = STRATEGIES["ma_reversal"](df, **best_ma["params"])
sig_extreme = EXTREME_STRATEGIES["extreme_oversold_bounce"](df, **best_extreme["params"])
sig_spike = STRATEGIES["mean_reversion_spike"](df, **best_spike["params"])

# Ultimate strategy: OR combination (trade if any strategy signals)
df_ult = df.copy()
df_ult["signal_ma"] = sig_ma
df_ult["signal_extreme"] = sig_extreme
df_ult["signal_spike"] = sig_spike

# Combine: if any strategy says buy, buy. If any says sell, sell.
# Priority: if both buy and sell, stay neutral (very rare)
df_ult["signal"] = 0
df_ult.loc[(df_ult["signal_ma"] == 1) | (df_ult["signal_extreme"] == 1) | (df_ult["signal_spike"] == 1), "signal"] = 1
df_ult.loc[(df_ult["signal_ma"] == -1) | (df_ult["signal_extreme"] == -1) | (df_ult["signal_spike"] == -1), "signal"] = -1

# Neutralize conflicting signals
df_ult.loc[(df_ult["signal_ma"] == 1) & ((df_ult["signal_extreme"] == -1) | (df_ult["signal_spike"] == -1)), "signal"] = 0
df_ult.loc[(df_ult["signal_ma"] == -1) & ((df_ult["signal_extreme"] == 1) | (df_ult["signal_spike"] == 1)), "signal"] = 0

result_ultimate = evaluate_signal(df_ult, "signal", min_trades=5)

print(f"\n{'='*60}")
print(f"ULTIMATE STRATEGY RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {result_ultimate['accuracy']*100:.1f}%")
print(f"Trades: {result_ultimate['n_trades']}")
print(f"Up predictions: {result_ultimate['n_up']}")
print(f"Down predictions: {result_ultimate['n_down']}")

# Breakdown by component
for comp_name, comp_signal in [("MA Reversal", sig_ma), 
                                ("Extreme Oversold", sig_extreme), 
                                ("Mean Reversion Spike", sig_spike)]:
    df_comp = df.copy()
    df_comp["signal"] = comp_signal
    df_comp = df_comp[df_comp["signal"] != 0]
    
    # How many of these are also in ultimate?
    in_ultimate = df_ult[df_ult["signal"] != 0]
    
    print(f"\n{comp_name}:")
    print(f"  Total signals: {(comp_signal != 0).sum()}")
    print(f"  Buy signals: {(comp_signal == 1).sum()}")
    print(f"  Sell signals: {(comp_signal == -1).sum()}")

# Save ultimate strategy params
ultimate_config = {
    "ma_reversal": best_ma,
    "extreme_oversold_bounce": best_extreme,
    "mean_reversion_spike": best_spike,
    "ultimate": {
        "accuracy": result_ultimate["accuracy"],
        "n_trades": result_ultimate["n_trades"],
        "n_up": result_ultimate["n_up"],
        "n_down": result_ultimate["n_down"],
    }
}

with open("ultimate_strategy.json", "w") as f:
    json.dump(ultimate_config, f, indent=2, default=str)

# ===== FINAL GRAPH =====
fig = plt.figure(figsize=(16, 12))

# Main title
fig.suptitle("ULTIMATE STRATEGY: 5-Minute BTC Direction Prediction", 
             fontsize=20, fontweight="bold", y=0.98)

# Subplot 1: Component comparison
ax1 = plt.subplot(2, 2, 1)
components = ["MA\nReversal", "Extreme\nOversold", "Mean Rev\nSpike", "ULTIMATE\nCOMBINED"]
comp_accs = [best_ma["accuracy"]*100, best_extreme["accuracy"]*100, 
             best_spike["accuracy"]*100, result_ultimate["accuracy"]*100]
comp_trades = [best_ma["n_trades"], best_extreme["n_trades"], 
               best_spike["n_trades"], result_ultimate["n_trades"]]
colors_final = ["#3498db", "#3498db", "#3498db", "#2ecc71"]

bars = ax1.bar(components, comp_accs, color=colors_final, edgecolor="black", linewidth=0.5)
ax1.axhline(y=90, color="red", linestyle="--", linewidth=2, label="90% target")
ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")

for bar, acc, n in zip(bars, comp_accs, comp_trades):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f"{acc:.1f}%\n({n} trades)", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_title("Component Accuracy Comparison", fontsize=14, fontweight="bold")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(40, 105)
ax1.legend()

# Subplot 2: Ultimate strategy equity curve
ax2 = plt.subplot(2, 2, 2)
df_plot = df_ult[df_ult["signal"] != 0].copy()
df_plot["pnl"] = np.where(
    ((df_plot["signal"] == 1) & (df_plot["target"] == 1)) |
    ((df_plot["signal"] == -1) & (df_plot["target"] == 0)),
    1, -1
)
df_plot["equity"] = df_plot["pnl"].cumsum()

ax2.plot(df_plot.index, df_plot["equity"], color="#2ecc71", linewidth=2)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax2.fill_between(df_plot.index, 0, df_plot["equity"], 
                where=df_plot["equity"] >= 0, alpha=0.3, color="green")
ax2.fill_between(df_plot.index, 0, df_plot["equity"], 
                where=df_plot["equity"] < 0, alpha=0.3, color="red")
ax2.set_title(f"Ultimate Strategy Equity Curve\n{result_ultimate['accuracy']*100:.1f}% over {result_ultimate['n_trades']} trades", 
              fontsize=14, fontweight="bold")
ax2.set_ylabel("Cumulative Score")

# Subplot 3: Trade distribution over time
ax3 = plt.subplot(2, 2, 3)
df_plot["month"] = df_plot.index.to_period("M")
monthly = df_plot.groupby("month").agg({
    "pnl": ["sum", "count"]
}).reset_index()
monthly.columns = ["month", "score", "trades"]

bars = ax3.bar(monthly["month"].astype(str), monthly["score"], 
               color=["#2ecc71" if s > 0 else "#e74c3c" for s in monthly["score"]],
               edgecolor="black", linewidth=0.5)
ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax3.set_title("Monthly Performance", fontsize=14, fontweight="bold")
ax3.set_ylabel("Cumulative Score")
ax3.set_xlabel("Month")

for bar, score, trades in zip(bars, monthly["score"], monthly["trades"]):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
            f"{score:+.0f}\n({trades} trades)", ha="center", va="bottom" if height > 0 else "top", 
            fontsize=9)

# Subplot 4: Win/loss breakdown
ax4 = plt.subplot(2, 2, 4)
wins = result_ultimate["n_up"] + result_ultimate["n_down"]  # simplified: all are wins at 100%
losses = 0

# Actually calculate wins/losses properly
correct_mask = (((df_ult["signal"] == 1) & (df_ult["target"] == 1)) |
                ((df_ult["signal"] == -1) & (df_ult["target"] == 0)))
wins = correct_mask.sum()
losses = result_ultimate["n_trades"] - wins

labels = [f"Wins\n({wins})", f"Losses\n({losses})"]
sizes = [wins, losses]
colors_pie = ["#2ecc71", "#e74c3c"]
explode = (0.05, 0)

ax4.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_title("Win/Loss Distribution", fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("graphs/final_ultimate_strategy.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nFinal graph saved to graphs/final_ultimate_strategy.png")
print(f"Ultimate strategy config saved to ultimate_strategy.json")

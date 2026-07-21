"""Create final comprehensive summary graph."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

# Load results
with open("final_10_strategies.json", "r") as f:
    final_10 = json.load(f)

with open("optimization_round_2.json", "r") as f:
    rare = json.load(f)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("BTC 5M Polymarket Trading Analysis: March-June 2026\nFinal Results Summary", 
             fontsize=22, fontweight="bold", y=0.98)

# === Panel 1: 10 Strategies with 25%+ Coverage ===
ax1 = plt.subplot(2, 3, 1)
names = [r["name"].replace("_", " ").title() for r in final_10]
accs = [r["accuracy"] * 100 for r in final_10]
trades = [r["n_trades"] for r in final_10]
coverage = [t / 35088 * 100 for t in trades]

bars = ax1.barh(range(len(names)), accs, color=["#3498db" if a > 50 else "#e74c3c" for a in accs], 
                edgecolor="black", linewidth=0.5)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=10)
ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
ax1.axvline(x=90, color="red", linestyle="--", alpha=0.7, linewidth=2, label="90% target")
ax1.invert_yaxis()

for i, (bar, acc, cov) in enumerate(zip(bars, accs, coverage)):
    ax1.text(acc + 0.5, i, f"{acc:.1f}% ({cov:.0f}% cov)", va="center", fontsize=9)

ax1.set_title("10 Strategies: 25%+ Coverage\n(Best: 52.2%)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Accuracy (%)")
ax1.legend(loc="lower right")
ax1.set_xlim(40, 95)

# === Panel 2: Rare High-Accuracy Strategies ===
ax2 = plt.subplot(2, 3, 2)

rare_results = []
for name, res in rare.items():
    if res and name != "ensembles":
        rare_results.append({
            "name": name.replace("_", " ").title(),
            "accuracy": res[0]["accuracy"] * 100,
            "trades": res[0]["n_trades"],
            "coverage": res[0]["n_trades"] / 35088 * 100
        })

rare_results.sort(key=lambda x: x["accuracy"], reverse=True)
rare_results = rare_results[:5]

names_rare = [r["name"] for r in rare_results]
accs_rare = [r["accuracy"] for r in rare_results]

bars = ax2.barh(range(len(names_rare)), accs_rare, color="#2ecc71", 
                edgecolor="black", linewidth=0.5)
ax2.set_yticks(range(len(names_rare)))
ax2.set_yticklabels(names_rare, fontsize=10)
ax2.axvline(x=90, color="red", linestyle="--", linewidth=2)
ax2.invert_yaxis()

for i, (bar, r) in enumerate(zip(bars, rare_results)):
    ax2.text(r["accuracy"] + 1, i, f"{r['accuracy']:.0f}%\n({r['trades']} trades, {r['coverage']:.2f}% cov)", 
             va="center", fontsize=9)

ax2.set_title("Rare High-Accuracy Strategies\n(90-100% but <0.05% coverage)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Accuracy (%)")
ax2.set_xlim(70, 105)

# === Panel 3: Coverage vs Accuracy Scatter ===
ax3 = plt.subplot(2, 3, 3)

# All round 2 results
all_points = []
for name, res in rare.items():
    if res and name != "ensembles":
        for r in res:
            all_points.append({
                "coverage": r["n_trades"] / 35088 * 100,
                "accuracy": r["accuracy"] * 100,
                "strategy": name
            })

for p in all_points:
    ax3.scatter(p["coverage"], p["accuracy"], alpha=0.6, s=50, edgecolors="black", linewidth=0.3)

# Add the 25% strategies
for r in final_10:
    cov = r["n_trades"] / 35088 * 100
    acc = r["accuracy"] * 100
    ax3.scatter(cov, acc, alpha=0.9, s=200, color="#e74c3c", edgecolors="black", linewidth=1, zorder=5)

ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.7)
ax3.axhline(y=90, color="red", linestyle="--", linewidth=2)
ax3.axvline(x=25, color="blue", linestyle="--", linewidth=2, label="25% coverage threshold")
ax3.set_xscale("log")
ax3.set_title("Accuracy vs Coverage (Log Scale)\nRed dots = 25% coverage strategies", fontsize=13, fontweight="bold")
ax3.set_xlabel("Coverage (%)")
ax3.set_ylabel("Accuracy (%)")
ax3.legend()

# === Panel 4: Key Finding Text ===
ax4 = plt.subplot(2, 3, 4)
ax4.axis("off")

finding_text = """
KEY FINDINGS

Dataset: 35,088 5m BTC candles (Mar-Jun 2026)
Baseline: 49.8% up moves

1. HIGH FREQUENCY (25%+ coverage):
   Best accuracy achievable: 52.2%
   Strategy: Bollinger Band Mean Reversion
   This is only a 2.4% edge over baseline

2. RARE SETUPS (<0.1% coverage):
   MA Reversal: 100% (5 trades)
   Extreme Oversold Bounce: 100% (6 trades)
   Mean Reversion Spike: 85.7% (7 trades)
   
   These are too rare for practical Polymarket
   trading (1-2 trades per month)

3. ML APPROACH:
   Tested Logistic Regression, Random Forest,
   Gradient Boosting with 256 features
   At 25% coverage: 55.3% accuracy max

4. CONCLUSION:
   90% accuracy + 25% coverage is NOT
   achievable on this dataset with technical
   indicators alone. Previous success likely
   relied on additional data sources.
"""

ax4.text(0.05, 0.95, finding_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="black", linewidth=1))

# === Panel 5: Best Strategy Parameters ===
ax5 = plt.subplot(2, 3, 5)
ax5.axis("off")

with open("optimization_round_2.json", "r") as f:
    round2 = json.load(f)

params_text = """
BEST STRATEGY PARAMETERS

MA Reversal (100%, 5 trades):
  MA Period: 10
  Dist Threshold: -7.3%
  Overbought Dist: +2.2%
  Require Reversal: False

Extreme Oversold (100%, 6 trades):
  RSI Max: 29
  Stoch Max: 15
  BB Position Max: 0.175
  Wick Min: 31.6%
  Volume Min: 3.7x avg
  Prev Drop: 0.47%
  N Prev Candles: 3

Mean Rev Spike (85.7%, 7 trades):
  Spike Period: 1
  Spike Threshold: 0.37%
  Reversal Threshold: 0.28%
  Require Wick: True
  Wick Min: 40.2%
"""

ax5.text(0.05, 0.95, params_text, transform=ax5.transAxes, fontsize=10,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fff0", edgecolor="green", linewidth=1))

# === Panel 6: Recommendation ===
ax6 = plt.subplot(2, 3, 6)
ax6.axis("off")

rec_text = """
RECOMMENDATIONS

For Polymarket 5m BTC trading:

Option A: High-Frequency (25%+ coverage)
  Expected accuracy: 52-55%
  With Polymarket's ~2% fee, this may
  not be profitable without proper
  bankroll management

Option B: Rare Event Trading
  Only trade the 100% setups
  ~1-2 trades per month
  Very high confidence but low volume

Option C: Add External Data
  To achieve 90% x 25%, you need:
  - Order book imbalance
  - Funding rate data
  - Liquidation cascades
  - Cross-exchange flows
  
  These were likely in your previous
  successful setup from a year ago.
"""

ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8f0", edgecolor="orange", linewidth=1))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("graphs/final_comprehensive_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("Final comprehensive summary saved to graphs/final_comprehensive_summary.png")

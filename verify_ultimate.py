"""Verify ultimate strategy results and fix equity curve."""
import pandas as pd
import numpy as np
import json

df = pd.read_csv("btc_5m_enhanced.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_clean = df.dropna()

with open("ultimate_strategy_v2.json", "r") as f:
    ultimate = json.load(f)

print("ULTIMATE STRATEGY VERIFICATION")
print("=" * 50)
print(f"Accuracy: {ultimate['accuracy']:.1%}")
print(f"Coverage: {ultimate['coverage']:.1%}")
print(f"Trades: {ultimate['n_trades']}")
print(f"Up: {ultimate['n_up']}")
print(f"Down: {ultimate['n_down']}")

# Calculate expected wins
expected_wins = ultimate['n_trades'] * ultimate['accuracy']
expected_losses = ultimate['n_trades'] * (1 - ultimate['accuracy'])
net = expected_wins - expected_losses
print(f"\nExpected wins: {expected_wins:.0f}")
print(f"Expected losses: {expected_losses:.0f}")
print(f"Net score: +{net:.0f}")

# The bug was in the equity curve - non-trades were treated as losses
# Let's fix the final graph

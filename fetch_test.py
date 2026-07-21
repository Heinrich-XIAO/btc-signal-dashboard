import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Try to fetch BTC 5m data for March-June 2026
print("Fetching BTC data with yfinance...")

# For 5m interval, yfinance typically limits to recent data
# Let's try different approaches

# Approach 1: Direct download for the period
try:
    df = yf.download("BTC-USD", start="2026-03-01", end="2026-07-01", interval="5m")
    if df is not None and not df.empty:
        print(f"Approach 1 (5m direct): Got {len(df)} rows")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(df.head())
    else:
        print("Approach 1: Empty or None result")
except Exception as e:
    print(f"Approach 1 failed: {e}")

# Approach 2: Try 1m interval
try:
    df_1m = yf.download("BTC-USD", start="2026-03-01", end="2026-07-01", interval="1m")
    if df_1m is not None and not df_1m.empty:
        print(f"\nApproach 2 (1m direct): Got {len(df_1m)} rows")
        print(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")
    else:
        print("\nApproach 2: Empty or None result")
except Exception as e:
    print(f"\nApproach 2 failed: {e}")

# Approach 3: Recent 5m data (last 60 days max)
try:
    df_recent = yf.download("BTC-USD", period="60d", interval="5m")
    if df_recent is not None and not df_recent.empty:
        print(f"\nApproach 3 (5m 60d period): Got {len(df_recent)} rows")
        print(f"Date range: {df_recent.index.min()} to {df_recent.index.max()}")
    else:
        print("\nApproach 3: Empty or None result")
except Exception as e:
    print(f"\nApproach 3 failed: {e}")

# Approach 4: 1h interval for longer history
try:
    df_1h = yf.download("BTC-USD", start="2026-03-01", end="2026-07-01", interval="1h")
    if df_1h is not None and not df_1h.empty:
        print(f"\nApproach 4 (1h direct): Got {len(df_1h)} rows")
        print(f"Date range: {df_1h.index.min()} to {df_1h.index.max()}")
    else:
        print("\nApproach 4: Empty or None result")
except Exception as e:
    print(f"\nApproach 4 failed: {e}")

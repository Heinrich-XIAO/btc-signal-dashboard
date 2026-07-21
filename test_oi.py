"""Fix open interest fetch."""
import requests
import pandas as pd
from datetime import datetime, timezone

# Test different endpoints
FUTURES_BASE = "https://fapi.binance.com/fapi/v1"

start = datetime(2026, 3, 1, tzinfo=timezone.utc)
end = datetime(2026, 7, 1, tzinfo=timezone.utc)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

# Try /openInterestHist
print("Testing /openInterestHist...")
resp = requests.get(f"{FUTURES_BASE}/openInterestHist", 
                    params={"symbol": "BTCUSDT", "period": "5m", "limit": 500, 
                            "startTime": start_ms, "endTime": end_ms},
                    timeout=30)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:200]}")

# Try /openInterest directly (current snapshot)
print("\nTesting /openInterest (snapshot)...")
resp2 = requests.get(f"{FUTURES_BASE}/openInterest", 
                     params={"symbol": "BTCUSDT"},
                     timeout=30)
print(f"Status: {resp2.status_code}")
print(f"Response: {resp2.text[:200]}")

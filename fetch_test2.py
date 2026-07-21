import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing multiple request strategies for yfinance 5m data...")

# Test 1: Try to get data in chunks going backwards from today
# Today is July 21, 2026
# 60 days before that is around May 22

end_date = datetime(2026, 7, 21)
chunk_size = timedelta(days=59)  # Just under 60 days

all_data = []

# Try to fetch 4 chunks to cover March - June
for i in range(4):
    chunk_end = end_date - timedelta(days=i * 60)
    chunk_start = chunk_end - chunk_size
    
    print(f"\nChunk {i+1}: {chunk_start.date()} to {chunk_end.date()}")
    try:
        df = yf.download(
            "BTC-USD", 
            start=chunk_start.strftime("%Y-%m-%d"), 
            end=chunk_end.strftime("%Y-%m-%d"), 
            interval="5m",
            progress=False
        )
        if df is not None and not df.empty:
            print(f"  SUCCESS: Got {len(df)} rows, range: {df.index.min()} to {df.index.max()}")
            all_data.append(df)
        else:
            print(f"  FAILED: Empty/None result")
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\n\nTotal successful chunks: {len(all_data)}")
if all_data:
    combined = pd.concat(all_data)
    print(f"Combined data: {len(combined)} rows")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

# Test 2: Try different period strings
print("\n\nTesting different period strings with 5m:")
for period in ["1mo", "2mo", "3mo", "6mo", "ytd", "1y", "max"]:
    try:
        df = yf.download("BTC-USD", period=period, interval="5m", progress=False)
        if df is not None and not df.empty:
            print(f"  period='{period}': {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
        else:
            print(f"  period='{period}': Empty/None")
    except Exception as e:
        print(f"  period='{period}': {str(e)[:80]}")

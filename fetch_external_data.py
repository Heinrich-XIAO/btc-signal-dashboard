"""Fetch external data: funding rates, open interest, multi-timeframe."""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

SPOT_BASE = "https://api.binance.com/api/v3"
FUTURES_BASE = "https://fapi.binance.com/fapi/v1"

def fetch_klines(symbol="BTCUSDT", interval="1m", start_date=None, end_date=None):
    """Fetch klines from Binance spot."""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + pd.Timedelta(days=1)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    print(f"Fetching {symbol} {interval} klines...")
    while current_start < end_ms:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if current_start:
            params["startTime"] = int(current_start)
        if end_ms:
            params["endTime"] = int(end_ms)
        
        resp = requests.get(f"{SPOT_BASE}/klines", params=params, timeout=30)
        data = resp.json()
        
        if not data or not isinstance(data, list):
            break
        
        all_data.extend(data)
        last_close = data[-1][6]
        current_start = last_close + 1
        
        if len(data) < 1000:
            break
        time.sleep(0.05)
    
    print(f"  Got {len(all_data)} {interval} candles")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                    "trades", "taker_buy_volume", "taker_buy_quote_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume", "trades", "taker_buy_volume", "quote_volume"]]

def fetch_funding_rates(symbol="BTCUSDT", start_date=None, end_date=None):
    """Fetch funding rate history from Binance futures."""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + pd.Timedelta(days=1)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    print("Fetching funding rates...")
    while current_start < end_ms:
        params = {"symbol": symbol, "limit": 1000}
        if current_start:
            params["startTime"] = int(current_start)
        if end_ms:
            params["endTime"] = int(end_ms)
        
        resp = requests.get(f"{FUTURES_BASE}/fundingRate", params=params, timeout=30)
        data = resp.json()
        
        if not data or not isinstance(data, list):
            break
        
        all_data.extend(data)
        last_time = pd.to_datetime(data[-1]["fundingTime"], unit="ms").timestamp() * 1000
        current_start = int(last_time) + 1
        
        if len(data) < 1000:
            break
        time.sleep(0.05)
    
    print(f"  Got {len(all_data)} funding rate records")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["fundingRate"]]

def fetch_open_interest(symbol="BTCUSDT", period="5m", start_date=None, end_date=None):
    """Fetch open interest history from Binance futures."""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + pd.Timedelta(days=1)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    print(f"Fetching open interest ({period})...")
    while current_start < end_ms:
        params = {"symbol": symbol, "period": period, "limit": 500}
        if current_start:
            params["startTime"] = int(current_start)
        if end_ms:
            params["endTime"] = int(end_ms)
        
        resp = requests.get(f"{FUTURES_BASE}/openInterest", params=params, timeout=30)
        data = resp.json()
        
        if not data or not isinstance(data, list):
            break
        
        all_data.extend(data)
        last_time = pd.to_datetime(data[-1]["timestamp"], unit="ms").timestamp() * 1000
        current_start = int(last_time) + 1
        
        if len(data) < 500:
            break
        time.sleep(0.05)
    
    print(f"  Got {len(all_data)} OI records")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["sumOpenInterest", "sumOpenInterestValue"]]

if __name__ == "__main__":
    # Fetch all external data
    start = "2026-03-01"
    end = "2026-07-01"
    
    # Multi-timeframe klines
    df_1m = fetch_klines("BTCUSDT", "1m", start, end)
    df_1m.to_csv("btc_1m_data.csv")
    
    df_15m = fetch_klines("BTCUSDT", "15m", start, end)
    df_15m.to_csv("btc_15m_data.csv")
    
    df_1h = fetch_klines("BTCUSDT", "1h", start, end)
    df_1h.to_csv("btc_1h_data.csv")
    
    df_4h = fetch_klines("BTCUSDT", "4h", start, end)
    df_4h.to_csv("btc_4h_data.csv")
    
    # Funding rates
    df_funding = fetch_funding_rates("BTCUSDT", start, end)
    df_funding.to_csv("btc_funding_rates.csv")
    
    # Open interest
    df_oi = fetch_open_interest("BTCUSDT", "5m", start, end)
    df_oi.to_csv("btc_open_interest.csv")
    
    print("\nAll external data saved!")
    print(f"1m data: {len(df_1m)} rows")
    print(f"15m data: {len(df_15m)} rows")
    print(f"1h data: {len(df_1h)} rows")
    print(f"4h data: {len(df_4h)} rows")
    print(f"Funding rates: {len(df_funding)} rows")
    print(f"Open interest: {len(df_oi)} rows")

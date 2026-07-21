"""Fetch 5m BTC data from Binance API."""
import requests
import pandas as pd
import time
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol="BTCUSDT", interval="5m", start_ms=None, end_ms=None, limit=1000):
    """Fetch a single chunk of kline data from Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ms:
        params["startTime"] = int(start_ms)
    if end_ms:
        params["endTime"] = int(end_ms)
    
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def fetch_all_klines(symbol="BTCUSDT", interval="5m", start_date=None, end_date=None):
    """
    Fetch all klines for a date range by paginating through the API.
    
    start_date/end_date should be datetime objects or strings like "2026-03-01"
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + pd.Timedelta(days=1)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    print(f"Fetching {symbol} {interval} data from {start_date} to {end_date}...")
    
    while current_start < end_ms:
        chunk = fetch_klines(symbol, interval, current_start, end_ms)
        if not chunk:
            break
        
        all_data.extend(chunk)
        
        # Last candle's close time + 1ms
        last_close_time = chunk[-1][6]  # index 6 is close time
        current_start = last_close_time + 1
        
        print(f"  Fetched {len(chunk)} candles, total: {len(all_data)}, current time: {pd.to_datetime(current_start, unit='ms')}")
        
        # Rate limit protection
        time.sleep(0.1)
        
        if len(chunk) < 1000:
            break
    
    print(f"Total candles fetched: {len(all_data)}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    # Binance kline format:
    # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    # Convert types
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", 
                    "trades", "taker_buy_volume", "taker_buy_quote_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Set timestamp
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    
    # Drop duplicates
    df = df[~df.index.duplicated(keep="first")]
    
    # Keep only needed columns
    df = df[["open", "high", "low", "close", "volume", "trades", 
             "taker_buy_volume", "quote_volume"]]
    
    return df

if __name__ == "__main__":
    # Fetch March 1, 2026 to June 30, 2026
    df = fetch_all_klines("BTCUSDT", "5m", "2026-03-01", "2026-06-30")
    df.to_csv("btc_5m_data.csv")
    print(f"\nSaved to btc_5m_data.csv")
    print(df.head())
    print(df.tail())
    print(f"Shape: {df.shape}")

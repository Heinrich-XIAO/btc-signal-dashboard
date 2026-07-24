"""Fetch live kline data from Binance for multiple timeframes."""
import requests
import pandas as pd
import numpy as np
from typing import Optional

SPOT_BASE = "https://api.binance.com/api/v3"
FUTURES_BASE = "https://fapi.binance.com/fapi/v1"

PROXIES = {"http": "socks5h://5.255.103.55:1080", "https": "socks5h://5.255.103.55:1080"}

def fetch_klines(symbol="BTCUSDT", interval="5m", limit=200) -> Optional[pd.DataFrame]:
    """Fetch recent klines from Binance spot."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(f"{SPOT_BASE}/klines", params=params, timeout=30, proxies=PROXIES)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching {interval} klines: {e}")
        return None

    if not data or not isinstance(data, list):
        return None

    df = pd.DataFrame(data, columns=[
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
    return df[["open", "high", "low", "close", "volume", "trades",
               "taker_buy_volume", "quote_volume"]]


def fetch_funding_rates(symbol="BTCUSDT", limit=50) -> Optional[pd.DataFrame]:
    """Fetch recent funding rates from Binance futures."""
    params = {"symbol": symbol, "limit": limit}
    try:
        resp = requests.get(f"{FUTURES_BASE}/fundingRate", params=params, timeout=30, proxies=PROXIES)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching funding rates: {e}")
        return None

    if not data or not isinstance(data, list):
        return None

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["fundingRate"]]


def fetch_all_timeframes(symbol="BTCUSDT") -> dict:
    """Fetch all required timeframes for feature computation."""
    return {
        "5m": fetch_klines(symbol, "5m", limit=200),
        "1m": fetch_klines(symbol, "1m", limit=1500),
        "15m": fetch_klines(symbol, "15m", limit=100),
        "1h": fetch_klines(symbol, "1h", limit=100),
        "4h": fetch_klines(symbol, "4h", limit=50),
        "funding": fetch_funding_rates(symbol, limit=50),
    }

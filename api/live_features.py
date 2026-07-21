"""Compute live features from Binance kline data for model prediction."""
import pandas as pd
import numpy as np
from typing import Optional, Dict


def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add base technical features (from features.py)."""
    df = df.copy()

    # Basic returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Price relative to moving averages
    for window in [3, 5, 10, 20, 50]:
        ma = df["close"].rolling(window=window).mean()
        df[f"close_ma{window}_ratio"] = df["close"] / ma
        df[f"close_ma{window}_dist"] = (df["close"] - ma) / ma

    # EMA ratios
    for span in [3, 5, 10, 20]:
        ema = df["close"].ewm(span=span, adjust=False).mean()
        df[f"close_ema{span}_ratio"] = df["close"] / ema

    # RSI
    def compute_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    for window in [7, 14, 21]:
        df[f"rsi_{window}"] = compute_rsi(df["close"], window)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    for window in [10, 20]:
        ma = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        df[f"bb_{window}_upper"] = ma + 2 * std
        df[f"bb_{window}_lower"] = ma - 2 * std
        df[f"bb_{window}_width"] = (df[f"bb_{window}_upper"] - df[f"bb_{window}_lower"]) / ma
        df[f"bb_{window}_position"] = (df["close"] - df[f"bb_{window}_lower"]) / (df[f"bb_{window}_upper"] - df[f"bb_{window}_lower"])

    # ATR
    for window in [7, 14, 21]:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{window}"] = tr.rolling(window=window).mean()
        df[f"atr_{window}_ratio"] = df[f"atr_{window}"] / df["close"]

    # Stochastic
    for window in [7, 14]:
        low_min = df["low"].rolling(window=window).min()
        high_max = df["high"].rolling(window=window).max()
        df[f"stoch_k_{window}"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df[f"stoch_d_{window}"] = df[f"stoch_k_{window}"].rolling(window=3).mean()

    # Williams %R
    for window in [7, 14]:
        high_max = df["high"].rolling(window=window).max()
        low_min = df["low"].rolling(window=window).min()
        df[f"williams_r_{window}"] = -100 * (high_max - df["close"]) / (high_max - low_min)

    # CCI
    for window in [14, 20]:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma_tp = tp.rolling(window=window).mean()
        std_tp = tp.rolling(window=window).std()
        df[f"cci_{window}"] = (tp - ma_tp) / (0.015 * std_tp)

    # Volume features
    df["volume_ma10"] = df["volume"].rolling(window=10).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma10"]
    df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"]
    df["taker_buy_ratio_ma10"] = df["taker_buy_ratio"].rolling(window=10).mean()

    # Candlestick features
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"] / df["open"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"]
    df["body_range_ratio"] = np.abs(df["body"]) / df["range"]

    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f"returns_lag{lag}"] = df["returns"].shift(lag)
        df[f"volume_ratio_lag{lag}"] = df["volume_ratio"].shift(lag)
        df[f"body_pct_lag{lag}"] = df["body_pct"].shift(lag)

    # Cumulative returns
    for window in [1, 2, 3, 5, 10, 20]:
        df[f"cumret_{window}"] = df["returns"].rolling(window=window).sum()

    # Volatility
    for window in [10, 20]:
        df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()

    # ADX
    for window in [14]:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([
            df["high"] - df["low"],
            np.abs(df["high"] - df["close"].shift(1)),
            np.abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * plus_dm.rolling(window=window).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=window).mean() / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df[f"adx_{window}"] = dx.rolling(window=window).mean()

    # Time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    return df


def _add_enhanced_features(
    df: pd.DataFrame,
    df_1m: Optional[pd.DataFrame] = None,
    df_15m: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    df_funding: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add enhanced features from multi-timeframe + funding rate data."""

    # === FUNDING RATE FEATURES ===
    if df_funding is not None and not df_funding.empty:
        df_funding = df_funding.copy()
        df_funding.index = pd.to_datetime(df_funding.index)
        df_funding_resampled = df_funding.resample("5min").ffill()
        df = df.join(df_funding_resampled, how="left")
        df["fundingRate"] = df["fundingRate"].ffill()

        if "fundingRate" in df.columns:
            df["funding_rate"] = df["fundingRate"]
            df["funding_rate_abs"] = df["funding_rate"].abs()
            df["funding_rate_sign"] = np.sign(df["funding_rate"])
            df["funding_rate_ma3"] = df["funding_rate"].rolling(3).mean()
            df["funding_rate_trend"] = df["funding_rate"].diff(3)
            df["funding_extreme"] = (df["funding_rate_abs"] > df["funding_rate_abs"].quantile(0.9)).astype(int)
            df["funding_positive"] = (df["funding_rate"] > 0).astype(int)

    # === 1M MICROSTRUCTURE FEATURES ===
    if df_1m is not None and not df_1m.empty:
        df_1m = df_1m.copy()
        df_1m["returns_1m"] = df_1m["close"].pct_change()
        df_1m["range_1m"] = df_1m["high"] - df_1m["low"]
        df_1m["volume_1m"] = df_1m["volume"]

        agg_1m = df_1m.resample("5min").agg({
            "returns_1m": ["sum", "std", "min", "max", "count"],
            "range_1m": ["sum", "mean", "max"],
            "volume_1m": ["sum", "mean"],
            "taker_buy_volume": ["sum"],
        })
        agg_1m.columns = ["_".join(c).strip() for c in agg_1m.columns]
        agg_1m = agg_1m.add_prefix("1m_")
        df = df.join(agg_1m, how="left")

        if "1m_returns_1m_sum" in df.columns:
            df["1m_momentum"] = df["1m_returns_1m_sum"]
            df["1m_volatility"] = df["1m_returns_1m_std"]
            df["1m_range_sum"] = df["1m_range_1m_sum"]
            df["1m_max_range"] = df["1m_range_1m_max"]
            df["1m_candle_count"] = df["1m_returns_1m_count"]
            df["1m_taker_ratio"] = df["1m_taker_buy_volume_sum"] / (df["1m_volume_1m_sum"] + 1e-8)

    # === MULTI-TIMEFRAME FEATURES ===
    def add_tf_features(tf_data, prefix):
        """Add features from higher timeframe."""
        tf_data = tf_data.copy()
        tf_data["returns"] = tf_data["close"].pct_change()
        tf_data["range"] = tf_data["high"] - tf_data["low"]
        tf_data["body"] = tf_data["close"] - tf_data["open"]
        tf_data["body_pct"] = tf_data["body"] / tf_data["open"]

        # RSI
        delta = tf_data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        tf_data["rsi"] = 100 - (100 / (1 + rs))

        # EMAs
        for span in [5, 10, 20]:
            tf_data[f"ema{span}"] = tf_data["close"].ewm(span=span, adjust=False).mean()

        tf_resampled = tf_data[["close", "returns", "range", "body_pct", "rsi", "ema5", "ema10", "ema20"]].resample("5min").ffill()
        tf_resampled = tf_resampled.add_prefix(f"{prefix}_")
        return tf_resampled

    if df_15m is not None and not df_15m.empty:
        df_15m_tf = add_tf_features(df_15m, "15m")
        df = df.join(df_15m_tf, how="left")

    if df_1h is not None and not df_1h.empty:
        df_1h_tf = add_tf_features(df_1h, "1h")
        df = df.join(df_1h_tf, how="left")

    if df_4h is not None and not df_4h.empty:
        df_4h_tf = add_tf_features(df_4h, "4h")
        df = df.join(df_4h_tf, how="left")

    # Multi-timeframe alignment features
    if "15m_rsi" in df.columns and "1h_rsi" in df.columns:
        df["rsi_5m_vs_15m"] = df.get("rsi_14", pd.Series(np.nan, index=df.index)) - df["15m_rsi"]
        df["rsi_5m_vs_1h"] = df.get("rsi_14", pd.Series(np.nan, index=df.index)) - df["1h_rsi"]
        df["rsi_15m_vs_1h"] = df["15m_rsi"] - df["1h_rsi"]

        df["rsi_5m_oversold"] = (df.get("rsi_14", pd.Series(50, index=df.index)) < 30).astype(int)
        df["rsi_15m_oversold"] = (df["15m_rsi"] < 30).astype(int)
        df["rsi_1h_oversold"] = (df["1h_rsi"] < 30).astype(int)
        df["rsi_4h_oversold"] = (df["4h_rsi"] < 30).astype(int)
        df["rsi_all_oversold"] = df["rsi_5m_oversold"] + df["rsi_15m_oversold"] + df["rsi_1h_oversold"] + df["rsi_4h_oversold"]

        df["rsi_5m_overbought"] = (df.get("rsi_14", pd.Series(50, index=df.index)) > 70).astype(int)
        df["rsi_15m_overbought"] = (df["15m_rsi"] > 70).astype(int)
        df["rsi_1h_overbought"] = (df["1h_rsi"] > 70).astype(int)
        df["rsi_4h_overbought"] = (df["4h_rsi"] > 70).astype(int)
        df["rsi_all_overbought"] = df["rsi_5m_overbought"] + df["rsi_15m_overbought"] + df["rsi_1h_overbought"] + df["rsi_4h_overbought"]

    # Higher timeframe trend direction
    if "4h_ema20" in df.columns:
        df["trend_4h_up"] = (df["4h_close"] > df["4h_ema20"]).astype(int)
        df["trend_1h_up"] = (df["1h_close"] > df["1h_ema20"]).astype(int)
        df["trend_15m_up"] = (df["15m_close"] > df["15m_ema20"]).astype(int)
        df["trend_confluence_up"] = df["trend_4h_up"] + df["trend_1h_up"] + df["trend_15m_up"]
        df["trend_confluence_down"] = 3 - df["trend_confluence_up"]

    if "1h_ema10" in df.columns:
        df["price_vs_1h_ema10"] = df["close"] / df["1h_ema10"]
        df["price_vs_4h_ema20"] = df["close"] / df["4h_ema20"]
        df["price_vs_15m_ema5"] = df["close"] / df["15m_ema5"]

    if "4h_returns" in df.columns:
        df["mom_4h"] = df["4h_returns"].rolling(5).sum()
        df["mom_1h"] = df["1h_returns"].rolling(5).sum()
        df["mom_15m"] = df["15m_returns"].rolling(5).sum()

    if "15m_range" in df.columns:
        df["volatility_15m"] = df["15m_range"].rolling(10).mean() / df["15m_close"]
        df["volatility_1h"] = df["1h_range"].rolling(10).mean() / df["1h_close"]

    # === CROSS-FEATURES ===
    if "funding_rate" in df.columns and "rsi_14" in df.columns:
        df["funding_x_rsi"] = df["funding_rate"] * df.get("rsi_14", 0)
        df["funding_x_price_change"] = df["funding_rate"] * df["returns"]

    if "funding_rate" in df.columns and "volume_ratio" in df.columns:
        df["funding_x_volume"] = df["funding_rate"] * df["volume_ratio"]

    if "1m_momentum" in df.columns and "trend_4h_up" in df.columns:
        df["micro_trend_align"] = np.where(
            df["trend_4h_up"] == 1,
            df["1m_momentum"],
            -df["1m_momentum"]
        )

    return df


def compute_live_features(data: Dict[str, Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Compute full feature set from live Binance data."""
    df_5m = data.get("5m")
    if df_5m is None or df_5m.empty:
        return None

    df = _add_base_features(df_5m)
    df = _add_enhanced_features(
        df,
        df_1m=data.get("1m"),
        df_15m=data.get("15m"),
        df_1h=data.get("1h"),
        df_4h=data.get("4h"),
        df_funding=data.get("funding"),
    )

    return df

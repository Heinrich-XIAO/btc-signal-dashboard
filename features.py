"""Feature engineering for 5m BTC prediction."""
import numpy as np
import pandas as pd

def add_features(df):
    """Add all technical features to the dataframe."""
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
    
    # Cumulative returns over recent windows
    for window in [1, 2, 3, 5, 10, 20]:
        df[f"cumret_{window}"] = df["returns"].rolling(window=window).sum()
    
    # Volatility
    for window in [10, 20]:
        df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()
    
    # Trend strength (ADX simplified)
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

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_data.csv", index_col="timestamp", parse_dates=True)
    df = add_features(df)
    df.to_csv("btc_5m_features.csv")
    print(f"Features added. Shape: {df.shape}")
    print(df.columns.tolist())

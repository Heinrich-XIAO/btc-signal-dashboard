"""Strategy implementations for 5m BTC prediction."""
import numpy as np
import pandas as pd

def evaluate_signal(df, signal_col, min_trades=30):
    """
    Evaluate a binary signal column.
    Returns dict with accuracy, n_trades, n_up, n_down.
    """
    trades = df[df[signal_col] != 0].copy()
    if len(trades) < min_trades:
        return {"accuracy": 0.5, "n_trades": len(trades), "n_up": 0, "n_down": 0, "direction": 0}
    
    # signal_col: 1 = buy (predict up), -1 = sell (predict down)
    correct = ((trades[signal_col] == 1) & (trades["target"] == 1)) | \
              ((trades[signal_col] == -1) & (trades["target"] == 0))
    
    accuracy = correct.mean()
    n_up = (trades[signal_col] == 1).sum()
    n_down = (trades[signal_col] == -1).sum()
    
    return {
        "accuracy": accuracy,
        "n_trades": len(trades),
        "n_up": n_up,
        "n_down": n_down,
        "direction": 1 if n_up > n_down else -1
    }

# ===== Strategy 1: RSI Extreme Reversal =====
def strategy_rsi_reversal(df, rsi_period=14, oversold=30, overbought=70, volume_min=1.0):
    """Buy when RSI oversold, sell when overbought. Optional volume filter."""
    df = df.copy()
    rsi_col = f"rsi_{rsi_period}"
    
    df["signal"] = 0
    df.loc[(df[rsi_col] < oversold) & (df["volume_ratio"] >= volume_min), "signal"] = 1
    df.loc[(df[rsi_col] > overbought) & (df["volume_ratio"] >= volume_min), "signal"] = -1
    
    return df["signal"]

# ===== Strategy 2: Bollinger Band Reversal =====
def strategy_bb_reversal(df, bb_period=20, bb_position_low=0.1, bb_position_high=0.9,
                         require_wick=False, wick_threshold=0.3):
    """Buy at lower BB, sell at upper BB. Optional wick confirmation."""
    df = df.copy()
    bb_pos_col = f"bb_{bb_period}_position"
    
    df["signal"] = 0
    
    buy_condition = df[bb_pos_col] < bb_position_low
    sell_condition = df[bb_pos_col] > bb_position_high
    
    if require_wick:
        buy_condition &= df["lower_wick_ratio"] > wick_threshold
        sell_condition &= df["upper_wick_ratio"] > wick_threshold
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 3: Stochastic Extreme =====
def strategy_stoch_extreme(df, stoch_period=14, low_threshold=20, high_threshold=80,
                           use_d=False, d_threshold=3):
    """Buy when stochastic is very low, sell when very high."""
    df = df.copy()
    k_col = f"stoch_k_{stoch_period}"
    d_col = f"stoch_d_{stoch_period}"
    
    df["signal"] = 0
    
    buy_condition = df[k_col] < low_threshold
    sell_condition = df[k_col] > high_threshold
    
    if use_d:
        buy_condition &= (df[d_col] - df[k_col]).abs() < d_threshold
        sell_condition &= (df[d_col] - df[k_col]).abs() < d_threshold
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 4: Volume-Price Momentum =====
def strategy_volume_momentum(df, volume_threshold=1.5, momentum_period=3, 
                              momentum_threshold=0.001, require_taker_buy=True,
                              taker_threshold=0.55):
    """Buy when volume spike + upward momentum. Sell when volume spike + downward momentum."""
    df = df.copy()
    mom_col = f"cumret_{momentum_period}"
    
    df["signal"] = 0
    
    buy_condition = (df["volume_ratio"] > volume_threshold) & (df[mom_col] > momentum_threshold)
    sell_condition = (df["volume_ratio"] > volume_threshold) & (df[mom_col] < -momentum_threshold)
    
    if require_taker_buy:
        buy_condition &= df["taker_buy_ratio"] > taker_threshold
        sell_condition &= df["taker_buy_ratio"] < (1 - taker_threshold)
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 5: MA Distance Reversal =====
def strategy_ma_reversal(df, ma_period=20, dist_threshold=-0.02, 
                         overbought_dist=0.02, require_reversal=True):
    """Buy when price far below MA, sell when far above."""
    df = df.copy()
    dist_col = f"close_ma{ma_period}_dist"
    
    df["signal"] = 0
    
    buy_condition = df[dist_col] < dist_threshold
    sell_condition = df[dist_col] > overbought_dist
    
    if require_reversal:
        buy_condition &= df["returns_lag1"] > 0  # started moving up
        sell_condition &= df["returns_lag1"] < 0  # started moving down
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 6: MACD Histogram Reversal =====
def strategy_macd_reversal(df, macd_turn_threshold=0, require_extreme=False,
                           hist_extreme=None):
    """Buy when MACD histogram turns positive, sell when turns negative."""
    df = df.copy()
    
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    
    df["signal"] = 0
    
    buy_condition = (df["macd_hist_prev"] < macd_turn_threshold) & (df["macd_hist"] > macd_turn_threshold)
    sell_condition = (df["macd_hist_prev"] > macd_turn_threshold) & (df["macd_hist"] < macd_turn_threshold)
    
    if require_extreme and hist_extreme is not None:
        buy_condition &= df["macd_hist_prev"] < -hist_extreme
        sell_condition &= df["macd_hist_prev"] > hist_extreme
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 7: Candlestick Reversal Pattern =====
def strategy_candlestick(df, body_threshold=0.001, require_lower_wick=True,
                         lower_wick_min=0.5, require_engulfing=False,
                         prev_body_max=0.0005):
    """Detect hammer-like and engulfing patterns."""
    df = df.copy()
    
    # Hammer: small body, long lower wick
    hammer = (df["body_range_ratio"] < 0.3) & (df["lower_wick_ratio"] > lower_wick_min)
    
    # Engulfing: current body covers previous body, opposite direction
    prev_body = df["body_pct"].shift(1)
    engulfing = (df["body_pct"] * prev_body < 0) & (df["body_pct"].abs() > prev_body.abs())
    
    df["signal"] = 0
    
    buy_condition = hammer if not require_engulfing else (hammer | engulfing)
    sell_condition = (df["body_range_ratio"] < 0.3) & (df["upper_wick_ratio"] > lower_wick_min)
    
    if require_engulfing:
        sell_condition |= ((df["body_pct"] * prev_body < 0) & (df["body_pct"].abs() > prev_body.abs()) & (df["body_pct"] < 0))
    
    # Require small previous candle for better engulfing
    if require_engulfing and prev_body_max:
        buy_condition &= prev_body.abs() < prev_body_max
        sell_condition &= prev_body.abs() < prev_body_max
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 8: Multi-Indicator Confluence =====
def strategy_confluence(df, rsi_max=35, stoch_max=25, bb_pos_max=0.15,
                        volume_min=1.0, require_all=True):
    """Buy when multiple indicators agree on oversold. Sell when overbought."""
    df = df.copy()
    
    buy_rsi = df["rsi_14"] < rsi_max
    buy_stoch = df["stoch_k_14"] < stoch_max
    buy_bb = df["bb_20_position"] < bb_pos_max
    buy_vol = df["volume_ratio"] > volume_min
    
    sell_rsi = df["rsi_14"] > (100 - rsi_max)
    sell_stoch = df["stoch_k_14"] > (100 - stoch_max)
    sell_bb = df["bb_20_position"] > (1 - bb_pos_max)
    sell_vol = df["volume_ratio"] > volume_min
    
    if require_all:
        buy_condition = buy_rsi & buy_stoch & buy_bb & buy_vol
        sell_condition = sell_rsi & sell_stoch & sell_bb & sell_vol
    else:
        buy_condition = (buy_rsi.astype(int) + buy_stoch.astype(int) + buy_bb.astype(int) + buy_vol.astype(int)) >= 3
        sell_condition = (sell_rsi.astype(int) + sell_stoch.astype(int) + sell_bb.astype(int) + sell_vol.astype(int)) >= 3
    
    df["signal"] = 0
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 9: Mean Reversion After Spike =====
def strategy_mean_reversion_spike(df, spike_period=1, spike_threshold=0.008,
                                  reversal_threshold=0.002, require_wick=True,
                                  wick_min=0.3):
    """Buy after a large drop with immediate reversal."""
    df = df.copy()
    spike_col = f"returns_lag{spike_period}"
    
    df["signal"] = 0
    
    # Large drop followed by immediate bounce
    buy_condition = (df[spike_col] < -spike_threshold) & (df["returns"] > reversal_threshold)
    sell_condition = (df[spike_col] > spike_threshold) & (df["returns"] < -reversal_threshold)
    
    if require_wick:
        buy_condition &= df["lower_wick_ratio"] > wick_min
        sell_condition &= df["upper_wick_ratio"] > wick_min
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy 10: Trend Following with Pullback =====
def strategy_trend_pullback(df, trend_ma=50, pullback_ma=10, 
                            pullback_threshold=-0.005, require_volume=False,
                            volume_min=1.0):
    """Buy in uptrend after pullback to short MA. Sell in downtrend after bounce."""
    df = df.copy()
    
    # Use dist column for trend direction
    dist_trend = df[f"close_ma{trend_ma}_dist"]
    trend_up = dist_trend > 0
    trend_down = dist_trend < 0
    
    # Pullback: price below short MA
    pullback_dist = df[f"close_ma{pullback_ma}_dist"]
    
    df["signal"] = 0
    
    buy_condition = trend_up & (pullback_dist < pullback_threshold)
    sell_condition = trend_down & (pullback_dist > -pullback_threshold)
    
    if require_volume:
        buy_condition &= df["volume_ratio"] > volume_min
        sell_condition &= df["volume_ratio"] > volume_min
    
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# ===== Strategy registry =====
STRATEGIES = {
    "rsi_reversal": strategy_rsi_reversal,
    "bb_reversal": strategy_bb_reversal,
    "stoch_extreme": strategy_stoch_extreme,
    "volume_momentum": strategy_volume_momentum,
    "ma_reversal": strategy_ma_reversal,
    "macd_reversal": strategy_macd_reversal,
    "candlestick": strategy_candlestick,
    "confluence": strategy_confluence,
    "mean_reversion_spike": strategy_mean_reversion_spike,
    "trend_pullback": strategy_trend_pullback,
}

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    print("Testing strategies with default parameters:\n")
    for name, func in STRATEGIES.items():
        signal = func(df)
        df_test = df.copy()
        df_test["signal"] = signal
        result = evaluate_signal(df_test, "signal", min_trades=10)
        print(f"{name:25s}: accuracy={result['accuracy']:.3f}, trades={result['n_trades']}, "
              f"up={result['n_up']}, down={result['n_down']}")

"""Additional extreme strategies for round 2 optimization."""
import numpy as np
import pandas as pd

def strategy_extreme_oversold_bounce(df, rsi_max=20, stoch_max=15, bb_pos_max=0.1,
                                     wick_min=0.5, volume_min=1.5, 
                                     prev_drop_threshold=0.005,
                                     n_prev_candles=2):
    """Buy only when multiple extreme oversold conditions align with reversal signs."""
    df = df.copy()
    
    # Extreme oversold
    extreme_rsi = df["rsi_14"] < rsi_max
    extreme_stoch = df["stoch_k_14"] < stoch_max
    extreme_bb = df["bb_20_position"] < bb_pos_max
    
    # Reversal signs
    long_wick = df["lower_wick_ratio"] > wick_min
    vol_spike = df["volume_ratio"] > volume_min
    
    # Previous drop
    prev_drop = df[f"cumret_{n_prev_candles}"] < -prev_drop_threshold
    
    buy_condition = extreme_rsi & extreme_stoch & extreme_bb & long_wick & vol_spike & prev_drop
    
    # Extreme overbought for sells
    extreme_rsi_sell = df["rsi_14"] > (100 - rsi_max)
    extreme_stoch_sell = df["stoch_k_14"] > (100 - stoch_max)
    extreme_bb_sell = df["bb_20_position"] > (1 - bb_pos_max)
    long_wick_upper = df["upper_wick_ratio"] > wick_min
    prev_rise = df[f"cumret_{n_prev_candles}"] > prev_drop_threshold
    
    sell_condition = extreme_rsi_sell & extreme_stoch_sell & extreme_bb_sell & long_wick_upper & vol_spike & prev_rise
    
    df["signal"] = 0
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

def strategy_consecutive_drop_reversal(df, n_consecutive=3, total_drop=0.01,
                                       require_hammer=True, wick_min=0.4,
                                       volume_min=1.0):
    """Buy after N consecutive down candles with significant total drop and hammer candle."""
    df = df.copy()
    
    # Consecutive down candles
    df["is_red"] = df["close"] < df["open"]
    
    # Rolling sum of consecutive reds
    df["consecutive_reds"] = 0
    count = 0
    for i in range(len(df)):
        if df["is_red"].iloc[i]:
            count += 1
        else:
            count = 0
        df.iloc[i, df.columns.get_loc("consecutive_reds")] = count
    
    # Check if previous N candles were all red
    prev_n_red = df["consecutive_reds"] >= n_consecutive
    
    # Total drop over N+1 candles (including current)
    total_return = df[f"cumret_{n_consecutive+1}"] if f"cumret_{n_consecutive+1}" in df.columns else df["cumret_5"]
    large_drop = total_return < -total_drop
    
    # Current candle reversal
    current_green = df["close"] > df["open"]
    hammer = df["lower_wick_ratio"] > wick_min if require_hammer else True
    vol = df["volume_ratio"] > volume_min
    
    buy_condition = prev_n_red & large_drop & current_green & hammer & vol
    
    # Sell side: consecutive greens then drop
    df["is_green"] = df["close"] > df["open"]
    df["consecutive_greens"] = 0
    count = 0
    for i in range(len(df)):
        if df["is_green"].iloc[i]:
            count += 1
        else:
            count = 0
        df.iloc[i, df.columns.get_loc("consecutive_greens")] = count
    
    prev_n_green = df["consecutive_greens"] >= n_consecutive
    large_rise = total_return > total_drop
    current_red = df["close"] < df["open"]
    shooting_star = df["upper_wick_ratio"] > wick_min if require_hammer else True
    
    sell_condition = prev_n_green & large_rise & current_red & shooting_star & vol
    
    df["signal"] = 0
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

def strategy_volume_exhaustion(df, volume_threshold=3.0, prev_move_threshold=0.01,
                               reversal_body_min=0.002, require_opposite_direction=True,
                               n_prev=3):
    """Trade when huge volume occurs after a large move, signaling exhaustion."""
    df = df.copy()
    
    prev_return_col = f"cumret_{n_prev}" if f"cumret_{n_prev}" in df.columns else "cumret_5"
    
    vol_extreme = df["volume_ratio"] > volume_threshold
    prev_large_drop = df[prev_return_col] < -prev_move_threshold
    prev_large_rise = df[prev_return_col] > prev_move_threshold
    
    # Current candle must be reversing
    reversal_up = df["body_pct"] > reversal_body_min
    reversal_down = df["body_pct"] < -reversal_body_min
    
    if require_opposite_direction:
        buy_condition = vol_extreme & prev_large_drop & reversal_up
        sell_condition = vol_extreme & prev_large_rise & reversal_down
    else:
        buy_condition = vol_extreme & prev_large_drop
        sell_condition = vol_extreme & prev_large_rise
    
    df["signal"] = 0
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

def strategy_bb_squeeze_breakout(df, bb_period=20, squeeze_threshold=0.01,
                                 breakout_threshold=0.003, volume_min=1.5,
                                 direction="both"):
    """Trade breakout after Bollinger Band squeeze."""
    df = df.copy()
    
    bb_width_col = f"bb_{bb_period}_width"
    squeeze = df[bb_width_col] < squeeze_threshold
    
    # Breakout
    break_up = df["returns"] > breakout_threshold
    break_down = df["returns"] < -breakout_threshold
    
    vol = df["volume_ratio"] > volume_min
    
    # Squeeze must have been in recent candles
    squeeze_recent = df[bb_width_col].rolling(window=5).min() < squeeze_threshold
    
    df["signal"] = 0
    
    if direction in ["both", "up"]:
        df.loc[squeeze_recent & break_up & vol, "signal"] = 1
    if direction in ["both", "down"]:
        df.loc[squeeze_recent & break_down & vol, "signal"] = -1
    
    return df["signal"]

def strategy_ultimate_confluence(df, rsi_max=25, stoch_max=20, bb_pos_max=0.15,
                                 wick_min=0.4, volume_min=2.0, 
                                 prev_candles_drop=0.008, n_prev=3,
                                 ma_dist_max=-0.01, require_green=True):
    """The most restrictive confluence strategy requiring ALL conditions."""
    df = df.copy()
    
    # All oversold conditions
    cond1 = df["rsi_14"] < rsi_max
    cond2 = df["stoch_k_14"] < stoch_max
    cond3 = df["bb_20_position"] < bb_pos_max
    cond4 = df["lower_wick_ratio"] > wick_min
    cond5 = df["volume_ratio"] > volume_min
    
    prev_ret_col = f"cumret_{n_prev}" if f"cumret_{n_prev}" in df.columns else "cumret_5"
    cond6 = df[prev_ret_col] < -prev_candles_drop
    cond7 = df["close_ma20_dist"] < ma_dist_max
    cond8 = (df["close"] > df["open"]) if require_green else True
    
    buy_condition = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
    
    # Sell conditions (mirror)
    cond1s = df["rsi_14"] > (100 - rsi_max)
    cond2s = df["stoch_k_14"] > (100 - stoch_max)
    cond3s = df["bb_20_position"] > (1 - bb_pos_max)
    cond4s = df["upper_wick_ratio"] > wick_min
    cond5s = df["volume_ratio"] > volume_min
    cond6s = df[prev_ret_col] > prev_candles_drop
    cond7s = df["close_ma20_dist"] > -ma_dist_max
    cond8s = (df["close"] < df["open"]) if require_green else True
    
    sell_condition = cond1s & cond2s & cond3s & cond4s & cond5s & cond6s & cond7s & cond8s
    
    df["signal"] = 0
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1
    
    return df["signal"]

# Add new strategies to registry
EXTREME_STRATEGIES = {
    "extreme_oversold_bounce": strategy_extreme_oversold_bounce,
    "consecutive_drop_reversal": strategy_consecutive_drop_reversal,
    "volume_exhaustion": strategy_volume_exhaustion,
    "bb_squeeze_breakout": strategy_bb_squeeze_breakout,
    "ultimate_confluence": strategy_ultimate_confluence,
}

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    print("Testing extreme strategies with default parameters:\n")
    for name, func in EXTREME_STRATEGIES.items():
        signal = func(df)
        df_test = df.copy()
        df_test["signal"] = signal
        from strategies import evaluate_signal
        result = evaluate_signal(df_test, "signal", min_trades=5)
        print(f"{name:25s}: accuracy={result['accuracy']:.3f}, trades={result['n_trades']}, "
              f"up={result['n_up']}, down={result['n_down']}")

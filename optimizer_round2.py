"""Round 2: Aggressive optimization with extreme parameters."""
import numpy as np
import pandas as pd
import json
from strategies import STRATEGIES, evaluate_signal
from extreme_strategies import EXTREME_STRATEGIES
import random

random.seed(42)
np.random.seed(42)

def optimize_strategy_random(df, strategy_name, strategy_func, param_ranges, 
                              min_trades=10, n_iterations=2000, top_n=10):
    """Random search optimization."""
    results = []
    
    print(f"Random search: {n_iterations} iterations for {strategy_name}")
    for i in range(n_iterations):
        params = {}
        for key, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, bool):
                params[key] = random.choice([True, False])
            elif isinstance(min_val, int) and isinstance(max_val, int):
                params[key] = random.randint(min_val, max_val)
            elif isinstance(min_val, float) or isinstance(max_val, float):
                params[key] = random.uniform(min_val, max_val)
            else:
                params[key] = random.choice([min_val, max_val])
        
        try:
            signal = strategy_func(df, **params)
            df_test = df.copy()
            df_test["signal"] = signal
            result = evaluate_signal(df_test, "signal", min_trades=min_trades)
            
            if result["n_trades"] >= min_trades:
                results.append({
                    "params": params,
                    "accuracy": result["accuracy"],
                    "n_trades": result["n_trades"],
                    "n_up": result["n_up"],
                    "n_down": result["n_down"],
                })
        except Exception as e:
            pass
        
        if (i + 1) % 500 == 0:
            best_so_far = max(results, key=lambda x: x["accuracy"]) if results else None
            if best_so_far:
                print(f"  Processed {i+1}/{n_iterations}, best so far: {best_so_far['accuracy']:.3f} (n={best_so_far['n_trades']})")
            else:
                print(f"  Processed {i+1}/{n_iterations}, no valid results yet")
    
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results[:top_n]

def run_round_2(df):
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION ROUND 2 - EXTREME PARAMETERS")
    print(f"{'='*60}\n")
    
    # Original strategies with more extreme ranges
    
    # 1. RSI Reversal - more extreme
    print("Optimizing RSI Reversal (extreme)...")
    all_results["rsi_reversal"] = optimize_strategy_random(
        df, "rsi_reversal", STRATEGIES["rsi_reversal"],
        {
            "rsi_period": (7, 21),
            "oversold": (5, 25),
            "overbought": (75, 95),
            "volume_min": (0.5, 5.0),
        }, min_trades=10, n_iterations=3000)
    
    # 2. BB Reversal
    print("Optimizing BB Reversal (extreme)...")
    all_results["bb_reversal"] = optimize_strategy_random(
        df, "bb_reversal", STRATEGIES["bb_reversal"],
        {
            "bb_period": (10, 20),
            "bb_position_low": (0.0, 0.25),
            "bb_position_high": (0.75, 1.0),
            "require_wick": (True, False),
            "wick_threshold": (0.2, 0.8),
        }, min_trades=10, n_iterations=3000)
    
    # 3. Stoch Extreme
    print("Optimizing Stoch Extreme (extreme)...")
    all_results["stoch_extreme"] = optimize_strategy_random(
        df, "stoch_extreme", STRATEGIES["stoch_extreme"],
        {
            "stoch_period": (7, 14),
            "low_threshold": (3, 25),
            "high_threshold": (75, 97),
            "use_d": (True, False),
            "d_threshold": (1, 15),
        }, min_trades=10, n_iterations=3000)
    
    # 4. Volume Momentum
    print("Optimizing Volume Momentum (extreme)...")
    all_results["volume_momentum"] = optimize_strategy_random(
        df, "volume_momentum", STRATEGIES["volume_momentum"],
        {
            "volume_threshold": (1.0, 8.0),
            "momentum_period": (1, 5),
            "momentum_threshold": (0.0, 0.01),
            "require_taker_buy": (True, False),
            "taker_threshold": (0.5, 0.8),
        }, min_trades=10, n_iterations=3000)
    
    # 5. MA Reversal
    print("Optimizing MA Reversal (extreme)...")
    all_results["ma_reversal"] = optimize_strategy_random(
        df, "ma_reversal", STRATEGIES["ma_reversal"],
        {
            "ma_period": (10, 50),
            "dist_threshold": (-0.08, -0.001),
            "overbought_dist": (0.001, 0.08),
            "require_reversal": (True, False),
        }, min_trades=10, n_iterations=3000)
    
    # 6. MACD Reversal
    print("Optimizing MACD Reversal (extreme)...")
    all_results["macd_reversal"] = optimize_strategy_random(
        df, "macd_reversal", STRATEGIES["macd_reversal"],
        {
            "macd_turn_threshold": (-0.001, 0.001),
            "require_extreme": (True, False),
            "hist_extreme": (0.0, 0.01),
        }, min_trades=10, n_iterations=2000)
    
    # 7. Candlestick
    print("Optimizing Candlestick (extreme)...")
    all_results["candlestick"] = optimize_strategy_random(
        df, "candlestick", STRATEGIES["candlestick"],
        {
            "body_threshold": (0.0001, 0.005),
            "require_lower_wick": (True, False),
            "lower_wick_min": (0.2, 0.9),
            "require_engulfing": (True, False),
            "prev_body_max": (0.0001, 0.003),
        }, min_trades=10, n_iterations=3000)
    
    # 8. Confluence
    print("Optimizing Confluence (extreme)...")
    all_results["confluence"] = optimize_strategy_random(
        df, "confluence", STRATEGIES["confluence"],
        {
            "rsi_max": (10, 40),
            "stoch_max": (5, 35),
            "bb_pos_max": (0.02, 0.4),
            "volume_min": (0.5, 4.0),
            "require_all": (True, False),
        }, min_trades=10, n_iterations=3000)
    
    # 9. Mean Reversion Spike
    print("Optimizing Mean Reversion Spike (extreme)...")
    all_results["mean_reversion_spike"] = optimize_strategy_random(
        df, "mean_reversion_spike", STRATEGIES["mean_reversion_spike"],
        {
            "spike_period": (1, 3),
            "spike_threshold": (0.002, 0.03),
            "reversal_threshold": (0.0, 0.01),
            "require_wick": (True, False),
            "wick_min": (0.1, 0.7),
        }, min_trades=10, n_iterations=3000)
    
    # 10. Trend Pullback
    print("Optimizing Trend Pullback (extreme)...")
    all_results["trend_pullback"] = optimize_strategy_random(
        df, "trend_pullback", STRATEGIES["trend_pullback"],
        {
            "trend_ma": (20, 50),
            "pullback_ma": (5, 20),
            "pullback_threshold": (-0.02, -0.0005),
            "require_volume": (True, False),
            "volume_min": (0.5, 3.0),
        }, min_trades=10, n_iterations=3000)
    
    # === Extreme strategies ===
    
    # 11. Extreme Oversold Bounce
    print("Optimizing Extreme Oversold Bounce...")
    all_results["extreme_oversold_bounce"] = optimize_strategy_random(
        df, "extreme_oversold_bounce", EXTREME_STRATEGIES["extreme_oversold_bounce"],
        {
            "rsi_max": (5, 30),
            "stoch_max": (5, 25),
            "bb_pos_max": (0.0, 0.2),
            "wick_min": (0.2, 0.8),
            "volume_min": (0.5, 5.0),
            "prev_drop_threshold": (0.003, 0.03),
            "n_prev_candles": (1, 5),
        }, min_trades=10, n_iterations=5000)
    
    # 12. Consecutive Drop Reversal
    print("Optimizing Consecutive Drop Reversal...")
    all_results["consecutive_drop_reversal"] = optimize_strategy_random(
        df, "consecutive_drop_reversal", EXTREME_STRATEGIES["consecutive_drop_reversal"],
        {
            "n_consecutive": (2, 5),
            "total_drop": (0.005, 0.03),
            "require_hammer": (True, False),
            "wick_min": (0.2, 0.8),
            "volume_min": (0.5, 4.0),
        }, min_trades=10, n_iterations=5000)
    
    # 13. Volume Exhaustion
    print("Optimizing Volume Exhaustion...")
    all_results["volume_exhaustion"] = optimize_strategy_random(
        df, "volume_exhaustion", EXTREME_STRATEGIES["volume_exhaustion"],
        {
            "volume_threshold": (2.0, 10.0),
            "prev_move_threshold": (0.005, 0.03),
            "reversal_body_min": (0.0, 0.01),
            "require_opposite_direction": (True, False),
            "n_prev": (1, 5),
        }, min_trades=10, n_iterations=5000)
    
    # 14. BB Squeeze Breakout
    print("Optimizing BB Squeeze Breakout...")
    all_results["bb_squeeze_breakout"] = optimize_strategy_random(
        df, "bb_squeeze_breakout", EXTREME_STRATEGIES["bb_squeeze_breakout"],
        {
            "bb_period": (10, 20),
            "squeeze_threshold": (0.005, 0.03),
            "breakout_threshold": (0.001, 0.01),
            "volume_min": (0.5, 4.0),
            "direction": ("both", "both"),
        }, min_trades=10, n_iterations=3000)
    
    # 15. Ultimate Confluence
    print("Optimizing Ultimate Confluence...")
    all_results["ultimate_confluence"] = optimize_strategy_random(
        df, "ultimate_confluence", EXTREME_STRATEGIES["ultimate_confluence"],
        {
            "rsi_max": (5, 35),
            "stoch_max": (5, 30),
            "bb_pos_max": (0.0, 0.25),
            "wick_min": (0.2, 0.8),
            "volume_min": (0.5, 5.0),
            "prev_candles_drop": (0.003, 0.03),
            "n_prev": (1, 5),
            "ma_dist_max": (-0.05, -0.001),
            "require_green": (True, False),
        }, min_trades=10, n_iterations=5000)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ROUND 2 SUMMARY")
    print(f"{'='*60}")
    best_overall = None
    for name, results in all_results.items():
        if results:
            best = results[0]
            print(f"{name:30s}: best={best['accuracy']:.3f} (n={best['n_trades']})")
            if best_overall is None or best["accuracy"] > best_overall["accuracy"]:
                best_overall = {"name": name, **best}
        else:
            print(f"{name:30s}: no valid results")
    
    if best_overall:
        print(f"\nBEST OVERALL: {best_overall['name']} = {best_overall['accuracy']:.3f} (n={best_overall['n_trades']})")
    
    return all_results

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    results = run_round_2(df)
    
    with open("optimization_round_2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to optimization_round_2.json")

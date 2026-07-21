"""Strategy optimizer with grid search and random search."""
import numpy as np
import pandas as pd
import json
from itertools import product
from strategies import STRATEGIES, evaluate_signal
import random

def optimize_strategy(df, strategy_name, param_grid, min_trades=50, top_n=5, 
                       search_type="grid", n_random=500):
    """
    Optimize a strategy's parameters.
    
    param_grid: dict of param_name -> list of values
    search_type: 'grid' or 'random'
    """
    strategy_func = STRATEGIES[strategy_name]
    results = []
    
    if search_type == "grid":
        # Compute total combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        total = np.prod([len(v) for v in values])
        
        if total > 50000:
            print(f"Grid too large ({total}), switching to random search with {n_random} iterations")
            search_type = "random"
        else:
            print(f"Grid search: {total} combinations for {strategy_name}")
            for i, combo in enumerate(product(*values)):
                params = dict(zip(keys, combo))
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
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1}/{total}")
    
    if search_type == "random":
        print(f"Random search: {n_random} iterations for {strategy_name}")
        for i in range(n_random):
            params = {k: random.choice(v) for k, v in param_grid.items()}
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
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_random}")
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results[:top_n]

def run_optimization_round(df, round_num=1, min_trades=30):
    """Run one round of optimization for all 10 strategies."""
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION ROUND {round_num}")
    print(f"{'='*60}\n")
    
    # Strategy 1: RSI Reversal
    print("Optimizing RSI Reversal...")
    results = optimize_strategy(df, "rsi_reversal", {
        "rsi_period": [7, 14, 21],
        "oversold": [10, 15, 20, 25, 30, 35],
        "overbought": [65, 70, 75, 80, 85, 90],
        "volume_min": [0.5, 1.0, 1.5, 2.0, 3.0],
    }, min_trades=min_trades, search_type="grid")
    all_results["rsi_reversal"] = results
    
    # Strategy 2: BB Reversal
    print("Optimizing BB Reversal...")
    results = optimize_strategy(df, "bb_reversal", {
        "bb_period": [10, 20],
        "bb_position_low": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        "bb_position_high": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        "require_wick": [True, False],
        "wick_threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
    }, min_trades=min_trades, search_type="grid")
    all_results["bb_reversal"] = results
    
    # Strategy 3: Stoch Extreme
    print("Optimizing Stoch Extreme...")
    results = optimize_strategy(df, "stoch_extreme", {
        "stoch_period": [7, 14],
        "low_threshold": [5, 10, 15, 20, 25],
        "high_threshold": [75, 80, 85, 90, 95],
        "use_d": [True, False],
        "d_threshold": [2, 3, 5, 10],
    }, min_trades=min_trades, search_type="grid")
    all_results["stoch_extreme"] = results
    
    # Strategy 4: Volume Momentum
    print("Optimizing Volume Momentum...")
    results = optimize_strategy(df, "volume_momentum", {
        "volume_threshold": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "momentum_period": [1, 2, 3, 5],
        "momentum_threshold": [0.000, 0.001, 0.002, 0.003, 0.005],
        "require_taker_buy": [True, False],
        "taker_threshold": [0.5, 0.55, 0.6, 0.65, 0.7],
    }, min_trades=min_trades, search_type="grid")
    all_results["volume_momentum"] = results
    
    # Strategy 5: MA Reversal
    print("Optimizing MA Reversal...")
    results = optimize_strategy(df, "ma_reversal", {
        "ma_period": [10, 20, 50],
        "dist_threshold": [-0.05, -0.03, -0.02, -0.01, -0.005],
        "overbought_dist": [0.005, 0.01, 0.02, 0.03, 0.05],
        "require_reversal": [True, False],
    }, min_trades=min_trades, search_type="grid")
    all_results["ma_reversal"] = results
    
    # Strategy 6: MACD Reversal
    print("Optimizing MACD Reversal...")
    results = optimize_strategy(df, "macd_reversal", {
        "macd_turn_threshold": [-0.0001, 0, 0.0001],
        "require_extreme": [True, False],
        "hist_extreme": [None, 0.001, 0.002, 0.005],
    }, min_trades=min_trades, search_type="grid")
    all_results["macd_reversal"] = results
    
    # Strategy 7: Candlestick
    print("Optimizing Candlestick...")
    results = optimize_strategy(df, "candlestick", {
        "body_threshold": [0.0005, 0.001, 0.002],
        "require_lower_wick": [True, False],
        "lower_wick_min": [0.3, 0.4, 0.5, 0.6, 0.7],
        "require_engulfing": [True, False],
        "prev_body_max": [0.0003, 0.0005, 0.001],
    }, min_trades=min_trades, search_type="grid")
    all_results["candlestick"] = results
    
    # Strategy 8: Confluence
    print("Optimizing Confluence...")
    results = optimize_strategy(df, "confluence", {
        "rsi_max": [20, 25, 30, 35, 40],
        "stoch_max": [15, 20, 25, 30],
        "bb_pos_max": [0.05, 0.1, 0.15, 0.2, 0.3],
        "volume_min": [0.5, 1.0, 1.5, 2.0],
        "require_all": [True, False],
    }, min_trades=min_trades, search_type="grid")
    all_results["confluence"] = results
    
    # Strategy 9: Mean Reversion Spike
    print("Optimizing Mean Reversion Spike...")
    results = optimize_strategy(df, "mean_reversion_spike", {
        "spike_period": [1, 2, 3],
        "spike_threshold": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02],
        "reversal_threshold": [0.000, 0.001, 0.002, 0.003, 0.005],
        "require_wick": [True, False],
        "wick_min": [0.2, 0.3, 0.4, 0.5],
    }, min_trades=min_trades, search_type="grid")
    all_results["mean_reversion_spike"] = results
    
    # Strategy 10: Trend Pullback
    print("Optimizing Trend Pullback...")
    results = optimize_strategy(df, "trend_pullback", {
        "trend_ma": [20, 50],
        "pullback_ma": [5, 10, 20],
        "pullback_threshold": [-0.01, -0.005, -0.003, -0.002, -0.001],
        "require_volume": [True, False],
        "volume_min": [1.0, 1.5, 2.0],
    }, min_trades=min_trades, search_type="grid")
    all_results["trend_pullback"] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ROUND {round_num} SUMMARY")
    print(f"{'='*60}")
    for name, results in all_results.items():
        if results:
            best = results[0]
            print(f"{name:25s}: best={best['accuracy']:.3f} (n={best['n_trades']})")
        else:
            print(f"{name:25s}: no valid results")
    
    return all_results

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    # Run first optimization round
    results = run_optimization_round(df, round_num=1, min_trades=30)
    
    # Save results
    with open("optimization_round_1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to optimization_round_1.json")

"""Round 2: Focused optimization on top strategies + ensemble testing."""
import numpy as np
import pandas as pd
import json
from strategies import STRATEGIES, evaluate_signal
from extreme_strategies import EXTREME_STRATEGIES
import random

random.seed(42)
np.random.seed(42)

def optimize_strategy_random(df, strategy_name, strategy_func, param_ranges, 
                              min_trades=5, n_iterations=1000, top_n=10):
    """Random search optimization - focused version."""
    results = []
    
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
        except Exception:
            pass
    
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results[:top_n]

def test_ensemble(df, strategies_with_params):
    """Test ensemble where all strategies must agree."""
    signals = []
    for name, (func, params) in strategies_with_params.items():
        sig = func(df, **params)
        signals.append(sig)
    
    # Only trade when all agree
    ensemble = pd.DataFrame({f"s{i}": s for i, s in enumerate(signals)})
    ensemble["sum"] = ensemble.sum(axis=1)
    
    # All buy signals
    buy = (ensemble["sum"] == len(strategies_with_params))
    # All sell signals  
    sell = (ensemble["sum"] == -len(strategies_with_params))
    
    df_test = df.copy()
    df_test["signal"] = 0
    df_test.loc[buy, "signal"] = 1
    df_test.loc[sell, "signal"] = -1
    
    return evaluate_signal(df_test, "signal", min_trades=5)

def test_pairwise_ensembles(df, top_results):
    """Test all pairs of top strategies."""
    pairs = []
    names = list(top_results.keys())
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            if not top_results[name1] or not top_results[name2]:
                continue
            
            best1 = top_results[name1][0]
            best2 = top_results[name2][0]
            
            func1 = STRATEGIES.get(name1) or EXTREME_STRATEGIES.get(name1)
            func2 = STRATEGIES.get(name2) or EXTREME_STRATEGIES.get(name2)
            
            result = test_ensemble(df, {
                name1: (func1, best1["params"]),
                name2: (func2, best2["params"])
            })
            
            if result["n_trades"] >= 5:
                pairs.append({
                    "pair": f"{name1} + {name2}",
                    **result
                })
    
    pairs.sort(key=lambda x: x["accuracy"], reverse=True)
    return pairs

def run_round_2_focused(df):
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"ROUND 2 - FOCUSED OPTIMIZATION")
    print(f"{'='*60}\n")
    
    # 1. Mean Reversion Spike - the best from round 1
    print("1. Mean Reversion Spike (top performer)...")
    all_results["mean_reversion_spike"] = optimize_strategy_random(
        df, "mean_reversion_spike", STRATEGIES["mean_reversion_spike"],
        {
            "spike_period": (1, 3),
            "spike_threshold": (0.001, 0.05),
            "reversal_threshold": (0.0, 0.02),
            "require_wick": (True, False),
            "wick_min": (0.1, 0.9),
        }, min_trades=5, n_iterations=2000)
    
    # 2. MA Reversal
    print("2. MA Reversal...")
    all_results["ma_reversal"] = optimize_strategy_random(
        df, "ma_reversal", STRATEGIES["ma_reversal"],
        {
            "ma_period": (10, 50),
            "dist_threshold": (-0.1, -0.001),
            "overbought_dist": (0.001, 0.1),
            "require_reversal": (True, False),
        }, min_trades=5, n_iterations=2000)
    
    # 3. RSI Reversal
    print("3. RSI Reversal...")
    all_results["rsi_reversal"] = optimize_strategy_random(
        df, "rsi_reversal", STRATEGIES["rsi_reversal"],
        {
            "rsi_period": (7, 21),
            "oversold": (5, 30),
            "overbought": (70, 95),
            "volume_min": (0.5, 5.0),
        }, min_trades=5, n_iterations=2000)
    
    # 4. Confluence
    print("4. Confluence...")
    all_results["confluence"] = optimize_strategy_random(
        df, "confluence", STRATEGIES["confluence"],
        {
            "rsi_max": (5, 35),
            "stoch_max": (5, 30),
            "bb_pos_max": (0.01, 0.3),
            "volume_min": (0.5, 4.0),
            "require_all": (True, False),
        }, min_trades=5, n_iterations=2000)
    
    # 5. Extreme Oversold Bounce
    print("5. Extreme Oversold Bounce...")
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
        }, min_trades=5, n_iterations=2000)
    
    # 6. Ultimate Confluence
    print("6. Ultimate Confluence...")
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
        }, min_trades=5, n_iterations=2000)
    
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
    
    # Test pairwise ensembles
    print(f"\n{'='*60}")
    print(f"TESTING PAIRWISE ENSEMBLES")
    print(f"{'='*60}")
    ensembles = test_pairwise_ensembles(df, all_results)
    if ensembles:
        for e in ensembles[:10]:
            print(f"{e['pair']:40s}: {e['accuracy']:.3f} (n={e['n_trades']})")
        all_results["ensembles"] = ensembles[:10]
    
    if best_overall:
        print(f"\nBEST SINGLE: {best_overall['name']} = {best_overall['accuracy']:.3f} (n={best_overall['n_trades']})")
    
    if ensembles and ensembles[0]["accuracy"] > (best_overall["accuracy"] if best_overall else 0):
        print(f"BEST ENSEMBLE: {ensembles[0]['pair']} = {ensembles[0]['accuracy']:.3f} (n={ensembles[0]['n_trades']})")
    
    return all_results

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    results = run_round_2_focused(df)
    
    with open("optimization_round_2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to optimization_round_2.json")

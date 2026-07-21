"""Round 3: Fine-tuning around best parameters + ensemble creation."""
import numpy as np
import pandas as pd
import json
from strategies import STRATEGIES, evaluate_signal
from extreme_strategies import EXTREME_STRATEGIES
import random

random.seed(123)
np.random.seed(123)

def fine_tune(df, strategy_name, strategy_func, base_params, param_variations,
              min_trades=5, top_n=10):
    """Fine-tune by varying base parameters slightly."""
    results = []
    
    # Generate variations
    for _ in range(500):
        params = base_params.copy()
        for key, (min_mult, max_mult) in param_variations.items():
            base_val = params[key]
            if isinstance(base_val, bool):
                continue
            elif isinstance(base_val, int):
                params[key] = max(1, int(base_val * random.uniform(min_mult, max_mult)))
            elif isinstance(base_val, float):
                params[key] = base_val * random.uniform(min_mult, max_mult)
        
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

def test_ensemble(df, strategies_with_params, require_agreement=2):
    """Test ensemble where at least N strategies must agree."""
    signals = []
    for name, (func, params) in strategies_with_params.items():
        sig = func(df, **params)
        signals.append(sig)
    
    ensemble = pd.DataFrame({f"s{i}": s for i, s in enumerate(signals)})
    
    # Count buy and sell signals per row
    buy_count = (ensemble == 1).sum(axis=1)
    sell_count = (ensemble == -1).sum(axis=1)
    
    df_test = df.copy()
    df_test["signal"] = 0
    df_test.loc[buy_count >= require_agreement, "signal"] = 1
    df_test.loc[sell_count >= require_agreement, "signal"] = -1
    
    return evaluate_signal(df_test, "signal", min_trades=5)

def run_round_3(df):
    with open("optimization_round_2.json", "r") as f:
        round2 = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"ROUND 3 - FINE TUNING + ENSEMBLES")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    # Fine-tune ma_reversal (100% accuracy)
    print("1. Fine-tuning MA Reversal...")
    base = round2["ma_reversal"][0]["params"]
    all_results["ma_reversal_ft"] = fine_tune(
        df, "ma_reversal", STRATEGIES["ma_reversal"], base,
        {
            "dist_threshold": (0.5, 1.5),
            "overbought_dist": (0.5, 1.5),
            "ma_period": (0.8, 1.2),
        }, min_trades=5)
    
    # Fine-tune extreme_oversold_bounce (100% accuracy)
    print("2. Fine-tuning Extreme Oversold Bounce...")
    base = round2["extreme_oversold_bounce"][0]["params"]
    all_results["extreme_oversold_bounce_ft"] = fine_tune(
        df, "extreme_oversold_bounce", EXTREME_STRATEGIES["extreme_oversold_bounce"], base,
        {
            "rsi_max": (0.7, 1.3),
            "stoch_max": (0.7, 1.3),
            "bb_pos_max": (0.5, 2.0),
            "wick_min": (0.7, 1.3),
            "volume_min": (0.7, 1.3),
            "prev_drop_threshold": (0.5, 1.5),
        }, min_trades=5)
    
    # Fine-tune mean_reversion_spike (85.7%)
    print("3. Fine-tuning Mean Reversion Spike...")
    base = round2["mean_reversion_spike"][0]["params"]
    all_results["mean_reversion_spike_ft"] = fine_tune(
        df, "mean_reversion_spike", STRATEGIES["mean_reversion_spike"], base,
        {
            "spike_threshold": (0.5, 1.5),
            "reversal_threshold": (0.5, 1.5),
            "wick_min": (0.5, 1.5),
        }, min_trades=5)
    
    # Fine-tune rsi_reversal (80%)
    print("4. Fine-tuning RSI Reversal...")
    base = round2["rsi_reversal"][0]["params"]
    all_results["rsi_reversal_ft"] = fine_tune(
        df, "rsi_reversal", STRATEGIES["rsi_reversal"], base,
        {
            "oversold": (0.7, 1.3),
            "overbought": (0.9, 1.1),
            "volume_min": (0.7, 1.3),
        }, min_trades=5)
    
    # Fine-tune confluence (80%)
    print("5. Fine-tuning Confluence...")
    base = round2["confluence"][0]["params"]
    all_results["confluence_ft"] = fine_tune(
        df, "confluence", STRATEGIES["confluence"], base,
        {
            "rsi_max": (0.7, 1.3),
            "stoch_max": (0.7, 1.3),
            "bb_pos_max": (0.5, 2.0),
            "volume_min": (0.7, 1.3),
        }, min_trades=5)
    
    # Test ensembles
    print("\n6. Testing Ensembles...")
    
    top_strategies = {
        "ma_reversal": (STRATEGIES["ma_reversal"], round2["ma_reversal"][0]["params"]),
        "extreme_oversold_bounce": (EXTREME_STRATEGIES["extreme_oversold_bounce"], round2["extreme_oversold_bounce"][0]["params"]),
        "mean_reversion_spike": (STRATEGIES["mean_reversion_spike"], round2["mean_reversion_spike"][0]["params"]),
        "rsi_reversal": (STRATEGIES["rsi_reversal"], round2["rsi_reversal"][0]["params"]),
        "confluence": (STRATEGIES["confluence"], round2["confluence"][0]["params"]),
    }
    
    ensemble_results = []
    for req in [2, 3, 4, 5]:
        result = test_ensemble(df, top_strategies, require_agreement=req)
        if result["n_trades"] >= 5:
            ensemble_results.append({
                "ensemble": f"Top 5 strategies (agreement={req})",
                **result
            })
            print(f"  Agreement={req}: {result['accuracy']:.3f} (n={result['n_trades']})")
    
    # Test 2-strategy ensembles
    names = list(top_strategies.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pair = {names[i]: top_strategies[names[i]], names[j]: top_strategies[names[j]]}
            result = test_ensemble(df, pair, require_agreement=2)
            if result["n_trades"] >= 5:
                ensemble_results.append({
                    "ensemble": f"{names[i]} + {names[j]}",
                    **result
                })
    
    ensemble_results.sort(key=lambda x: x["accuracy"], reverse=True)
    all_results["ensembles"] = ensemble_results[:15]
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ROUND 3 SUMMARY")
    print(f"{'='*60}")
    best_overall = None
    for name, results in all_results.items():
        if name == "ensembles":
            continue
        if results:
            best = results[0]
            print(f"{name:35s}: best={best['accuracy']:.3f} (n={best['n_trades']})")
            if best_overall is None or best["accuracy"] > best_overall["accuracy"]:
                best_overall = {"name": name, **best}
        else:
            print(f"{name:35s}: no valid results")
    
    if ensemble_results:
        best_ens = ensemble_results[0]
        print(f"\nBEST ENSEMBLE: {best_ens['ensemble']} = {best_ens['accuracy']:.3f} (n={best_ens['n_trades']})")
        if best_ens["accuracy"] > (best_overall["accuracy"] if best_overall else 0):
            best_overall = {"name": best_ens["ensemble"], **best_ens}
    
    if best_overall:
        print(f"\nBEST OVERALL: {best_overall['name']} = {best_overall['accuracy']:.3f} (n={best_overall['n_trades']})")
    
    return all_results

if __name__ == "__main__":
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    results = run_round_3(df)
    
    with open("optimization_round_3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to optimization_round_3.json")

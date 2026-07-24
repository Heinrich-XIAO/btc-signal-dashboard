"""Simulate predictions at each 5m candle boundary and backfill stats."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from api.binance_client import fetch_all_timeframes
from api.live_features import compute_live_features
from api.predictor import Predictor
from api.main import _wilson_score_interval

LOCAL_OFFSET = timedelta(hours=8)

def main():
    predictor = Predictor()
    print("Fetching data...")
    data = fetch_all_timeframes("BTCUSDT")
    df_5m = data.get("5m")
    if df_5m is None or df_5m.empty:
        print("No 5m data")
        return

    print(f"Got {len(df_5m)} 5m candles, latest: {df_5m.index[-1]}")

    # Backfill last 200 candles
    num_predictions = min(200, len(df_5m) - 52)
    end_idx = len(df_5m) - 2
    start_idx = end_idx - num_predictions + 1
    if start_idx < 50:
        start_idx = 50

    # Stats tracking
    correct = 0
    total_predictions = 0
    holds = 0
    total_candles = 0
    equity = 0
    peak = 0
    max_drawdown = 0.0
    true_positives = false_positives = true_negatives = false_negatives = 0
    equity_history = []

    print(f"\nSimulating {num_predictions} candles ({start_idx}..{end_idx})...\n")
    print(f"{'Time':<12} {'Signal':<6} {'Actual':<7} {'Result':<7} {'Equity':>6}")
    print("-" * 45)

    for offset in range(start_idx, end_idx + 1):
        candle_t = df_5m.iloc[offset]
        candle_t1 = df_5m.iloc[offset + 1]
        candle_t_time = df_5m.index[offset]

        # Predict at candle OPEN using data up to T-1 (exclude forming candle T)
        # This matches the live system: completed_data["5m"] = tf_df.iloc[:-1]
        df_slice = df_5m.iloc[:offset].copy()
        cutoff_time = candle_t_time

        sliced_data = {}
        for tf, tf_df in data.items():
            if tf_df is None:
                sliced_data[tf] = None
            elif tf == "5m":
                sliced_data[tf] = df_slice
            else:
                sliced_data[tf] = tf_df[tf_df.index <= cutoff_time]

        df_features = compute_live_features(sliced_data)
        if df_features is None or df_features.empty:
            continue

        prediction = predictor.predict(df_features)
        signal = prediction["signal"]

        # Actual outcome
        price_t = float(candle_t["close"])
        price_t1 = float(candle_t1["close"])
        actual_up = price_t1 > price_t
        actual = "UP" if actual_up else "DOWN"

        # Resolve
        total_candles += 1
        result = None
        if signal == "UP":
            total_predictions += 1
            if actual_up:
                result = "TP"
                correct += 1
                true_positives += 1
            else:
                result = "FP"
                false_positives += 1
        elif signal == "DOWN":
            total_predictions += 1
            if not actual_up:
                result = "TN"
                correct += 1
                true_negatives += 1
            else:
                result = "FN"
                false_negatives += 1
        else:
            holds += 1

        pnl = 0
        if result in ("TP", "TN"):
            pnl = 1
        elif result in ("FP", "FN"):
            pnl = -1
        equity += pnl
        equity_history.append(equity)
        if equity > peak:
            peak = equity
        dd = equity - peak
        if dd < max_drawdown:
            max_drawdown = dd

        symbol = "✓" if result in ("TP", "TN") else ("✗" if result in ("FP", "FN") else "-")
        china_time = (candle_t_time + LOCAL_OFFSET).strftime("%I:%M %p").lstrip("0")

        print(f"{china_time:<12} {signal:<6} {actual:<7} {symbol:<7} {equity:>6}")

    # Summary
    print("\n" + "=" * 45)
    print(f"Total candles: {total_candles}")
    print(f"Predictions: {total_predictions}  Holds: {holds}")
    print(f"TP={true_positives}  FP={false_positives}  TN={true_negatives}  FN={false_negatives}")
    if total_predictions > 0:
        acc = correct / total_predictions * 100
        cov = total_predictions / total_candles * 100
        ci_low, ci_high = _wilson_score_interval(correct, total_predictions)
        print(f"Accuracy: {correct}/{total_predictions} = {acc:.1f}% ±{(ci_high-ci_low)/2:.1f}")
        print(f"Coverage: {total_predictions}/{total_candles} = {cov:.1f}%")
    print(f"Equity: {equity}  Peak: {peak}  MaxDD: {max_drawdown}")

    # Write to stats.json for deployment
    stats = {
        "total_predictions": total_predictions,
        "total_candles": total_candles,
        "correct": correct,
        "accuracy": round(correct / total_predictions * 100, 1) if total_predictions > 0 else 0,
        "coverage": round(total_predictions / total_candles * 100, 1) if total_candles > 0 else 0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "holds": holds,
        "equity": equity,
        "peak": peak,
        "max_drawdown": max_drawdown,
    }
    stats_path = Path(__file__).parent / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"\nWrote stats to {stats_path}")


if __name__ == "__main__":
    main()

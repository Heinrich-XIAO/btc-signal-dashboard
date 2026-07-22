"""Simulate what the predictor would have said at the start of each candle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from api.binance_client import fetch_all_timeframes
from api.live_features import compute_live_features
from api.predictor import Predictor


# China = UTC+8
LOCAL_OFFSET = timedelta(hours=+8)


def parse_target_time(arg: str) -> datetime:
    """Parse a time string like '21:30' as today in UTC."""
    now = datetime.now(timezone.utc)
    parts = arg.split(":")
    hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target > now:
        target -= timedelta(days=1)
    return target


def main():
    predictor = Predictor()

    target_utc = None
    if len(sys.argv) > 1:
        target_utc = parse_target_time(sys.argv[1])
        china_time = target_utc + LOCAL_OFFSET
        print(f"Target time: {target_utc.strftime('%H:%M UTC')} ({china_time.strftime('%I:%M %p')} China)")

    print("Fetching data...")
    data = fetch_all_timeframes("BTCUSDT")
    df_5m = data.get("5m")
    if df_5m is None or df_5m.empty:
        print("No 5m data")
        return

    print(f"Got {len(df_5m)} 5m candles, latest: {df_5m.index[-1]}")

    num_predictions = min(200, len(df_5m) - 52)
    min_valid_idx = 50
    max_valid_idx = len(df_5m) - 2

    if target_utc:
        diffs = abs(df_5m.index - target_utc)
        closest_idx = diffs.argmin()
        end_idx = closest_idx
        start_idx = end_idx - num_predictions + 1
        if start_idx < min_valid_idx:
            start_idx = min_valid_idx
            end_idx = min_valid_idx + num_predictions - 1
        if end_idx > max_valid_idx:
            end_idx = max_valid_idx
            start_idx = max_valid_idx - num_predictions + 1
    else:
        end_idx = max_valid_idx
        start_idx = end_idx - num_predictions + 1
        if start_idx < min_valid_idx:
            start_idx = min_valid_idx

    # Stats tracking
    ens_tp = ens_fp = ens_tn = ens_fn = ens_hold = 0
    lr_tp = lr_fp = lr_tn = lr_fn = lr_hold = 0
    total = 0

    print(f"\nSimulating predictions at candle starts (candles {start_idx}..{end_idx})...\n")
    print(f"{'Time (China)':<12} {'Ensemble':<7} {'✓/✗':>4} │ {'LR_Comb':<7} {'✓/✗':>4} │ {'Actual':>7} {'Ens%':>7} {'LR%':>7}")
    print("-" * 75)

    for offset in range(start_idx, end_idx + 1):
        candle_t = df_5m.iloc[offset]
        candle_t1 = df_5m.iloc[offset + 1]
        candle_t_time = df_5m.index[offset]
        candle_t1_time = df_5m.index[offset + 1]

        # Predict at the START of candle T (using data up to end of T-1)
        # This matches how the live system should work: signal locked at candle open
        df_slice = df_5m.iloc[: offset].copy()
        cutoff_time = candle_t_time  # End of T-1

        sliced_data = {}
        for tf, tf_df in data.items():
            if tf_df is None:
                sliced_data[tf] = None
                continue
            if tf == "5m":
                sliced_data[tf] = df_slice
            else:
                sliced_data[tf] = tf_df[tf_df.index <= cutoff_time]

        df_features = compute_live_features(sliced_data)
        if df_features is None or df_features.empty:
            continue

        prediction = predictor.predict(df_features)
        ensemble_signal = prediction["signal"]
        ensemble_proba = prediction["ensemble_proba"]
        
        # Get lr_comb standalone signal using its own thresholds
        lr_comb_proba = prediction["models"]["lr_comb"]["proba"]
        lr_comb_up_thresh = predictor.meta["model_results"]["lr_comb"]["up_threshold"] * 100
        lr_comb_down_thresh = predictor.meta["model_results"]["lr_comb"]["down_threshold"] * 100
        
        if lr_comb_proba > lr_comb_up_thresh:
            lr_comb_signal = "UP"
        elif lr_comb_proba < lr_comb_down_thresh:
            lr_comb_signal = "DOWN"
        else:
            lr_comb_signal = "HOLD"

        # Actual: did T+1 close higher than T?
        price_t = float(candle_t["close"])
        price_t1 = float(candle_t1["close"])
        actual_up = price_t1 > price_t
        actual_signal = "UP" if actual_up else "DOWN"

        if ensemble_signal == "HOLD":
            ens_result = "-"
            ens_hold += 1
        elif ensemble_signal == actual_signal:
            ens_result = "✓"
            if actual_up:
                ens_tp += 1
            else:
                ens_tn += 1
        else:
            ens_result = "✗"
            if actual_up:
                ens_fn += 1
            else:
                ens_fp += 1
            
        if lr_comb_signal == "HOLD":
            lr_result = "-"
            lr_hold += 1
        elif lr_comb_signal == actual_signal:
            lr_result = "✓"
            if actual_up:
                lr_tp += 1
            else:
                lr_tn += 1
        else:
            lr_result = "✗"
            if actual_up:
                lr_fn += 1
            else:
                lr_fp += 1

        total += 1
        china_time = (candle_t_time + LOCAL_OFFSET).strftime("%I:%M %p").lstrip("0")

        print(
            f"{china_time:<12} {ensemble_signal:<7} {ens_result:>4} │ "
            f"{lr_comb_signal:<7} {lr_result:>4} │ {actual_signal:>7} "
            f"{ensemble_proba:>7.2f} {lr_comb_proba:>7.2f}"
        )

    print("-" * 75)

    # Summary
    ens_total_pred = ens_tp + ens_fp + ens_tn + ens_fn
    lr_total_pred = lr_tp + lr_fp + lr_tn + lr_fn
    ens_correct = ens_tp + ens_tn
    lr_correct = lr_tp + lr_tn

    print(f"\n{'='*75}")
    print(f"ENSEMBLE ({ens_total_pred} predictions, {ens_hold} holds, {total} total candles)")
    print(f"  TP={ens_tp}  FP={ens_fp}  TN={ens_tn}  FN={ens_fn}")
    if ens_total_pred > 0:
        print(f"  Accuracy: {ens_correct}/{ens_total_pred} = {ens_correct/ens_total_pred*100:.1f}%")
        print(f"  Coverage: {ens_total_pred}/{total} = {ens_total_pred/total*100:.1f}%")
    else:
        print(f"  No predictions made (all HOLD)")

    print(f"\nLR_COMB ({lr_total_pred} predictions, {lr_hold} holds, {total} total candles)")
    print(f"  TP={lr_tp}  FP={lr_fp}  TN={lr_tn}  FN={lr_fn}")
    if lr_total_pred > 0:
        print(f"  Accuracy: {lr_correct}/{lr_total_pred} = {lr_correct/lr_total_pred*100:.1f}%")
        print(f"  Coverage: {lr_total_pred}/{total} = {lr_total_pred/total*100:.1f}%")
    else:
        print(f"  No predictions made (all HOLD)")

    print(f"\nNote: Predictions are made at candle OPEN using data up to previous candle close.")
    print(f"Ensemble thresholds: UP>66.97%, DOWN<32.44% | LR_Comb thresholds: UP>65.08%, DOWN<34.59%")


if __name__ == "__main__":
    main()

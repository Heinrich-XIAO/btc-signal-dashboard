"""Precompute all features once, then grid-search thresholds for profit."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd
import numpy as np
from itertools import product

from api.live_features import compute_live_features
from api.predictor import Predictor

SPOT_BASE = "https://api.binance.com/api/v3"
FUTURES_BASE = "https://fapi.binance.com/fapi/v1"


def fetch_klines(symbol="BTCUSDT", interval="5m", limit=1000):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(f"{SPOT_BASE}/klines", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_volume",
        "taker_buy_quote_volume","ignore"
    ])
    for col in ["open","high","low","close","volume","quote_volume","trades","taker_buy_volume","taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open","high","low","close","volume","trades","taker_buy_volume","quote_volume"]]


def fetch_funding_rates(symbol="BTCUSDT", limit=100):
    params = {"symbol": symbol, "limit": limit}
    resp = requests.get(f"{FUTURES_BASE}/fundingRate", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["fundingRate"]]


def fetch_all(symbol="BTCUSDT"):
    print("Fetching 5m (2000 candles)...")
    df_5m_new = fetch_klines(symbol, "5m", 1000)
    oldest = df_5m_new.index[0]
    start_ms = int((oldest - pd.Timedelta(minutes=5)).timestamp() * 1000)
    params = {"symbol": symbol, "interval": "5m", "limit": 1000, "endTime": start_ms}
    resp = requests.get(f"{SPOT_BASE}/klines", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df_5m_old = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_volume",
        "taker_buy_quote_volume","ignore"
    ])
    for col in ["open","high","low","close","volume","quote_volume","trades","taker_buy_volume","taker_buy_quote_volume"]:
        df_5m_old[col] = pd.to_numeric(df_5m_old[col], errors="coerce")
    df_5m_old["timestamp"] = pd.to_datetime(df_5m_old["open_time"], unit="ms", utc=True)
    df_5m_old = df_5m_old.set_index("timestamp").sort_index()
    df_5m_old = df_5m_old[~df_5m_old.index.duplicated(keep="first")]
    df_5m_old = df_5m_old[["open","high","low","close","volume","trades","taker_buy_volume","quote_volume"]]
    df_5m = pd.concat([df_5m_old, df_5m_new]).sort_index()
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]

    print(f"5m: {len(df_5m)} candles ({df_5m.index[0]} to {df_5m.index[-1]})")
    return {
        "5m": df_5m,
        "1m": fetch_klines(symbol, "1m", 1500),
        "15m": fetch_klines(symbol, "15m", 1000),
        "1h": fetch_klines(symbol, "1h", 500),
        "4h": fetch_klines(symbol, "4h", 250),
        "funding": fetch_funding_rates(symbol, limit=100),
    }


def main():
    predictor = Predictor()
    data = fetch_all()
    df_5m = data["5m"]

    # Precompute features ONCE for the entire dataset
    print("\nPrecomputing features (one pass)...")
    df_features = compute_live_features(data)
    if df_features is None or df_features.empty:
        print("Failed to compute features")
        return
    print(f"Features: {len(df_features)} rows, {len(df_features.columns)} columns")

    # Precompute all probas using the LR_Comb model directly
    lr_model = predictor.lr_comb
    lr_scaler = predictor.scaler_comb
    feature_cols = predictor.combined_features

    probas = []
    valid_indices = []

    for i in range(len(df_features)):
        row = df_features.iloc[i]
        x = row[feature_cols].values.astype(float)
        if np.any(np.isnan(x)):
            x = np.nan_to_num(x, nan=0.0)
        if lr_scaler is not None:
            x = lr_scaler.transform(x.reshape(1, -1))
        p = lr_model.predict_proba(x)[0][1]
        probas.append(p)
        valid_indices.append(i)

    probas = np.array(probas)
    print(f"\nComputed {len(probas)} probas")
    print(f"LR_Comb proba dist: mean={probas.mean():.4f} std={probas.std():.4f} "
          f"min={probas.min():.4f} max={probas.max():.4f}")
    print(f"  25th={np.percentile(probas,25):.4f} 50th={np.percentile(probas,50):.4f} 75th={np.percentile(probas,75):.4f}")

    # Build trade data: for each prediction at offset i, actual = close[i+1] vs close[i]
    # df_features index aligns with df_5m index (same 5m timestamps)
    # We need offset 50..len-2 (need i+1 to exist)
    min_idx = 50
    max_idx = len(df_5m) - 2

    # Map feature index -> df_5m index
    # Features are computed from the full 5m slice, so feature row i corresponds to df_5m row i
    # But only if that row exists in df_features
    results = []
    for i in range(min_idx, max_idx + 1):
        if i not in valid_indices:
            continue
        proba_idx = valid_indices.index(i)
        p = probas[proba_idx]
        price_t = float(df_5m.iloc[i]["close"])
        price_t1 = float(df_5m.iloc[i + 1]["close"])
        results.append({"proba": p, "price_t": price_t, "price_t1": price_t1})

    print(f"\n{len(results)} valid prediction points")

    # Grid search
    print(f"\nGrid searching lr_comb thresholds (50-70% UP, 28-50% DOWN)...")
    best_score = -1
    grid = []

    for up_t in range(50, 72):
        for down_t in range(28, 51):
            if up_t <= down_t:
                continue

            trades = []
            for r in results:
                if r["proba"] > up_t / 100:
                    sig = "UP"
                elif r["proba"] < down_t / 100:
                    sig = "DOWN"
                else:
                    continue

                ret = (r["price_t1"] - r["price_t"]) / r["price_t"]
                if sig == "DOWN":
                    ret = -ret
                trades.append(ret)

            n = len(trades)
            if n < 10:
                continue

            arr = np.array(trades)
            wins = (arr > 0).sum()
            win_rate = wins / n * 100
            total_ret = float(arr.sum()) * 100
            avg_ret = float(arr.mean()) * 100
            std_ret = float(arr.std())
            sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0

            # Equity curve for max drawdown
            eq = np.cumprod(1 + arr)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / peak
            max_dd = float(dd.max()) * 100

            # Profit factor
            gross_profit = arr[arr > 0].sum() if (arr > 0).any() else 0
            gross_loss = abs(arr[arr <= 0].sum()) if (arr <= 0).any() else 0.00001
            pf = gross_profit / gross_loss

            # Calmar
            calmar = float((eq[-1] - 1)) / (max_dd / 100) if max_dd > 0 else 0

            # Score: profit factor * sqrt(n) * sharpe (balance profitability + frequency + consistency)
            score = pf * min(n, 500) ** 0.3 * max(sharpe, 0)

            grid.append({
                "up": up_t, "down": down_t, "n": n,
                "win_rate": win_rate, "total_ret": total_ret,
                "avg_ret": avg_ret, "sharpe": sharpe,
                "max_dd": max_dd, "pf": pf, "calmar": calmar, "score": score,
            })

            if score > best_score:
                best_score = score
                best = grid[-1]

    grid.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*95}")
    print(f"TOP 20 THRESHOLDS BY PROFIT SCORE (PF * n^0.3 * Sharpe)")
    print(f"{'='*95}")
    print(f"{'UP%':>5} {'DOWN%':>6} {'#Trades':>8} {'WinR%':>7} {'TotRet%':>9} {'AvgRet%':>9} "
          f"{'MaxDD%':>8} {'Sharpe':>7} {'PF':>6} {'Calmar':>8} {'Score':>7}")
    print("-" * 95)
    for g in grid[:20]:
        print(f"{g['up']:>5} {g['down']:>6} {g['n']:>8} {g['win_rate']:>7.1f} "
              f"{g['total_ret']:>9.3f} {g['avg_ret']:>9.3f} {g['max_dd']:>8.2f} "
              f"{g['sharpe']:>7.3f} {g['pf']:>6.2f} {g['calmar']:>8.2f} {g['score']:>7.1f}")

    # Accuracy-optimized
    acc_grid = [g for g in grid if g["n"] >= 15]
    acc_grid.sort(key=lambda x: x["win_rate"], reverse=True)
    print(f"\n{'='*95}")
    print(f"TOP 10 BY WIN RATE (15+ trades)")
    print(f"{'='*95}")
    print(f"{'UP%':>5} {'DOWN%':>6} {'#Trades':>8} {'WinR%':>7} {'TotRet%':>9} {'AvgRet%':>9} "
          f"{'MaxDD%':>8} {'Sharpe':>7} {'PF':>6}")
    print("-" * 85)
    for g in acc_grid[:10]:
        print(f"{g['up']:>5} {g['down']:>6} {g['n']:>8} {g['win_rate']:>7.1f} "
              f"{g['total_ret']:>9.3f} {g['avg_ret']:>9.3f} {g['max_dd']:>8.2f} "
              f"{g['sharpe']:>7.3f} {g['pf']:>6.2f}")

    # Current thresholds
    cur_up = predictor.meta["model_results"]["lr_comb"]["up_threshold"] * 100
    cur_down = predictor.meta["model_results"]["lr_comb"]["down_threshold"] * 100
    print(f"\n{'='*95}")
    print(f"CURRENT THRESHOLDS: UP>{cur_up:.2f}%  DOWN<{cur_down:.2f}%")
    print(f"{'='*95}")
    cur_trades = []
    for r in results:
        if r["proba"] > cur_up / 100:
            sig = "UP"
        elif r["proba"] < cur_down / 100:
            sig = "DOWN"
        else:
            continue
        ret = (r["price_t1"] - r["price_t"]) / r["price_t"]
        if sig == "DOWN":
            ret = -ret
        cur_trades.append(ret)
    if cur_trades:
        arr = np.array(cur_trades)
        eq = np.cumprod(1 + arr)
        pk = np.maximum.accumulate(eq)
        mdd = float(((pk - eq) / pk).max()) * 100
        gp = arr[arr > 0].sum() if (arr > 0).any() else 0
        gl = abs(arr[arr <= 0].sum()) if (arr <= 0).any() else 0.00001
        print(f"  Trades={len(cur_trades)}  WinR={((arr>0).sum()/len(cur_trades)*100):.1f}%  "
              f"TotRet={arr.sum()*100:.3f}%  MaxDD={mdd:.2f}%  "
              f"Sharpe={arr.mean()/arr.std():.3f}  PF={gp/gl:.2f}")

    # Show expected return by probability bucket
    print(f"\n{'='*95}")
    print(f"EXPECTED RETURN BY LR_COMB PROBABILITY BUCKET")
    print(f"{'='*95}")
    print(f"{'Bucket':>12} {'Trades':>7} {'WinR%':>7} {'AvgRet%':>9} {'TotRet%':>9} {'Sharpe':>7}")
    print("-" * 60)
    for lo in range(0, 101, 5):
        hi = lo + 5
        bucket = [(r["price_t1"] - r["price_t"]) / r["price_t"]
                  if r["proba"] > lo/100 and r["proba"] <= hi/100
                  else None
                  for r in results]
        bucket = [x for x in bucket if x is not None]
        if len(bucket) >= 5:
            arr = np.array(bucket)
            wr = (arr > 0).sum() / len(arr) * 100
            print(f"  ({lo:3d}-{hi:3d}] {len(bucket):>7} {wr:>7.1f} {arr.mean()*100:>+9.4f} "
                  f"{arr.sum()*100:>9.3f} {arr.mean()/arr.std():>7.3f}" if arr.std() > 0 else
                  f"  ({lo:3d}-{hi:3d}] {len(bucket):>7} {wr:>7.1f} {arr.mean()*100:>+9.4f} "
                  f"{arr.sum()*100:>9.3f} {'N/A':>7}")

    print(f"\nBEST PROFIT THRESHOLDS: UP>{best['up']}%  DOWN<{best['down']}%")
    print(f"  → {best['n']} trades, {best['win_rate']:.1f}% win rate, "
          f"{best['total_ret']:.3f}% total return, PF={best['pf']:.2f}")


if __name__ == "__main__":
    main()

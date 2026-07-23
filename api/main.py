"""FastAPI backend with WebSocket live predictions and live accuracy tracking."""
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.binance_client import fetch_all_timeframes
from api.live_features import compute_live_features
from api.predictor import Predictor

def _wilson_score_interval(correct: int, total: int, z: float = 1.96):
    """Compute Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) as percentages (0-100).
    If total == 0, returns (0, 100).
    """
    if total == 0:
        return (0.0, 100.0)
    p_hat = correct / total
    denominator = 1 + z ** 2 / total
    center = (p_hat + z ** 2 / (2 * total)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * total)) / total) / denominator
    lower = max(0.0, (center - spread) * 100)
    upper = min(100.0, (center + spread) * 100)
    return (round(lower, 1), round(upper, 1))


app = FastAPI(title="BTC Signal Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
predictor = Predictor()
latest_prediction: Optional[dict] = None
latest_price: Optional[dict] = None
clients: List[WebSocket] = []

# Track which candle we last predicted for
last_predicted_candle_time: Optional[int] = None

# Live stats tracking
# pending_signals: keyed by the candle open_time (ms) we predicted for
# Value is the prediction dict with signal, timestamp, price
pending_signals: Dict[int, dict] = {}

# Resolved signals history
resolved_signals: List[dict] = []
max_resolved = 500

# Live stats
live_stats = {
    "total_predictions": 0,      # Total times we predicted (UP + DOWN)
    "total_candles": 0,        # Total candles observed (for coverage)
    "correct": 0,              # Correct predictions
    "accuracy": 0.0,
    "coverage": 0.0,
    "true_positives": 0,       # Predicted UP, actual UP
    "false_positives": 0,      # Predicted UP, actual DOWN
    "true_negatives": 0,       # Predicted DOWN, actual DOWN
    "false_negatives": 0,      # Predicted DOWN, actual UP
    "holds": 0,                # Times we held (no signal)
    "equity": 0,               # Running equity score (+1 for correct, -1 for wrong)
    "peak": 0,                 # Highest equity reached
    "max_drawdown": 0.0,      # Max drawdown from peak (negative %)
}

# Equity history for sparkline (list of ints)
equity_history: List[int] = []

# Stats persistence
STATS_FILE = Path(__file__).parent / "stats.json"
RESOLVED_FILE = Path(__file__).parent / "resolved.json"


def _load_stats():
    """Load persisted stats if available."""
    global live_stats, resolved_signals, equity_history
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                loaded = json.load(f)
                # Equity history is stored separately, not in live_stats dict
                equity_history = loaded.pop("equity_history", [])
                live_stats.update(loaded)
        except Exception:
            pass
    if RESOLVED_FILE.exists():
        try:
            with open(RESOLVED_FILE, "r") as f:
                resolved_signals = json.load(f)[-max_resolved:]
        except Exception:
            pass


def _save_stats():
    """Persist stats to disk."""
    try:
        with open(STATS_FILE, "w") as f:
            to_save = live_stats.copy()
            to_save["equity_history"] = equity_history[-500:]
            json.dump(to_save, f)
        with open(RESOLVED_FILE, "w") as f:
            json.dump(resolved_signals[-max_resolved:], f)
    except Exception as e:
        print(f"Failed to save stats: {e}")


def _compute_next_candle_countdown() -> int:
    """Seconds until next 5m candle close."""
    now = datetime.now(timezone.utc)
    minutes = now.minute
    seconds = now.second
    next_5m = ((minutes // 5) + 1) * 5
    if next_5m == 60:
        return (60 - minutes) * 60 - seconds
    return (next_5m - minutes) * 60 - seconds


def _resolve_pending_signals(df_5m: Any):
    """Resolve ALL pending predictions whose verification candle has closed.

    A prediction made during candle T (keyed by T's open time) says whether
    candle T+1 will close higher than candle T. We verify after T+1 closes,
    i.e. when the dataframe has a candle at T+2 (the current forming candle).

    For each pending signal keyed by T (ms), we find T and T+1 in the
    dataframe and resolve if T+1 is a completed candle.
    """
    global live_stats, resolved_signals, pending_signals, equity_history

    if df_5m is None or len(df_5m) < 3:
        return

    # Build a map of candle time (ms) -> close price for all completed candles
    # (exclude the last one which is still forming)
    completed = df_5m.iloc[:-1]
    candle_map = {}
    for ts in completed.index:
        key = int(ts.timestamp() * 1000)
        candle_map[key] = float(completed.loc[ts, "close"])

    # Find all resolvable signals
    resolved_any = False
    for key in list(pending_signals.keys()):
        # We need candle T (key) and candle T+1 (key + 300000) to both exist
        close_t = candle_map.get(key)
        close_t_plus_1 = candle_map.get(key + 5 * 60 * 1000)  # T+1 is 5min later

        if close_t is None or close_t_plus_1 is None:
            continue

        signal_info = pending_signals.pop(key)
        predicted_signal = signal_info.get("signal", "HOLD")
        actual_up = close_t_plus_1 > close_t

        result = None
        if predicted_signal == "UP":
            live_stats["total_predictions"] += 1
            if actual_up:
                result = "TP"
                live_stats["correct"] += 1
                live_stats["true_positives"] += 1
            else:
                result = "FP"
                live_stats["false_positives"] += 1
        elif predicted_signal == "DOWN":
            live_stats["total_predictions"] += 1
            if not actual_up:
                result = "TN"
                live_stats["correct"] += 1
                live_stats["true_negatives"] += 1
            else:
                result = "FN"
                live_stats["false_negatives"] += 1
        else:  # HOLD
            live_stats["holds"] += 1

        live_stats["total_candles"] += 1

        pnl = 0
        if result in ("TP", "TN"):
            pnl = 1
        elif result in ("FP", "FN"):
            pnl = -1
        live_stats["equity"] += pnl
        equity_history.append(live_stats["equity"])
        if len(equity_history) > 500:
            equity_history.pop(0)
        if live_stats["equity"] > live_stats["peak"]:
            live_stats["peak"] = live_stats["equity"]
        drawdown = live_stats["equity"] - live_stats["peak"]
        if drawdown < live_stats["max_drawdown"]:
            live_stats["max_drawdown"] = drawdown

        resolved_any = True
        resolved = {
            "predicted_at": signal_info.get("timestamp"),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "predicted_signal": predicted_signal,
            "actual": "UP" if actual_up else "DOWN",
            "result": result,
            "price_at_prediction": signal_info.get("price"),
            "price_now": close_t_plus_1,
            "confidence": signal_info.get("confidence"),
            "equity": live_stats["equity"],
            "drawdown": drawdown,
        }
        resolved_signals.append(resolved)
        if len(resolved_signals) > max_resolved:
            resolved_signals.pop(0)

        print(
            f"Resolved: predicted {predicted_signal}, actual {'UP' if actual_up else 'DOWN'} -> {result} | "
            f"Accuracy: {live_stats['accuracy']}% ({live_stats['correct']}/{live_stats['total_predictions']}) | "
            f"Coverage: {live_stats['coverage']}% | "
            f"Equity: {live_stats['equity']} | MaxDD: {live_stats['max_drawdown']}"
        )

    if resolved_any:
        # Recalculate derived stats
        if live_stats["total_predictions"] > 0:
            live_stats["accuracy"] = round(
                live_stats["correct"] / live_stats["total_predictions"] * 100, 1
            )
        total_observed = live_stats["total_predictions"] + live_stats["holds"]
        if total_observed > 0:
            live_stats["coverage"] = round(
                live_stats["total_predictions"] / total_observed * 100, 1
            )
        _save_stats()


async def _run_prediction_cycle():
    """Fetch data, compute features, run predictions, resolve past signals, broadcast."""
    global latest_prediction, latest_price, last_predicted_candle_time

    try:
        data = fetch_all_timeframes("BTCUSDT")
        df_5m = data.get("5m")

        if df_5m is None or df_5m.empty:
            print("No 5m data available")
            return

        latest_price = {
            "price": float(df_5m["close"].iloc[-1]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Resolve any pending predictions from previous candles
        _resolve_pending_signals(df_5m)

        # Only predict if this is a new candle we haven't seen yet
        current_candle_time = int(df_5m.index[-1].timestamp() * 1000)
        
        if current_candle_time == last_predicted_candle_time:
            # Same candle, just broadcast current state without re-predicting
            if latest_prediction:
                latest_prediction["price"] = latest_price["price"]
                latest_prediction["timestamp"] = datetime.now(timezone.utc).isoformat()
                latest_prediction["countdown"] = _compute_next_candle_countdown()

                # Refresh stats snapshot
                stats_snapshot = live_stats.copy()
                n = stats_snapshot["total_predictions"]
                stats_snapshot["ci_low"], stats_snapshot["ci_high"] = _wilson_score_interval(
                    stats_snapshot["correct"], n
                )
                n_cov = stats_snapshot["total_predictions"] + stats_snapshot["holds"]
                stats_snapshot["cov_ci_low"], stats_snapshot["cov_ci_high"] = _wilson_score_interval(
                    stats_snapshot["total_predictions"], n_cov
                )
                stats_snapshot["pending_count"] = len(pending_signals)
                stats_snapshot["equity_history"] = equity_history[-100:]
                latest_prediction["live_stats"] = stats_snapshot

                message = json.dumps({"type": "prediction", "data": latest_prediction})
                disconnected = []
                for client in clients:
                    try:
                        await client.send_text(message)
                    except Exception:
                        disconnected.append(client)
                
                for client in disconnected:
                    if client in clients:
                        clients.remove(client)
            return

        # New candle - generate prediction at the start
        last_predicted_candle_time = current_candle_time
        print(f"New candle at {df_5m.index[-1]} UTC - generating prediction...")

        # Compute features using ONLY completed candles (exclude current forming candle)
        # This matches how the model was trained - on completed candle data
        completed_data = {}
        current_candle_start = df_5m.index[-1]
        for tf, tf_df in data.items():
            if tf_df is None:
                completed_data[tf] = None
            elif tf == "5m":
                # Exclude last (forming) candle
                completed_data[tf] = tf_df.iloc[:-1].copy()
            else:
                # Exclude any data from the current candle's time window
                completed_data[tf] = tf_df[tf_df.index < current_candle_start].copy()
        
        df_features = compute_live_features(completed_data)
        if df_features is None or df_features.empty:
            print("Feature computation failed")
            return

        # Run prediction (at candle start using only prior completed candles)
        prediction = predictor.predict(df_features)
        prediction["timestamp"] = datetime.now(timezone.utc).isoformat()
        prediction["countdown"] = _compute_next_candle_countdown()
        prediction["price"] = latest_price["price"]

        # Enrich stats with Wilson CI for both accuracy and coverage
        stats_snapshot = live_stats.copy()
        n_acc = stats_snapshot["total_predictions"]
        stats_snapshot["ci_low"], stats_snapshot["ci_high"] = _wilson_score_interval(
            stats_snapshot["correct"], n_acc
        )
        n_cov = stats_snapshot["total_predictions"] + stats_snapshot["holds"]
        stats_snapshot["cov_ci_low"], stats_snapshot["cov_ci_high"] = _wilson_score_interval(
            stats_snapshot["total_predictions"], n_cov
        )
        stats_snapshot["pending_count"] = len(pending_signals)
        stats_snapshot["equity_history"] = equity_history[-100:]
        prediction["live_stats"] = stats_snapshot

        # Store this prediction as pending (keyed by current candle time)
        pending_signals[current_candle_time] = {
            "timestamp": prediction["timestamp"],
            "signal": prediction["signal"],
            "price": prediction["price"],
            "confidence": prediction["confidence"],
        }

        # Clean old pending signals that can't be resolved (candles fell off dataframe)
        cutoff = current_candle_time - 30 * 60 * 1000
        for key in list(pending_signals.keys()):
            if key < cutoff:
                pending_signals.pop(key, None)

        latest_prediction = prediction

        # Broadcast to all connected clients
        message = json.dumps({"type": "prediction", "data": prediction})
        disconnected = []
        for client in clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            if client in clients:
                clients.remove(client)

    except Exception as e:
        print(f"Prediction cycle error: {e}")
        import traceback
        traceback.print_exc()


async def prediction_loop():
    """Background loop: run predictions once at the start of each 5m candle."""
    while True:
        # Sleep until the next 5-minute boundary
        now = datetime.now(timezone.utc)
        seconds = now.minute * 60 + now.second
        next_5m_seconds = ((seconds // 300) + 1) * 300
        sleep_time = next_5m_seconds - seconds
        if sleep_time <= 0:
            sleep_time = 300
        await asyncio.sleep(sleep_time)
        await _run_prediction_cycle()


async def stats_broadcast_loop():
    """Background loop: broadcast updated stats every 30 seconds (no re-prediction)."""
    while True:
        await asyncio.sleep(30)
        try:
            if latest_prediction and clients:
                stats_snapshot = live_stats.copy()
                n = stats_snapshot["total_predictions"]
                stats_snapshot["ci_low"], stats_snapshot["ci_high"] = _wilson_score_interval(
                    stats_snapshot["correct"], n
                )
                n_cov = stats_snapshot["total_predictions"] + stats_snapshot["holds"]
                stats_snapshot["cov_ci_low"], stats_snapshot["cov_ci_high"] = _wilson_score_interval(
                    stats_snapshot["total_predictions"], n_cov
                )
                stats_snapshot["pending_count"] = len(pending_signals)
                stats_snapshot["equity_history"] = equity_history[-100:]

                msg = json.dumps({"type": "stats_update", "live_stats": stats_snapshot})
                disconnected = []
                for client in clients:
                    try:
                        await client.send_text(msg)
                    except Exception:
                        disconnected.append(client)
                for client in disconnected:
                    if client in clients:
                        clients.remove(client)
        except Exception as e:
            print(f"Stats broadcast error: {e}")
        await asyncio.sleep(30)


@app.on_event("startup")
async def startup():
    _load_stats()
    asyncio.create_task(prediction_loop())
    asyncio.create_task(stats_broadcast_loop())


@app.get("/api/signal")
async def get_signal():
    """REST endpoint for current signal."""
    if latest_prediction is None:
        return {"error": "No prediction available yet"}
    return latest_prediction


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get recent resolved prediction history."""
    return resolved_signals[-limit:]


@app.get("/api/stats")
async def get_stats():
    """Get live accuracy stats with Wilson score confidence interval."""
    n = live_stats["total_predictions"]
    correct = live_stats["correct"]
    ci_low, ci_high = _wilson_score_interval(correct, n)
    n_cov = live_stats["total_predictions"] + live_stats["holds"]
    cov_ci_low, cov_ci_high = _wilson_score_interval(live_stats["total_predictions"], n_cov)
    result = live_stats.copy()
    result["ci_low"] = ci_low
    result["ci_high"] = ci_high
    result["cov_ci_low"] = cov_ci_low
    result["cov_ci_high"] = cov_ci_high
    result["pending_count"] = len(pending_signals)
    result["equity_history"] = equity_history[-100:]
    return result


@app.post("/api/reset")
async def reset_stats():
    """Reset all live stats and history."""
    global live_stats, resolved_signals, pending_signals, equity_history

    live_stats.update({
        "total_predictions": 0,
        "total_candles": 0,
        "correct": 0,
        "accuracy": 0.0,
        "coverage": 0.0,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "holds": 0,
        "equity": 0,
        "peak": 0,
        "max_drawdown": 0.0,
    })
    resolved_signals.clear()
    pending_signals.clear()
    equity_history.clear()

    # Delete persisted files
    try:
        if STATS_FILE.exists():
            STATS_FILE.unlink()
        if RESOLVED_FILE.exists():
            RESOLVED_FILE.unlink()
    except Exception as e:
        print(f"Failed to delete stat files: {e}")

    return {"success": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)

    # Send current state immediately
    if latest_prediction:
        await websocket.send_text(json.dumps({"type": "prediction", "data": latest_prediction}))

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
    except Exception:
        if websocket in clients:
            clients.remove(websocket)


# Serve built frontend
BUILD_DIR = Path(__file__).parent.parent / "dashboard" / "dist"
if BUILD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(BUILD_DIR / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        if path.startswith("api/") or path == "ws":
            return {"detail": "Not Found"}
        index_file = BUILD_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"detail": "Frontend not built yet"}

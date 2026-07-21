"""FastAPI backend with WebSocket live predictions and live accuracy tracking."""
import asyncio
import json
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
}

# Stats persistence
STATS_FILE = Path(__file__).parent / "stats.json"
RESOLVED_FILE = Path(__file__).parent / "resolved.json"


def _load_stats():
    """Load persisted stats if available."""
    global live_stats, resolved_signals
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                loaded = json.load(f)
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
            json.dump(live_stats, f)
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
    """Check if any pending predictions can now be resolved."""
    global live_stats, resolved_signals, pending_signals

    if df_5m is None or len(df_5m) < 2:
        return

    # The latest candle just closed or is still forming
    # We need at least 2 candles to compare
    latest_candle_time = int(df_5m.index[-1].timestamp() * 1000)  # open time in ms
    latest_close = float(df_5m["close"].iloc[-1])
    prev_close = float(df_5m["close"].iloc[-2])
    actual_up = latest_close > prev_close

    # Check if we have any pending prediction for the candle that just preceded this one
    # The prediction was made for the transition from candle[-2] to candle[-1]
    # We use the open time of candle[-2] as the key
    prev_candle_time = int(df_5m.index[-2].timestamp() * 1000)

    # Find and resolve matching pending signal
    signal_to_resolve = None
    for key in list(pending_signals.keys()):
        # Resolve any signal that was for a candle before the latest one
        if key <= prev_candle_time:
            signal_to_resolve = pending_signals.pop(key)
            break

    if signal_to_resolve is None:
        # Also check if there's a signal for the exact prev candle
        if prev_candle_time in pending_signals:
            signal_to_resolve = pending_signals.pop(prev_candle_time)

    if signal_to_resolve:
        predicted_signal = signal_to_resolve.get("signal", "HOLD")

        # Determine correctness
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

        # Calculate derived stats
        if live_stats["total_predictions"] > 0:
            live_stats["accuracy"] = round(
                live_stats["correct"] / live_stats["total_predictions"] * 100, 1
            )
        total_observed = live_stats["total_predictions"] + live_stats["holds"]
        if total_observed > 0:
            live_stats["coverage"] = round(
                live_stats["total_predictions"] / total_observed * 100, 1
            )

        # Record resolved signal
        resolved = {
            "predicted_at": signal_to_resolve.get("timestamp"),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "predicted_signal": predicted_signal,
            "actual": "UP" if actual_up else "DOWN",
            "result": result,
            "price_at_prediction": signal_to_resolve.get("price"),
            "price_now": latest_close,
            "confidence": signal_to_resolve.get("confidence"),
        }
        resolved_signals.append(resolved)
        if len(resolved_signals) > max_resolved:
            resolved_signals.pop(0)

        _save_stats()
        print(
            f"Resolved: predicted {predicted_signal}, actual {'UP' if actual_up else 'DOWN'} -> {result} | "
            f"Accuracy: {live_stats['accuracy']}% ({live_stats['correct']}/{live_stats['total_predictions']}) | "
            f"Coverage: {live_stats['coverage']}%"
        )


async def _run_prediction_cycle():
    """Fetch data, compute features, run predictions, resolve past signals, broadcast."""
    global latest_prediction, latest_price

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

        # Compute features
        df_features = compute_live_features(data)
        if df_features is None or df_features.empty:
            print("Feature computation failed")
            return

        # Run prediction
        prediction = predictor.predict(df_features)
        prediction["timestamp"] = datetime.now(timezone.utc).isoformat()
        prediction["countdown"] = _compute_next_candle_countdown()
        prediction["price"] = latest_price["price"]
        prediction["live_stats"] = live_stats.copy()

        # Store this prediction as pending (keyed by current candle time)
        current_candle_time = int(df_5m.index[-1].timestamp() * 1000)
        pending_signals[current_candle_time] = {
            "timestamp": prediction["timestamp"],
            "signal": prediction["signal"],
            "price": prediction["price"],
            "confidence": prediction["confidence"],
        }

        # Clean old pending signals (older than 15 minutes)
        cutoff = current_candle_time - 15 * 60 * 1000
        for key in list(pending_signals.keys()):
            if key < cutoff:
                pending_signals.pop(key, None)
                live_stats["holds"] += 1
                live_stats["total_candles"] += 1

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
    """Background loop: run predictions every 30 seconds."""
    while True:
        await _run_prediction_cycle()
        await asyncio.sleep(30)


@app.on_event("startup")
async def startup():
    _load_stats()
    asyncio.create_task(prediction_loop())


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
    """Get live accuracy stats."""
    return live_stats


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

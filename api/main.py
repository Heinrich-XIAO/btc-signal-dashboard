"""FastAPI backend with WebSocket live predictions."""
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.binance_client import fetch_all_timeframes
from api.live_features import compute_live_features
from api.predictor import Predictor

app = FastAPI(title="BTC Signal Dashboard API")

# CORS - allow all origins for now (lock down in production)
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
history: List[dict] = []
max_history = 100
clients: List[WebSocket] = []

# Try to load existing history
HISTORY_FILE = Path(__file__).parent / "history.json"
if HISTORY_FILE.exists():
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except Exception:
        history = []


def _save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-max_history:], f)
    except Exception as e:
        print(f"Failed to save history: {e}")


def _compute_next_candle_countdown() -> int:
    """Seconds until next 5m candle close."""
    now = datetime.now(timezone.utc)
    minutes = now.minute
    seconds = now.second
    next_5m = ((minutes // 5) + 1) * 5
    if next_5m == 60:
        return (60 - minutes) * 60 - seconds
    return (next_5m - minutes) * 60 - seconds


async def _run_prediction_cycle():
    """Fetch data, compute features, run predictions, broadcast."""
    global latest_prediction, latest_price

    try:
        data = fetch_all_timeframes("BTCUSDT")
        df_5m = data.get("5m")

        if df_5m is None or df_5m.empty:
            print("No 5m data available")
            return

        latest_price = {
            "price": float(df_5m["close"].iloc[-1]),
            "change_24h": None,  # Would need 24h comparison
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

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

        # Add to history if signal is UP or DOWN
        if prediction["signal"] in ("UP", "DOWN"):
            history_entry = {
                "timestamp": prediction["timestamp"],
                "price": prediction["price"],
                "signal": prediction["signal"],
                "confidence": prediction["confidence"],
                "ensemble_proba": prediction["ensemble_proba"],
            }
            # Avoid duplicates within same minute
            if not history or history[-1]["timestamp"] != history_entry["timestamp"]:
                history.append(history_entry)
                if len(history) > max_history:
                    history.pop(0)
                _save_history()

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
    # Start background loop (first prediction will happen immediately in the task)
    asyncio.create_task(prediction_loop())


@app.get("/api/signal")
async def get_signal():
    """REST endpoint for current signal."""
    if latest_prediction is None:
        return {"error": "No prediction available yet"}
    return latest_prediction


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get recent prediction history."""
    return history[-limit:]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)

    # Send current state immediately
    if latest_prediction:
        await websocket.send_text(json.dumps({"type": "prediction", "data": latest_prediction}))

    try:
        while True:
            # Keep connection alive, handle pings if needed
            data = await websocket.receive_text()
            # Echo back as ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
    except Exception:
        if websocket in clients:
            clients.remove(websocket)


# Serve built frontend (for VPS deployment)
BUILD_DIR = Path(__file__).parent.parent / "dashboard" / "dist"
if BUILD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(BUILD_DIR / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        # API routes handled above
        if path.startswith("api/") or path == "ws":
            return {"detail": "Not Found"}
        index_file = BUILD_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"detail": "Frontend not built yet"}

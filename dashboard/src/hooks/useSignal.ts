import { useState, useEffect, useRef, useCallback } from 'react';
import type { PredictionData, HistoryEntry } from '../types';

const WS_URL = import.meta.env.DEV
  ? `ws://${window.location.hostname}:8000/ws`
  : `ws://${window.location.host}/ws`;

const API_URL = import.meta.env.DEV
  ? 'http://localhost:8000/api'
  : `/api`;

export function useSignal() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/history?limit=20`);
      const data = await res.json();
      setHistory(data);
    } catch (e) {
      console.error('Failed to fetch history:', e);
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      fetchHistory();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'prediction' && msg.data) {
          setPrediction(msg.data);
          setLastUpdate(new Date());
        }
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [fetchHistory]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Keep-alive ping
  useEffect(() => {
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  return { prediction, history, connected, lastUpdate };
}

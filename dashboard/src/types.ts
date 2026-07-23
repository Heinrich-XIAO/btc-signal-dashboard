export interface ModelVote {
  proba: number;
  signal: 'UP' | 'DOWN' | 'HOLD';
}

export interface LiveStats {
  total_predictions: number;
  total_candles: number;
  correct: number;
  accuracy: number;
  coverage: number;
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
  holds: number;
  equity: number;
  peak: number;
  max_drawdown: number;
  ci_low: number;
  ci_high: number;
  cov_ci_low: number;
  cov_ci_high: number;
  pending_count: number;
  equity_history: number[];
}

export interface PredictionData {
  signal: 'UP' | 'DOWN' | 'HOLD';
  confidence: number;
  ensemble_proba: number;
  up_threshold: number;
  down_threshold: number;
  up_votes: number;
  down_votes: number;
  total_models: number;
  models: Record<string, ModelVote>;
  timestamp?: string;
  countdown?: number;
  price?: number;
  live_stats?: LiveStats;
}

export interface HistoryEntry {
  timestamp?: string;
  predicted_at?: string;
  resolved_at?: string;
  price?: number;
  signal?: 'UP' | 'DOWN';
  confidence?: number;
  ensemble_proba?: number;
  predicted_signal?: 'UP' | 'DOWN';
  actual?: 'UP' | 'DOWN';
  result?: 'TP' | 'FP' | 'TN' | 'FN';
  equity?: number;
  drawdown?: number;
}

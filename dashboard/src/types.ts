export interface ModelVote {
  proba: number;
  signal: 'UP' | 'DOWN' | 'HOLD';
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
}

export interface HistoryEntry {
  timestamp: string;
  price: number;
  signal: 'UP' | 'DOWN';
  confidence: number;
  ensemble_proba: number;
  result?: 'win' | 'loss';
}

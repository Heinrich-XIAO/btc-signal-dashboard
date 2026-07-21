import { SignalPill } from './SignalPill';
import { CountdownTimer } from './CountdownTimer';
import { PriceHeader } from './PriceHeader';
import type { PredictionData, HistoryEntry } from '../types';

interface MobileSignalBarProps {
  prediction: PredictionData | null;
  history: HistoryEntry[];
  connected: boolean;
}

const MODEL_NAMES: Record<string, string> = {
  lr_comb: 'LR',
  xgb: 'XGB',
  lgb: 'LGB',
  lr_all: 'LR2',
  cat: 'CAT',
  rf: 'RF',
  hgb: 'HGB',
};

export function MobileSignalBar({ prediction, connected }: MobileSignalBarProps) {
  return (
    <div className="flex flex-col h-screen bg-bg">
      <PriceHeader prediction={prediction} connected={connected} />

      <div className="flex-1 flex flex-col items-center justify-center gap-6 px-4">
        {prediction ? (
          <>
            <SignalPill prediction={prediction} size="lg" />

            <div className="flex flex-col items-center gap-1">
              <span className="text-text-dim text-sm">
                {prediction.confidence > 0
                  ? `${prediction.confidence.toFixed(1)}% confident`
                  : 'No clear signal'}
              </span>
              <CountdownTimer seconds={prediction.countdown} />
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-4 animate-pulse">
            <div className="w-32 h-14 bg-surface-2 rounded-full" />
            <div className="w-24 h-4 bg-surface-2 rounded" />
          </div>
        )}
      </div>

      <div className="border-t border-border bg-surface p-3">
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
          {prediction ? (
            Object.entries(prediction.models).map(([key, model]) => (
              <div
                key={key}
                className={`flex-shrink-0 px-3 py-1.5 rounded-lg border text-xs font-medium ${
                  model.signal === 'UP'
                    ? 'bg-up/10 border-up/30 text-up'
                    : model.signal === 'DOWN'
                    ? 'bg-down/10 border-down/30 text-down'
                    : 'bg-surface-2 border-border text-text-dim'
                }`}
              >
                <span className="block text-center">{MODEL_NAMES[key] || key}</span>
                <span className="block text-center opacity-70">{model.proba.toFixed(0)}%</span>
              </div>
            ))
          ) : (
            <div className="flex gap-2">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="w-12 h-10 bg-surface-2 rounded-lg" />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import { Activity } from 'lucide-react';
import type { PredictionData } from '../types';

interface PriceHeaderProps {
  prediction: PredictionData | null;
  connected: boolean;
}

export function PriceHeader({ prediction, connected }: PriceHeaderProps) {
  const price = prediction?.price;

  return (
    <div className="flex items-center justify-between px-4 py-3 bg-surface border-b border-border">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-up animate-pulse' : 'bg-down'}`} />
          <span className="text-xs text-text-dim uppercase tracking-wider">
            {connected ? 'Live' : 'Disconnected'}
          </span>
        </div>
        <h1 className="text-sm font-semibold text-text">
          BTC/USDT
        </h1>
      </div>

      <div className="flex items-center gap-4">
        {price ? (
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold font-mono text-text">
              ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
        ) : (
          <div className="h-6 w-24 bg-surface-2 rounded animate-pulse" />
        )}
        <Activity size={16} className="text-text-dim" />
      </div>
    </div>
  );
}

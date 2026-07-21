import { SignalPill } from './SignalPill';
import { CountdownTimer } from './CountdownTimer';
import { PriceHeader } from './PriceHeader';
import type { PredictionData, HistoryEntry } from '../types';

interface DesktopDashboardProps {
  prediction: PredictionData | null;
  history: HistoryEntry[];
  connected: boolean;
}

const MODEL_NAMES: Record<string, string> = {
  lr_comb: 'LR Combined',
  xgb: 'XGBoost',
  lgb: 'LightGBM',
  lr_all: 'LR All',
  cat: 'CatBoost',
  rf: 'Random Forest',
  hgb: 'HistGradient',
};

export function DesktopDashboard({ prediction, history, connected }: DesktopDashboardProps) {
  const stats = prediction?.live_stats;

  return (
    <div className="flex flex-col h-screen bg-bg overflow-hidden">
      <PriceHeader prediction={prediction} connected={connected} />

      <div className="flex-1 grid grid-cols-12 gap-4 p-4 overflow-hidden">
        {/* Left panel: Model breakdown + Stats */}
        <div className="col-span-3 flex flex-col gap-3 overflow-y-auto">
          <div className="bg-surface rounded-xl border border-border p-4">
            <h3 className="text-sm font-semibold text-text-dim uppercase tracking-wider mb-3">
              Model Votes
            </h3>
            <div className="flex flex-col gap-2">
              {prediction ? (
                Object.entries(prediction.models).map(([key, model]) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-sm text-text">{MODEL_NAMES[key] || key}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-surface-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            model.signal === 'UP' ? 'bg-up' : model.signal === 'DOWN' ? 'bg-down' : 'bg-hold'
                          }`}
                          style={{ width: `${model.proba}%` }}
                        />
                      </div>
                      <SignalPill prediction={{ ...prediction, signal: model.signal, confidence: 0 }} size="sm" />
                    </div>
                  </div>
                ))
              ) : (
                <div className="space-y-2">
                  {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                    <div key={i} className="h-6 bg-surface-2 rounded animate-pulse" />
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="bg-surface rounded-xl border border-border p-4">
            <h3 className="text-sm font-semibold text-text-dim uppercase tracking-wider mb-3">
              Live Stats
            </h3>
            {stats ? (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-dim">Live Accuracy</span>
                  <span className={`font-mono font-semibold ${stats.accuracy >= 50 ? 'text-up' : 'text-down'}`}>
                    {stats.accuracy.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs text-text-dim">
                  <span>{stats.correct}/{stats.total_predictions} correct</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-dim">Coverage</span>
                  <span className="text-text font-mono">{stats.coverage.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-dim">Signals / Holds</span>
                  <span className="text-text font-mono">{stats.total_predictions} / {stats.holds}</span>
                </div>

                <div className="border-t border-border pt-2 mt-2">
                  <p className="text-xs text-text-dim uppercase tracking-wider mb-2">Confusion Matrix</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-up/10 rounded p-2 text-center">
                      <div className="text-up font-bold text-lg">{stats.true_positives}</div>
                      <div className="text-text-dim">True Positives</div>
                    </div>
                    <div className="bg-down/10 rounded p-2 text-center">
                      <div className="text-down font-bold text-lg">{stats.false_positives}</div>
                      <div className="text-text-dim">False Positives</div>
                    </div>
                    <div className="bg-down/10 rounded p-2 text-center">
                      <div className="text-down font-bold text-lg">{stats.false_negatives}</div>
                      <div className="text-text-dim">False Negatives</div>
                    </div>
                    <div className="bg-up/10 rounded p-2 text-center">
                      <div className="text-up font-bold text-lg">{stats.true_negatives}</div>
                      <div className="text-text-dim">True Negatives</div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-5 bg-surface-2 rounded animate-pulse" />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center panel: Main signal */}
        <div className="col-span-5 flex flex-col gap-4">
          <div className="bg-surface rounded-xl border border-border p-6 flex flex-col items-center justify-center flex-1">
            {prediction ? (
              <>
                <SignalPill prediction={prediction} size="lg" />
                <div className="mt-4 text-center">
                  <p className="text-3xl font-bold font-mono text-text">
                    {prediction.confidence.toFixed(1)}%
                  </p>
                  <p className="text-text-dim text-sm mt-1">confidence</p>
                </div>
                <div className="mt-6">
                  <CountdownTimer seconds={prediction.countdown} />
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center gap-4 animate-pulse">
                <div className="w-40 h-16 bg-surface-2 rounded-full" />
                <div className="w-24 h-8 bg-surface-2 rounded" />
                <div className="w-20 h-5 bg-surface-2 rounded" />
              </div>
            )}
          </div>

          <div className="bg-surface rounded-xl border border-border p-4">
            <h3 className="text-sm font-semibold text-text-dim uppercase tracking-wider mb-3">
              Vote Distribution
            </h3>
            {prediction ? (
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="flex justify-between text-xs text-text-dim mb-1">
                    <span>UP {prediction.up_votes}/{prediction.total_models}</span>
                    <span>DOWN {prediction.down_votes}/{prediction.total_models}</span>
                  </div>
                  <div className="flex h-3 rounded-full overflow-hidden bg-surface-2">
                    <div
                      className="bg-up"
                      style={{ width: `${(prediction.up_votes / prediction.total_models) * 100}%` }}
                    />
                    <div
                      className="bg-down"
                      style={{ width: `${(prediction.down_votes / prediction.total_models) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-10 bg-surface-2 rounded animate-pulse" />
            )}
          </div>
        </div>

        {/* Right panel: Resolved History */}
        <div className="col-span-4 flex flex-col gap-3">
          <div className="bg-surface rounded-xl border border-border p-4 flex-1 overflow-hidden flex flex-col">
            <h3 className="text-sm font-semibold text-text-dim uppercase tracking-wider mb-3">
              Resolved Predictions
            </h3>
            <div className="flex-1 overflow-y-auto">
              {history.length > 0 ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-text-dim text-xs uppercase">
                      <th className="text-left pb-2">Time</th>
                      <th className="text-center pb-2">Pred</th>
                      <th className="text-center pb-2">Actual</th>
                      <th className="text-center pb-2">Result</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {[...history].reverse().map((entry, i) => (
                      <tr key={i} className="text-text">
                        <td className="py-2 text-text-dim font-mono text-xs">
                          {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td className="py-2 text-center">
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
                              entry.signal === 'UP'
                                ? 'bg-up/20 text-up'
                                : 'bg-down/20 text-down'
                            }`}
                          >
                            {entry.signal}
                          </span>
                        </td>
                        <td className="py-2 text-center text-xs">
                          <span className={entry.actual === 'UP' ? 'text-up' : 'text-down'}>
                            {entry.actual}
                          </span>
                        </td>
                        <td className="py-2 text-center">
                          <span
                            className={`inline-block w-5 h-5 rounded-full text-xs leading-5 font-bold ${
                              entry.result === 'TP' || entry.result === 'TN'
                                ? 'bg-up/20 text-up'
                                : entry.result === 'FP' || entry.result === 'FN'
                                ? 'bg-down/20 text-down'
                                : 'bg-hold/20 text-hold'
                            }`}
                          >
                            {entry.result === 'TP' || entry.result === 'TN' ? '✓' : entry.result === 'FP' || entry.result === 'FN' ? '✗' : '-'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="text-text-dim text-sm text-center py-8">
                  Waiting for candles to close and predictions to resolve...
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

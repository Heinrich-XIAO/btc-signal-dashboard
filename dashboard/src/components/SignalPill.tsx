import type { PredictionData } from '../types';

interface SignalPillProps {
  prediction: PredictionData;
  size?: 'sm' | 'md' | 'lg';
}

export function SignalPill({ prediction, size = 'md' }: SignalPillProps) {
  const { signal, confidence } = prediction;

  const colors = {
    UP: 'bg-up/20 text-up border-up/50',
    DOWN: 'bg-down/20 text-down border-down/50',
    HOLD: 'bg-hold/20 text-hold border-hold/50',
  };

  const glows = {
    UP: 'shadow-[0_0_30px_rgba(34,197,94,0.3)]',
    DOWN: 'shadow-[0_0_30px_rgba(239,68,68,0.3)]',
    HOLD: 'shadow-[0_0_30px_rgba(107,114,128,0.2)]',
  };

  const sizes = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-6 py-2 text-lg',
    lg: 'px-10 py-4 text-4xl font-bold',
  };

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border ${colors[signal]} ${sizes[size]} ${size === 'lg' ? glows[signal] : ''}`}
    >
      {signal === 'UP' && (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M7 17l5-5 5 5M12 12V3" />
        </svg>
      )}
      {signal === 'DOWN' && (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M7 7l5 5 5-5M12 12v9" />
        </svg>
      )}
      <span>{signal}</span>
      {confidence > 0 && size !== 'sm' && (
        <span className="opacity-80 text-sm font-normal">
          {confidence.toFixed(1)}%
        </span>
      )}
    </div>
  );
}

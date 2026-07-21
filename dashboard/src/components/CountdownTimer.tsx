import { useEffect, useState } from 'react';

interface CountdownTimerProps {
  seconds?: number;
}

export function CountdownTimer({ seconds }: CountdownTimerProps) {
  const [remaining, setRemaining] = useState(seconds ?? 300);

  useEffect(() => {
    if (seconds !== undefined) {
      setRemaining(seconds);
    }
  }, [seconds]);

  useEffect(() => {
    const interval = setInterval(() => {
      setRemaining((r) => (r > 0 ? r - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const mins = Math.floor(remaining / 60);
  const secs = remaining % 60;

  return (
    <div className="flex items-center gap-1.5 text-text-dim text-sm font-mono">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 6v6l4 2" />
      </svg>
      <span>
        {mins}:{secs.toString().padStart(2, '0')}
      </span>
    </div>
  );
}

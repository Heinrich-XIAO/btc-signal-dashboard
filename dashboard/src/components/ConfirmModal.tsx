import { useState, useRef } from 'react';

interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  description: string;
  confirmPhrase: string;
  actionLabel: string;
}

export function ConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  confirmPhrase,
  actionLabel,
}: ConfirmModalProps) {
  const [input, setInput] = useState('');
  const [error, setError] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() === confirmPhrase) {
      onConfirm();
      setInput('');
      setError(false);
    } else {
      setError(true);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-surface border border-down/50 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl shadow-down/20">
        <div className="flex items-center gap-2 mb-3">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <h2 className="text-lg font-bold text-down">{title}</h2>
        </div>

        <p className="text-text-dim text-sm mb-4">{description}</p>

        <div className="bg-down/10 border border-down/30 rounded-lg p-3 mb-4">
          <p className="text-xs text-text-dim mb-1">Type exactly:</p>
          <code className="block text-sm font-mono text-down break-all select-all">
            {confirmPhrase}
          </code>
        </div>

        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              setError(false);
            }}
            autoFocus
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            className={`w-full bg-surface-2 border rounded-lg px-3 py-2 text-text font-mono text-sm mb-3 focus:outline-none focus:ring-2 ${
              error
                ? 'border-down focus:ring-down/50'
                : 'border-border focus:ring-accent/50'
            }`}
            placeholder="Type the confirmation phrase..."
          />

          {error && (
            <p className="text-down text-xs mb-3">
              Phrase does not match. Try again.
            </p>
          )}

          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => {
                setInput('');
                setError(false);
                onClose();
              }}
              className="flex-1 px-4 py-2 rounded-lg border border-border text-text-dim text-sm hover:bg-surface-2 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={input.trim() !== confirmPhrase}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                input.trim() === confirmPhrase
                  ? 'bg-down text-white hover:bg-down/90'
                  : 'bg-down/30 text-down/50 cursor-not-allowed'
              }`}
            >
              {actionLabel}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

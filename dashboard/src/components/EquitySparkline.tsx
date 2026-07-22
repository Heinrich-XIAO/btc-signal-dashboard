interface EquitySparklineProps {
  data: number[];
  width?: number;
  height?: number;
}

export function EquitySparkline({ data, width = 200, height = 40 }: EquitySparklineProps) {
  if (data.length < 2) {
    return (
      <div
        className="bg-surface-2 rounded flex items-center justify-center text-text-dim text-xs"
        style={{ width, height }}
      >
        Collecting data...
      </div>
    );
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = 2;
  const innerW = width - padding * 2;
  const innerH = height - padding * 2;

  const points = data.map((v, i) => {
    const x = padding + (i / (data.length - 1)) * innerW;
    const y = padding + innerH - ((v - min) / range) * innerH;
    return `${x},${y}`;
  });

  const polyline = points.join(' ');

  const lastVal = data[data.length - 1];
  const color = lastVal >= 0 ? '#22c55e' : '#ef4444';

  // Zero line position
  const zeroY = padding + innerH - ((0 - min) / range) * innerH;
  const showZero = min < 0 && max > 0;

  return (
    <svg width={width} height={height} className="rounded">
      <rect width={width} height={height} fill="rgb(30,30,40)" rx="4" />
      {showZero && (
        <line
          x1={padding}
          y1={zeroY}
          x2={width - padding}
          y2={zeroY}
          stroke="rgb(80,80,100)"
          strokeWidth="1"
          strokeDasharray="3,3"
        />
      )}
      <polyline
        points={polyline}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={padding + innerW}
        cy={padding + innerH - ((lastVal - min) / range) * innerH}
        r="3"
        fill={color}
      />
    </svg>
  );
}

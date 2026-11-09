interface MeterProps {
  label: string;
  /** 0..1 fill fraction. */
  fraction: number;
  /** Text shown on the right (e.g. "0.82" or "+12 Hz"). */
  value: string;
  /** Signed meters fill from the centre; unsigned fill from the left. */
  signed?: boolean;
  /** Sign for colouring a signed meter. */
  positive?: boolean;
}

export function Meter({ label, fraction, value, signed = false, positive = true }: MeterProps) {
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  const toneClass = signed ? (positive ? "meter-pos" : "meter-neg") : "meter-neutral";

  return (
    <div className="meter">
      <div className="meter-head">
        <span className="meter-label">{label}</span>
        <span className="meter-value">{value}</span>
      </div>
      <div className={`meter-track ${signed ? "signed" : ""}`}>
        {signed && <span className="meter-center" aria-hidden />}
        <span
          className={`meter-fill ${toneClass} ${signed ? (positive ? "from-center-right" : "from-center-left") : ""}`}
          style={{ width: `${signed ? pct / 2 : pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={label}
        />
      </div>
    </div>
  );
}

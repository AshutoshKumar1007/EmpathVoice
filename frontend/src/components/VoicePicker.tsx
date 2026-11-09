import type { Voice } from "../types";

interface Props {
  voices: Voice[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}

export function VoicePicker({ voices, value, onChange, disabled }: Props) {
  return (
    <div className="voice-picker">
      <label htmlFor="voice">Voice</label>
      <select
        id="voice"
        value={value}
        disabled={disabled || voices.length === 0}
        onChange={(e) => onChange(e.target.value)}
      >
        {voices.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
          </option>
        ))}
      </select>
    </div>
  );
}

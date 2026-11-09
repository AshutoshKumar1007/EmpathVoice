import { EXAMPLES, MAX_CHARS } from "../constants";

interface Props {
  value: string;
  onChange: (text: string) => void;
  disabled?: boolean;
}

export function TextComposer({ value, onChange, disabled }: Props) {
  const over = value.length > MAX_CHARS;

  return (
    <section className="composer" aria-label="Text to speak">
      <div className="textarea-wrap">
        <label htmlFor="text" className="sr-only">
          Text to convert to speech
        </label>
        <textarea
          id="text"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Type something with feeling — a message, a line, a thought…"
          rows={5}
          aria-describedby="charcount"
        />
        <span id="charcount" className={`char-count ${over ? "over" : ""}`}>
          {value.length} / {MAX_CHARS}
        </span>
      </div>

      <div className="examples" role="group" aria-label="Example inputs">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            className="chip"
            disabled={disabled}
            onClick={() => onChange(ex.text)}
          >
            {ex.label}
          </button>
        ))}
      </div>
    </section>
  );
}

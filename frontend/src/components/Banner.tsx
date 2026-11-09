interface Props {
  kind: "error" | "info";
  message: string;
  onDismiss?: () => void;
}

export function Banner({ kind, message, onDismiss }: Props) {
  return (
    <div className={`banner banner-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span>{message}</span>
      {onDismiss && (
        <button type="button" className="banner-close" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  );
}

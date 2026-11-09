interface Props {
  url: string;
  expiresIn: number;
}

export function AudioPlayer({ url, expiresIn }: Props) {
  const minutes = Math.round(expiresIn / 60);
  return (
    <section className="player" aria-label="Generated audio">
      {/* key=url forces the element to reload when a new clip is generated */}
      <audio key={url} controls autoPlay src={url} className="audio-el">
        Your browser does not support audio playback.
      </audio>
      <div className="player-actions">
        <a className="download-btn" href={url} download>
          Download MP3
        </a>
        <span className="expiry-note">Link expires in ~{minutes} min</span>
      </div>
    </section>
  );
}

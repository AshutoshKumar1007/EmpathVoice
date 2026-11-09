// Mirror of the backend prosody clamp bounds (app/emotion/prosody.py), used
// only to scale the visualization bars. Display-only; not sent to the server.
export const PITCH_HZ_MAX = 40;
export const RATE_PCT_MAX = 35;
export const ENERGY_PCT_MAX = 30;

export const MAX_CHARS = 2000;

export const EXAMPLES: { label: string; text: string }[] = [
  { label: "Joyful", text: "I can't believe it — we actually won! This is the best day ever!" },
  { label: "Somber", text: "It's been quiet since they left. I keep expecting them to walk back in." },
  { label: "Urgent", text: "Please, we need to move now — there isn't much time left before it closes." },
  { label: "Tender", text: "Rest now. I'm right here, and everything is going to be alright." },
  { label: "Calm", text: "Take a slow breath. The water is still, and there's nowhere else to be." },
];

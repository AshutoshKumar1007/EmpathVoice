async function generate() {
  const textEl = document.getElementById("text");
  const player = document.getElementById("player");

  if (!textEl || !player) {
    return;
  }

  const text = textEl.value.trim();
  if (!text) {
    alert("Please enter text first.");
    return;
  }

  try {
    const url = `/speak?text=${encodeURIComponent(text)}`;
    const response = await fetch(url, { method: "POST" });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(err || `Request failed with status ${response.status}`);
    }

    const payload = await response.json();
    if (!payload.audio) {
      throw new Error("Backend did not return an audio path.");
    }

    // Cache-bust so repeated generations are always fresh.
    player.src = `${payload.audio}?t=${Date.now()}`;
    await player.play();
  } catch (error) {
    console.error("Generate failed:", error);
    alert("Could not generate audio. Check backend logs for details.");
  }
}

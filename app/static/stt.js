/* Speech-to-text: record a short microphone clip with the MediaRecorder API
 * and POST the raw audio bytes to /api/v1/chat/stt, then drop the transcribed
 * text into the chat `message` signal.
 *
 * Loaded only from chat.html. Reads/writes the Datastar signals through the
 * `mergePatch` export of the vendored datastar.js client.
 */
import { mergePatch } from "/static/datastar.js";

const STT_URL = "/api/v1/chat/stt";
const MAX_RECORD_MS = 60_000;
const MIC_SELECTOR = "[data-mic-button]";

let recorder = null;
let mediaStream = null;
let chunks = [];
let stopTimer = null;
let starting = false;

function pickMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function setMicState(button, state) {
  button.dataset.micState = state;
  button.setAttribute("aria-pressed", String(state === "recording"));
}

function setStatus(text) {
  mergePatch({ stt_status: text || "", stt_error: "" });
}

function setError(text) {
  mergePatch({ stt_status: "", stt_error: text });
}

function stopTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

async function transcribe(button, blob) {
  setMicState(button, "transcribing");
  setStatus("Transcribing…");
  try {
    const response = await fetch(STT_URL, {
      method: "POST",
      headers: { "Datastar-Request": "true" },
      body: blob,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setError(payload.detail || "Speech-to-text failed. Try again.");
      return;
    }
    const text = (payload.text || "").trim();
    if (text) {
      mergePatch({ message: text, stt_status: "", stt_error: "" });
    } else {
      setError("No speech was detected. Try again.");
    }
  } catch {
    setError("Speech-to-text failed. Try again.");
  } finally {
    setMicState(button, "idle");
  }
}

function finishRecording() {
  clearTimeout(stopTimer);
  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  }
}

async function startRecording(button) {
  if (starting) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setError("Speech recognition is not supported in this browser.");
    return;
  }
  starting = true;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setError("Microphone access was denied. Allow it and try again.");
    return;
  } finally {
    starting = false;
  }

  const mimeType = pickMimeType();
  recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
  chunks = [];
  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };
  recorder.onerror = () => setError("Recording failed. Try again.");
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    chunks = [];
    stopTracks();
    transcribe(button, blob);
  };
  recorder.start();
  setMicState(button, "recording");
  setStatus("Listening…");
  stopTimer = setTimeout(finishRecording, MAX_RECORD_MS);
}

document.addEventListener("click", (event) => {
  const button = event.target.closest(MIC_SELECTOR);
  if (!button) return;
  event.preventDefault();
  if (button.dataset.micState === "recording") {
    finishRecording();
  } else {
    startRecording(button);
  }
});

/* Speak page helpers: live character counter, MP3 download with a loading
 * state, and automatic voice selection based on the pasted text's language.
 *
 * Loaded only from speak.html. The analysis itself is server-side; this only
 * wires up the small pieces of client-side polish. formatReadingTime lives in
 * an inline script in speak.html so it exists before Datastar scans the page.
 */
import { mergePatch } from "/static/datastar.js";

const DOWNLOAD_SELECTOR = ".speak-download-btn";
const DOWNLOAD_FILENAME = "transcript.mp3";

/* Lightweight language hints. Script ranges handle the big non-Latin families;
 * Latin-based languages are told apart by a few common stop words. Good enough
 * to pick a Piper voice automatically — it only runs when the detected
 * language actually changes, so it never fights the user's manual choice.
 */
const SCRIPT_RANGES = [
  [0x0900, 0x097f, "hi"], // Devanagari (Hindi, Marathi…)
  [0x0600, 0x06ff, "ar"], // Arabic
  [0x0590, 0x05ff, "he"], // Hebrew
  [0x0400, 0x04ff, "ru"], // Cyrillic
  [0x0e00, 0x0e7f, "th"], // Thai
  [0x0e80, 0x0eff, "lo"], // Lao
  [0x3040, 0x30ff, "ja"], // Hiragana / Katakana
  [0x4e00, 0x9fff, "zh"], // CJK unified ideographs
  [0xac00, 0xd7af, "ko"], // Hangul
];

const STOPWORDS = {
  en: new Set(["the", "and", "is", "a", "to", "of", "in", "for", "this", "that", "with", "you", "it", "we", "on", "be"]),
  de: new Set(["der", "die", "das", "und", "ist", "ein", "eine", "nicht", "ich", "sie", "es", "mit", "auf", "auch"]),
  es: new Set(["el", "la", "los", "las", "y", "que", "de", "en", "es", "un", "una", "por", "con", "se", "no"]),
  fr: new Set(["le", "la", "les", "et", "des", "une", "est", "pas", "pour", "dans", "avec", "sur", "que", "je"]),
  it: new Set(["il", "lo", "la", "i", "gli", "e", "che", "di", "non", "una", "per", "con", "sono", "questo"]),
  pt: new Set(["o", "a", "os", "as", "e", "que", "de", "em", "para", "não", "um", "uma", "por", "com", "se"]),
  nl: new Set(["de", "het", "een", "en", "is", "van", "niet", "op", "met", "dat", "voor", "in", "je"]),
  sv: new Set(["och", "att", "det", "som", "en", "är", "på", "med", "för", "av", "inte", "till", "den"]),
};

function detectLanguage(text) {
  const sample = (text || "").trim();
  if (!sample) return null;
  const total = (sample.match(/[^\s]/g) || []).length;
  if (!total) return null;

  const counts = {};
  for (const ch of sample) {
    const code = ch.codePointAt(0);
    for (const [start, end, lang] of SCRIPT_RANGES) {
      if (code >= start && code <= end) {
        counts[lang] = (counts[lang] || 0) + 1;
        break;
      }
    }
  }
  let scriptLang = null;
  let scriptCount = 0;
  for (const [lang, count] of Object.entries(counts)) {
    if (count > scriptCount) {
      scriptLang = lang;
      scriptCount = count;
    }
  }
  if (scriptLang && scriptCount / total >= 0.2) return scriptLang;

  const words = sample.toLowerCase().split(/[^a-zà-ÿ]+/).filter((w) => w.length > 1);
  const scores = {};
  for (const word of words) {
    for (const [lang, set] of Object.entries(STOPWORDS)) {
      scores[lang] = (scores[lang] || 0) + (set.has(word) ? 1 : 0);
    }
  }
  let best = "en";
  let bestScore = 0;
  for (const [lang, score] of Object.entries(scores)) {
    if (score > bestScore) {
      best = lang;
      bestScore = score;
    }
  }
  return best;
}

function pickVoiceForLang(lang) {
  const select = document.getElementById("speak-voice");
  if (!select) return null;
  const defaultVoice = select.dataset.defaultVoice || "";
  let candidate = null;
  for (const option of select.options) {
    if (option.dataset.cloned === "true") continue;
    if (option.dataset.lang !== lang) continue;
    candidate = candidate || option.value;
    if (option.value === defaultVoice) return option.value;
  }
  return candidate;
}

function hasClonedVoiceSelected(select) {
  return (
    select.selectedOptions.length > 0 &&
    select.selectedOptions[0].dataset.cloned === "true"
  );
}

function setDownloading(loading) {
  mergePatch({ tts_mp3_loading: loading });
}

async function downloadMp3(anchor) {
  const href = anchor.getAttribute("href");
  if (!href || href === "#" || anchor.dataset.mp3Loading === "true") return;
  anchor.dataset.mp3Loading = "true";
  setDownloading(true);
  try {
    const response = await fetch(href);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      mergePatch({ tts_status: payload.detail || "Download failed. Try again.", tts_error: "" });
      return;
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = DOWNLOAD_FILENAME;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  } catch {
    mergePatch({ tts_status: "Download failed. Try again.", tts_error: "" });
  } finally {
    anchor.dataset.mp3Loading = "false";
    setDownloading(false);
  }
}

document.addEventListener("click", (event) => {
  const anchor = event.target.closest(DOWNLOAD_SELECTOR);
  if (!anchor) return;
  event.preventDefault();
  downloadMp3(anchor);
});

(() => {
  const textarea = document.querySelector('[data-bind="tts_text"]');
  const counter = document.getElementById("char-count");
  if (textarea && counter) {
    const max = textarea.getAttribute("data-tts-max") || "0";
    const update = () => {
      counter.textContent = `${textarea.value.length} / ${max}`;
    };
    textarea.addEventListener("input", update);
    update();
  }

  let autoVoiceTimer = null;
  let lastAutoLang = null;

  function applyAutoVoice() {
    const select = document.getElementById("speak-voice");
    if (!textarea || !select) return;
    if (hasClonedVoiceSelected(select)) return;
    const lang = detectLanguage(textarea.value);
    if (!lang || lang === lastAutoLang) return;
    lastAutoLang = lang;
    const voice = pickVoiceForLang(lang);
    if (!voice || select.value === voice) return;
    mergePatch({ voice, tts_url: "", tts_mp3_url: "", tts_status: "" });
  }

  if (textarea) {
    textarea.addEventListener("input", () => {
      clearTimeout(autoVoiceTimer);
      autoVoiceTimer = setTimeout(applyAutoVoice, 400);
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    setTimeout(applyAutoVoice, 250);
    setTimeout(applyAutoVoice, 800);
  });
})();

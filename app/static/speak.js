/* Speak page helpers: live character counter and reading-time formatting.
 *
 * Loaded only from speak.html. The analysis itself is server-side; this only
 * wires up the small pieces of client-side polish.
 */
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

  window.formatReadingTime = (seconds) => {
    if (typeof seconds !== "number") return "";
    if (seconds < 60) return `${Math.round(seconds)} sec`;
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.round(seconds % 60);
    return remainder ? `${minutes} min ${remainder} sec` : `${minutes} min`;
  };
})();

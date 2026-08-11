/* Speak page helpers: live character counter.
 *
 * Loaded only from speak.html. The analysis itself is server-side; this only
 * wires up the small pieces of client-side polish. formatReadingTime lives in
 * an inline script in speak.html so it exists before Datastar scans the page.
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
})();

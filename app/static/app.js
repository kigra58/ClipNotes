/* SPA-style navigation helpers for Datastar @get links.
 *
 * Links that navigate to other pages carry both an `href` (no-JS fallback)
 * and a Datastar `data-on:click__prevent="@get('/path')"` action. The @get
 * action fetches the page and swaps <head>/<body> in place, so this module
 * only needs to keep the address bar in sync and reset the scroll position.
 *
 * When the target URL carries a fragment hash (e.g. a search result linking
 * to "#t-12.5") the Datastar morph does not auto-scroll, so a MutationObserver
 * waits for the anchor to appear and scrolls it into view, briefly flashing
 * it so the user can see where they landed.
 *
 * Loaded with `type="module"`, so it runs exactly once even though Datastar
 * re-inserts the script tag into <head> on every page swap.
 */
(() => {
  const LINK_SELECTOR = 'a[data-on\\:click__prevent*="@get("]';

  function spaUrl(element) {
    const value = element.getAttribute("data-on:click__prevent") || "";
    const match = value.match(/@get\(\s*['"]([^'"]+)['"]\s*\)/);
    return match ? match[1] : null;
  }

  document.addEventListener("click", (event) => {
    const link = event.target.closest(LINK_SELECTOR);
    if (!link) return;
    const url = spaUrl(link);
    if (!url) return;
    history.pushState({}, "", url);
    window.scrollTo(0, 0);
  });

  function closeExportMenus() {
    document.querySelectorAll(".export-menu.open").forEach((menu) => {
      menu.classList.remove("open");
      const toggle = menu.querySelector(".export-menu-toggle");
      if (toggle) toggle.setAttribute("aria-expanded", "false");
    });
  }

  document.addEventListener("click", (event) => {
    const toggle = event.target.closest(".export-menu-toggle");
    if (toggle) {
      const menu = toggle.closest(".export-menu");
      if (!menu) return;
      const open = menu.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (!event.target.closest(".export-menu")) {
      closeExportMenus();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeExportMenus();
  });

  function isClickOutside(dialog, event) {
    const rect = dialog.getBoundingClientRect();
    return (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    );
  }

  document.addEventListener("click", (event) => {
    const opener = event.target.closest("[data-open-category-modal]");
    if (opener) {
      const dialog = document.getElementById(
        `category-modal-${opener.dataset.openCategoryModal}`
      );
      if (dialog && !dialog.open) dialog.showModal();
      return;
    }
    const closer = event.target.closest("[data-close-category-modal]");
    if (closer) {
      const dialog = closer.closest("dialog");
      if (dialog) dialog.close();
      return;
    }
    if (event.target instanceof HTMLDialogElement && event.target.open) {
      if (isClickOutside(event.target, event)) event.target.close();
    }
  });

  document.addEventListener("click", (event) => {
    const copy = event.target.closest("[data-copy-social-post]");
    if (!copy) return;
    const container = copy.closest("#social-post-result");
    if (!container) return;
    const body = container.querySelector("#social-post-body");
    if (!body) return;
    navigator.clipboard
      .writeText(body.value)
      .then(() => {
        copy.textContent = "Copied!";
        setTimeout(() => (copy.textContent = "Copy post"), 1500);
      })
      .catch(() => {});
  });

  function scrollToHash() {
    const hash = window.location.hash;
    if (!hash) return;
    const target = document.getElementById(hash.slice(1));
    if (!target) return;
    target.scrollIntoView({ block: "center" });
    target.classList.add("flash-highlight");
    setTimeout(() => target.classList.remove("flash-highlight"), 2400);
  }

  new MutationObserver(scrollToHash).observe(document.body, {
    childList: true,
    subtree: true,
  });

  /* Client-side filter for the transcript card on the video page. Hides
   * segments whose text does not match the query. Re-applies on every body
   * mutation so it survives Datastar page swaps. */
  const TRANSCRIPT_CARD_SELECTOR = "section.transcript.card";
  const TRANSCRIPT_SEARCH_SELECTOR = "[data-transcript-search]";
  const TRANSCRIPT_EMPTY_SELECTOR = "[data-transcript-search-empty]";

  function filterTranscripts() {
    document.querySelectorAll(TRANSCRIPT_CARD_SELECTOR).forEach((card) => {
      const input = card.querySelector(TRANSCRIPT_SEARCH_SELECTOR);
      if (!input) return;
      const query = input.value.trim().toLowerCase();
      let visible = 0;
      card.querySelectorAll(".segment-item").forEach((item) => {
        const text = item.querySelector(".segment-text");
        const match =
          !query || (text && text.textContent.toLowerCase().includes(query));
        item.hidden = !match;
        if (match) visible += 1;
      });
      const empty = card.querySelector(TRANSCRIPT_EMPTY_SELECTOR);
      if (empty) empty.hidden = query.length === 0 || visible > 0;
    });
  }

  document.addEventListener("input", (event) => {
    if (event.target.closest(TRANSCRIPT_SEARCH_SELECTOR)) {
      filterTranscripts();
    }
  });

  new MutationObserver(filterTranscripts).observe(document.body, {
    childList: true,
    subtree: true,
  });

  window.addEventListener("popstate", () => {
    window.location.reload();
  });
})();

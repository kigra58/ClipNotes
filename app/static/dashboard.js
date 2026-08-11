/* Instant client-side search for the dashboard videos panel.
 *
 * The query is kept in sessionStorage so it survives Datastar re-renders of
 * #videos-panel (background poller, delete actions, SPA navigation). A
 * MutationObserver on <body> re-applies the filter after every panel swap.
 */
(() => {
  const STORAGE_KEY = "vidnotes-video-search";
  const PANEL_SELECTOR = "#videos-panel";
  const SEARCH_SELECTOR = "[data-video-search]";
  const EMPTY_SELECTOR = "[data-video-search-empty]";
  const CARD_SELECTOR = ".video-card";

  function apply() {
    const panel = document.querySelector(PANEL_SELECTOR);
    if (!panel) return;

    const input = panel.querySelector(SEARCH_SELECTOR);
    const stored = sessionStorage.getItem(STORAGE_KEY) || "";

    if (input && input.value !== stored) {
      input.value = stored;
    }

    const query = stored.trim().toLowerCase();
    const cards = panel.querySelectorAll(CARD_SELECTOR);
    let visible = 0;

    cards.forEach((card) => {
      const match = !query || (card.dataset.search || "").includes(query);
      card.hidden = !match;
      if (match) visible += 1;
    });

    const empty = panel.querySelector(EMPTY_SELECTOR);
    if (empty) {
      empty.hidden = query.length === 0 || visible > 0;
    }
  }

  document.addEventListener("input", (event) => {
    const input = event.target.closest(SEARCH_SELECTOR);
    if (!input || !input.closest(PANEL_SELECTOR)) return;
    sessionStorage.setItem(STORAGE_KEY, input.value);
    apply();
  });

  new MutationObserver(apply).observe(document.body, { childList: true, subtree: true });
  window.addEventListener("pageshow", apply);

  apply();
})();

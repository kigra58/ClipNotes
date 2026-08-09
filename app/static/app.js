/* SPA-style navigation helpers for Datastar @get links.
 *
 * Links that navigate to other pages carry both an `href` (no-JS fallback)
 * and a Datastar `data-on:click__prevent="@get('/path')"` action. The @get
 * action fetches the page and swaps <head>/<body> in place, so this module
 * only needs to keep the address bar in sync and reset the scroll position.
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

  window.addEventListener("popstate", () => {
    window.location.reload();
  });
})();

/* Video page: embedded YouTube player synced with the transcript.
 *
 * Loaded as a classic (non-module) script from video.html, so it runs once on
 * a fresh page load and re-runs whenever Datastar swaps this <head> in during
 * SPA navigation. State lives on `window.__ytTranscriptPlayer` so a re-run can
 * tear down the previous player/observer and rebuild for the new page without
 * leaking intervals or listeners.
 *
 * The player is initialized lazily: on a fresh load the <head> script runs
 * before <body> exists, and during SPA navigation the head patch can arrive
 * before or after the body patch, so a MutationObserver watches for the
 * #youtube-player container instead of assuming a DOM order.
 */
(() => {
  const CONTAINER_ID = "youtube-player";
  const PLAYING = 1;
  const PAUSED = 2;
  const ENDED = 0;
  const POLL_MS = 500;

  const state = window.__ytTranscriptPlayer || {};
  window.__ytTranscriptPlayer = state;

  function clearPlayer() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    if (state.player) {
      try {
        state.player.destroy();
      } catch {
        /* already torn down */
      }
      state.player = null;
    }
    state.playerReady = false;
    state.videoId = null;
    state.container = null;
    if (state.activeEl) {
      state.activeEl.classList.remove("segment-active");
      state.activeEl = null;
    }
  }

  function teardown() {
    clearPlayer();
    if (state.observer) {
      state.observer.disconnect();
      state.observer = null;
    }
  }

  function showError(message) {
    const box = document.getElementById("player-error");
    if (!box) return;
    box.hidden = false;
    box.textContent = message;
  }

  function hideError() {
    const box = document.getElementById("player-error");
    if (box) box.hidden = true;
  }

  function highlight(seconds) {
    let next = null;
    for (const el of document.querySelectorAll(".segment-item")) {
      const start = parseFloat(el.dataset.start);
      const end = parseFloat(el.dataset.end);
      if (seconds >= start && seconds < end) {
        next = el;
        break;
      }
    }
    if (next === state.activeEl) return;
    if (state.activeEl) state.activeEl.classList.remove("segment-active");
    state.activeEl = next;
    if (state.activeEl) {
      state.activeEl.classList.add("segment-active");
      state.activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  function tick() {
    if (!state.player || !state.playerReady) return;
    if (!state.container || !state.container.isConnected) {
      clearPlayer();
      return;
    }
    let seconds;
    try {
      seconds = state.player.getCurrentTime();
    } catch {
      return;
    }
    if (typeof seconds !== "number" || Number.isNaN(seconds)) return;
    highlight(seconds);
  }

  function buildPlayer(container) {
    state.container = container;
    state.videoId = container.dataset.videoId;
    if (!state.videoId) return;

    const onReady = () => {
      state.playerReady = true;
      hideError();
      const hash = window.location.hash;
      if (hash.startsWith("#t-")) {
        const start = parseFloat(hash.slice(3));
        if (!Number.isNaN(start) && start > 0) {
          state.player.seekTo(start, true);
        }
      }
    };

    const onStateChange = (event) => {
      switch (event.data) {
        case PLAYING:
          if (state.timer === null) state.timer = setInterval(tick, POLL_MS);
          tick();
          break;
        case PAUSED:
          if (state.timer !== null) {
            clearInterval(state.timer);
            state.timer = null;
          }
          tick();
          break;
        case ENDED:
          if (state.timer !== null) {
            clearInterval(state.timer);
            state.timer = null;
          }
          highlight(Number.POSITIVE_INFINITY);
          break;
        default:
          break;
      }
    };

    const onError = () => {
      showError("The embedded player couldn't load — this video may not allow embedding.");
    };

    try {
      state.player = new YT.Player(container, {
        videoId: state.videoId,
        width: "100%",
        height: "100%",
        playerVars: { playsinline: 1, rel: 0 },
        events: { onReady, onStateChange, onError },
      });
    } catch {
      showError("Couldn't start the embedded player.");
    }
  }

  function ensureApiAndBuild(container) {
    if (window.YT && typeof window.YT.Player === "function") {
      buildPlayer(container);
      return;
    }
    window.onYouTubeIframeAPIReady = () => {
      const current = document.getElementById(CONTAINER_ID);
      if (current) buildPlayer(current);
    };
    if (window.__ytTranscriptApiLoading) return;
    window.__ytTranscriptApiLoading = true;
    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    script.async = true;
    script.onerror = () =>
      showError("Couldn't load the YouTube player — check your internet connection.");
    document.head.appendChild(script);
  }

  function check() {
    const container = document.getElementById(CONTAINER_ID);
    if (!container) {
      if (state.player || state.timer) clearPlayer();
      return;
    }
    if (state.player && state.videoId === container.dataset.videoId) return;
    clearPlayer();
    ensureApiAndBuild(container);
  }

  function init() {
    teardown();
    state.observer = new MutationObserver(check);
    state.observer.observe(document.documentElement, { childList: true, subtree: true });
    check();
  }

  if (!state.clickBound) {
    state.clickBound = true;
    document.addEventListener("click", (event) => {
      const seek = event.target.closest("[data-seek]");
      if (!seek) return;
      const seconds = parseFloat(seek.dataset.seek);
      if (!state.player || !state.playerReady || Number.isNaN(seconds)) return;

      state.player.seekTo(seconds, true);
      state.player.playVideo();

      const href = seek.getAttribute("href") || "";
      if (/^https?:/i.test(href)) {
        event.preventDefault();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();

(() => {
  const state = (window.__socialPostEditor = window.__socialPostEditor || {});

  function plainToHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .split(/\n\s*\n/)
      .map((paragraph) => "<p>" + paragraph.replace(/\n/g, "<br>") + "</p>")
      .join("");
  }

  function mount() {
    const host = document.getElementById("social-post-editor");
    const source = document.getElementById("social-post-body");
    if (!host || !source) return;

    if (typeof Quill === "undefined") {
      source.removeAttribute("hidden");
      return;
    }

    if (state.quill && state.quill.container === host) return;

    if (state.quill) {
      const oldContainer = state.quill.container;
      const toolbar = oldContainer && oldContainer.previousElementSibling;
      if (toolbar && toolbar.classList && toolbar.classList.contains("ql-toolbar")) {
        toolbar.remove();
      }
      if (oldContainer && oldContainer.parentNode) {
        oldContainer.remove();
      }
      state.quill = null;
    }

    host.innerHTML = "";

    const quill = new Quill(host, {
      theme: "snow",
      placeholder: "Your post will appear here…",
      modules: {
        toolbar: [
          ["bold", "italic", "underline", "strike"],
          [{ list: "ordered" }, { list: "bullet" }],
          ["link"],
          ["clean"],
        ],
      },
    });

    quill.clipboard.dangerouslyPasteHTML(plainToHtml(source.value));

    source.hidden = true;

    quill.on("text-change", () => {
      source.value = quill.root.innerHTML;
    });

    source.value = quill.root.innerHTML;

    state.quill = quill;
  }

  const observer = new MutationObserver(() => {
    const host = document.getElementById("social-post-editor");
    if (host && (!state.quill || state.quill.container !== host)) mount();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();

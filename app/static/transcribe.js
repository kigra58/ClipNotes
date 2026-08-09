(() => {
    const form = document.getElementById("transcribe-form");
    if (!form) return;

    const overlay = document.createElement("div");
    overlay.className = "loading-overlay";
    overlay.innerHTML = `
        <div class="loading-box">
            <div class="spinner"></div>
            <p>Transcribing… this can take a few minutes.</p>
        </div>
    `;

    form.addEventListener("submit", () => {
        document.body.appendChild(overlay);
        form.querySelector("button").disabled = true;
    });
})();

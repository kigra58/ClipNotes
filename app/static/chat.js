(() => {
    const transcriptId = window.TRANSCRIPT_ID;
    const chatAvailable = window.CHAT_AVAILABLE;

    const scrollEl = document.getElementById("chat-scroll");
    const messagesEl = document.getElementById("chat-messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    const banner = document.getElementById("chat-banner");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const sidebar = document.getElementById("sidebar");

    const ICONS = {
        bot: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>',
        user: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        copy: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>',
        check: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
    };

    const THINKING_PHASES = [
        "Reading the transcript…",
        "Finding the right parts…",
        "Forming an answer…",
    ];

    const escapeHtml = (text) =>
        String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");

    /* ---------- Minimal Markdown renderer (streaming-safe) ---------- */

    const inlineMd = (text) =>
        text
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/__(.+?)__/g, "<strong>$1</strong>")
            .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>")
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    function renderMarkdown(raw) {
        const lines = escapeHtml(raw).split("\n");
        const out = [];
        let inCode = false;
        let codeAcc = [];
        let list = null;

        const closeList = () => {
            if (list) {
                out.push(`</${list}>`);
                list = null;
            }
        };

        const closeCode = () => {
            if (inCode) {
                out.push(`<pre><code>${codeAcc.join("\n")}</code></pre>`);
                inCode = false;
                codeAcc = [];
            }
        };

        for (const line of lines) {
            if (/^```/.test(line)) {
                if (!inCode) {
                    closeList();
                    inCode = true;
                    codeAcc = [];
                } else {
                    closeCode();
                }
                continue;
            }
            if (inCode) {
                codeAcc.push(line);
                continue;
            }
            if (!line.trim()) {
                closeList();
                continue;
            }

            const heading = line.match(/^(#{1,3})\s+(.*)$/);
            if (heading) {
                closeList();
                const level = heading[1].length;
                out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
                continue;
            }

            const quote = line.match(/^&gt;\s?(.*)$/);
            if (quote) {
                closeList();
                out.push(`<blockquote>${inlineMd(quote[1])}</blockquote>`);
                continue;
            }

            const item = line.match(/^(\s*)[-*]\s+(.*)$/) || line.match(/^(\s*)\d+\.\s+(.*)$/);
            if (item) {
                const tag = /^\d+\./.test(item[0]) ? "ol" : "ul";
                if (list !== tag) {
                    closeList();
                    out.push(`<${tag}>`);
                    list = tag;
                }
                out.push(`<li>${inlineMd(item[2])}</li>`);
                continue;
            }

            closeList();
            out.push(`<p>${inlineMd(line)}</p>`);
        }
        closeList();
        closeCode();
        return out.join("\n");
    }

    /* ---------- Helpers ---------- */

    const formatTime = (seconds) => {
        const total = Math.max(0, Math.round(seconds));
        const m = Math.floor(total / 60);
        const s = total % 60;
        return `${m}:${String(s).padStart(2, "0")}`;
    };

    const clock = () => {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    };

    const isNearBottom = () =>
        scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 120;

    const scrollToBottom = () => {
        scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
    };

    const setStatus = (state, text) => {
        statusDot.className = `status-dot ${state}`;
        statusText.textContent = text;
    };

    const showBanner = (message, isError) => {
        banner.hidden = false;
        banner.textContent = message;
        banner.className = `chat-banner${isError ? " error" : ""}`;
    };

    const hideBanner = () => {
        banner.hidden = true;
    };

    const setBusy = (busy) => {
        sendBtn.disabled = busy;
        input.disabled = busy;
        if (busy) {
            setStatus("busy", "Thinking…");
        } else {
            setStatus("connected", "Connected");
        }
    };

    /* ---------- Message construction ---------- */

    const makeAvatar = (role) => {
        const avatar = document.createElement("div");
        avatar.className = `avatar ${role}`;
        avatar.innerHTML = ICONS[role];
        return avatar;
    };

    const makeMessage = (role) => {
        const wrap = document.createElement("div");
        wrap.className = `message ${role}`;
        const body = document.createElement("div");
        body.className = "message-body";
        wrap.appendChild(makeAvatar(role));
        wrap.appendChild(body);
        messagesEl.appendChild(wrap);
        return body;
    };

    const addUserMessage = (text) => {
        const body = makeMessage("user");
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;
        body.appendChild(bubble);
        const foot = document.createElement("div");
        foot.className = "message-foot";
        foot.innerHTML = `<span>${clock()}</span>`;
        body.appendChild(foot);
    };

    const makeThinking = (label) => {
        const bubble = document.createElement("div");
        bubble.className = "bubble thinking";
        bubble.innerHTML =
            '<div class="dots"><span></span><span></span><span></span></div>' +
            `<span class="thinking-label">${escapeHtml(label)}</span>`;
        return bubble;
    };

    const makeFoot = (copyCallback) => {
        const foot = document.createElement("div");
        foot.className = "message-foot";
        const time = document.createElement("span");
        time.textContent = clock();
        const copy = document.createElement("button");
        copy.className = "copy-btn";
        copy.innerHTML = `${ICONS.copy} Copy`;
        copy.addEventListener("click", () => copyCallback(copy));
        foot.appendChild(time);
        foot.appendChild(copy);
        return foot;
    };

    const makeSources = (sources) => {
        const el = document.createElement("div");
        el.className = "sources";
        const label = document.createElement("span");
        label.className = "sources-label";
        label.textContent = "Sources";
        el.appendChild(label);
        sources.forEach((source) => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "source-chip";
            chip.textContent = `${formatTime(source.start)}–${formatTime(source.end)}`;
            chip.title = source.text;
            el.appendChild(chip);
        });
        return el;
    };

    /* ---------- State ---------- */

    const current = {
        body: null,
        sourcesEl: null,
        thinkingEl: null,
        bubble: null,
        raw: "",
        done: false,
    };

    const resetCurrent = () => {
        current.body = null;
        current.sourcesEl = null;
        current.thinkingEl = null;
        current.bubble = null;
        current.raw = "";
        current.done = false;
    };

    const phaseTimer = {
        timer: null,
        index: 0,
        start() {
            this.stop();
            this.index = 0;
            this.tick();
            this.timer = setInterval(() => this.tick(), 2600);
        },
        tick() {
            const label = current.thinkingEl?.querySelector(".thinking-label");
            if (label) {
                label.textContent = THINKING_PHASES[this.index % THINKING_PHASES.length];
                this.index += 1;
            }
        },
        stop() {
            if (this.timer) {
                clearInterval(this.timer);
                this.timer = null;
            }
        },
    };

    /* ---------- WebSocket ---------- */

    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${wsProtocol}//${location.host}/ws/chat/${transcriptId}`);

    socket.addEventListener("open", () => {
        setStatus("connected", "Connected");
        hideBanner();
    });

    socket.addEventListener("close", () => {
        setStatus("offline", "Disconnected");
        setBusy(false);
        phaseTimer.stop();
        if (!chatAvailable) return;
        if (!current.done) {
            showBanner("Connection lost. Reload the page to reconnect.", true);
        }
    });

    socket.addEventListener("error", () => {
        setStatus("offline", "Offline");
        showBanner("Could not reach the server.", true);
    });

    socket.addEventListener("message", (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch {
            return;
        }

        switch (data.type) {
            case "sources":
                current.sourcesEl = makeSources(data.data);
                current.body.appendChild(current.sourcesEl);
                break;
            case "token":
                if (!current.bubble) {
                    phaseTimer.stop();
                    current.thinkingEl.remove();
                    current.thinkingEl = null;
                    current.bubble = document.createElement("div");
                    current.bubble.className = "bubble streaming";
                    current.body.appendChild(current.bubble);
                }
                current.raw += data.data;
                current.bubble.innerHTML = renderMarkdown(current.raw);
                if (isNearBottom()) scrollToBottom();
                break;
            case "error":
                phaseTimer.stop();
                setBusy(false);
                if (current.thinkingEl) {
                    current.thinkingEl.remove();
                    current.thinkingEl = null;
                }
                if (current.bubble) {
                    current.bubble.classList.remove("streaming");
                    current.done = true;
                    showBanner(data.data || "An error occurred.", true);
                } else {
                    const bubble = document.createElement("div");
                    bubble.className = "bubble error-bubble";
                    bubble.textContent = data.data || "Sorry, something went wrong.";
                    current.body.appendChild(bubble);
                }
                break;
            case "done":
                phaseTimer.stop();
                current.bubble?.classList.remove("streaming");
                current.done = true;
                setBusy(false);
                if (current.bubble) {
                    const raw = current.raw;
                    const foot = makeFoot((btn) => copyAnswer(btn, raw));
                    current.body.appendChild(foot);
                }
                scrollToBottom();
                break;
            default:
                break;
        }
    });

    const copyAnswer = async (btn, text) => {
        try {
            await navigator.clipboard.writeText(text);
            btn.classList.add("copied");
            btn.innerHTML = `${ICONS.check} Copied`;
            setTimeout(() => {
                btn.classList.remove("copied");
                btn.innerHTML = `${ICONS.copy} Copy`;
            }, 1800);
        } catch {
            /* clipboard unavailable */
        }
    };

    /* ---------- Input ---------- */

    const autoResize = () => {
        input.style.height = "auto";
        input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    };

    input.addEventListener("input", autoResize);
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const question = input.value.trim();
        if (!question || socket.readyState !== WebSocket.OPEN || current.body) return;

        addUserMessage(question);
        const body = makeMessage("bot");
        current.body = body;
        current.thinkingEl = makeThinking(THINKING_PHASES[0]);
        body.appendChild(current.thinkingEl);
        phaseTimer.start();

        input.value = "";
        autoResize();
        setBusy(true);
        scrollToBottom();

        socket.send(JSON.stringify({ message: question }));
    });

    /* ---------- Suggestions & sidebar ---------- */

    document.querySelectorAll(".chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            input.value = chip.dataset.question;
            autoResize();
            form.requestSubmit();
        });
    });

    document.getElementById("sidebar-open").addEventListener("click", () => {
        sidebar.classList.add("open");
    });
    document.getElementById("sidebar-close").addEventListener("click", () => {
        sidebar.classList.remove("open");
    });

    /* ---------- Init ---------- */

    if (!chatAvailable) {
        setStatus("offline", "Chat unavailable");
        input.disabled = true;
        sendBtn.disabled = true;
        showBanner(
            "Chat is disabled: set GEMINI_API_KEY in .env and restart the server.",
            true,
        );
        socket.close();
    }
})();

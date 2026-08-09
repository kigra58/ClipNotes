(() => {
    const transcriptId = window.TRANSCRIPT_ID;
    const messagesEl = document.getElementById("chat-messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    const statusEl = document.getElementById("chat-status");

    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${wsProtocol}//${location.host}/ws/chat/${transcriptId}`);

    let botBubble = null;
    let sourceList = null;

    const escapeHtml = (text) =>
        text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#039;");

    const addMessage = (role, text) => {
        const div = document.createElement("div");
        div.className = `message ${role}`;
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.innerHTML = text;
        div.appendChild(bubble);
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return bubble;
    };

    const showSources = (sources) => {
        if (!sources || !sources.length) return;
        sourceList = document.createElement("div");
        sourceList.className = "sources";
        const heading = document.createElement("div");
        heading.className = "sources-heading";
        heading.textContent = "Sources";
        sourceList.appendChild(heading);
        sources.forEach((source) => {
            const span = document.createElement("span");
            span.className = "source";
            span.title = source.text;
            span.textContent = `${formatTime(source.start)}–${formatTime(source.end)}`;
            sourceList.appendChild(span);
        });
        botBubble.parentElement.appendChild(sourceList);
    };

    const formatTime = (seconds) => {
        const total = Math.max(0, Math.round(seconds));
        const m = Math.floor(total / 60);
        const s = total % 60;
        return `${m}:${String(s).padStart(2, "0")}`;
    };

    const setStatus = (text) => {
        statusEl.hidden = !text;
        statusEl.textContent = text;
    };

    const setBusy = (busy) => {
        sendBtn.disabled = busy;
        input.disabled = busy;
        setStatus(busy ? "Generating answer…" : "");
        if (busy) input.focus();
    };

    socket.addEventListener("open", () => {
        setStatus("");
    });

    socket.addEventListener("close", () => {
        setStatus("Connection closed. Reload the page to reconnect.");
    });

    socket.addEventListener("error", () => {
        setStatus("WebSocket error. Check that the server is running.");
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
                showSources(data.data);
                break;
            case "token":
                botBubble.innerHTML += escapeHtml(data.data);
                messagesEl.scrollTop = messagesEl.scrollHeight;
                break;
            case "error":
                setBusy(false);
                setStatus(data.data || "An error occurred.");
                if (botBubble && !botBubble.textContent.trim()) {
                    botBubble.textContent = "Sorry, I couldn't generate an answer.";
                }
                break;
            case "done":
                setBusy(false);
                break;
            default:
                break;
        }
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const question = input.value.trim();
        if (!question || socket.readyState !== WebSocket.OPEN) return;

        addMessage("user", escapeHtml(question));
        botBubble = addMessage("bot", "");
        sourceList = null;
        input.value = "";
        setBusy(true);

        socket.send(JSON.stringify({ message: question }));
    });
})();

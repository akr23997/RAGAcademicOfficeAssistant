const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const chatBox = document.getElementById("chat-box");
const statusText = document.getElementById("status");
const sendButton = document.getElementById("send-button");


function addMessage(text, className) {
    const message = document.createElement("div");

    message.className = `message ${className}`;
    message.textContent = text;

    chatBox.appendChild(message);

    return message;
}


function addSources(message, sources) {
    if (!sources || sources.length === 0) {
        return;
    }

    const sourceContainer = document.createElement("div");
    sourceContainer.className = "source-list";

    const heading = document.createElement("strong");
    heading.textContent = "Sources:";

    sourceContainer.appendChild(heading);

    const list = document.createElement("ul");

    sources.forEach(source => {
        const item = document.createElement("li");

        // Use textContent to prevent HTML injection.
        item.textContent =
            `${source.document}, page ${source.page}`;

        list.appendChild(item);
    });

    sourceContainer.appendChild(list);
    message.appendChild(sourceContainer);
}


form.addEventListener("submit", async event => {
    event.preventDefault();

    const question = input.value.trim();

    if (!question) {
        return;
    }

    addMessage(question, "user");

    input.value = "";
    sendButton.disabled = true;
    statusText.textContent = "Searching institutional documents...";

    try {
        const response = await fetch("/api/chat", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "The request failed."
            );
        }

        const message = addMessage(
            data.answer,
            "bot"
        );

        addSources(message, data.sources);

    } catch (error) {
        addMessage(
            "Sorry, the assistant could not process your question. " +
            "Please try again.",
            "bot"
        );

        console.error(error);

    } finally {
        sendButton.disabled = false;
        statusText.textContent = "";
        chatBox.scrollTop = chatBox.scrollHeight;
        input.focus();
    }
});
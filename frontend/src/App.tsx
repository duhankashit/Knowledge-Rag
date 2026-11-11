import { Files, MessageSquareText, Plus } from "lucide-react";
import { useEffect, useState } from "react";

import ChatPage from "./components/ChatPage";
import DocumentsPage from "./components/DocumentsPage";
import type { ChatMessage, Page } from "./types";

const CHAT_STORAGE_KEY = "atlas-chat-messages";

function readSavedMessages(): ChatMessage[] {
  try {
    const saved = sessionStorage.getItem(CHAT_STORAGE_KEY);
    if (!saved) return [];
    const parsed: unknown = JSON.parse(saved);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (message): message is ChatMessage =>
        typeof message === "object" &&
        message !== null &&
        "role" in message &&
        (message.role === "user" || message.role === "assistant") &&
        "content" in message &&
        typeof message.content === "string",
    );
  } catch {
    return [];
  }
}

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [messages, setMessages] = useState<ChatMessage[]>(readSavedMessages);

  useEffect(() => {
    try {
      sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // Chat still works when browser storage is disabled or full.
    }
  }, [messages]);

  function startNewChat() {
    setMessages([]);
    setPage("chat");
  }

  return (
    <div className="app-shell">
      <main className="main-panel">
        <header className="topbar">
          <nav className="top-nav" aria-label="Main navigation">
            <button
              className={`top-nav-item ${page === "files" ? "active" : ""}`}
              onClick={() => setPage("files")}
              type="button"
              aria-current={page === "files" ? "page" : undefined}
            >
              <Files size={16} />
              <span>Knowledge base</span>
            </button>
            <button
              className={`top-nav-item ${page === "chat" ? "active" : ""}`}
              onClick={() => setPage("chat")}
              type="button"
              aria-current={page === "chat" ? "page" : undefined}
            >
              <MessageSquareText size={17} />
              <span>Conversation</span>
            </button>
          </nav>
          {page === "chat" && (
            <button
              className="topbar-new-chat"
              onClick={startNewChat}
              type="button"
            >
              <Plus size={16} />
              <span>New conversation</span>
            </button>
          )}
        </header>

        {page === "chat" ? (
          <ChatPage messages={messages} setMessages={setMessages} />
        ) : (
          <DocumentsPage />
        )}
      </main>
    </div>
  );
}

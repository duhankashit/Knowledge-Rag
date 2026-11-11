import { ArrowDown, ArrowUp, BookOpen, FileText, Plus, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Dispatch, FormEvent, KeyboardEvent, SetStateAction } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { sendMessage } from "../api";
import type { ChatMessage } from "../types";

interface ChatPageProps {
  messages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
}

const suggestions = [
  {
    icon: "01",
    title: "Give me the key takeaways",
    prompt: "Give me the key takeaways from my documents.",
  },
  {
    icon: "02",
    title: "Compare important details",
    prompt: "Compare the most important details across my documents.",
  },
  {
    icon: "03",
    title: "Find a specific answer",
    prompt: "What are the main topics covered in my documents?",
  },
];

export default function ChatPage({ messages, setMessages }: ChatPageProps) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const scrollArea = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const area = scrollArea.current;
    if (area) area.scrollTo({ top: area.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  function updateQuestion(value: string) {
    setQuestion(value);
    const input = textarea.current;
    if (input) {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    }
  }

  async function submitQuestion(value = question) {
    const prompt = value.trim();
    if (!prompt || busy) return;

    setError("");
    setQuestion("");
    if (textarea.current) textarea.current.style.height = "auto";
    const history = messages.slice(-12).map(({ role, content }) => ({ role, content }));
    setMessages((current) => [...current, { role: "user", content: prompt }]);
    setBusy(true);

    try {
      const response = await sendMessage(prompt, history);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.answer, sources: response.sources },
      ]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Something went wrong while sending your message.",
      );
    } finally {
      setBusy(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <section className={`chat-page ${isEmpty ? "chat-page-empty" : ""}`}>
      {isEmpty ? (
        <div className="welcome-content">
          <div className="welcome-kicker"><span /> YOUR KNOWLEDGE, CONNECTED</div>
          <h1>Answers are already<br /><span>in your files.</span></h1>
          <p className="welcome-description">
            Ask a question. Your assistant will find the right information and show you
            exactly where it came from.
          </p>

          <div className="suggestion-grid">
            {suggestions.map((suggestion) => (
              <button
                className="suggestion-card"
                key={suggestion.icon}
                onClick={() => void submitQuestion(suggestion.prompt)}
              >
                <span className="suggestion-index">{suggestion.icon}</span>
                <span className="suggestion-title">{suggestion.title}</span>
                <ArrowUp size={15} className="suggestion-arrow" />
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="conversation-scroll" ref={scrollArea}>
          <div className="conversation-inner">
            <div className="conversation-heading">
              <span className="conversation-mark"><Sparkles size={15} /></span>
              <span>Conversation</span>
              <span className="conversation-rule" />
              <button
                className="conversation-new"
                onClick={() => setMessages([])}
                type="button"
              >
                <Plus size={14} /> New chat
              </button>
            </div>
            {messages.map((message, index) => (
              <article
                className={`message-row ${message.role}`}
                key={`${index}-${message.role}`}
              >
                <div className={`message-avatar ${message.role}`}>
                  {message.role === "assistant" ? <Sparkles size={15} /> : "A"}
                </div>
                <div className="message-body">
                  <div className="message-author">
                    {message.role === "assistant" ? "Assistant" : "You"}
                    {message.role === "assistant" && (
                      <span className="message-meta">RAG assistant</span>
                    )}
                  </div>
                  {message.role === "assistant" ? (
                    <div className="markdown-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="user-message-text">{message.content}</p>
                  )}
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources-block">
                      <div className="sources-heading">
                        <BookOpen size={13} />
                        <span>Sources</span>
                        <span className="source-count">{message.sources.length}</span>
                      </div>
                      <div className="source-list">
                        {message.sources.map((source, sourceIndex) => (
                          <details className="source-item" key={`${source.filename}-${source.chunk}-${sourceIndex}`}>
                            <summary>
                              <FileText size={14} />
                              <span className="source-name">{source.filename}</span>
                              <span className="source-chunk">Chunk {source.chunk}</span>
                            </summary>
                            <p>{source.excerpt}</p>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </article>
            ))}
            {busy && (
              <div className="message-row assistant">
                <div className="message-avatar assistant"><Sparkles size={15} /></div>
                <div className="message-body">
                  <div className="message-author">Assistant <span className="message-meta">Searching your knowledge</span></div>
                  <div className="typing-indicator"><i /><i /><i /></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {messages.length > 3 && (
        <button
          className="latest-button"
          onClick={() => scrollArea.current?.scrollTo({ top: scrollArea.current.scrollHeight, behavior: "smooth" })}
          type="button"
        >
          <ArrowDown size={14} /> Latest
        </button>
      )}

      {isEmpty && (
        <div className="empty-note">
          <FileText size={14} />
          <span>Answers are grounded in your indexed documents</span>
          <span className="note-separator">·</span>
          <span>Sources included</span>
        </div>
      )}
      <div className={`composer-wrap ${isEmpty ? "composer-wrap-empty" : ""}`}>
        {!isEmpty && error && <div className="composer-error" role="alert">{error}</div>}
        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="Ask your documents"
            ref={textarea}
            rows={1}
            value={question}
            onChange={(event) => updateQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your documents..."
            maxLength={8000}
            disabled={busy}
          />
          <div className="composer-footer">
            <span className="composer-hint">Answers grounded in your files</span>
            <button
              className="send-button"
              type="submit"
              disabled={!question.trim() || busy}
              aria-label="Send message"
            >
              <ArrowUp size={17} />
            </button>
          </div>
        </form>
        <p className="composer-disclaimer">
          AI can make mistakes. Verify important details with the source documents.
        </p>
      </div>
    </section>
  );
}

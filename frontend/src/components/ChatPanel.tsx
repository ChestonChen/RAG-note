import { useEffect, useMemo, useRef, useState } from "react";
import type { Citation, Message } from "../types";
import { getLastUserMessage, renderMarkdown } from "../ui";

type ChatPanelProps = {
  title: string;
  messages: Message[];
  streaming: boolean;
  draftAnswer: string;
  streamError: string | null;
  editRequest: { content: string; token: number } | null;
  selectedMessageId: number | null;
  onSelectMessage: (message: Message) => void;
  onSubmit: (question: string, allowWebFallback: boolean) => void;
  onStop: () => void;
  onRetry: (message: Message) => void;
  onEdit: (message: Message) => void;
  onOpenNavigation: () => void;
  onToggleEvidence: () => void;
};

function Confidence({ message }: { message: Message }) {
  if (!message.confidence) return null;
  const score = message.confidence_score == null ? "" : ` · ${Math.round(message.confidence_score * 100)}%`;
  return <span className={`confidence confidence-${message.confidence.toLowerCase()}`}>{message.confidence}{score}</span>;
}

function MessageActions({ message, onRetry, onEdit }: { message: Message; onRetry: (message: Message) => void; onEdit: (message: Message) => void }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard?.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };
  if (message.role === "user") {
    return <div className="message-actions"><button type="button" onClick={() => onEdit(message)}>Edit & resend</button></div>;
  }
  return <div className="message-actions"><button type="button" onClick={copy}>{copied ? "Copied" : "Copy"}</button><button type="button" onClick={() => onRetry(message)}>Retry</button></div>;
}

function CitationStrip({ citations, onOpen }: { citations: Citation[]; onOpen: () => void }) {
  if (!citations.length) return null;
  return <button className="citation-strip" type="button" onClick={onOpen}>
    <span className="citation-icon" aria-hidden="true">⌕</span>
    <span><strong>{citations.length} source{citations.length === 1 ? "" : "s"}</strong><small>{citations.slice(0, 2).map((citation) => citation.title).join(" · ")}</small></span>
    <span aria-hidden="true">›</span>
  </button>;
}

export function ChatPanel({ title, messages, streaming, draftAnswer, streamError, editRequest, selectedMessageId, onSelectMessage, onSubmit, onStop, onRetry, onEdit, onOpenNavigation, onToggleEvidence }: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [allowWeb, setAllowWeb] = useState(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const canSubmit = question.trim().length > 0 && !streaming;
  const retryMessage = getLastUserMessage(messages);
  const currentMessages = useMemo(() => streaming && draftAnswer ? [...messages, { id: -1, conversation_id: 0, role: "assistant" as const, content: draftAnswer, status: "streaming", citations: [], retrieval_debug: {}, confidence: null, confidence_score: null, mode: "", web_sources: [], error_message: null, created_at: new Date().toISOString() }] : messages, [messages, streaming, draftAnswer]);

  // An edit request is versioned so the same question can be edited twice.
  useEffect(() => {
    if (!editRequest) return;
    setQuestion(editRequest.content);
    const focusTimer = window.setTimeout(() => textareaRef.current?.focus(), 0);
    return () => window.clearTimeout(focusTimer);
  }, [editRequest]);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit(question.trim(), allowWeb);
    setQuestion("");
  };

  return <section className="chat-pane">
    <header className="chat-header">
      <button className="icon-button mobile-only" onClick={onOpenNavigation} aria-label="Open navigation">☰</button>
      <div><span className="chat-kicker">Conversation</span><h1>{title || "New conversation"}</h1></div>
      <button className="evidence-toggle" type="button" onClick={onToggleEvidence}><span aria-hidden="true">⌕</span> Evidence</button>
    </header>
    <div className="message-scroll" aria-live="polite">
      {currentMessages.length === 0 ? <div className="chat-empty"><div className="empty-orbit" aria-hidden="true">✦</div><h2>What are you exploring?</h2><p>Ask about your indexed knowledge. Answers stay grounded with sources you can inspect.</p><div className="prompt-suggestions"><button type="button" onClick={() => setQuestion("Summarize the key ideas in my knowledge base")}>Summarize my knowledge base</button><button type="button" onClick={() => setQuestion("What changed in the latest documents?")}>What changed recently?</button></div></div> : null}
      {currentMessages.map((message) => <article className={`message-row ${message.role}`} key={message.id} onClick={() => message.role === "assistant" && message.id > 0 ? onSelectMessage(message) : undefined}>
        <div className="avatar" aria-hidden="true">{message.role === "assistant" ? "R" : "You"}</div>
        <div className="message-body">
          <div className="message-meta"><strong>{message.role === "assistant" ? "Research assistant" : "You"}</strong>{message.role === "assistant" ? <Confidence message={message} /> : null}{message.status === "streaming" ? <span className="streaming-label"><i /> Writing</span> : null}</div>
          {message.status === "error" ? <div className="message-error"><strong>Answer interrupted</strong><span>{message.error_message || "Something went wrong. Try again."}</span><button type="button" onClick={() => onRetry(message)}>Retry</button></div> : <div className="message-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} />}
          {message.role === "assistant" && message.status !== "streaming" ? <CitationStrip citations={message.citations} onOpen={onToggleEvidence} /> : null}
          {message.status !== "streaming" ? <MessageActions message={message} onRetry={onRetry} onEdit={onEdit} /> : null}
        </div>
      </article>)}
      {streamError ? <div className="stream-error" role="alert"><strong>Stream error</strong><span>{streamError}</span>{retryMessage ? <button type="button" onClick={() => onRetry(retryMessage)}>Retry last question</button> : null}</div> : null}
    </div>
    <form className="composer" onSubmit={submit}>
      <textarea ref={textareaRef} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(event); } }} placeholder="Ask a question about your knowledge base..." rows={1} disabled={streaming} aria-label="Question" />
      <div className="composer-toolbar"><label className="toggle-label"><input type="checkbox" checked={allowWeb} onChange={(event) => setAllowWeb(event.target.checked)} /> <span>Web fallback</span></label>{streaming ? <button className="stop-button" type="button" onClick={onStop}><span aria-hidden="true">■</span> Stop</button> : <button className="send-button" type="submit" disabled={!canSubmit} aria-label="Send question"><span aria-hidden="true">↑</span></button>}</div>
    </form>
    <p className="composer-hint">Shift + Enter for a new line · Answers may contain mistakes</p>
  </section>;
}

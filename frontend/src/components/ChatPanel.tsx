import { useEffect, useRef, useState } from "react";
import type { Citation, Message } from "../types";
import { getConfidenceLabel, getLastUserMessage, getMessageBubbleClassName, getMessageRowClassName, renderMarkdown } from "../ui";

type ChatPanelProps = {
  title: string;
  messages: Message[];
  streaming: boolean;
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
  return <span className={`confidence confidence-${message.confidence.toLowerCase()}`}>{getConfidenceLabel(message.confidence)}{score}</span>;
}

function MessageActions({ message, onRetry, onEdit }: { message: Message; onRetry: (message: Message) => void; onEdit: (message: Message) => void }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard?.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };
  if (message.role === "user") {
    return <div className="message-actions"><button type="button" onClick={() => onEdit(message)}>编辑并重发</button></div>;
  }
  return <div className="message-actions"><button type="button" onClick={copy}>{copied ? "已复制" : "复制"}</button><button type="button" onClick={() => onRetry(message)}>重试</button></div>;
}

function CitationStrip({ citations, onOpen }: { citations: Citation[]; onOpen: () => void }) {
  if (!citations.length) return null;
  return <button className="citation-strip" type="button" onClick={onOpen}>
    <span className="citation-icon" aria-hidden="true">⌕</span>
    <span><strong>{citations.length} 个来源</strong><small>{citations.slice(0, 2).map((citation) => citation.title).join(" · ")}</small></span>
    <span aria-hidden="true">›</span>
  </button>;
}

export function ChatPanel({ title, messages, streaming, streamError, editRequest, selectedMessageId, onSelectMessage, onSubmit, onStop, onRetry, onEdit, onOpenNavigation, onToggleEvidence }: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [allowWeb, setAllowWeb] = useState(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const canSubmit = question.trim().length > 0 && !streaming;
  const retryMessage = getLastUserMessage(messages);

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
      <button className="icon-button mobile-only" onClick={onOpenNavigation} aria-label="打开导航">☰</button>
      <div><span className="chat-kicker">会话</span><h1>{title || "新建会话"}</h1></div>
      <button className="evidence-toggle" type="button" onClick={onToggleEvidence}><span aria-hidden="true">⌕</span> 查看证据</button>
    </header>
    <div className="message-scroll" aria-live="polite">
      {messages.length === 0 ? <div className="chat-empty"><div className="empty-orbit" aria-hidden="true">✦</div><h2>你想探索什么？</h2><p>向知识库提问，回答会尽量基于可查看的资料来源。</p><div className="prompt-suggestions"><button type="button" onClick={() => setQuestion("总结我的知识库中的关键观点")}>总结知识库要点</button><button type="button" onClick={() => setQuestion("最新文档有哪些变化？")}>查看最近变化</button></div></div> : null}
      {messages.map((message) => <article className={getMessageRowClassName(message.role)} key={message.id} onClick={() => message.role === "assistant" && message.id > 0 ? onSelectMessage(message) : undefined}>
        {message.role === "user" ? <div className="avatar" aria-hidden="true">你</div> : null}
        <div className="message-body">
          <div className="message-meta"><strong>{message.role === "assistant" ? "研究助手" : "你"}</strong>{message.role === "assistant" ? <Confidence message={message} /> : null}{message.status === "streaming" ? <span className="streaming-label"><i /> {message.content ? "正在生成回答" : "正在检索知识库"}</span> : null}</div>
          <div className={getMessageBubbleClassName(message.role)}>{message.status === "error" ? <div className="message-error"><strong>回答已中断</strong><span>{message.error_message || "出现问题，请重试。"}</span><button type="button" onClick={() => onRetry(message)}>重试</button></div> : <div className="message-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} />}</div>
          {message.role === "assistant" && message.status !== "streaming" ? <CitationStrip citations={message.citations} onOpen={onToggleEvidence} /> : null}
          {message.status !== "streaming" ? <MessageActions message={message} onRetry={onRetry} onEdit={onEdit} /> : null}
        </div>
      </article>)}
      {streamError ? <div className="stream-error" role="alert"><strong>回答流错误</strong><span>{streamError}</span>{retryMessage ? <button type="button" onClick={() => onRetry(retryMessage)}>重试上一问题</button> : null}</div> : null}
    </div>
    <form className="composer" onSubmit={submit}>
      <textarea ref={textareaRef} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(event); } }} placeholder="请输入关于知识库的问题..." rows={1} disabled={streaming} aria-label="问题" />
      <div className="composer-toolbar"><label className="toggle-label"><input type="checkbox" checked={allowWeb} onChange={(event) => setAllowWeb(event.target.checked)} /> <span>联网补充</span></label>{streaming ? <button className="stop-button" type="button" onClick={onStop}><span aria-hidden="true">■</span> 停止生成</button> : <button className="send-button" type="submit" disabled={!canSubmit} aria-label="发送问题"><span aria-hidden="true">↑</span></button>}</div>
    </form>
    <p className="composer-hint">Shift + Enter 换行 · 回答可能存在错误</p>
  </section>;
}

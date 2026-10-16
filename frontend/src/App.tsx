import { useCallback, useEffect, useRef, useState } from "react";
import { createConversation, deleteConversation, deleteDocument, getConversation, ingestDocuments, listConversations, listDocuments, sendMessageStream, updateConversation, uploadDocument } from "./api";
import type { Conversation, ConversationSummary, IngestResult, KnowledgeDocument, Message, StreamEventMap } from "./types";
import { ChatPanel } from "./components/ChatPanel";
import { ConversationList } from "./components/ConversationList";
import { EvidencePanel } from "./components/EvidencePanel";
import { KnowledgePanel } from "./components/KnowledgePanel";
import { createOptimisticMessagePair, getLatestAssistantMessage, getQuestionForMessage, markStreamError, patchMessage, replaceMessageId, waitForPaint } from "./ui";

type View = "chat" | "knowledge";

export default function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [editRequest, setEditRequest] = useState<{ content: string; token: number } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [view, setView] = useState<View>("chat");
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);
  const [deletingDocumentId, setDeletingDocumentId] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const selectedIdRef = useRef<number | null>(null);
  const lastQuestionRef = useRef<string>("");
  const editSequenceRef = useRef(0);
  const temporaryMessageIdRef = useRef(-1);
  const streamingRef = useRef(false);
  const streamStateRef = useRef<{ conversationId: number; userId: number; assistantId: number }>({ conversationId: 0, userId: 0, assistantId: 0 });

  const selectConversation = (id: number | null) => {
    selectedIdRef.current = id;
    setSelectedId(id);
  };

  const loadConversations = useCallback(async () => {
    try { setLoadError(null); const items = await listConversations(); setConversations(items); if (selectedIdRef.current == null && items[0]) selectConversation(items[0].id); } catch (error) { setLoadError(error instanceof Error ? error.message : "无法加载会话"); }
  }, []);

  const loadConversation = useCallback(async (id: number) => {
    try {
      const item = await getConversation(id);
      if (selectedIdRef.current !== id) return;
      if (streamingRef.current && streamStateRef.current.conversationId === id) return;
      setConversation(item); setMessages(item.messages); setSelectedMessage([...item.messages].reverse().find((message) => message.role === "assistant") || null);
    } catch (error) { if (selectedIdRef.current === id) setLoadError(error instanceof Error ? error.message : "无法加载会话内容"); }
  }, []);

  useEffect(() => { void loadConversations(); void loadDocuments(); }, [loadConversations]);
  useEffect(() => {
    const preserveActiveStream = selectedId != null && streamingRef.current && streamStateRef.current.conversationId === selectedId;
    if (!preserveActiveStream) {
      setConversation(null);
      setMessages([]);
      setSelectedMessage(null);
      if (selectedId != null) void loadConversation(selectedId);
    }
  }, [selectedId, loadConversation]);

  async function loadDocuments() { setDocumentsLoading(true); try { setKnowledgeError(null); setDocuments(await listDocuments()); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "无法加载知识库文件"); } finally { setDocumentsLoading(false); } }

  const handleNew = async () => { try { const item = await createConversation(); setConversations((items) => [item, ...items]); selectConversation(item.id); setView("chat"); } catch (error) { setLoadError(error instanceof Error ? error.message : "无法创建会话"); } };
  const handleRename = async (item: ConversationSummary) => { const title = window.prompt("重命名会话", item.title); if (!title?.trim()) return; try { const updated = await updateConversation(item.id, { title: title.trim() }); setConversations((items) => items.map((entry) => entry.id === item.id ? updated : entry)); setConversation((current) => current?.id === item.id ? { ...current, ...updated } : current); } catch (error) { setLoadError(error instanceof Error ? error.message : "无法重命名会话"); } };
  const handleArchive = async (item: ConversationSummary) => { try { await updateConversation(item.id, { archived: true }); setConversations((items) => items.filter((entry) => entry.id !== item.id)); if (selectedIdRef.current === item.id) { selectConversation(null); setConversation(null); setMessages([]); } } catch (error) { setLoadError(error instanceof Error ? error.message : "无法归档会话"); } };
  const handleDelete = async (item: ConversationSummary) => { if (!window.confirm(`确定删除“${item.title || "未命名会话"}”吗？`)) return; try { await deleteConversation(item.id); setConversations((items) => items.filter((entry) => entry.id !== item.id)); if (selectedIdRef.current === item.id) { selectConversation(null); setConversation(null); setMessages([]); } } catch (error) { setLoadError(error instanceof Error ? error.message : "无法删除会话"); } };

  const submitQuestion = async (question: string, allowWebFallback = true) => {
    let id = selectedId;
    const temporaryUserId = temporaryMessageIdRef.current--;
    const temporaryAssistantId = temporaryMessageIdRef.current--;
    const optimisticPair = createOptimisticMessagePair(question, id ?? 0, temporaryUserId, temporaryAssistantId);
    streamStateRef.current = { conversationId: id ?? 0, userId: temporaryUserId, assistantId: temporaryAssistantId };
    lastQuestionRef.current = question;
    setMessages((current) => [...current, optimisticPair.user, optimisticPair.assistant]);
    setStreaming(true);
    streamingRef.current = true;
    setStreamError(null); setSelectedMessage(null); setEvidenceOpen(false);
    const controller = new AbortController(); abortRef.current = controller;
    let streamFailed = false;
    let streamCompleted = false;
    try {
      // Give the optimistic user/assistant pair one browser frame before any
      // conversation creation, retrieval, or model request begins.
      await waitForPaint();
      if (controller.signal.aborted) throw new Error("stream aborted before start");
      if (id == null) {
        const created = await createConversation();
        setConversations((items) => [created, ...items]);
        setConversation(created);
        id = created.id;
        const conversationId = id;
        streamStateRef.current.conversationId = id;
        setMessages((current) => current.map((message) => message.id === temporaryUserId || message.id === temporaryAssistantId ? { ...message, conversation_id: conversationId } : message));
        selectConversation(id);
      }
      await sendMessageStream(id, question, (event, data) => {
        if (event === "message_start") {
          const start = data as StreamEventMap["message_start"];
          if (start.user_message_id) {
            streamStateRef.current.userId = start.user_message_id;
            setMessages((current) => replaceMessageId(current, temporaryUserId, start.user_message_id));
          }
          return;
        }
        if (event === "token") {
          const token = (data as StreamEventMap["token"]).text ?? (data as StreamEventMap["token"]).content ?? "";
          if (!token) return;
          setMessages((current) => current.map((message) => message.id === streamStateRef.current.assistantId ? { ...message, content: `${message.content}${token}`, status: "streaming" } : message));
          setSelectedMessage((current) => current ? { ...current, content: `${current.content}${token}`, status: "streaming" } : null);
        }
        if (event === "metadata") {
          const metadata = data as StreamEventMap["metadata"];
          const previousAssistantId = streamStateRef.current.assistantId;
          if (metadata.assistant_message_id) {
            streamStateRef.current.assistantId = metadata.assistant_message_id;
            setMessages((current) => {
              const withId = replaceMessageId(current, previousAssistantId, metadata.assistant_message_id);
              return patchMessage(withId, metadata.assistant_message_id, {
                citations: metadata.citations,
                confidence: metadata.confidence,
                confidence_score: metadata.confidence_score,
                retrieval_debug: metadata.retrieval_debug,
                mode: metadata.mode,
                web_sources: metadata.web_sources,
              });
            });
          }
          setSelectedMessage((current) => current ? { ...current, citations: metadata.citations, confidence: metadata.confidence, confidence_score: metadata.confidence_score, retrieval_debug: metadata.retrieval_debug, mode: metadata.mode, web_sources: metadata.web_sources } : null);
        }
        if (event === "complete") {
          const complete = data as StreamEventMap["complete"];
          streamCompleted = true;
          const previousAssistantId = streamStateRef.current.assistantId;
          if (complete.assistant_message_id) streamStateRef.current.assistantId = complete.assistant_message_id;
          setMessages((current) => {
            const withId = complete.assistant_message_id ? replaceMessageId(current, previousAssistantId, complete.assistant_message_id) : current;
            return patchMessage(withId, complete.assistant_message_id || previousAssistantId, { content: complete.answer, status: "complete", citations: complete.citations, confidence: complete.confidence, confidence_score: complete.confidence_score, retrieval_debug: complete.retrieval_debug, mode: complete.mode, web_sources: complete.web_sources, error_message: null });
          });
          setSelectedMessage((current) => current ? { ...current, id: complete.assistant_message_id || current.id, content: complete.answer, status: "complete", citations: complete.citations, confidence: complete.confidence, confidence_score: complete.confidence_score, retrieval_debug: complete.retrieval_debug, mode: complete.mode, web_sources: complete.web_sources, error_message: null } : null);
        }
        if (event === "error") {
          const streamErrorEvent = data as StreamEventMap["error"];
          streamFailed = true;
          setStreamError(streamErrorEvent.error);
          setMessages((current) => markStreamError(current, streamStateRef.current.userId, streamStateRef.current.assistantId, streamErrorEvent.error));
        }
      }, controller.signal, allowWebFallback);
      if (!streamCompleted && !streamFailed) throw new Error("回答流在完成前结束");
      streamingRef.current = false;
      if (!streamFailed) {
        await loadConversation(id); await loadConversations();
      }
    } catch (error) {
      streamFailed = true;
      const message = controller.signal.aborted ? "已停止生成。" : error instanceof Error ? error.message : "回答生成失败";
      setStreamError(message);
      setMessages((current) => markStreamError(current, streamStateRef.current.userId, streamStateRef.current.assistantId, message));
      streamingRef.current = false;
    }
    finally { setStreaming(false); streamingRef.current = false; abortRef.current = null; }
  };

  const retry = (message: Message) => {
    const question = message.role === "user" ? message.content : getQuestionForMessage(messages, message.id)?.content || lastQuestionRef.current;
    if (question) void submitQuestion(question);
  };
  const edit = (message: Message) => {
    if (message.role !== "user") return;
    lastQuestionRef.current = message.content;
    editSequenceRef.current += 1;
    setEditRequest({ content: message.content, token: editSequenceRef.current });
  };
  const handleUpload = async (file: File) => { setUploading(true); setKnowledgeError(null); try { await uploadDocument(file); await loadDocuments(); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "上传文件失败"); } finally { setUploading(false); } };
  const handleIngest = async () => { setIngesting(true); setKnowledgeError(null); try { setIngestResult(await ingestDocuments()); await loadDocuments(); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "建立索引失败"); } finally { setIngesting(false); } };
  const handleDeleteDocument = async (document: KnowledgeDocument) => { setDeletingDocumentId(document.id); setKnowledgeError(null); try { await deleteDocument(document.id); setDocuments((items) => items.filter((item) => item.id !== document.id)); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "删除知识库文件失败"); } finally { setDeletingDocumentId(null); } };
  const toggleEvidence = () => { if (!evidenceOpen) setSelectedMessage((current) => current ?? getLatestAssistantMessage(messages)); setEvidenceOpen((open) => !open); };

  return <main className={`app-shell ${evidenceOpen ? "evidence-visible" : ""}`}>
    {loadError ? <div className="global-error" role="alert">{loadError}<button type="button" onClick={() => void loadConversations()}>重试</button></div> : null}
    <div className={`nav-scrim ${mobileNavOpen ? "is-visible" : ""}`} onClick={() => setMobileNavOpen(false)} />
    <div className={`left-drawer ${mobileNavOpen ? "is-open" : ""}`}><ConversationList conversations={conversations} selectedId={selectedId} loading={!conversations.length && !loadError} onSelect={(id) => { selectConversation(id); setView("chat"); }} onNew={handleNew} onRename={handleRename} onArchive={handleArchive} onDelete={handleDelete} onKnowledge={() => setView("knowledge")} onCloseMobile={() => setMobileNavOpen(false)} /></div>
    <div className="workspace-main">{view === "chat" ? <ChatPanel title={conversation?.title || "新建会话"} messages={messages} streaming={streaming} streamError={streamError} editRequest={editRequest} selectedMessageId={selectedMessage?.id || null} onSelectMessage={(message) => { setSelectedMessage(message); setEvidenceOpen(true); }} onSubmit={submitQuestion} onStop={() => abortRef.current?.abort()} onRetry={retry} onEdit={edit} onOpenNavigation={() => setMobileNavOpen(true)} onToggleEvidence={toggleEvidence} /> : <KnowledgePanel documents={documents} loading={documentsLoading} uploading={uploading} ingesting={ingesting} ingestResult={ingestResult} error={knowledgeError} onUpload={handleUpload} onIngest={handleIngest} onRefresh={() => void loadDocuments()} onBack={() => setView("chat")} deletingDocumentId={deletingDocumentId} onDelete={handleDeleteDocument} />}</div>
    {view === "chat" ? <EvidencePanel message={selectedMessage} documents={documents} open={evidenceOpen} onClose={() => setEvidenceOpen(false)} /> : null}
  </main>;
}

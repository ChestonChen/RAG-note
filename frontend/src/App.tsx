import { useCallback, useEffect, useRef, useState } from "react";
import { createConversation, deleteConversation, getConversation, ingestDocuments, listConversations, listDocuments, sendMessageStream, updateConversation, uploadDocument } from "./api";
import type { Conversation, ConversationSummary, IngestResult, KnowledgeDocument, Message, StreamEventMap } from "./types";
import { ChatPanel } from "./components/ChatPanel";
import { ConversationList } from "./components/ConversationList";
import { EvidencePanel } from "./components/EvidencePanel";
import { KnowledgePanel } from "./components/KnowledgePanel";
import { getQuestionForMessage } from "./ui";

type View = "chat" | "knowledge";

export default function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [draftAnswer, setDraftAnswer] = useState("");
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
  const abortRef = useRef<AbortController | null>(null);
  const selectedIdRef = useRef<number | null>(null);
  const lastQuestionRef = useRef<string>("");
  const editSequenceRef = useRef(0);

  const selectConversation = (id: number | null) => {
    selectedIdRef.current = id;
    setSelectedId(id);
  };

  const loadConversations = useCallback(async () => {
    try { setLoadError(null); const items = await listConversations(); setConversations(items); if (selectedIdRef.current == null && items[0]) selectConversation(items[0].id); } catch (error) { setLoadError(error instanceof Error ? error.message : "Could not load conversations"); }
  }, []);

  const loadConversation = useCallback(async (id: number) => {
    try {
      const item = await getConversation(id);
      if (selectedIdRef.current !== id) return;
      setConversation(item); setMessages(item.messages); setSelectedMessage([...item.messages].reverse().find((message) => message.role === "assistant") || null);
    } catch (error) { if (selectedIdRef.current === id) setLoadError(error instanceof Error ? error.message : "Could not load conversation"); }
  }, []);

  useEffect(() => { void loadConversations(); void loadDocuments(); }, [loadConversations]);
  useEffect(() => {
    setConversation(null);
    setMessages([]);
    setSelectedMessage(null);
    if (selectedId != null) void loadConversation(selectedId);
  }, [selectedId, loadConversation]);

  async function loadDocuments() { setDocumentsLoading(true); try { setKnowledgeError(null); setDocuments(await listDocuments()); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "Could not load documents"); } finally { setDocumentsLoading(false); } }

  const handleNew = async () => { try { const item = await createConversation(); setConversations((items) => [item, ...items]); selectConversation(item.id); setView("chat"); } catch (error) { setLoadError(error instanceof Error ? error.message : "Could not create conversation"); } };
  const handleRename = async (item: ConversationSummary) => { const title = window.prompt("Rename conversation", item.title); if (!title?.trim()) return; try { const updated = await updateConversation(item.id, { title: title.trim() }); setConversations((items) => items.map((entry) => entry.id === item.id ? updated : entry)); setConversation((current) => current?.id === item.id ? { ...current, ...updated } : current); } catch (error) { setLoadError(error instanceof Error ? error.message : "Could not rename conversation"); } };
  const handleArchive = async (item: ConversationSummary) => { try { await updateConversation(item.id, { archived: true }); setConversations((items) => items.filter((entry) => entry.id !== item.id)); if (selectedIdRef.current === item.id) { selectConversation(null); setConversation(null); setMessages([]); } } catch (error) { setLoadError(error instanceof Error ? error.message : "Could not archive conversation"); } };
  const handleDelete = async (item: ConversationSummary) => { if (!window.confirm(`Delete “${item.title || "Untitled conversation"}”?`)) return; try { await deleteConversation(item.id); setConversations((items) => items.filter((entry) => entry.id !== item.id)); if (selectedIdRef.current === item.id) { selectConversation(null); setConversation(null); setMessages([]); } } catch (error) { setLoadError(error instanceof Error ? error.message : "Could not delete conversation"); } };

  const submitQuestion = async (question: string, allowWebFallback = true) => {
    let id = selectedId;
    if (id == null) { const created = await createConversation(); setConversations((items) => [created, ...items]); selectConversation(created.id); id = created.id; }
    lastQuestionRef.current = question; setStreaming(true); setDraftAnswer(""); setStreamError(null); setSelectedMessage(null); setEvidenceOpen(false);
    const controller = new AbortController(); abortRef.current = controller;
    try {
      await sendMessageStream(id, question, (event, data) => {
        if (event === "token") {
          const token = (data as StreamEventMap["token"]).text ?? (data as StreamEventMap["token"]).content ?? "";
          if (!token) return;
          setDraftAnswer((current) => current + token);
          setSelectedMessage((current) => ({
            id: current?.id ?? -1,
            conversation_id: current?.conversation_id ?? id,
            role: "assistant",
            content: `${current?.content ?? ""}${token}`,
            status: "streaming",
            citations: current?.citations ?? [],
            retrieval_debug: current?.retrieval_debug ?? {},
            confidence: current?.confidence ?? null,
            confidence_score: current?.confidence_score ?? null,
            mode: current?.mode ?? "",
            web_sources: current?.web_sources ?? [],
            error_message: null,
            created_at: current?.created_at ?? new Date().toISOString(),
          }));
        }
        if (event === "metadata") { const metadata = data as StreamEventMap["metadata"]; setSelectedMessage((current) => ({
          id: current?.id ?? metadata.assistant_message_id,
          conversation_id: current?.conversation_id ?? id,
          role: "assistant",
          content: current?.content ?? "",
          status: current?.status ?? "streaming",
          citations: metadata.citations,
          confidence: metadata.confidence,
          confidence_score: metadata.confidence_score,
          retrieval_debug: metadata.retrieval_debug,
          mode: metadata.mode,
          web_sources: metadata.web_sources,
          error_message: null,
          created_at: current?.created_at ?? new Date().toISOString(),
        })); }
        if (event === "error") setStreamError((data as StreamEventMap["error"]).error);
      }, controller.signal, allowWebFallback);
      await loadConversation(id); await loadConversations();
    } catch (error) {
      if (!controller.signal.aborted) setStreamError(error instanceof Error ? error.message : "The answer stream failed");
      try { await loadConversation(id); } catch { /* Keep the transport error visible if refresh also fails. */ }
    }
    finally { setDraftAnswer(""); setStreaming(false); abortRef.current = null; }
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
  const handleUpload = async (file: File) => { setUploading(true); setKnowledgeError(null); try { await uploadDocument(file); await loadDocuments(); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "Upload failed"); } finally { setUploading(false); } };
  const handleIngest = async () => { setIngesting(true); setKnowledgeError(null); try { setIngestResult(await ingestDocuments()); await loadDocuments(); } catch (error) { setKnowledgeError(error instanceof Error ? error.message : "Ingestion failed"); } finally { setIngesting(false); } };

  return <main className={`app-shell ${evidenceOpen ? "evidence-visible" : ""}`}>
    {loadError ? <div className="global-error" role="alert">{loadError}<button type="button" onClick={() => void loadConversations()}>Retry</button></div> : null}
    <div className={`nav-scrim ${mobileNavOpen ? "is-visible" : ""}`} onClick={() => setMobileNavOpen(false)} />
    <div className={`left-drawer ${mobileNavOpen ? "is-open" : ""}`}><ConversationList conversations={conversations} selectedId={selectedId} loading={!conversations.length && !loadError} onSelect={(id) => { selectConversation(id); setView("chat"); }} onNew={handleNew} onRename={handleRename} onArchive={handleArchive} onDelete={handleDelete} onKnowledge={() => setView("knowledge")} onCloseMobile={() => setMobileNavOpen(false)} /></div>
    <div className="workspace-main">{view === "chat" ? <ChatPanel title={conversation?.title || "New conversation"} messages={messages} streaming={streaming} draftAnswer={draftAnswer} streamError={streamError} editRequest={editRequest} selectedMessageId={selectedMessage?.id || null} onSelectMessage={(message) => { setSelectedMessage(message); setEvidenceOpen(true); }} onSubmit={submitQuestion} onStop={() => abortRef.current?.abort()} onRetry={retry} onEdit={edit} onOpenNavigation={() => setMobileNavOpen(true)} onToggleEvidence={() => setEvidenceOpen((open) => !open)} /> : <KnowledgePanel documents={documents} loading={documentsLoading} uploading={uploading} ingesting={ingesting} ingestResult={ingestResult} error={knowledgeError} onUpload={handleUpload} onIngest={handleIngest} onRefresh={() => void loadDocuments()} onBack={() => setView("chat")} />}</div>
    {view === "chat" ? <EvidencePanel message={selectedMessage} documents={documents} open={evidenceOpen} onClose={() => setEvidenceOpen(false)} /> : null}
  </main>;
}

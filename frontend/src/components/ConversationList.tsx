import type { ConversationSummary } from "../types";
import { formatRelativeTime } from "../ui";

type ConversationListProps = {
  conversations: ConversationSummary[];
  selectedId: number | null;
  loading: boolean;
  onSelect: (id: number) => void;
  onNew: () => void;
  onRename: (conversation: ConversationSummary) => void;
  onArchive: (conversation: ConversationSummary) => void;
  onDelete: (conversation: ConversationSummary) => void;
  onKnowledge: () => void;
  onCloseMobile?: () => void;
};

export function ConversationList({
  conversations,
  selectedId,
  loading,
  onSelect,
  onNew,
  onRename,
  onArchive,
  onDelete,
  onKnowledge,
  onCloseMobile,
}: ConversationListProps) {
  return (
    <aside className="conversation-pane" aria-label="Conversations">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">R</div>
        <div>
          <strong>RAG workspace</strong>
          <span>Knowledge, in context</span>
        </div>
        {onCloseMobile ? <button className="icon-button mobile-only" onClick={onCloseMobile} aria-label="Close navigation">×</button> : null}
      </div>
      <button className="new-chat-button" onClick={onNew} type="button"><span aria-hidden="true">+</span> New conversation</button>
      <div className="pane-label"><span>Recent</span><span className="count-badge">{conversations.length}</span></div>
      <div className="conversation-items" aria-live="polite">
        {loading ? <div className="list-skeleton" aria-label="Loading conversations"><i /><i /><i /></div> : null}
        {!loading && conversations.length === 0 ? <p className="muted-empty">No conversations yet.<br />Start with a question.</p> : null}
        {conversations.map((conversation) => (
          <div className={`conversation-item ${selectedId === conversation.id ? "is-selected" : ""}`} key={conversation.id}>
            <button className="conversation-select" onClick={() => { onSelect(conversation.id); onCloseMobile?.(); }} type="button">
              <span className="conversation-title">{conversation.title || "Untitled conversation"}</span>
              <span className="conversation-time">{formatRelativeTime(conversation.updated_at)}</span>
            </button>
            <details className="conversation-actions">
              <summary aria-label={`Actions for ${conversation.title || "conversation"}`}>•••</summary>
              <div className="action-menu">
                <button type="button" onClick={() => onRename(conversation)}>Rename</button>
                <button type="button" onClick={() => onArchive(conversation)}>Archive</button>
                <button className="danger-action" type="button" onClick={() => onDelete(conversation)}>Delete</button>
              </div>
            </details>
          </div>
        ))}
      </div>
      <button className="knowledge-link" onClick={() => { onKnowledge(); onCloseMobile?.(); }} type="button">
        <span className="knowledge-icon" aria-hidden="true">⌁</span>
        <span><strong>Knowledge base</strong><small>Sources & ingestion</small></span>
        <span aria-hidden="true">›</span>
      </button>
      <div className="pane-footer"><span className="status-dot" /> API connected locally</div>
    </aside>
  );
}

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
    <aside className="conversation-pane" aria-label="会话列表">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <img src="/brand-icon.svg" alt="" />
        </div>
        <div>
          <strong>RAG 知识工作台</strong>
          <span>让知识有据可查</span>
        </div>
        {onCloseMobile ? <button className="icon-button mobile-only" onClick={onCloseMobile} aria-label="关闭导航">×</button> : null}
      </div>
      <button className="new-chat-button" onClick={onNew} type="button"><span aria-hidden="true">+</span> 新建会话</button>
      <div className="pane-label"><span>最近会话</span><span className="count-badge">{conversations.length}</span></div>
      <div className="conversation-items" aria-live="polite">
        {loading ? <div className="list-skeleton" aria-label="正在加载会话"><i /><i /><i /></div> : null}
        {!loading && conversations.length === 0 ? <p className="muted-empty">还没有会话。<br />先提出一个问题吧。</p> : null}
        {conversations.map((conversation) => (
          <div className={`conversation-item ${selectedId === conversation.id ? "is-selected" : ""}`} key={conversation.id}>
            <button className="conversation-select" onClick={() => { onSelect(conversation.id); onCloseMobile?.(); }} type="button">
              <span className="conversation-title">{conversation.title || "未命名会话"}</span>
              <span className="conversation-time">{formatRelativeTime(conversation.updated_at)}</span>
            </button>
            <details className="conversation-actions">
              <summary aria-label={`会话操作：${conversation.title || "未命名会话"}`}>•••</summary>
              <div className="action-menu">
                <button type="button" onClick={() => onRename(conversation)}>重命名</button>
                <button type="button" onClick={() => onArchive(conversation)}>归档</button>
                <button className="danger-action" type="button" onClick={() => onDelete(conversation)}>删除</button>
              </div>
            </details>
          </div>
        ))}
      </div>
      <button className="knowledge-link" onClick={() => { onKnowledge(); onCloseMobile?.(); }} type="button">
        <span className="knowledge-icon" aria-hidden="true">⌁</span>
        <span><strong>知识库</strong><small>文件与索引</small></span>
        <span aria-hidden="true">›</span>
      </button>
      <div className="pane-footer"><span className="status-dot" /> 本地服务已连接</div>
    </aside>
  );
}

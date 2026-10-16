import type { KnowledgeDocument, Message, SearchHit } from "../types";
import { documentFileUrl } from "../api";
import { findKnowledgeDocument, getConfidenceLabel, getEvidencePanelClassName, getRetrievalChannelLabel, getRetrievalModeLabel } from "../ui";

type EvidencePanelProps = { message: Message | null; documents: KnowledgeDocument[]; open: boolean; onClose: () => void };

function debugHits(message: Message): SearchHit[] {
  const value = message.retrieval_debug?.hits;
  return Array.isArray(value) ? value as SearchHit[] : [];
}

export function EvidencePanel({ message, documents, open, onClose }: EvidencePanelProps) {
  const hits = message ? debugHits(message) : [];
  const citations = message?.citations ?? [];
  return <aside className={getEvidencePanelClassName(open)} aria-label="回答证据" aria-hidden={!open}>
    <div className="evidence-header"><div><span className="chat-kicker">追溯回答依据</span><h2>证据</h2></div><button className="icon-button" type="button" onClick={onClose} aria-label="关闭证据">×</button></div>
    {!message ? <div className="evidence-empty"><span aria-hidden="true">⌕</span><h3>选择一条回答</h3><p>点击研究助手的回答，查看来源、分数和检索过程。</p></div> : <div className="evidence-scroll">
      <div className="evidence-summary"><span className="confidence-dot" /><div><strong>{getConfidenceLabel(message.confidence)}置信度</strong><span>{message.confidence_score == null ? "暂无置信度分数" : `${Math.round(message.confidence_score * 100)}% 回答置信度`} · {getRetrievalModeLabel(message.mode)}</span></div></div>
      <section className="evidence-section"><div className="section-heading"><h3>来源</h3><span>{citations.length}</span></div>{citations.length ? citations.map((citation) => {
        const localDocument = citation.source.startsWith("http") ? null : findKnowledgeDocument(citation.source, documents);
        const sourceUrl = citation.source.startsWith("http") ? citation.source : localDocument ? documentFileUrl(localDocument.id) : null;
        return <div className="evidence-source" key={citation.citation_id}><div className="source-index">{citation.citation_id}</div><div><strong>{citation.title || citation.source}</strong><span>{citation.heading?.join(" › ") || "文档来源"}{citation.page_from ? ` · 第 ${citation.page_from} 页` : ""}</span><div className="source-score"><span>相似度</span><strong>{Number(citation.score ?? 0).toFixed(3)}</strong></div>{sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer">打开来源 ↗</a> : <span className="source-unavailable">索引文件不可用</span>}</div></div>;
      }) : <p className="muted-empty">这条回答没有返回引用来源。</p>}</section>
      {hits.length ? <section className="evidence-section"><div className="section-heading"><h3>检索分片</h3><span>{hits.length}</span></div>{hits.slice(0, 8).map((hit) => <details className="retrieval-hit" key={hit.chunk_id}><summary><span>#{hit.rank ?? hit.chunk_id}</span><strong>{hit.title || hit.source}</strong><b>{Number(hit.score ?? 0).toFixed(3)}</b></summary><p>{hit.content}</p><small>{hit.matched_channels?.map(getRetrievalChannelLabel).join(" · ") || "混合检索"} · 向量 {hit.vector_score == null ? "-" : Number(hit.vector_score).toFixed(2)} · 关键词 {hit.keyword_score == null ? "-" : Number(hit.keyword_score).toFixed(2)}</small></details>)}</section> : null}
      {message.web_sources?.length ? <section className="evidence-section"><div className="section-heading"><h3>联网来源</h3><span>{message.web_sources.length}</span></div>{message.web_sources.map((source) => <a className="web-source" key={source.url} href={source.url} target="_blank" rel="noreferrer"><strong>{source.title}</strong><span>{source.url}</span></a>)}</section> : null}
    </div>}
  </aside>;
}

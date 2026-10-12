import type { KnowledgeDocument, Message, SearchHit } from "../types";
import { documentFileUrl } from "../api";
import { findKnowledgeDocument } from "../ui";

type EvidencePanelProps = { message: Message | null; documents: KnowledgeDocument[]; open: boolean; onClose: () => void };

function debugHits(message: Message): SearchHit[] {
  const value = message.retrieval_debug?.hits;
  return Array.isArray(value) ? value as SearchHit[] : [];
}

export function EvidencePanel({ message, documents, open, onClose }: EvidencePanelProps) {
  const hits = message ? debugHits(message) : [];
  const citations = message?.citations ?? [];
  return <aside className={`evidence-pane ${open ? "is-open" : ""}`} aria-label="Evidence">
    <div className="evidence-header"><div><span className="chat-kicker">Trace the answer</span><h2>Evidence</h2></div><button className="icon-button" onClick={onClose} aria-label="Close evidence">×</button></div>
    {!message ? <div className="evidence-empty"><span aria-hidden="true">⌕</span><h3>Select an answer</h3><p>Click an assistant response to inspect its sources, scores, and retrieval trace.</p></div> : <div className="evidence-scroll">
      <div className="evidence-summary"><span className="confidence-dot" /><div><strong>{message.confidence || "Unrated"} confidence</strong><span>{message.confidence_score == null ? "No confidence score" : `${Math.round(message.confidence_score * 100)}% answer confidence`} · {message.mode || "hybrid retrieval"}</span></div></div>
      <section className="evidence-section"><div className="section-heading"><h3>Sources</h3><span>{citations.length}</span></div>{citations.length ? citations.map((citation) => {
        const localDocument = citation.source.startsWith("http") ? null : findKnowledgeDocument(citation.source, documents);
        const sourceUrl = citation.source.startsWith("http") ? citation.source : localDocument ? documentFileUrl(localDocument.id) : null;
        return <div className="evidence-source" key={citation.citation_id}><div className="source-index">{citation.citation_id}</div><div><strong>{citation.title || citation.source}</strong><span>{citation.heading?.join(" › ") || "Document source"}{citation.page_from ? ` · p. ${citation.page_from}` : ""}</span><div className="source-score"><span>Score</span><strong>{Number(citation.score ?? 0).toFixed(3)}</strong></div>{sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer">Open source ↗</a> : <span className="source-unavailable">Indexed file unavailable</span>}</div></div>;
      }) : <p className="muted-empty">No citations were returned for this answer.</p>}</section>
      {hits.length ? <section className="evidence-section"><div className="section-heading"><h3>Retrieval hits</h3><span>{hits.length}</span></div>{hits.slice(0, 8).map((hit) => <details className="retrieval-hit" key={hit.chunk_id}><summary><span>#{hit.rank ?? hit.chunk_id}</span><strong>{hit.title || hit.source}</strong><b>{Number(hit.score ?? 0).toFixed(3)}</b></summary><p>{hit.content}</p><small>{hit.matched_channels?.join(" · ") || "hybrid"} · vector {hit.vector_score == null ? "-" : Number(hit.vector_score).toFixed(2)} · keyword {hit.keyword_score == null ? "-" : Number(hit.keyword_score).toFixed(2)}</small></details>)}</section> : null}
      {message.web_sources?.length ? <section className="evidence-section"><div className="section-heading"><h3>Web sources</h3><span>{message.web_sources.length}</span></div>{message.web_sources.map((source) => <a className="web-source" key={source.url} href={source.url} target="_blank" rel="noreferrer"><strong>{source.title}</strong><span>{source.url}</span></a>)}</section> : null}
    </div>}
  </aside>;
}

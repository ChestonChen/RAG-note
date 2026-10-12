import { useRef, useState } from "react";
import type { IngestResult, KnowledgeDocument } from "../types";
import { documentFileUrl } from "../api";
import { canStartIngestion } from "../ui";

type KnowledgePanelProps = { documents: KnowledgeDocument[]; loading: boolean; uploading: boolean; ingesting: boolean; ingestResult: IngestResult | null; error: string | null; onUpload: (file: File) => void; onIngest: () => void; onRefresh: () => void; onBack: () => void };

export function KnowledgePanel({ documents, loading, uploading, ingesting, ingestResult, error, onUpload, onIngest, onRefresh, onBack }: KnowledgePanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const selectFile = (file: File | undefined) => {
    if (!file) return;
    if (!/\.(md|markdown|pdf)$/i.test(file.name)) { setFileError("Choose a Markdown or PDF file."); return; }
    setFileError(null); onUpload(file);
  };
  return <section className="knowledge-panel">
    <header className="knowledge-header"><button className="icon-button" onClick={onBack} aria-label="Back to conversations">←</button><div><span className="chat-kicker">Workspace</span><h1>Knowledge base</h1></div><button className="icon-button" onClick={onRefresh} aria-label="Refresh documents">↻</button></header>
    <div className="knowledge-content">
      <div className="knowledge-intro"><p>Keep answers grounded in the files your team trusts.</p><div className="knowledge-actions"><button className="secondary-button" type="button" onClick={() => inputRef.current?.click()} disabled={uploading}><span aria-hidden="true">↑</span>{uploading ? "Uploading..." : "Upload file"}</button><button className="primary-button" type="button" onClick={onIngest} disabled={!canStartIngestion(ingesting, uploading)}><span aria-hidden="true">↻</span>{ingesting ? "Indexing..." : "Ingest files"}</button></div><input ref={inputRef} className="visually-hidden" type="file" accept=".md,.markdown,.pdf,application/pdf,text/markdown" onChange={(event) => { selectFile(event.target.files?.[0]); event.currentTarget.value = ""; }} />{fileError || error ? <p className="inline-error" role="alert">{fileError || error}</p> : null}</div>
      {ingestResult ? <div className="ingest-result"><strong>Indexing complete</strong><span>{ingestResult.processed_files} processed · {ingestResult.chunk_count} chunks · {ingestResult.failed_files} failed</span></div> : null}
      <div className="document-list-heading"><h2>Indexed files</h2><span>{documents.length} files</span></div>
      {loading ? <div className="document-skeleton"><i /><i /><i /></div> : null}
      {!loading && documents.length === 0 ? <div className="knowledge-empty"><span aria-hidden="true">⌁</span><h3>Your sources will appear here</h3><p>Upload a Markdown or PDF file to start building your searchable context.</p></div> : <div className="document-list">{documents.map((document) => <article className="document-row" key={document.id}><div className={`file-icon file-${document.type}`}>{document.type === "pdf" ? "PDF" : "MD"}</div><div className="document-details"><strong title={document.title}>{document.title}</strong><span>{document.chunk_count} chunks · {document.source_path}</span>{document.error_message ? <small className="document-error">{document.error_message}</small> : null}</div><span className={`document-status status-${document.status}`}>{document.status}</span><a className="icon-button" href={documentFileUrl(document.id)} target="_blank" rel="noreferrer" aria-label={`Open ${document.title}`}>↗</a></article>)}</div>}
    </div>
  </section>;
}

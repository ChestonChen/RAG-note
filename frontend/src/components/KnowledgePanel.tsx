import { useRef, useState } from "react";
import type { IngestResult, KnowledgeDocument } from "../types";
import { documentFileUrl } from "../api";
import { canStartIngestion, getDocumentStatusLabel, knowledgeWorkflowCopy } from "../ui";

type KnowledgePanelProps = { documents: KnowledgeDocument[]; loading: boolean; uploading: boolean; ingesting: boolean; ingestResult: IngestResult | null; error: string | null; onUpload: (file: File) => void; onIngest: () => void; onRefresh: () => void; onBack: () => void; deletingDocumentId: number | null; onDelete: (document: KnowledgeDocument) => void };

export function KnowledgePanel({ documents, loading, uploading, ingesting, ingestResult, error, onUpload, onIngest, onRefresh, onBack, deletingDocumentId, onDelete }: KnowledgePanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const selectFile = (file: File | undefined) => {
    if (!file) return;
    if (!/\.(md|markdown|pdf)$/i.test(file.name)) { setFileError("请选择 Markdown 或 PDF 文件。"); return; }
    setFileError(null); onUpload(file);
  };
  const requestDelete = (document: KnowledgeDocument) => {
    if (!window.confirm(`确定删除“${document.title}”吗？删除后该文件及其索引将不再参与问答。`)) return;
    void onDelete(document);
  };
  return <section className="knowledge-panel">
    <header className="knowledge-header"><button className="icon-button" onClick={onBack} aria-label="返回会话">←</button><div><span className="chat-kicker">工作区</span><h1>知识库</h1></div><button className="icon-button" onClick={onRefresh} aria-label="刷新文件列表">↻</button></header>
    <div className="knowledge-content">
      <div className="knowledge-intro"><p>上传可信的 Markdown 或 PDF 文件，建立索引后即可用于问答。</p><div className="knowledge-actions"><button className="primary-button" type="button" onClick={() => inputRef.current?.click()} disabled={uploading}><span aria-hidden="true">↑</span>{uploading ? "正在上传..." : knowledgeWorkflowCopy.uploadPrimary}</button><button className="secondary-button" type="button" onClick={onIngest} disabled={!canStartIngestion(ingesting, uploading, documents.length > 0)}><span aria-hidden="true">↻</span>{ingesting ? "正在建立索引..." : knowledgeWorkflowCopy.ingest}</button></div><p className="knowledge-action-note"><span>{knowledgeWorkflowCopy.supportedFormats}</span><span>{knowledgeWorkflowCopy.indexHint}</span></p><input ref={inputRef} className="visually-hidden" type="file" accept=".md,.markdown,.pdf,application/pdf,text/markdown" onChange={(event) => { selectFile(event.target.files?.[0]); event.currentTarget.value = ""; }} />{fileError || error ? <p className="inline-error" role="alert">{fileError || error}</p> : null}</div>
      {ingestResult ? <div className="ingest-result"><strong>索引完成</strong><span>已处理 {ingestResult.processed_files} 个文件 · 生成 {ingestResult.chunk_count} 个分片 · {ingestResult.failed_files} 个失败</span></div> : null}
      <div className="document-list-heading"><h2>已入库文件</h2><span>{documents.length} 个文件</span></div>
      {loading ? <div className="document-skeleton"><i /><i /><i /></div> : null}
      {!loading && documents.length === 0 ? <div className="knowledge-empty"><span aria-hidden="true">⌁</span><h3>还没有知识库文件</h3><p>先上传文件，再点击“开始建立索引”，文件才会参与问答。</p><button className="primary-button" type="button" onClick={() => inputRef.current?.click()} disabled={uploading}><span aria-hidden="true">↑</span>{uploading ? "正在上传..." : knowledgeWorkflowCopy.uploadFirst}</button></div> : <div className="document-list">{documents.map((document) => <article className="document-row" key={document.id}><div className={`file-icon file-${document.type}`}>{document.type === "pdf" ? "PDF" : "MD"}</div><div className="document-details"><strong title={document.title}>{document.title}</strong><span>{document.chunk_count} 个分片 · {document.source_path}</span>{document.error_message ? <small className="document-error">{document.error_message}</small> : null}</div><span className={`document-status status-${document.status}`}>{getDocumentStatusLabel(document.status)}</span><a className="icon-button" href={documentFileUrl(document.id)} target="_blank" rel="noreferrer" aria-label={`打开 ${document.title}`}>↗</a><button className="icon-button document-delete" type="button" onClick={() => requestDelete(document)} disabled={deletingDocumentId === document.id} aria-label={`${deletingDocumentId === document.id ? "正在删除" : "删除"} ${document.title}`}>{deletingDocumentId === document.id ? "…" : "×"}</button></article>)}</div>}
    </div>
  </section>;
}

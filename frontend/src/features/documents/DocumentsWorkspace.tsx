import { AlertTriangle, CheckCircle2, FileArchive, FileText, Filter, LoaderCircle, Search, UploadCloud } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { api, ApiError } from "../../api/client";
import type { DocumentRecord } from "../../api/types";
import { Metric } from "../../components/Metric";
import { StatusPill } from "../../components/StatusPill";

const formatBytes = (bytes: number) => bytes < 1048576 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

export function DocumentsWorkspace({ documents, onDocumentsChange }: { documents: DocumentRecord[]; onDocumentsChange: (items: DocumentRecord[]) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "ready" | "flagged" | "failed">("all");
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const visible = useMemo(() => documents.filter((item) => item.name.toLowerCase().includes(query.toLowerCase()) && (filter === "all" || (filter === "ready" ? item.status === "ready" : filter === "failed" ? item.status === "failed" : item.qualityFlags.length > 0))), [documents, query, filter]);
  const ready = documents.filter((item) => item.status === "ready").length;
  const flagged = documents.filter((item) => item.qualityFlags.length > 0).length;
  const chunks = documents.reduce((total, item) => total + item.chunkCount, 0);

  const upload = async (files: FileList | File[]) => {
    const selected = Array.from(files); if (!selected.length) return;
    setUploading(true); setMessage(null); const updated = [...documents];
    try {
      for (const file of selected) { const { document } = await api.uploadDocument(file); const index = updated.findIndex((item) => item.id === document.id); if (index >= 0) updated[index] = document; else updated.unshift(document); }
      onDocumentsChange(updated); setMessage(`${selected.length} document${selected.length === 1 ? "" : "s"} processed.`);
    } catch (reason) { setMessage(reason instanceof ApiError ? reason.message : "The upload could not be completed."); }
    finally { setUploading(false); }
  };

  return <div className="page documents-page">
    <div className="page-heading"><div><span className="eyebrow">Corpus operations</span><h1>Knowledge base</h1><p>Ingest, inspect, and govern the evidence available to Atlas.</p></div><button className="primary-button" onClick={() => inputRef.current?.click()} disabled={uploading}><UploadCloud size={17} /> Add documents</button></div>
    <section className="metric-grid"><Metric label="Ready documents" value={String(ready)} detail={`${documents.length} total`} tone="success" /><Metric label="Indexed chunks" value={chunks.toLocaleString()} detail="Citation-ready evidence" /><Metric label="Quality flags" value={String(flagged)} detail="Noise, duplicate, or parser alerts" tone={flagged ? "warning" : "default"} /><Metric label="Supported formats" value="5" detail="PDF · TXT · HTML · DOCX · PPTX" /></section>
    <section className={`upload-zone ${dragging ? "upload-zone--active" : ""}`} aria-label="Document upload" onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void upload(event.dataTransfer.files); }}><input ref={inputRef} type="file" multiple accept=".pdf,.txt,.html,.htm,.docx,.pptx" aria-label="Choose documents to upload" onChange={(event) => event.target.files && void upload(event.target.files)} /><div className="upload-zone__icon" aria-hidden="true">{uploading ? <LoaderCircle className="spin" /> : <UploadCloud />}</div><div><strong>{uploading ? "Processing documents…" : "Drop documents to ingest"}</strong><span>or <button type="button" onClick={() => inputRef.current?.click()}>browse local files</button> · up to 50 MB each</span></div><small>Validated, normalized, deduplicated, chunked, embedded, and quality checked.</small></section>
    {message && <div className="notice" role="status">{message}</div>}
    <section className="quality-card" aria-label="Ingestion failures"><h2>Ingestion failures ({documents.filter(item => item.status === "failed").length})</h2>{documents.filter(item => item.status === "failed").length ? <ul>{documents.filter(item => item.status === "failed").map(item => <li key={item.id}>{item.name} — {item.errorCode || "Unknown failure"}</li>)}</ul> : <p>No failed documents reported.</p>}<button type="button" onClick={() => setFilter("failed")}>Show failed documents</button></section>
    <section className="table-card"><div className="table-toolbar"><div><h2>Document inventory</h2><span>{visible.length} records</span></div><div className="table-tools"><label><Search size={16} aria-hidden="true" /><span className="sr-only">Search documents</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search documents" /></label><button className={filter !== "all" ? "filter-active" : ""} aria-label={`Document filter: ${filter}`} title={`Filter: ${filter}`} onClick={() => setFilter((current) => current === "all" ? "ready" : current === "ready" ? "flagged" : "all")}><Filter size={17} aria-hidden="true" /><span>{filter}</span></button></div></div><div className="document-table-wrap"><table className="document-table"><caption className="sr-only">Documents available to the retrieval system</caption><thead><tr><th>Document</th><th>Status</th><th>Evidence</th><th>Quality</th><th>Added</th></tr></thead><tbody>{visible.map((document) => <tr key={document.id}><td><div className="document-name"><div aria-hidden="true"><FileText size={18} /></div><span><strong title={document.name}>{document.name}</strong><small>{document.mediaType} · {formatBytes(document.sizeBytes)}</small></span></div></td><td><StatusPill status={document.status} /></td><td><strong>{document.chunkCount.toLocaleString()}</strong><small>{document.segmentCount} source segments</small></td><td>{document.qualityFlags.length ? <span className="quality-warning"><AlertTriangle size={14} aria-hidden="true" />{document.qualityFlags.length} flags</span> : <span className="quality-good"><CheckCircle2 size={14} aria-hidden="true" />Clean</span>}</td><td>{new Date(document.createdAt).toLocaleDateString()}</td></tr>)}{!visible.length && <tr><td colSpan={5}><div className="table-empty"><FileArchive size={22} aria-hidden="true" /><strong>No matching documents</strong><span>Adjust your search or add a new source.</span></div></td></tr>}</tbody></table></div></section>
  </div>;
}

import { ArrowUp, CheckCircle2, Copy, FileText, LoaderCircle, RotateCcw, ShieldCheck, Sparkles } from "lucide-react";
import { useMemo, useState, type FormEvent } from "react";
import { api, ApiError } from "../../api/client";
import type { Citation, DocumentRecord, QuestionResponse } from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { StatusPill } from "../../components/StatusPill";

interface ChatTurn { question: string; response: QuestionResponse; }
const suggestions = [
  "How does five-level paging extend address translation?",
  "Compare memory consistency with cache coherence.",
  "What evidence discusses AVX-512 sorting performance?",
];

export function ChatWorkspace({ documents }: { documents: DocumentRecord[] }) {
  const [question, setQuestion] = useState("");
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const readyDocuments = useMemo(() => documents.filter((item) => item.status === "ready"), [documents]);

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    const value = question.trim();
    if (!value || loading) return;
    setLoading(true); setError(null);
    try {
      const response = await api.askQuestion(value, selectedDocuments);
      setTurns((current) => [...current, { question: value, response }]);
      setQuestion("");
      if (response.citations[0]) setSelectedCitation(response.citations[0]);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Atlas could not complete the request.");
    } finally { setLoading(false); }
  };

  return <div className="chat-layout">
    <section className="chat-panel" aria-label="Document conversation">
      <div className="page-heading page-heading--chat"><div><span className="eyebrow"><Sparkles size={14} /> Grounded document intelligence</span><h1>Ask the knowledge base</h1><p>Answers are constrained to retrieved evidence and verified citations.</p></div><label className="scope-select"><span>Search scope</span><select value={selectedDocuments.length === 1 ? selectedDocuments[0] : "all"} onChange={(event) => setSelectedDocuments(event.target.value === "all" ? [] : [event.target.value])}><option value="all">All ready documents</option>{readyDocuments.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>
      <div className="conversation" aria-live="polite">
        {turns.length === 0 ? <div className="chat-welcome"><div className="chat-welcome__mark"><Sparkles size={24} /></div><h2>Research across your technical library</h2><p>Ask a specific question. Atlas will retrieve, rerank, validate, and cite the supporting source.</p><div className="suggestion-grid">{suggestions.map((item) => <button key={item} onClick={() => setQuestion(item)}>{item}<ArrowUp size={15} /></button>)}</div></div> : turns.map((turn, index) => <article className="chat-turn" key={`${turn.question}-${index}`}><div className="user-message"><span>You</span><p>{turn.question}</p></div><div className="assistant-message"><div className="assistant-message__header"><div><span className="atlas-avatar"><Sparkles size={14} /></span><strong>Atlas</strong><StatusPill status={turn.response.status} /></div><div className="message-actions"><button aria-label="Copy answer" onClick={() => navigator.clipboard.writeText(turn.response.answer)}><Copy size={15} /></button><button aria-label="Retry question" onClick={() => setQuestion(turn.question)}><RotateCcw size={15} /></button></div></div><p className="answer-copy">{turn.response.answer}</p>{turn.response.citations.length > 0 && <div className="citation-row">{turn.response.citations.map((citation) => <button key={citation.id} onClick={() => setSelectedCitation(citation)}><FileText size={14} />[{citation.id}] {citation.documentName}<span>{citation.locationType} {citation.location}</span></button>)}</div>}<div className="answer-meta"><span><ShieldCheck size={14} /> Output verified</span><span>{turn.response.evidenceUsed} evidence chunks</span><span>Request {turn.response.requestId.slice(-8)}</span></div></div></article>)}
        {loading && <div className="thinking"><LoaderCircle size={17} className="spin" /><span>Retrieving and validating evidence…</span></div>}
      </div>
      <div className="composer-wrap">{error && <div className="inline-error" role="alert">{error}</div>}<form className="composer" onSubmit={ask}><textarea value={question} onChange={(event) => { setQuestion(event.target.value); if (error) setError(null); }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder="Ask a question grounded in your documents…" aria-label="Question" rows={2} maxLength={4000} /><div className="composer__footer"><span><ShieldCheck size={14} /> Guardrails enabled</span><button className="send-button" disabled={!question.trim() || loading} aria-label="Send question">{loading ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={18} />}</button></div></form><p className="composer-note">Atlas may abstain when the retrieved evidence is insufficient.</p></div>
    </section>
    <aside className="evidence-panel" aria-label="Source evidence"><div className="evidence-panel__header"><div><span>Evidence inspector</span><strong>Source context</strong></div>{selectedCitation && <CheckCircle2 size={18} />}</div>{selectedCitation ? <div className="evidence-detail"><div className="evidence-file"><div><FileText size={19} /></div><span><strong>{selectedCitation.documentName}</strong><small>{selectedCitation.locationType} {selectedCitation.location}</small></span></div><div className="relevance"><span>Reranked relevance</span><strong>{Math.round(selectedCitation.relevance * 100)}%</strong><div><i style={{ width: `${selectedCitation.relevance * 100}%` }} /></div></div><div className="excerpt"><span>Supporting excerpt</span><blockquote>{selectedCitation.excerpt}</blockquote></div><dl className="evidence-metadata"><div><dt>Chunk</dt><dd>{selectedCitation.chunkId.slice(-10)}</dd></div><div><dt>Document</dt><dd>{selectedCitation.documentId.slice(-10)}</dd></div></dl></div> : <EmptyState icon={FileText} title="No source selected" description="Ask a question, then select a citation to inspect its exact evidence." />}</aside>
  </div>;
}

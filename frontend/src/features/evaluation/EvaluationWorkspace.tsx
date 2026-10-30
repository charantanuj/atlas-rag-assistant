import { AlertCircle, CheckCircle2, FlaskConical, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { EvaluationSummary } from "../../api/types";
import { Metric } from "../../components/Metric";
import { StatusPill } from "../../components/StatusPill";

const percentage = (value?: number) => value === undefined ? "Not run" : `${Math.round(value * 100)}%`;

export function EvaluationWorkspace({ evaluation }: { evaluation: EvaluationSummary | null }) {
  const [cache, setCache] = useState<{items: number; hits: number; misses: number} | null>(null);
  useEffect(() => { void api.cacheSummary().then(result => setCache(result.answer)).catch(() => setCache(null)); }, []);
  const measured = Boolean(evaluation?.lastRunAt);
  const metrics = [
    ["Recall@k", evaluation?.recallAtK, 0.8], ["Reciprocal rank", evaluation?.reciprocalRank, 0.75],
    ["Citation precision", evaluation?.citationPrecision, 1], ["Citation completeness", evaluation?.citationCompleteness, 0.9],
    ["Abstention accuracy", evaluation?.abstentionAccuracy, 0.9], ["Attack block rate", evaluation?.attackBlockRate, 0.95],
  ] as const;
  return <div className="page evaluation-page">
    <div className="page-heading"><div><span className="eyebrow">Quality and safety</span><h1>Evaluation control room</h1><p>Evaluation results and local cache activity.</p></div></div>
    <section className="quality-card"><h2>Answer cache</h2><p>{cache ? `${cache.hits} hits · ${cache.misses} misses · ${cache.items} cached entries` : "Cache statistics unavailable."}</p></section>
    <section className="metric-grid"><Metric label="Quality gate" value={evaluation?.passed ? "Passed" : "Pending"} detail={evaluation?.lastRunAt ? `Last run ${new Date(evaluation.lastRunAt).toLocaleString()}` : "No verified corpus run"} tone={evaluation?.passed ? "success" : "warning"} /><Metric label="Attack coverage" value={percentage(measured ? evaluation?.attackBlockRate : undefined)} detail="Prompt-injection benchmark" /><Metric label="False positives" value={percentage(measured ? evaluation?.falsePositiveRate : undefined)} detail="Benign requests blocked" /><Metric label="Failed gates" value={String(evaluation?.failedGates.length ?? 0)} detail="Requires investigation" tone={evaluation?.failedGates.length ? "warning" : "default"} /></section>
    <div className="evaluation-grid"><section className="quality-card"><div className="section-heading"><div><FlaskConical size={18} /><span><strong>Deterministic metrics</strong><small>CI-safe, no judge model required</small></span></div><StatusPill status={evaluation?.passed ? "passed" : "not_run"} /></div><div className="quality-list">{metrics.map(([label, value, target]) => { const passed = measured && value !== undefined && value >= target; return <div key={label}><div><span>{label}</span><strong>{percentage(measured ? value : undefined)}</strong></div><div className="quality-bar"><i style={{ width: `${measured ? (value ?? 0) * 100 : 0}%` }} /></div><small>{passed ? <><CheckCircle2 size={13} /> Meets {percentage(target)} gate</> : <><AlertCircle size={13} /> Target {percentage(target)}</>}</small></div>; })}</div></section>
      <section className="quality-card safety-card"><div className="section-heading"><div><ShieldCheck size={18} /><span><strong>Safety layers</strong><small>Fail-closed request path</small></span></div></div><ol className="safety-layers"><li><span>01</span><div><strong>Input policy</strong><small>Injection, length, PII and secret detection</small></div></li><li><span>02</span><div><strong>Evidence quarantine</strong><small>Indirect instructions removed before generation</small></div></li><li><span>03</span><div><strong>Grounded generation</strong><small>Context-only answer contract</small></div></li><li><span>04</span><div><strong>Citation integrity</strong><small>Chunk, document and location verification</small></div></li></ol></section>
    </div>
  </div>;
}

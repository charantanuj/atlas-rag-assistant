export function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase().replaceAll("_", "-");
  return <span className={`status-pill status-pill--${normalized}`}><span className="status-pill__dot" />{status.replaceAll("_", " ")}</span>;
}

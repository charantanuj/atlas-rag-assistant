interface Props { label: string; value: string; detail?: string; tone?: "default" | "success" | "warning"; }
export function Metric({ label, value, detail, tone = "default" }: Props) {
  return <article className={`metric metric--${tone}`}><p>{label}</p><strong>{value}</strong>{detail && <span>{detail}</span>}</article>;
}

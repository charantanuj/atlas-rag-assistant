import { BarChart3, BookOpenCheck, Database, FileStack, Menu, MessageSquareText, ShieldCheck, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { BrandMark } from "../components/BrandMark";

export type WorkspaceView = "chat" | "documents" | "evaluation";
const nav = [
  { id: "chat" as const, label: "Ask Atlas", icon: MessageSquareText },
  { id: "documents" as const, label: "Knowledge base", icon: FileStack },
  { id: "evaluation" as const, label: "Evaluation", icon: BarChart3 },
];

export function AppShell({ activeView, onViewChange, onSignOut, children, readyDocuments, systemStatus }: { activeView: WorkspaceView; onViewChange: (view: WorkspaceView) => void; onSignOut: () => void; children: ReactNode; readyDocuments: number; systemStatus: "checking" | "online" | "offline" }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const changeView = (view: WorkspaceView) => { onViewChange(view); setMobileOpen(false); };
  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <aside className={`sidebar ${mobileOpen ? "sidebar--open" : ""}`}>
      <div className="brand"><BrandMark /><div><strong>Atlas</strong><span>Knowledge intelligence</span></div><button className="icon-button sidebar__close" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={18} /></button></div>
      <nav aria-label="Workspace navigation"><p className="nav-label">Workspace</p>{nav.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${activeView === id ? "nav-item--active" : ""}`} aria-current={activeView === id ? "page" : undefined} onClick={() => changeView(id)}><Icon size={18} aria-hidden="true" /><span>{label}</span></button>)}</nav>
      <div className="sidebar__context"><p className="nav-label">Active corpus</p><div className="corpus-card"><div className="corpus-card__icon"><Database size={17} /></div><div><strong>Technical research</strong><span>{readyDocuments} indexed documents</span></div></div></div>
      <div className="sidebar__footer"><div className="safety-state"><ShieldCheck size={18} /><div><strong>Safety active</strong><span>Input · evidence · output</span></div></div><div className="build-label"><BookOpenCheck size={14} /> Local workspace</div><button className="nav-item" onClick={onSignOut}>Sign out</button></div>
    </aside>
    {mobileOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <div className="workspace"><header className="topbar"><button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20} aria-hidden="true" /></button><div><span>Enterprise RAG</span><strong>{nav.find((item) => item.id === activeView)?.label}</strong></div><div className={`topbar__status topbar__status--${systemStatus}`} role="status"><span aria-hidden="true" />{systemStatus === "checking" ? "Checking API" : systemStatus === "online" ? "API connected" : "API unavailable"}</div></header><main id="main-content" className="main-content" tabIndex={-1}>{children}</main></div>
  </div>;
}

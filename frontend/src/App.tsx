import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { DocumentRecord, EvaluationSummary } from "./api/types";
import { ChatWorkspace } from "./features/chat/ChatWorkspace";
import { DocumentsWorkspace } from "./features/documents/DocumentsWorkspace";
import { EvaluationWorkspace } from "./features/evaluation/EvaluationWorkspace";
import { AppShell, type WorkspaceView } from "./layout/AppShell";
import { Login } from "./features/auth/Login";

export function App() {
  const [authenticated, setAuthenticated] = useState(() => Boolean(localStorage.getItem("atlas_token")));
  const [view, setView] = useState<WorkspaceView>("chat");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [systemStatus, setSystemStatus] = useState<"checking" | "online" | "offline">("checking");
  useEffect(() => {
    if (!authenticated) return;
    void Promise.allSettled([api.listDocuments(), api.getEvaluation()]).then(([documentResult, evaluationResult]) => {
      if (documentResult.status === "fulfilled") setDocuments(documentResult.value.documents);
      if (evaluationResult.status === "fulfilled") setEvaluation(evaluationResult.value);
      setSystemStatus(documentResult.status === "fulfilled" ? "online" : "offline");
    });
  }, [authenticated]);
  useEffect(() => {
    const expired = () => { localStorage.removeItem("atlas_token"); setAuthenticated(false); };
    window.addEventListener("atlas-session-expired", expired);
    return () => window.removeEventListener("atlas-session-expired", expired);
  }, []);
  if (!authenticated) return <Login onLogin={() => setAuthenticated(true)} />;
  return <AppShell onSignOut={() => { localStorage.removeItem("atlas_token"); setDocuments([]); setEvaluation(null); setAuthenticated(false); }} activeView={view} onViewChange={setView} readyDocuments={documents.filter((item) => item.status === "ready").length} systemStatus={systemStatus}>
    {view === "chat" && <ChatWorkspace documents={documents} />}
    {view === "documents" && <DocumentsWorkspace documents={documents} onDocumentsChange={setDocuments} />}
    {view === "evaluation" && <EvaluationWorkspace evaluation={evaluation} />}
  </AppShell>;
}

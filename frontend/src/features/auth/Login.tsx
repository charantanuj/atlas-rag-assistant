import { FormEvent, useState } from "react";
import { api, ApiError } from "../../api/client";

export function Login({ onLogin }: { onLogin: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setError("");
    try { const result = await api.login(username, password); localStorage.setItem("atlas_token", result.access_token); onLogin(); }
    catch (reason) { setError(reason instanceof ApiError ? reason.message : "Login failed."); }
  };
  return <main className="login-page"><form className="login-card" onSubmit={submit}><span className="eyebrow">ATLAS KNOWLEDGE INTELLIGENCE</span><h1>Sign in</h1><p>Access your grounded document workspace.</p><label>Email<input type="email" value={username} onChange={e => setUsername(e.target.value)} required /></label><label>Password<input type="password" value={password} onChange={e => setPassword(e.target.value)} required /></label>{error && <div className="inline-error">{error}</div>}<button className="send-button login-submit">Sign in</button></form></main>;
}

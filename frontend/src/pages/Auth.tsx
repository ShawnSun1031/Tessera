import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { ErrorNotice, PageHead } from "../components/ui";

export function LoginPage() {
  const { user, login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/" replace />;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username, password);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="panel stack" onSubmit={submit}>
        <h1>
          <span className="brand-mark" aria-hidden />
          Tessera
        </h1>
        <p className="muted">Annotate your images, train a YOLO model on them, and run it on new images.</p>
        <label className="field">
          Username
          <input type="text" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />
        </label>
        <label className="field">
          Password
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        <ErrorNotice error={error} />
        <button className="btn primary" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <p className="small muted">No account? Ask your platform admin to create one.</p>
      </form>
    </div>
  );
}

export function UsersPage() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const users = useQuery({ queryKey: ["users"], queryFn: api.users });
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);
  const create = useMutation({
    mutationFn: () => api.createUser({ username, password, is_admin: isAdmin }),
    onSuccess: () => {
      setUsername("");
      setPassword("");
      setIsAdmin(false);
      qc.invalidateQueries({ queryKey: ["users"] });
    },
  });
  if (!user?.is_admin) return <Navigate to="/" replace />;

  return (
    <div className="page stack">
      <PageHead title="Users">
        <p>Everyone who can sign in. People only see projects they created or were added to.</p>
      </PageHead>
      <form
        className="panel stack"
        onSubmit={(e) => {
          e.preventDefault();
          create.mutate();
        }}
      >
        <h2>Add a user</h2>
        <div className="grid-2">
          <label className="field">
            Username
            <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
          </label>
          <label className="field">
            Password
            <span className="help">At least 8 characters. Share it with them privately.</span>
            <input type="password" value={password} minLength={8} onChange={(e) => setPassword(e.target.value)} required />
          </label>
        </div>
        <label className="row">
          <input type="checkbox" checked={isAdmin} onChange={(e) => setIsAdmin(e.target.checked)} />
          Admin (can add users and open every project)
        </label>
        <ErrorNotice error={create.error} />
        {create.isSuccess && <div className="notice ok">User added.</div>}
        <div>
          <button className="btn primary" disabled={create.isPending}>
            Add user
          </button>
        </div>
      </form>
      <div className="panel table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Username</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {users.data?.map((u) => (
              <tr key={u.id}>
                <td>{u.username}</td>
                <td>{u.is_admin ? "Admin" : "Member"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

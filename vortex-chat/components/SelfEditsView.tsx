import React, { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircle2,
  FileCode,
  FlaskConical,
  RefreshCw,
  XCircle,
  PlayCircle,
  AlertTriangle,
} from "lucide-react";
import ModificationExplorerModal from "./ModificationExplorerModal";
import { Language, LogEntry } from "../types";
import { translations } from "../translations";

type ProposalStatus = "pending" | "accepted" | "rejected" | "applied" | string;

type ProposalSummary = {
  id: string;
  created_at: number | null;
  title: string;
  summary: string;
  status: ProposalStatus;
  author: string;
  files: string[];
};

type ProposalDetail = ProposalSummary & {
  diff: string;
  fileChanges: { path: string; diff: string }[];
  sandbox?: any;
  apply?: any;
};

const fmtDateTime = (ts: number | null) => {
  if (!ts) return "-";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "-";
  }
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  const text = await resp.text().catch(() => "");
  if (!resp.ok) {
    let msg = text || `HTTP ${resp.status}`;
    try {
      const parsed = JSON.parse(text);
      msg = parsed?.error?.message || parsed?.detail || msg;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

interface SelfEditsViewProps {
  language: Language;
  onAddLog: (level: LogEntry["level"], message: string) => void;
  onPendingCountChange?: (count: number) => void;
}

const SelfEditsView: React.FC<SelfEditsViewProps> = ({ language, onAddLog, onPendingCountChange }) => {
  const t = translations[language];

  const [status, setStatus] = useState<"pending" | "all">("pending");
  const [items, setItems] = useState<ProposalSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<ProposalDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDiff, setShowDiff] = useState(false);

  const selectedSummary = useMemo(() => items.find((x) => x.id === selectedId) || null, [items, selectedId]);

  const refreshPendingCount = useCallback(async () => {
    try {
      const payload = await fetchJson<{ object: string; data: ProposalSummary[] }>("/v1/self-edits/proposals?status=pending");
      onPendingCountChange?.(payload.data?.length || 0);
    } catch {
      // ignore
    }
  }, [onPendingCountChange]);

  const refreshList = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      const url = status === "all" ? "/v1/self-edits/proposals?status=all" : "/v1/self-edits/proposals?status=pending";
      const payload = await fetchJson<{ object: string; data: ProposalSummary[] }>(url);
      setItems(payload.data || []);
      if (!selectedId && payload.data?.[0]?.id) {
        setSelectedId(payload.data[0].id);
      }
      await refreshPendingCount();
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }, [refreshPendingCount, selectedId, status]);

  const loadDetail = useCallback(async (id: string) => {
    setError(null);
    setSelected(null);
    try {
      const payload = await fetchJson<ProposalDetail>(`/v1/self-edits/proposals/${encodeURIComponent(id)}`);
      setSelected(payload);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }, []);

  useEffect(() => {
    void refreshList();
    const interval = window.setInterval(() => {
      void refreshPendingCount();
    }, 8000);
    return () => window.clearInterval(interval);
  }, [refreshList, refreshPendingCount]);

  useEffect(() => {
    if (!selectedId) return;
    void loadDetail(selectedId);
  }, [selectedId, loadDetail]);

  const runAction = useCallback(
    async (action: "accept" | "reject" | "apply" | "demo") => {
      setError(null);
      setBusy(true);
      try {
        if (action === "demo") {
          await fetchJson("/v1/self-edits/proposals/demo", { method: "POST" });
          onAddLog("SYSTEM", language === "es" ? "Propuesta demo creada." : "Demo proposal created.");
          await refreshList();
          return;
        }

        if (!selectedId) return;
        await fetchJson(`/v1/self-edits/proposals/${encodeURIComponent(selectedId)}/${action}`, { method: "POST" });
        onAddLog("SYSTEM", `${action.toUpperCase()} OK: ${selectedId}`);
        await refreshList();
        await loadDetail(selectedId);
      } catch (e: any) {
        const msg = String(e?.message || e);
        setError(msg);
        onAddLog("SYSTEM", msg);
      } finally {
        setBusy(false);
      }
    },
    [language, loadDetail, onAddLog, refreshList, selectedId]
  );

  const statusPill = (st: ProposalStatus) => {
    const base = "px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-[0.25em] border";
    if (st === "accepted") return <span className={`${base} bg-primary/10 border-primary/20 text-primary`}>accepted</span>;
    if (st === "applied") return <span className={`${base} bg-emerald-500/10 border-emerald-500/20 text-emerald-400`}>applied</span>;
    if (st === "rejected") return <span className={`${base} bg-red-500/10 border-red-500/20 text-red-400`}>rejected</span>;
    return <span className={`${base} bg-muted/10 border-border/30 text-muted-foreground`}>{st}</span>;
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="pt-32 px-10 max-w-[1600px] mx-auto w-full pb-40">
      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-8 pb-10 border-b border-border/40">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-primary rounded-[2.2rem] flex items-center justify-center text-primary-foreground shadow-2xl">
            <FileCode size={34} strokeWidth={1.5} />
          </div>
          <div className="space-y-1">
            <h2 className="text-4xl font-black tracking-tighter leading-none">{language === "es" ? "Auto-ediciones" : "Self-edits"}</h2>
            <p className="text-[11px] font-black uppercase tracking-[0.35em] text-muted-foreground/70">
              {language === "es" ? "Propuestas locales · Aceptar / Rechazar / Aplicar" : "Local proposals · Accept / Reject / Apply"}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center bg-muted/10 border border-border/40 rounded-2xl p-1 shadow-inner">
            <button
              onClick={() => setStatus("pending")}
              className={`px-5 py-3 rounded-xl text-[9px] font-black uppercase tracking-[0.25em] transition-all ${
                status === "pending" ? "bg-primary text-primary-foreground shadow-lg" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {language === "es" ? "Pendientes" : "Pending"}
            </button>
            <button
              onClick={() => setStatus("all")}
              className={`px-5 py-3 rounded-xl text-[9px] font-black uppercase tracking-[0.25em] transition-all ${
                status === "all" ? "bg-primary text-primary-foreground shadow-lg" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {language === "es" ? "Todas" : "All"}
            </button>
          </div>

          <button
            onClick={() => void refreshList()}
            disabled={busy}
            className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-muted/10 border border-border/40 text-[10px] font-black uppercase tracking-[0.25em] hover:bg-muted/20 transition-all disabled:opacity-50"
          >
            <RefreshCw size={16} className={busy ? "animate-spin" : ""} />
            {language === "es" ? "Actualizar" : "Refresh"}
          </button>

          <button
            onClick={() => void runAction("demo")}
            disabled={busy}
            className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-primary text-primary-foreground text-[10px] font-black uppercase tracking-[0.25em] hover:scale-105 active:scale-95 transition-all disabled:opacity-50"
          >
            <FlaskConical size={16} />
            {language === "es" ? "Demo" : "Demo"}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[520px_1fr] gap-10 pt-10">
        <aside className="space-y-4">
          <div className="p-6 bg-muted/5 border border-border/30 rounded-[2.5rem] glass-card">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-black uppercase tracking-[0.35em] text-muted-foreground">
                {language === "es" ? "Propuestas" : "Proposals"} ({items.length})
              </span>
              {busy && <span className="text-[9px] font-black uppercase tracking-[0.25em] text-primary">{language === "es" ? "Cargando" : "Loading"}</span>}
            </div>
          </div>

          <div className="space-y-3">
            <AnimatePresence>
              {items.map((p) => {
                const active = p.id === selectedId;
                return (
                  <motion.button
                    key={p.id}
                    layout
                    onClick={() => setSelectedId(p.id)}
                    className={`w-full text-left p-6 rounded-[2.2rem] border transition-all glass-card ${
                      active ? "border-primary/40 bg-primary/5 shadow-xl" : "border-border/30 bg-muted/5 hover:bg-muted/10 hover:border-border/50"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-[14px] font-black tracking-tight truncate">{p.title}</span>
                        </div>
                        <p className="text-[11px] text-muted-foreground/70 line-clamp-2">{p.summary}</p>
                      </div>
                      <div className="shrink-0">{statusPill(p.status)}</div>
                    </div>
                    <div className="mt-4 flex items-center justify-between gap-4">
                      <span className="text-[9px] font-mono font-black uppercase tracking-widest text-muted-foreground/50 truncate">
                        {fmtDateTime(p.created_at)}
                      </span>
                      <span className="text-[9px] font-black uppercase tracking-[0.25em] text-muted-foreground/70">
                        {p.files?.length || 0} {language === "es" ? "archivos" : "files"}
                      </span>
                    </div>
                  </motion.button>
                );
              })}
            </AnimatePresence>
          </div>
        </aside>

        <main className="space-y-6">
          {error && (
            <div className="p-6 bg-red-500/10 border border-red-500/20 rounded-[2.5rem] text-red-200 flex items-start gap-4">
              <AlertTriangle size={18} className="mt-0.5" />
              <div className="min-w-0">
                <div className="text-[10px] font-black uppercase tracking-[0.35em] mb-1">{language === "es" ? "Error" : "Error"}</div>
                <div className="text-[12px] font-mono break-words">{error}</div>
              </div>
            </div>
          )}

          {!selectedSummary && (
            <div className="p-10 bg-muted/5 border border-border/30 rounded-[3rem] glass-card">
              <div className="text-[12px] font-black uppercase tracking-[0.35em] text-muted-foreground">
                {language === "es" ? "Sin propuestas" : "No proposals"}
              </div>
            </div>
          )}

          {selectedSummary && (
            <div className="p-10 bg-muted/5 border border-border/30 rounded-[3rem] glass-card">
              <div className="flex items-start justify-between gap-6">
                <div className="min-w-0">
                  <div className="flex items-center gap-4 mb-3">
                    <h3 className="text-3xl font-black tracking-tighter truncate">{selectedSummary.title}</h3>
                    {statusPill(selectedSummary.status)}
                  </div>
                  <p className="text-[13px] text-muted-foreground/70 leading-relaxed">{selectedSummary.summary}</p>
                  <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-5 bg-background/60 border border-border/40 rounded-2xl">
                      <div className="text-[9px] font-black uppercase tracking-[0.35em] text-muted-foreground mb-1">
                        {language === "es" ? "ID" : "ID"}
                      </div>
                      <div className="text-[12px] font-mono break-all">{selectedSummary.id}</div>
                    </div>
                    <div className="p-5 bg-background/60 border border-border/40 rounded-2xl">
                      <div className="text-[9px] font-black uppercase tracking-[0.35em] text-muted-foreground mb-1">
                        {language === "es" ? "Creado" : "Created"}
                      </div>
                      <div className="text-[12px] font-mono">{fmtDateTime(selectedSummary.created_at)}</div>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-3 shrink-0">
                  <button
                    onClick={() => setShowDiff(true)}
                    disabled={busy || !selected?.fileChanges?.length}
                    className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-primary text-primary-foreground text-[10px] font-black uppercase tracking-[0.25em] hover:scale-105 active:scale-95 transition-all disabled:opacity-50"
                  >
                    <FileCode size={16} />
                    {language === "es" ? "Ver Diff" : "View Diff"}
                  </button>

                  {selectedSummary.status === "pending" && (
                    <button
                      onClick={() => void runAction("accept")}
                      disabled={busy}
                      className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-emerald-500 text-white text-[10px] font-black uppercase tracking-[0.25em] hover:scale-105 active:scale-95 transition-all disabled:opacity-50"
                    >
                      <CheckCircle2 size={16} />
                      {language === "es" ? "Aceptar" : "Accept"}
                    </button>
                  )}

                  {selectedSummary.status !== "applied" && (
                    <button
                      onClick={() => void runAction("reject")}
                      disabled={busy}
                      className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-red-500 text-white text-[10px] font-black uppercase tracking-[0.25em] hover:scale-105 active:scale-95 transition-all disabled:opacity-50"
                    >
                      <XCircle size={16} />
                      {language === "es" ? "Rechazar" : "Reject"}
                    </button>
                  )}

                  {selectedSummary.status === "accepted" && (
                    <button
                      onClick={() => void runAction("apply")}
                      disabled={busy}
                      className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-purple-600 text-white text-[10px] font-black uppercase tracking-[0.25em] hover:scale-105 active:scale-95 transition-all disabled:opacity-50"
                    >
                      <PlayCircle size={16} />
                      {language === "es" ? "Aplicar" : "Apply"}
                    </button>
                  )}
                </div>
              </div>

              <div className="mt-10">
                <div className="text-[10px] font-black uppercase tracking-[0.35em] text-muted-foreground mb-3">
                  {language === "es" ? "Archivos tocados" : "Touched files"}
                </div>
                <div className="flex flex-wrap gap-2">
                  {(selectedSummary.files || []).slice(0, 40).map((f) => (
                    <span
                      key={f}
                      className="px-3 py-2 rounded-2xl bg-background/70 border border-border/40 text-[10px] font-mono font-bold tracking-tight"
                    >
                      {f}
                    </span>
                  ))}
                  {(selectedSummary.files || []).length > 40 && (
                    <span className="px-3 py-2 rounded-2xl bg-background/70 border border-border/40 text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                      +{(selectedSummary.files || []).length - 40}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {showDiff && selected?.fileChanges?.length ? (
        <ModificationExplorerModal fileChanges={selected.fileChanges} onClose={() => setShowDiff(false)} language={language} />
      ) : null}
    </motion.div>
  );
};

export default SelfEditsView;


import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, GitPullRequest, MessageSquare, X, RefreshCw, ArrowLeft, Check, XCircle, Play, AlertTriangle, Sparkles } from 'lucide-react';
import ChatInput from './ChatInput';
import MessageBubble from './MessageBubble';
import { LlmSettings, Message, Role, FontSize, Language } from '../types';
import {
  SelfEditProposalSummary,
  SelfEditProposalDetail,
  acceptSelfEditProposal,
  rejectSelfEditProposal,
  applySelfEditProposal,
  getSelfEditProposal,
  createDemoSelfEditProposal,
} from '../services/selfEditsClient';
import { vortexService } from '../services/vortexService';

type Tab = 'chat' | 'edits';

interface PersonalAIDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  language: Language;
  api: LlmSettings;
  fontSize: FontSize;
  codeTheme: 'dark' | 'light' | 'match-app';
  isDarkMode: boolean;
  proposals: SelfEditProposalSummary[];
  pendingCount: number;
  backendError?: string | null;
  onRefreshProposals: () => void;
  onOpenModificationExplorer: (fileChanges: { path: string; diff: string }[]) => void;
}

const SEEN_KEY = 'self-edits-seen-ids';
const PERSONAL_CHAT_KEY = 'personal-ai-messages';

const loadSeenIds = (): Set<string> => {
  try {
    const raw = localStorage.getItem(SEEN_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
  } catch {
    return new Set();
  }
};

const saveSeenIds = (ids: Set<string>) => {
  try {
    localStorage.setItem(SEEN_KEY, JSON.stringify(Array.from(ids)));
  } catch {}
};

const loadPersonalMessages = (): Message[] => {
  try {
    const raw = localStorage.getItem(PERSONAL_CHAT_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as Message[]) : [];
  } catch {
    return [];
  }
};

export const PersonalAIDrawer: React.FC<PersonalAIDrawerProps> = ({
  isOpen,
  onClose,
  language,
  api,
  fontSize,
  codeTheme,
  isDarkMode,
  proposals,
  pendingCount,
  backendError,
  onRefreshProposals,
  onOpenModificationExplorer,
}) => {
  const [tab, setTab] = useState<Tab>('chat');

  const [seenIds, setSeenIds] = useState<Set<string>>(() => loadSeenIds());
  const unseenCount = useMemo(() => proposals.filter((p) => !seenIds.has(p.id)).length, [proposals, seenIds]);

  const [personalMessages, setPersonalMessages] = useState<Message[]>(() => loadPersonalMessages());
  const [personalIsLoading, setPersonalIsLoading] = useState(false);
  const personalAbortRef = useRef<AbortController | null>(null);
  const personalScrollRef = useRef<HTMLDivElement | null>(null);

  const [selectedProposalId, setSelectedProposalId] = useState<string | null>(null);
  const [selectedProposal, setSelectedProposal] = useState<SelfEditProposalDetail | null>(null);
  const [proposalLoading, setProposalLoading] = useState(false);
  const [proposalActionLoading, setProposalActionLoading] = useState<'accept' | 'reject' | 'apply' | 'demo' | null>(null);
  const [proposalError, setProposalError] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(PERSONAL_CHAT_KEY, JSON.stringify(personalMessages));
    } catch {}
  }, [personalMessages]);

  useEffect(() => {
    const el = personalScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [personalMessages.length, personalIsLoading, tab]);

  useEffect(() => {
    saveSeenIds(seenIds);
  }, [seenIds]);

  useEffect(() => {
    if (!selectedProposalId) {
      setSelectedProposal(null);
      setProposalError(null);
      return;
    }
    let isCancelled = false;
    (async () => {
      setProposalLoading(true);
      setProposalError(null);
      try {
        const detail = await getSelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
        if (isCancelled) return;
        setSelectedProposal(detail);
      } catch (err: any) {
        if (isCancelled) return;
        setProposalError(err?.message || 'Failed to load proposal');
      } finally {
        if (!isCancelled) setProposalLoading(false);
      }
    })();
    return () => {
      isCancelled = true;
    };
  }, [selectedProposalId, api.baseUrl, api.token]);

  const handleSendPersonal = async (message: string, useInternet: boolean, mode: any, useThinking: boolean) => {
    if (personalIsLoading) return;
    if (!api.baseUrl) return;

    personalAbortRef.current?.abort();
    personalAbortRef.current = new AbortController();

    const userMessage: Message = {
      id: `${Date.now()}-u`,
      role: Role.USER,
      content: message,
      timestamp: Date.now(),
    };
    const aiMessageId = `${Date.now()}-a`;
    const aiMessage: Message = {
      id: aiMessageId,
      role: Role.AI,
      content: '',
      timestamp: Date.now(),
    };
    setPersonalMessages((prev) => [...prev, userMessage, aiMessage]);
    setPersonalIsLoading(true);

    try {
      const history = [...personalMessages, userMessage].filter((m) => m.role !== Role.AI || !!m.content);
      const stream = vortexService.generateResponseStream({
        history,
        prompt: message,
        api,
        mode,
        useInternet,
        useThinking,
        signal: personalAbortRef.current.signal,
      });
      for await (const chunk of stream) {
        setPersonalMessages((prev) =>
          prev.map((m) =>
            m.id === aiMessageId
              ? {
                  ...m,
                  content: chunk.text,
                  thought: chunk.thought || m.thought,
                  sources: chunk.sources.length > 0 ? chunk.sources : m.sources,
                  fileChanges: chunk.fileChanges || m.fileChanges,
                }
              : m
          )
        );
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        setPersonalMessages((prev) =>
          prev.map((m) => (m.id === aiMessageId ? { ...m, content: language === 'es' ? 'Error: backend offline.' : 'Error: backend offline.' } : m))
        );
      }
    } finally {
      personalAbortRef.current = null;
      setPersonalIsLoading(false);
    }
  };

  const markSeen = (id: string) => {
    setSeenIds((prev) => {
      const next = new Set(prev);
      next.add(id);
      return next;
    });
  };

  const openProposal = (id: string) => {
    markSeen(id);
    setSelectedProposalId(id);
  };

  const handleAccept = async () => {
    if (!selectedProposalId) return;
    setProposalActionLoading('accept');
    setProposalError(null);
    try {
      await acceptSelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
      onRefreshProposals();
      const detail = await getSelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
      setSelectedProposal(detail);
    } catch (err: any) {
      setProposalError(err?.message || 'Accept failed');
    } finally {
      setProposalActionLoading(null);
    }
  };

  const handleReject = async () => {
    if (!selectedProposalId) return;
    setProposalActionLoading('reject');
    setProposalError(null);
    try {
      await rejectSelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
      onRefreshProposals();
      const detail = await getSelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
      setSelectedProposal(detail);
    } catch (err: any) {
      setProposalError(err?.message || 'Reject failed');
    } finally {
      setProposalActionLoading(null);
    }
  };

  const handleApply = async () => {
    if (!selectedProposalId) return;
    setProposalActionLoading('apply');
    setProposalError(null);
    try {
      await applySelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
      onRefreshProposals();
      const detail = await getSelfEditProposal({ baseUrl: api.baseUrl, token: api.token, id: selectedProposalId });
      setSelectedProposal(detail);
    } catch (err: any) {
      setProposalError(err?.message || 'Apply failed');
    } finally {
      setProposalActionLoading(null);
    }
  };

  const handleDemo = async () => {
    setProposalActionLoading('demo');
    setProposalError(null);
    try {
      await createDemoSelfEditProposal({ baseUrl: api.baseUrl, token: api.token });
      onRefreshProposals();
      setTab('edits');
    } catch (err: any) {
      setProposalError(err?.message || 'Demo failed');
    } finally {
      setProposalActionLoading(null);
    }
  };

  const spring = { type: 'spring' as const, damping: 28, stiffness: 220, mass: 0.9 };

  const proposalMessage: Message | null = useMemo(() => {
    if (!selectedProposal) return null;
    const tsMs = selectedProposal.created_at ? Math.floor(selectedProposal.created_at * 1000) : Date.now();
    const files = Array.isArray(selectedProposal.files) ? selectedProposal.files : [];
    const filesList = files.length ? files.map((f) => `- \`${f}\``).join('\n') : '';
    const statusLine = language === 'es' ? `**Estado:** ${selectedProposal.status}` : `**Status:** ${selectedProposal.status}`;
    const body = `### ${selectedProposal.title}\n\n${selectedProposal.summary || ''}\n\n${statusLine}\n\n${filesList ? `**Files**\n${filesList}` : ''}`.trim();
    return {
      id: `proposal-${selectedProposal.id}`,
      role: Role.AI,
      content: body,
      timestamp: tsMs,
      fileChanges: selectedProposal.fileChanges || [],
    };
  }, [selectedProposal, language]);

  return (
    <div className="h-full w-[420px] flex flex-col relative bg-zinc-950 overflow-hidden">
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="absolute top-0 right-0 w-full h-full bg-[radial-gradient(circle_at_100%_0%,rgba(var(--primary),0.15),transparent_50%)]" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-primary/5 blur-[100px] rounded-full" />
      </div>

      <header className="relative px-6 pt-10 pb-6 border-b border-white/5 bg-white/[0.02] shrink-0 z-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="relative group">
              <div className="absolute inset-0 bg-primary/40 blur-xl rounded-full animate-pulse" />
              <div className="w-12 h-12 bg-primary rounded-2xl flex items-center justify-center text-white shadow-2xl relative z-10">
                <Bot size={24} fill="currentColor" />
              </div>
            </div>
            <div>
              <h2 className="text-[10px] font-black tracking-[0.3em] text-primary uppercase leading-none mb-1.5">PERSONAL AI</h2>
              <div className="flex items-center gap-2">
                <div className={`w-1.5 h-1.5 rounded-full animate-pulse ${backendError ? 'bg-red-500' : 'bg-emerald-500'}`} />
                <span className="text-sm font-black text-white tracking-tight">
                  {backendError ? (language === 'es' ? 'Backend offline' : 'Backend offline') : (language === 'es' ? 'En línea' : 'Online')}
                </span>
              </div>
            </div>
          </div>
          <button onClick={onClose} className="p-2.5 bg-white/5 hover:bg-white/10 rounded-xl text-zinc-400 transition-all border border-white/5">
            <X size={18} />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <motion.button
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setTab('chat')}
            className={`p-4 rounded-2xl border transition-all text-left ${
              tab === 'chat' ? 'bg-primary/10 border-primary/30' : 'bg-white/[0.03] border-white/5 hover:bg-white/[0.05]'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-[8px] font-black uppercase tracking-widest text-zinc-500">
                <MessageSquare size={10} /> {language === 'es' ? 'Chat' : 'Chat'}
              </div>
            </div>
            <div className="mt-2 text-sm font-black text-white tracking-tight">{language === 'es' ? 'Personal' : 'Personal'}</div>
          </motion.button>

          <motion.button
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setTab('edits')}
            className={`p-4 rounded-2xl border transition-all text-left relative ${
              tab === 'edits' ? 'bg-primary/10 border-primary/30' : 'bg-white/[0.03] border-white/5 hover:bg-white/[0.05]'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-[8px] font-black uppercase tracking-widest text-zinc-500">
                <GitPullRequest size={10} /> {language === 'es' ? 'Auto-ediciones' : 'Auto-edits'}
              </div>
              {pendingCount > 0 && (
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center justify-center min-w-5 h-5 px-1.5 bg-red-500 text-white text-[9px] font-black rounded-full">
                    {pendingCount}
                  </span>
                </div>
              )}
            </div>
            <div className="mt-2 text-sm font-black text-white tracking-tight">
              {pendingCount > 0 ? (language === 'es' ? 'Pendientes' : 'Pending') : language === 'es' ? 'Sin pendientes' : 'Clear'}
            </div>
            {unseenCount > 0 && <div className="absolute top-3 right-3 w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_0_4px_rgba(239,68,68,0.15)]" />}
          </motion.button>
        </div>
      </header>

      <div className="relative flex-1 overflow-hidden z-10">
        <AnimatePresence mode="wait">
          {tab === 'chat' && (
            <motion.div
              key="chat"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={spring}
              className="h-full flex flex-col"
            >
              <div ref={personalScrollRef} className="flex-1 overflow-y-auto custom-scrollbar p-6">
                {personalMessages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center gap-6 text-zinc-500">
                    <div className="w-16 h-16 bg-white/5 border border-white/10 rounded-[2rem] flex items-center justify-center">
                      <Sparkles size={28} />
                    </div>
                    <div className="space-y-1">
                      <div className="text-[11px] font-black uppercase tracking-[0.3em] text-white/60">
                        {language === 'es' ? 'Chat Personal' : 'Personal Chat'}
                      </div>
                      <div className="text-[10px] font-medium text-white/30 max-w-[260px]">
                        {language === 'es'
                          ? 'Historial separado del chat principal. Ideal para notas rápidas.'
                          : 'Separate history from main chat. Great for quick notes.'}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="pb-28">
                    {personalMessages.map((msg, idx) => (
                      <MessageBubble
                        key={msg.id}
                        message={msg}
                        fontSize={fontSize}
                        codeTheme={codeTheme}
                        onOpenModificationExplorer={onOpenModificationExplorer}
                        isStreaming={personalIsLoading && idx === personalMessages.length - 1 && msg.role === Role.AI}
                        language={language}
                      />
                    ))}
                  </div>
                )}
              </div>

              <div className="shrink-0 border-t border-white/5 bg-zinc-950 p-4">
                <ChatInput
                  onSend={handleSendPersonal}
                  isLoading={personalIsLoading}
                  isDarkMode={isDarkMode}
                  onStop={() => personalAbortRef.current?.abort()}
                  language={language}
                />
              </div>
            </motion.div>
          )}

          {tab === 'edits' && (
            <motion.div
              key="edits"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={spring}
              className="h-full flex flex-col"
            >
              <div className="px-6 py-5 border-b border-white/5 bg-white/[0.02] flex items-center justify-between shrink-0">
                {selectedProposalId ? (
                  <button
                    onClick={() => setSelectedProposalId(null)}
                    className="flex items-center gap-3 text-[10px] font-black uppercase tracking-[0.3em] text-white/60 hover:text-white transition-colors"
                  >
                    <ArrowLeft size={14} /> {language === 'es' ? 'Volver' : 'Back'}
                  </button>
                ) : (
                  <div className="text-[10px] font-black uppercase tracking-[0.3em] text-white/40">
                    {language === 'es' ? 'Propuestas' : 'Proposals'}
                  </div>
                )}

                <div className="flex items-center gap-2">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={onRefreshProposals}
                    className="p-2.5 bg-white/5 hover:bg-white/10 rounded-xl text-zinc-300 transition-all border border-white/5"
                    title={language === 'es' ? 'Actualizar' : 'Refresh'}
                  >
                    <RefreshCw size={16} />
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleDemo}
                    disabled={proposalActionLoading === 'demo'}
                    className="px-4 py-2.5 bg-primary text-white rounded-xl text-[9px] font-black uppercase tracking-[0.2em] shadow-xl border border-primary/20 disabled:opacity-50"
                    title={language === 'es' ? 'Crear propuesta demo' : 'Create demo proposal'}
                  >
                    {proposalActionLoading === 'demo' ? (language === 'es' ? 'Creando…' : 'Creating…') : language === 'es' ? 'Demo' : 'Demo'}
                  </motion.button>
                </div>
              </div>

              <div className="flex-1 overflow-hidden">
                {!selectedProposalId ? (
                  <div className="h-full overflow-y-auto custom-scrollbar p-6 space-y-3">
                    {backendError && (
                      <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-300 text-[11px] font-bold">
                        {backendError}
                      </div>
                    )}

                    {proposals.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-center gap-6 text-zinc-500 py-10">
                        <div className="w-16 h-16 bg-white/5 border border-white/10 rounded-[2rem] flex items-center justify-center">
                          <GitPullRequest size={28} />
                        </div>
                        <div className="space-y-1">
                          <div className="text-[11px] font-black uppercase tracking-[0.3em] text-white/60">
                            {language === 'es' ? 'Sin propuestas' : 'No proposals'}
                          </div>
                          <div className="text-[10px] font-medium text-white/30 max-w-[280px]">
                            {language === 'es'
                              ? 'Cuando haya propuestas pendientes, aparecerán aquí con notificación.'
                              : 'Pending proposals will appear here with a notification badge.'}
                          </div>
                        </div>
                      </div>
                    ) : (
                      proposals.map((p) => {
                        const isUnseen = !seenIds.has(p.id);
                        const created = p.created_at ? new Date(p.created_at * 1000) : null;
                        const when = created ? created.toLocaleString() : '';
                        return (
                          <motion.button
                            key={p.id}
                            whileHover={{ y: -2 }}
                            whileTap={{ scale: 0.99 }}
                            onClick={() => openProposal(p.id)}
                            className="w-full text-left p-5 bg-white/[0.03] border border-white/5 hover:bg-white/[0.05] hover:border-primary/30 rounded-2xl transition-all relative"
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="min-w-0">
                                <div className="flex items-center gap-3">
                                  <div className="text-[12px] font-black text-white truncate">{p.title}</div>
                                  {isUnseen && <div className="w-2.5 h-2.5 rounded-full bg-red-500 shrink-0" />}
                                </div>
                                <div className="mt-1 text-[10px] font-medium text-white/40 line-clamp-2">{p.summary}</div>
                              </div>
                              <div className="shrink-0 text-[9px] font-black uppercase tracking-widest text-white/20">{when}</div>
                            </div>
                            <div className="mt-3 flex flex-wrap gap-2">
                              {(p.files || []).slice(0, 3).map((f) => (
                                <span key={f} className="px-2.5 py-1 bg-black/30 border border-white/5 rounded-full text-[9px] font-mono text-white/40">
                                  {f.split('/').slice(-2).join('/')}
                                </span>
                              ))}
                              {(p.files || []).length > 3 && (
                                <span className="px-2.5 py-1 bg-black/30 border border-white/5 rounded-full text-[9px] font-black text-white/30">
                                  +{(p.files || []).length - 3}
                                </span>
                              )}
                            </div>
                          </motion.button>
                        );
                      })
                    )}
                  </div>
                ) : (
                  <div className="h-full overflow-y-auto custom-scrollbar p-6">
                    {proposalLoading && (
                      <div className="p-4 bg-white/[0.03] border border-white/5 rounded-2xl text-white/40 text-[11px] font-bold">
                        {language === 'es' ? 'Cargando propuesta…' : 'Loading proposal…'}
                      </div>
                    )}
                    {proposalError && (
                      <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-300 text-[11px] font-bold">
                        {proposalError}
                      </div>
                    )}
                    {proposalMessage && (
                      <div className="pb-36">
                        <MessageBubble
                          message={proposalMessage}
                          fontSize={fontSize}
                          codeTheme={codeTheme}
                          onOpenModificationExplorer={onOpenModificationExplorer}
                          language={language}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>

              {selectedProposalId && (
                <div className="shrink-0 border-t border-white/5 bg-zinc-950 p-6">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleAccept}
                      disabled={proposalActionLoading !== null || selectedProposal?.status === 'accepted'}
                      className="flex-1 flex items-center justify-center gap-3 py-3.5 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] bg-emerald-500/15 border border-emerald-500/25 text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-40 transition-all"
                    >
                      <Check size={16} /> {language === 'es' ? 'Aceptar' : 'Accept'}
                    </button>
                    <button
                      onClick={handleReject}
                      disabled={proposalActionLoading !== null || selectedProposal?.status === 'rejected'}
                      className="flex-1 flex items-center justify-center gap-3 py-3.5 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] bg-red-500/15 border border-red-500/25 text-red-200 hover:bg-red-500/20 disabled:opacity-40 transition-all"
                    >
                      <XCircle size={16} /> {language === 'es' ? 'Rechazar' : 'Reject'}
                    </button>
                    <button
                      onClick={handleApply}
                      disabled={proposalActionLoading !== null || selectedProposal?.status !== 'accepted'}
                      className="flex-1 flex items-center justify-center gap-3 py-3.5 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] bg-primary/15 border border-primary/25 text-primary hover:bg-primary/20 disabled:opacity-40 transition-all"
                    >
                      <Play size={16} /> {language === 'es' ? 'Aplicar' : 'Apply'}
                    </button>
                  </div>
                  <div className="mt-4 flex items-start gap-3 text-[10px] font-medium text-white/30">
                    <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                    <div>
                      {language === 'es'
                        ? 'Aplicar ejecuta validación (pytest + skills validate + doctor --deep --mock) y hace rollback si falla. Requiere working tree limpio.'
                        : 'Apply runs validation (pytest + skills validate + doctor --deep --mock) and rolls back on failure. Requires a clean working tree.'}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default PersonalAIDrawer;

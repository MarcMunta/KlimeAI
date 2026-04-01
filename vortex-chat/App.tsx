
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { PanelLeft, Globe, Zap, MessageSquare, BarChart3, Terminal as TerminalIcon, FileCode } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatInput from './components/ChatInput';
import CommandPalette from './components/CommandPalette';
import SettingsModal from './components/SettingsModal';
import HelpModal from './components/HelpModal';
import ReasoningDrawer from './components/ReasoningDrawer';
import AnalysisView from './components/AnalysisView';
import TerminalView from './components/TerminalView';
import SelfEditsView from './components/SelfEditsView';
import VortexLogo from './components/VortexLogo';
import TopBarStackStatus from './components/TopBarStackStatus';
import VirtualizedMessageList from './components/VirtualizedMessageList';
import ModificationExplorerModal from './components/ModificationExplorerModal';
import { ChatSession, Message, Role, UserSettings, ViewType, LogEntry, AppMode, Source, Language, OperationalStatus, ControlStatus } from './types';
import { vortexService } from './services/vortexService';
import { controlService } from './services/controlService';
import { translations } from './translations';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';

const DEFAULT_SETTINGS: UserSettings = {
  categoryOrder: ['Acciones Rápidas', 'Preferencias', 'Interfaz', 'Datos', 'Chats Recientes', 'Sistema'],
  codeTheme: 'dark',
  fontSize: 'medium',
  language: 'es'
};

const VIEW_INDEX: Record<ViewType, number> = { 'chat': 0, 'analysis': 1, 'edits': 2, 'terminal': 3 };

const repairMojibakeText = (value: string | null | undefined): string => {
  if (!value || !/[ÃÂ]/.test(value)) return value ?? '';
  try {
    const bytes = Uint8Array.from(Array.from(value), (char) => char.charCodeAt(0) & 0xff);
    const decoded = new TextDecoder('utf-8').decode(bytes);
    return decoded.includes('\uFFFD') ? value : decoded;
  } catch {
    return value;
  }
};

const normalizeSession = (rawSession: unknown): ChatSession | null => {
  if (!rawSession || typeof rawSession !== 'object' || !Array.isArray((rawSession as ChatSession).messages)) {
    return null;
  }

  const session = rawSession as ChatSession;
  return {
    ...session,
    title: repairMojibakeText(session.title),
    messages: session.messages.map((message) => ({
      ...message,
      content: repairMojibakeText(message.content),
      thought: typeof message.thought === 'string' ? repairMojibakeText(message.thought) : message.thought,
    })),
  };
};

const normalizeSettings = (rawSettings: unknown): UserSettings => {
  if (!rawSettings || typeof rawSettings !== 'object') return DEFAULT_SETTINGS;

  const candidate = rawSettings as Partial<UserSettings>;
  return {
    ...DEFAULT_SETTINGS,
    ...candidate,
    categoryOrder: Array.isArray(candidate.categoryOrder)
      ? candidate.categoryOrder.map((entry) => repairMojibakeText(String(entry)))
      : DEFAULT_SETTINGS.categoryOrder,
  };
};

const createEmptySession = (language: Language): ChatSession => ({
  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  title: language === 'es' ? 'Nueva Conversación' : 'New Conversation',
  messages: [],
  updatedAt: Date.now(),
});

const getInitialDarkMode = (): boolean => {
  const savedMode = localStorage.getItem('dark-mode');
  if (savedMode !== null) return savedMode === 'true';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

const App: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewType>('chat');
  const [prevView, setPrevView] = useState<ViewType>('chat');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [selfEditsPendingCount, setSelfEditsPendingCount] = useState(0);
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(getInitialDarkMode());
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const [mode, setMode] = useState<AppMode>('ask');
  const [operationalStatus, setOperationalStatus] = useState<OperationalStatus | null>(null);
  const [controlStatus, setControlStatus] = useState<ControlStatus | null>(null);
  const [analysisFocusTab, setAnalysisFocusTab] = useState<'stack' | 'learning' | 'internet'>('stack');
  const [isComposerFocused, setIsComposerFocused] = useState(false);
  const [hasComposerDraft, setHasComposerDraft] = useState(false);
  
  const [headerVisible, setHeaderVisible] = useState(true);
  const [footerVisible, setFooterVisible] = useState(true);
  const [activeModificationFiles, setActiveModificationFiles] = useState<{ path: string, diff: string }[] | null>(null);
  
  const inactivityTimerRef = useRef<number | null>(null);
  const isAutoScrollingRef = useRef<boolean>(false);
  const lastScrollYRef = useRef(0);
  
  const [isReasoningOpen, setIsReasoningOpen] = useState(false);
  const [activeThoughtMessageId, setActiveThoughtMessageId] = useState<string | null>(null);
  const abortControllerRef = useRef<boolean>(false);
  const mainScrollRef = useRef<HTMLDivElement>(null);
  const AUTO_APPLY_SELF_EDITS = false;
  
  const { scrollY } = useScroll({ container: mainScrollRef });
  const t = translations[settings.language];
  const currentSession = sessions.find(s => s.id === currentSessionId);
  const hasMessages = currentSession && currentSession.messages && currentSession.messages.length > 0;
  const internetAllowlist = controlStatus?.internet?.allowlist || [];
  const canUseInternet = Boolean(controlStatus?.ok);
  const canStartTraining = Boolean(controlStatus?.ok);
  const sendDisabledReason = operationalStatus?.ok
    ? undefined
    : operationalStatus?.degraded_reason
      || operationalStatus?.engine_reason
      || operationalStatus?.model_reason
      || operationalStatus?.docker_reason
      || operationalStatus?.offline_reason
      || (settings.language === 'es' ? 'Stack local no listo.' : 'Local stack not ready.');
  const activeModelLabel = operationalStatus?.active_model || (settings.language === 'es' ? 'Modelo base pendiente' : 'Base model pending');
  const activeEngineLabel = (operationalStatus?.engine_kind || 'local').toUpperCase();
  const readyLabel = operationalStatus?.ok
    ? (settings.language === 'es' ? 'Listo' : 'Ready')
    : (settings.language === 'es' ? 'Pendiente' : 'Pending');
  const heroCards = settings.language === 'es'
    ? [
        { label: 'Modo', value: 'Consulta + agente' },
        { label: 'Internet', value: 'Solo por prompt' },
        { label: 'Entreno', value: 'Manual y visible' },
      ]
    : [
        { label: 'Mode', value: 'Query + agent' },
        { label: 'Internet', value: 'Prompt only' },
        { label: 'Training', value: 'Manual and visible' },
      ];

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const newLog: LogEntry = { id: Math.random().toString(36).substr(2, 9), timestamp: Date.now(), level, message };
    setLogs(prev => [...prev.slice(-149), newLog]);
  }, []);

  const activeThought = React.useMemo(() => {
    if (!activeThoughtMessageId || !currentSessionId) return undefined;
    return currentSession?.messages.find(m => m.id === activeThoughtMessageId)?.thought;
  }, [activeThoughtMessageId, currentSessionId, currentSession]);

  const isCurrentThoughtStreaming = React.useMemo(() => {
    if (!isLoading || !currentSession || !activeThoughtMessageId) return false;
    const lastMsg = currentSession.messages[currentSession.messages.length - 1];
    return lastMsg.id === activeThoughtMessageId;
  }, [isLoading, currentSession, activeThoughtMessageId]);

  const extractDiffBlocks = useCallback((content: string): string => {
    const blocks: string[] = [];
    const regex = /```(?:diff|patch)\n([\s\S]*?)```/gi;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(content)) !== null) {
      if (match[1]) blocks.push(match[1].trim());
    }
    return blocks.filter(Boolean).join("\n\n");
  }, []);

  const acceptAndApplyProposal = useCallback(async (proposalId: string) => {
    const resp = await fetch(`/v1/self-edits/proposals/${encodeURIComponent(proposalId)}/accept`, { method: 'POST' });
    if (!resp.ok) return { ok: false, stage: 'accept' };
    const apply = await fetch(`/v1/self-edits/proposals/${encodeURIComponent(proposalId)}/apply`, { method: 'POST' });
    if (!apply.ok) return { ok: false, stage: 'apply' };
    return { ok: true };
  }, []);

  const suggestPatchFromMessage = useCallback(async (messageId: string, reason: string) => {
    const session = sessions.find(s => s.id === currentSessionId);
    const msg = session?.messages.find(m => m.id === messageId);
    if (!msg || !msg.content) {
      addLog('SYSTEM', settings.language === 'es' ? 'No hay contenido para sugerir parche.' : 'No content available to suggest a patch.');
      return;
    }
    const diffText = extractDiffBlocks(msg.content);
    if (!diffText) {
      addLog('SYSTEM', settings.language === 'es' ? 'No se detectó un bloque diff en la respuesta.' : 'No diff block detected in the response.');
      return;
    }
    const title = settings.language === 'es' ? 'Parche sugerido (frontend)' : 'Suggested patch (frontend)';
    const summary = settings.language === 'es' ? `Generado desde chat · ${reason}` : `Generated from chat · ${reason}`;
    const proposal = await vortexService.proposeSelfEditFromDiff(diffText, title, summary);
    if (!proposal.ok || !proposal.id) {
      addLog('SYSTEM', settings.language === 'es' ? `No se pudo crear propuesta: ${proposal.error || 'error'}` : `Failed to create proposal: ${proposal.error || 'error'}`);
      return;
    }
    addLog('LEARN', settings.language === 'es' ? `Propuesta creada: ${proposal.id}` : `Proposal created: ${proposal.id}`);
    if (AUTO_APPLY_SELF_EDITS) {
      const applyRes = await acceptAndApplyProposal(proposal.id);
      if (applyRes.ok) {
        addLog('LEARN', settings.language === 'es' ? `Parche aplicado: ${proposal.id}` : `Patch applied: ${proposal.id}`);
      } else {
        addLog('SYSTEM', settings.language === 'es' ? `No se pudo aplicar: ${proposal.id}` : `Failed to apply: ${proposal.id}`);
      }
    }
  }, [acceptAndApplyProposal, addLog, currentSessionId, extractDiffBlocks, sessions, settings.language]);

  const resetInactivityTimer = useCallback(() => {
    if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current);
    if (activeModificationFiles) return;
    if (isLoading || isSearching) { setFooterVisible(true); return; }
    if (!hasMessages && activeView === 'chat') { setFooterVisible(true); return; }
    if (isComposerFocused || hasComposerDraft) { setFooterVisible(true); return; }
    if (hasMessages) setFooterVisible(true);
    inactivityTimerRef.current = window.setTimeout(() => {
      if (activeView === 'chat') {
        setFooterVisible(false);
      }
    }, 6000);
  }, [isLoading, isSearching, activeView, hasMessages, activeModificationFiles, isComposerFocused, hasComposerDraft]);

  useEffect(() => { resetInactivityTimer(); return () => { if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current); }; }, [resetInactivityTimer]);

  useEffect(() => {
    let disposed = false;

    const pollStatus = async () => {
      const [runtimeStatus, nextControlStatus] = await Promise.all([
        vortexService.fetchOperationalStatus(),
        controlService.fetchStatus(),
      ]);
      if (disposed) return;
      setOperationalStatus(runtimeStatus);
      setControlStatus(nextControlStatus);
    };

    pollStatus();
    const timer = window.setInterval(pollStatus, 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  // --- Poll backend /v1/status for REAL kernel activity ---
  useEffect(() => {
    let disposed = false;
    let prevEpisodes = -1;
    let prevRequests = -1;
    let prevKnowledge = -1;
    let prevBackendKey = '';
    let prevWebChunks = -1;
    let prevCodeChunks = -1;
    let prevAnalyses = -1;
    let prevProposals = -1;
    let prevDiscoveredUrls = -1;

    const poll = async () => {
      try {
        const resp = await fetch('/v1/status');
        if (!resp.ok || disposed) return;
        const data = await resp.json().catch(() => null);
        if (!data || disposed) return;

        const lang = settings.language;
        const backends = data.backends || [];
        const adaptersLoaded = data.adapters
          ? Object.values(data.adapters).filter(Boolean).length
          : 0;
        const metrics = data.metrics || {};
        const episodes = data.episodes || 0;
        const knowledge = data.knowledge_chunks || 0;
        const al = data.autolearn || {};

        // -- Backend & adapter status (only on CHANGE) --
        const backendKey = `${backends.join(',')}|${adaptersLoaded}`;
        if (backendKey !== prevBackendKey) {
          prevBackendKey = backendKey;
          if (adaptersLoaded > 0) {
            addLog('LEARN', lang === 'es'
              ? `Adaptadores LoRA activos: ${adaptersLoaded} en ${backends.join(', ')}.`
              : `Active LoRA adapters: ${adaptersLoaded} on ${backends.join(', ')}.`);
          } else if (backends.length > 0) {
            addLog('INFO', lang === 'es'
              ? `Backend: ${backends.join(', ')} — modelo cargado y listo.`
              : `Backend: ${backends.join(', ')} — model loaded and ready.`);
          }
        }

        // -- Episode growth --
        if (prevEpisodes >= 0 && episodes > prevEpisodes) {
          const diff = episodes - prevEpisodes;
          addLog('LEARN', lang === 'es'
            ? `+${diff} episodio${diff > 1 ? 's' : ''} registrado${diff > 1 ? 's' : ''} (total: ${episodes}).`
            : `+${diff} new episode${diff > 1 ? 's' : ''} logged (total: ${episodes}).`);
        } else if (prevEpisodes < 0 && episodes > 0) {
          addLog('INFO', lang === 'es'
            ? `Episodios almacenados: ${episodes}.`
            : `Stored episodes: ${episodes}.`);
        }
        prevEpisodes = episodes;

        // -- Knowledge chunks growth --
        if (prevKnowledge >= 0 && knowledge > prevKnowledge) {
          const diff = knowledge - prevKnowledge;
          addLog('LEARN', lang === 'es'
            ? `+${diff} chunk${diff > 1 ? 's' : ''} de conocimiento indexado${diff > 1 ? 's' : ''} (total: ${knowledge}).`
            : `+${diff} knowledge chunk${diff > 1 ? 's' : ''} indexed (total: ${knowledge}).`);
        } else if (prevKnowledge < 0 && knowledge > 0) {
          addLog('INFO', lang === 'es'
            ? `Base de conocimiento: ${knowledge} chunks indexados.`
            : `Knowledge base: ${knowledge} chunks indexed.`);
        }
        prevKnowledge = knowledge;

        // -- Autolearn: web ingest activity --
        const webChunks = al.total_web_chunks || 0;
        if (prevWebChunks >= 0 && webChunks > prevWebChunks) {
          const diff = webChunks - prevWebChunks;
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: +${diff} fragmentos web ingestados (total: ${webChunks}).`
            : `Autolearn: +${diff} web chunks ingested (total: ${webChunks}).`);
        } else if (prevWebChunks < 0 && webChunks > 0) {
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: ${webChunks} fragmentos web en base de conocimiento.`
            : `Autolearn: ${webChunks} web chunks in knowledge base.`);
        }
        prevWebChunks = webChunks;

        // -- Autolearn: code index activity --
        const codeChunks = al.total_code_chunks || 0;
        if (prevCodeChunks >= 0 && codeChunks > prevCodeChunks) {
          const diff = codeChunks - prevCodeChunks;
          addLog('LEARN', lang === 'es'
            ? `Autolearn: +${diff} fragmentos de código propio indexados.`
            : `Autolearn: +${diff} self-code chunks indexed.`);
        } else if (prevCodeChunks < 0 && codeChunks > 0) {
          addLog('LEARN', lang === 'es'
            ? `Autolearn: ${codeChunks} fragmentos de código propio indexados.`
            : `Autolearn: ${codeChunks} self-code chunks indexed.`);
        }
        prevCodeChunks = codeChunks;

        // -- Autolearn: analysis & proposals --
        const analyses = al.total_analyses || 0;
        const proposals = al.total_proposals || 0;
        if (prevAnalyses >= 0 && analyses > prevAnalyses) {
          const diff = analyses - prevAnalyses;
          addLog('LEARN', lang === 'es'
            ? `Autolearn: ${diff} archivo${diff > 1 ? 's' : ''} analizado${diff > 1 ? 's' : ''} — ${proposals} propuestas generadas.`
            : `Autolearn: ${diff} file${diff > 1 ? 's' : ''} analyzed — ${proposals} proposals generated.`);
        }
        prevAnalyses = analyses;
        if (prevProposals >= 0 && proposals > prevProposals) {
          const diff = proposals - prevProposals;
          addLog('SYSTEM', lang === 'es'
            ? `Autolearn: +${diff} propuesta${diff > 1 ? 's' : ''} de auto-mejora generada${diff > 1 ? 's' : ''}.`
            : `Autolearn: +${diff} self-improvement proposal${diff > 1 ? 's' : ''} generated.`);
        }
        prevProposals = proposals;

        // -- Autolearn: URL discovery --
        const discoveredUrls = (al.discovered_urls || []).length;
        if (prevDiscoveredUrls >= 0 && discoveredUrls > prevDiscoveredUrls) {
          const diff = discoveredUrls - prevDiscoveredUrls;
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: +${diff} URL${diff > 1 ? 's' : ''} descubierta${diff > 1 ? 's' : ''} por el modelo (total: ${discoveredUrls}).`
            : `Autolearn: +${diff} URL${diff > 1 ? 's' : ''} discovered by model (total: ${discoveredUrls}).`);
        } else if (prevDiscoveredUrls < 0 && discoveredUrls > 0) {
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: ${discoveredUrls} URLs descubiertas para aprendizaje autónomo.`
            : `Autolearn: ${discoveredUrls} URLs discovered for autonomous learning.`);
        }
        prevDiscoveredUrls = discoveredUrls;

        // -- Request metrics (only show when new requests arrived) --
        if (prevRequests >= 0 && metrics.chat_requests > prevRequests) {
          const diff = metrics.chat_requests - prevRequests;
          const lat = metrics.avg_latency_ms || 0;
          const tokens = metrics.completion_tokens_est || 0;
          addLog('INFO', lang === 'es'
            ? `+${diff} petición${diff > 1 ? 'es' : ''} procesada${diff > 1 ? 's' : ''} — latencia media: ${lat}ms, tokens generados: ${tokens}.`
            : `+${diff} request${diff > 1 ? 's' : ''} processed — avg latency: ${lat}ms, tokens generated: ${tokens}.`);
        }
        prevRequests = metrics.chat_requests || 0;

      } catch { /* backend offline */ }
    };

    // Initial poll after short delay
    const t1 = setTimeout(poll, 2000);
    // Then every 20s
    const interval = setInterval(poll, 20000);
    return () => { disposed = true; clearTimeout(t1); clearInterval(interval); };
  }, [addLog, settings.language]);

  useEffect(() => {
    let disposed = false;
    const fetchPending = async () => {
      try {
        const resp = await fetch("/v1/self-edits/proposals?status=pending");
        if (!resp.ok) return;
        const payload = await resp.json().catch(() => ({}));
        const nextCount = Array.isArray(payload?.data) ? payload.data.length : 0;
        if (!disposed) setSelfEditsPendingCount(nextCount);
      } catch {
        // ignore
      }
    };

    void fetchPending();
    const interval = window.setInterval(fetchPending, 8000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, []);

  useMotionValueEvent(scrollY, "change", (latest) => {
    if (activeModificationFiles) return;
    const container = mainScrollRef.current;
    if (!container || isAutoScrollingRef.current) return;
    const diff = latest - lastScrollYRef.current;
    lastScrollYRef.current = latest;
    if (latest < 10) { if (hasMessages) setHeaderVisible(true); return; }
    if (Math.abs(diff) < 10) return;
    if (diff > 15) setHeaderVisible(false);
    else if (diff < -20) setHeaderVisible(true);
  });

  useEffect(() => {
    const handleGlobalActivity = (e: MouseEvent) => {
      if (activeModificationFiles) return;
      if (e.clientY < 80 && lastScrollYRef.current < 10) {
        if (!headerVisible) setHeaderVisible(true);
      }
      if (e.clientY > window.innerHeight - 120) { if (!footerVisible) setFooterVisible(true); resetInactivityTimer(); }
    };
    window.addEventListener('mousemove', handleGlobalActivity);
    return () => window.removeEventListener('mousemove', handleGlobalActivity);
  }, [footerVisible, headerVisible, resetInactivityTimer, activeModificationFiles]);

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      resetInactivityTimer();
      if (!footerVisible) setFooterVisible(true);
      if (e.altKey && e.key.toLowerCase() === 'k') { e.preventDefault(); setIsCommandPaletteOpen(prev => !prev); return; }
      if (e.key === 'Escape') {
        if (activeModificationFiles) setActiveModificationFiles(null);
        else if (isSettingsOpen) setIsSettingsOpen(false);
        else if (isCommandPaletteOpen) setIsCommandPaletteOpen(false);
        else if (isHelpOpen) setIsHelpOpen(false);
        else if (isReasoningOpen) setIsReasoningOpen(false);
        else setIsSettingsOpen(true);
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [resetInactivityTimer, footerVisible, isCommandPaletteOpen, isHelpOpen, isReasoningOpen, isSettingsOpen, activeModificationFiles]);

  useEffect(() => {
    const savedSessions = localStorage.getItem('chat-sessions');
    const savedSettings = localStorage.getItem('user-settings');
    if (savedSessions) {
      try {
        const parsedSessions = JSON.parse(savedSessions);
        const normalizedSessions = Array.isArray(parsedSessions)
          ? parsedSessions
              .map((session) => normalizeSession(session))
              .filter((session): session is ChatSession => session !== null)
              .filter((session, index, all) => {
              const isEmptyDraft = session.messages.length === 0;
              if (!isEmptyDraft) return true;
              return all.findIndex(candidate =>
                candidate
                && candidate.title === session.title
                && candidate.messages.length === 0
              ) === index;
            })
          : [];
        if (normalizedSessions.length > 0) {
          setSessions(normalizedSessions);
          setCurrentSessionId(normalizedSessions[0].id);
        } else {
          handleNewChat();
        }
      } catch {
        handleNewChat();
      }
    } else handleNewChat();
    if (savedSettings) {
      try {
        setSettings(normalizeSettings(JSON.parse(savedSettings)));
      } catch {
        setSettings(DEFAULT_SETTINGS);
      }
    }
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    document.body.classList.toggle('dark', isDarkMode);
    document.documentElement.style.colorScheme = isDarkMode ? 'dark' : 'light';
    document.body.style.colorScheme = isDarkMode ? 'dark' : 'light';
    localStorage.setItem('dark-mode', String(isDarkMode));
  }, [isDarkMode]);
  useEffect(() => { if (sessions.length > 0) localStorage.setItem('chat-sessions', JSON.stringify(sessions)); }, [sessions]);
  useEffect(() => { localStorage.setItem('user-settings', JSON.stringify(settings)); }, [settings]);
  useEffect(() => {
    if (sessions.length === 0) {
      setCurrentSessionId(null);
      return;
    }
    if (!currentSessionId || !sessions.some(session => session.id === currentSessionId)) {
      setCurrentSessionId(sessions[0].id);
    }
  }, [sessions, currentSessionId]);

  useEffect(() => {
    if (activeView === 'chat' && currentSession?.messages.length) {
      const container = mainScrollRef.current;
      if (!container) return;
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 400;
      if (isAtBottom || isLoading) {
        isAutoScrollingRef.current = true;
        container.scrollTo({ top: container.scrollHeight, behavior: isLoading ? 'auto' : 'smooth' });
        const timer = setTimeout(() => { isAutoScrollingRef.current = false; }, 200);
        return () => clearTimeout(timer);
      }
    }
  }, [sessions, isLoading, isSearching, activeView, currentSessionId, currentSession]);

  const handleNewChat = useCallback(() => {
    const newSession = createEmptySession(settings.language);
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    handleSelectView('chat');
    setHeaderVisible(false); setFooterVisible(false);
    addLog('SYSTEM', settings.language === 'es' ? 'Sincronización de núcleo completada.' : 'Kernel sync complete.');
  }, [addLog, settings.language]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    const remainingSessions = sessions.filter(session => session.id !== sessionId);
    if (remainingSessions.length === 0) {
      const replacementSession = createEmptySession(settings.language);
      setSessions([replacementSession]);
      setCurrentSessionId(replacementSession.id);
      setActiveView(prev => { setPrevView(prev); return 'chat'; });
      setHeaderVisible(false);
      setFooterVisible(false);
      return;
    }
    setSessions(remainingSessions);
    if (!remainingSessions.some(session => session.id === currentSessionId)) {
      setCurrentSessionId(remainingSessions[0].id);
    }
  }, [sessions, currentSessionId, settings.language]);

  const handleClearHistory = useCallback(() => {
    const freshSession = createEmptySession(settings.language);
    setSessions([freshSession]);
    setCurrentSessionId(freshSession.id);
    setActiveView(prev => { setPrevView(prev); return 'chat'; });
    setHeaderVisible(false);
    setFooterVisible(false);
    addLog('SYSTEM', settings.language === 'es' ? 'Historial purgado. Conversación reiniciada.' : 'History cleared. Conversation reset.');
  }, [addLog, settings.language]);

  const handleSelectView = useCallback((newView: ViewType) => {
    if (newView === 'analysis') setAnalysisFocusTab('stack');
    setActiveView(prev => { setPrevView(prev); return newView; });
    setHeaderVisible(true);
  }, []);

  const openAnalysisTab = useCallback((tab: 'stack' | 'learning' | 'internet') => {
    setAnalysisFocusTab(tab);
    setActiveView(prev => { setPrevView(prev); return 'analysis'; });
    setHeaderVisible(true);
  }, []);

  const handleNavigateToChat = useCallback((sessionId: string, messageId?: string) => {
    setCurrentSessionId(sessionId); setActiveView('chat');
    if (messageId) { setTimeout(() => { document.getElementById(messageId)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 300); }
  }, []);

  const handleLoadDemo = useCallback(() => {
    if (!currentSessionId) return;
    const mockSources: Source[] = [
      { url: 'https://react.dev', title: t.analysis_library + ': React Hooks', domain: 'react.dev', kind: 'web', index: 0 },
      { url: 'https://fastapi.tiangolo.com', title: t.analysis_library + ': FastAPI', domain: 'fastapi.tiangolo.com', kind: 'web', index: 1 },
      { url: 'https://developer.mozilla.org/es/docs/Web/API/Element/scrollIntoView', title: 'MDN: Element.scrollIntoView()', domain: 'developer.mozilla.org', kind: 'web', index: 2 }
    ];
    
    const demoContentEs = `Protocolo activado. He analizado la estructura actual y optimizado los parámetros del kernel.

### Análisis de Componentes:
* **Motor de Animación**: Optimización de constantes de resorte (stiffness) para mayor fluidez.
* **Gestión de Estado**: Reducción de latencia en el ciclo de renderizado virtualizado.
* **Seguridad**: Verificación de firmas de integridad en parches dinámicos.

### Parámetros de Configuración Actualizados:
\`\`\`typescript
const VORTEX_CONFIG = {
  neuralPrecision: 0.98,
  latencyThreshold: "45ms",
  autoSync: true,
  engineVersion: "v2.5.0-beta",
  activeModules: ["Search", "PatchExplorer", "NeuralReasoning"]
};
\`\`\`

### Modificaciones de Archivo Propuestas:
\`\`\`file:App.tsx
- const timer = 100;
+ const timer = 60;
\`\`\`

\`\`\`file:components/Sidebar.tsx
- stiffness: 400;
+ stiffness: 500;
\`\`\``;

    const demoContentEn = `Protocol activated. I have analyzed the current structure and optimized kernel parameters.

### Component Analysis:
* **Animation Engine**: Optimization of spring constants (stiffness) for greater fluidity.
* **State Management**: Latency reduction in the virtualized rendering cycle.
* **Security**: Integrity signature verification in dynamic patches.

### Updated Configuration Parameters:
\`\`\`typescript
const VORTEX_CONFIG = {
  neuralPrecision: 0.98,
  latencyThreshold: "45ms",
  autoSync: true,
  engineVersion: "v2.5.0-beta",
  activeModules: ["Search", "PatchExplorer", "NeuralReasoning"]
};
\`\`\`

### Proposed File Modifications:
\`\`\`file:App.tsx
- const timer = 100;
+ const timer = 60;
\`\`\`

\`\`\`file:components/Sidebar.tsx
- stiffness: 400;
+ stiffness: 500;
\`\`\``;

    const demoMessages: Message[] = [
      { id: 'demo-1', role: Role.USER, content: settings.language === 'es' ? "Activar protocolo de demostración." : "Activate demo protocol.", timestamp: Date.now() - 60000 },
      { 
        id: 'demo-2', 
        role: Role.AI, 
        content: settings.language === 'es' ? demoContentEs : demoContentEn, 
        thought: settings.language === 'es' ? "Análisis completado. Se han identificado cuellos de botella en la renderización y se han ajustado las físicas del sidebar para una respuesta táctil superior." : "Analysis complete. Rendering bottlenecks identified and sidebar physics adjusted for superior tactile response.", 
        sources: mockSources, 
        fileChanges: [{ path: 'App.tsx', diff: `- const timer = 100;\n+ const timer = 60;` }, { path: 'components/Sidebar.tsx', diff: `- stiffness: 400;\n+ stiffness: 500;` }], 
        timestamp: Date.now() - 30000 
      }
    ];
    setSessions(prev => prev.map(s => s.id === currentSessionId ? { ...s, messages: demoMessages, updatedAt: Date.now() } : s));
    setHeaderVisible(true); setFooterVisible(true);
    addLog('SYSTEM', settings.language === 'es' ? 'Carga de demostración completada.' : 'Demo load complete.');
  }, [currentSessionId, addLog, settings.language, t]);

  const handleSendMessageLocalFirst = async (content: string, useInternet: boolean = false, selectedMode: AppMode = 'ask', useThinking: boolean = true, autoTrain: boolean = true) => {
    if (sendDisabledReason) {
      addLog('SYSTEM', sendDisabledReason);
      return;
    }
    let targetSessionId = currentSessionId;
    let targetSession = sessions.find(session => session.id === targetSessionId);
    if (!targetSession) {
      targetSession = createEmptySession(settings.language);
      targetSessionId = targetSession.id;
      setSessions(prev => [targetSession!, ...prev]);
      setCurrentSessionId(targetSessionId);
    }
    if (!targetSessionId) return;

    setMode(selectedMode);
    if (activeView !== 'chat') handleSelectView('chat');
    setHeaderVisible(true);
    setFooterVisible(true);
    resetInactivityTimer();
    addLog('INFO', settings.language === 'es' ? `Prompt enviado (${content.length} chars) · modo=${selectedMode}` : `Prompt sent (${content.length} chars) · mode=${selectedMode}`);
    if (useInternet) {
      addLog('SEARCH', settings.language === 'es' ? 'Internet activado para este prompt.' : 'Internet enabled for this prompt.');
    }

    const userMessage: Message = { id: Date.now().toString(), role: Role.USER, content, timestamp: Date.now() };
    const aiMessageId = (Date.now() + 1).toString();
    const initialAiMessage: Message = { id: aiMessageId, role: Role.AI, content: "", thought: "", requestId: undefined, sources: [], groundingSupports: [], timestamp: Date.now() };
    setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: [...s.messages, userMessage, initialAiMessage], updatedAt: Date.now() } : s));

    const currentMessages = targetSession.messages || [];
    if (currentMessages.length === 0) {
      vortexService.generateChatTitle(content, settings.language).then(result => {
        if (result.ok && result.title) {
          const normalizedTitle = repairMojibakeText(result.title);
          setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, title: normalizedTitle } : s));
        }
      }).catch(() => {});
    }

    setIsLoading(true);
    setIsSearching(useInternet);
    abortControllerRef.current = false;
    try {
      const history = targetSession.messages || [];
      const stream = vortexService.generateResponseStream(
        history,
        content,
        useInternet,
        useThinking,
        selectedMode,
        settings.language,
        internetAllowlist
      );
      let started = false;
      let aborted = false;
      let lastText = '';
      let lastRequestId: string | undefined;
      for await (const chunk of stream) {
        if (abortControllerRef.current) {
          aborted = true;
          break;
        }
        if (!started) {
          started = true;
          addLog('INFO', settings.language === 'es' ? 'Stream SSE conectado.' : 'SSE stream connected.');
        }
        setIsSearching(false);
        lastText = chunk.text || lastText;
        if (chunk.requestId) lastRequestId = chunk.requestId;
        setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: s.messages.map(m => m.id === aiMessageId ? { ...m, content: chunk.text, thought: chunk.thought || m.thought, requestId: chunk.requestId || m.requestId, sources: chunk.sources.length > 0 ? chunk.sources : m.sources, fileChanges: chunk.fileChanges || m.fileChanges } : m) } : s));
      }
      if (aborted) {
        addLog('SYSTEM', settings.language === 'es' ? 'Ejecución abortada por el usuario.' : 'Run aborted by user.');
      } else if (autoTrain && lastRequestId && lastText) {
        addLog('LEARN', settings.language === 'es' ? 'Auto-train: enviando feedback...' : 'Auto-train: sending feedback...');
        const feedback = await vortexService.submitFeedback(lastRequestId, lastText);
        if (feedback.ok && feedback.trainingEvent) {
          addLog('LEARN', settings.language === 'es' ? 'Auto-train registrado (training_event creado).' : 'Auto-train logged (training_event created).');
          setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: s.messages.map(m => m.id === aiMessageId ? { ...m, trainingEvent: true } : m) } : s));
          const quickTrain = await controlService.startTraining('quick').catch(() => null);
          if (quickTrain?.ok && quickTrain.run_id) {
            addLog('LEARN', settings.language === 'es' ? `Aprendizaje rápido lanzado: ${quickTrain.run_id}` : `Quick learning launched: ${quickTrain.run_id}`);
          }
          await suggestPatchFromMessage(aiMessageId, 'auto-train');
        } else if (feedback.ok) {
          addLog('SYSTEM', settings.language === 'es' ? 'Auto-train OK, pero sin training_event.' : 'Auto-train OK, but no training_event.');
        } else {
          addLog('SYSTEM', settings.language === 'es' ? `Auto-train falló: ${feedback.error || 'error'}` : `Auto-train failed: ${feedback.error || 'error'}`);
        }
      } else if (autoTrain) {
        addLog('SYSTEM', settings.language === 'es' ? 'Auto-train omitido: request_id ausente.' : 'Auto-train skipped: missing request_id.');
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : (settings.language === 'es' ? 'Interrupción de flujo.' : 'Flow interrupted.');
      addLog('SYSTEM', repairMojibakeText(detail));
    } finally {
      setIsLoading(false);
      setIsSearching(false);
      resetInactivityTimer();
    }
  };

  const handleOpenModificationExplorer = (files: { path: string, diff: string }[]) => {
    setActiveModificationFiles(files);
    setHeaderVisible(false);
    setFooterVisible(false);
  };

  const springConfig = { type: 'spring' as const, damping: 28, stiffness: 220, mass: 0.9 };
  const direction = VIEW_INDEX[activeView] > VIEW_INDEX[prevView] ? 1 : -1;

  return (
    <div className={`relative flex h-screen w-full bg-background transition-colors duration-1000 overflow-hidden text-foreground accelerated ${mode === 'agent' ? 'ring-[6px] ring-primary/10' : ''}`}>
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_14%_16%,rgba(0,194,255,0.16),transparent_22%),radial-gradient(circle_at_88%_10%,rgba(255,255,255,0.55),transparent_14%),linear-gradient(180deg,rgba(255,255,255,0.18),transparent)] dark:bg-[radial-gradient(circle_at_14%_16%,rgba(0,194,255,0.24),transparent_24%),radial-gradient(circle_at_88%_10%,rgba(255,255,255,0.08),transparent_14%),linear-gradient(180deg,rgba(255,255,255,0.06),transparent)]" />
        <div className="absolute inset-0 opacity-[0.05] vortex-grid-bg" />
        <div className="absolute -top-20 right-[-8%] h-[360px] w-[360px] rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute bottom-[-18%] left-[6%] h-[300px] w-[300px] rounded-full bg-foreground/5 blur-3xl dark:bg-white/5" />
      </div>
      <CommandPalette isOpen={isCommandPaletteOpen} onClose={() => setIsCommandPaletteOpen(false)} sessions={sessions} currentSessionId={currentSessionId} onSelectSession={setCurrentSessionId} onNewChat={handleNewChat} onDeleteSession={handleDeleteSession} onClearHistory={handleClearHistory} onExportChat={() => {}} isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} isSidebarOpen={isSidebarOpen} onToggleSidebar={() => { const next = !isSidebarOpen; setIsSidebarOpen(next); if (next) setIsReasoningOpen(false); }} onOpenSettings={() => setIsSettingsOpen(true)} onOpenHelp={() => setIsHelpOpen(true)} categoryOrder={settings.categoryOrder} language={settings.language} onSetFontSize={(size) => setSettings({ ...settings, fontSize: size })} />
      <AnimatePresence initial={false}>{isSidebarOpen && !activeModificationFiles && (
          <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 280, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full overflow-hidden shrink-0 z-50 flex border-r border-border/50 shadow-2xl relative"><Sidebar sessions={sessions} currentSessionId={currentSessionId} activeView={activeView} onSelectSession={setCurrentSessionId} onSelectView={handleSelectView} onNewChat={handleNewChat} onDeleteSession={handleDeleteSession} isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} onClose={() => setIsSidebarOpen(false)} onOpenSettings={() => setIsSettingsOpen(true)} isOpen={true} language={settings.language} selfEditsPendingCount={selfEditsPendingCount} /></motion.div>
      )}</AnimatePresence>
      <div className="flex-1 flex overflow-hidden relative">
        <main className="flex-1 flex flex-col h-full bg-background relative z-0 overflow-hidden">
          {!activeModificationFiles && (
            <motion.header initial={false} animate={{ y: headerVisible ? 0 : -100, opacity: headerVisible ? 1 : 0 }} transition={springConfig} className={`absolute top-0 left-0 right-0 h-24 border-b border-black/5 dark:border-white/10 flex items-center justify-between px-6 lg:px-10 bg-white/72 dark:bg-[#02060d]/80 backdrop-blur-3xl z-40 shrink-0 shadow-sm pointer-events-auto accelerated ${mode === 'agent' ? 'bg-primary/5 border-primary/20' : ''}`}>
              <div className="flex items-center gap-8">
                <AnimatePresence mode="wait">{!isSidebarOpen && (<motion.button initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.8, opacity: 0 }} whileHover={{ scale: 1.1, backgroundColor: 'hsla(var(--muted-foreground) / 0.1)' }} whileTap={{ scale: 0.9 }} onClick={() => { setIsSidebarOpen(true); setIsReasoningOpen(false); }} className="p-3.5 rounded-2xl transition-all"><PanelLeft size={24} /></motion.button>)}</AnimatePresence>
                <div className="flex items-center gap-4">
                  <motion.div whileHover={{ rotate: -8, scale: 1.04 }} transition={{ type: 'spring', stiffness: 320, damping: 18 }}>
                    <VortexLogo size={40} alt="Vortex" />
                  </motion.div>
                  <div className="flex flex-col">
                    <div className="flex items-center gap-3">
                      <h1 className="text-[18px] font-black tracking-tight leading-none">Vortex</h1>
                      <span className="rounded-full border border-border/70 bg-background/70 px-3 py-1 text-[9px] font-black uppercase tracking-[0.28em] text-primary">
                        {activeEngineLabel}
                      </span>
                    </div>
                    <span className="mt-2 text-[9px] font-black uppercase tracking-[0.32em] text-primary">{t.system_kernel}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <motion.button whileHover={{ scale: 1.1, backgroundColor: 'hsla(var(--primary) / 0.1)' }} whileTap={{ scale: 0.9 }} onClick={() => setSettings({ ...settings, language: settings.language === 'es' ? 'en' : 'es' })} className="w-12 h-12 flex items-center justify-center bg-muted/40 dark:bg-zinc-900/40 border border-border/50 rounded-2xl hover:border-primary/40 transition-all shadow-sm overflow-hidden"><img src={settings.language === 'es' ? 'https://flagcdn.com/w80/es.png' : 'https://flagcdn.com/w80/us.png'} alt={settings.language} className="w-7 h-auto object-contain rounded-sm select-none" /></motion.button>
                <div className="flex items-center gap-1 bg-muted/40 dark:bg-zinc-900/40 p-1 rounded-2xl border border-border/50 relative">{['chat', 'analysis', 'edits', 'terminal'].map(v => (<button key={v} onClick={() => handleSelectView(v as ViewType)} className={`relative p-2.5 rounded-xl transition-all z-10 ${activeView === v ? 'text-primary-foreground' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>{v === 'chat' ? <MessageSquare size={16} /> : v === 'analysis' ? <BarChart3 size={16} /> : v === 'edits' ? <FileCode size={16} /> : <TerminalIcon size={16} />}{activeView === v && <motion.div layoutId="header-nav-indicator" className="absolute inset-0 bg-primary rounded-xl shadow-lg -z-10" transition={springConfig} />}</button>))}</div>
                <TopBarStackStatus
                  status={operationalStatus}
                  controlStatus={controlStatus}
                  language={settings.language}
                  onBootstrap={() => controlService.bootstrap(false)}
                  onModelInit={() => controlService.initModel()}
                  onRestartRuntime={() => controlService.restartRuntime()}
                  onStartTraining={() => controlService.startTraining('quick')}
                  onOpenTraining={() => openAnalysisTab('learning')}
                />
                <motion.button whileHover={{ scale: 1.05, y: -2 }} whileTap={{ scale: 0.95 }} onClick={() => setIsCommandPaletteOpen(true)} className="flex items-center gap-3 px-5 py-2.5 bg-muted/50 dark:bg-zinc-900/50 hover:bg-primary/10 rounded-2xl border border-border/50 transition-all shadow-sm"><Zap size={16} className={'text-primary'} /><kbd className="hidden lg:inline-block px-2 py-0.5 bg-background border rounded-lg text-[8px] font-black opacity-40">ALT+K</kbd></motion.button>
              </div>
            </motion.header>
          )}

          <div ref={mainScrollRef} className="flex-1 overflow-y-auto custom-scrollbar flex flex-col relative h-full bg-background scroll-smooth accelerated">
            {hasMessages && !activeModificationFiles && <div className="pt-24 shrink-0" />}
            <AnimatePresence mode="popLayout" custom={direction}>
              {activeView === 'chat' && (
                <motion.div key="chat" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className={`mx-auto w-full flex-1 flex flex-col px-6 lg:px-16 min-h-full transition-all duration-500 ${!hasMessages ? 'justify-center pt-24 pb-40 max-w-[1420px]' : 'pt-6 max-w-full'}`}>
                  {!hasMessages ? (
                    <div className="grid items-center gap-8 lg:grid-cols-[1.08fr_0.92fr]">
                      <div className="relative z-10 space-y-8">
                        <div className="inline-flex items-center gap-3 rounded-full border border-primary/20 bg-primary/[0.08] px-5 py-3 text-[10px] font-black uppercase tracking-[0.35em] text-primary shadow-[0_12px_30px_-18px_rgba(0,194,255,0.75)]">
                          <span>{settings.language === 'es' ? 'Vortex local core' : 'Vortex local core'}</span>
                          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                          <span>{settings.language === 'es' ? 'Control manual visible' : 'Visible manual control'}</span>
                        </div>

                        <div className="space-y-5">
                          <h2 className="max-w-4xl text-5xl font-black tracking-[-0.06em] leading-[0.95] text-foreground lg:text-7xl">
                            {settings.language === 'es'
                              ? 'Vortex para código, control local y mejora continua.'
                              : 'Vortex for code, local control, and continuous improvement.'}
                          </h2>
                          <p className="max-w-2xl text-base font-medium leading-8 text-muted-foreground lg:text-lg">
                            {settings.language === 'es'
                              ? 'El punto de entrada ahora se comporta como una consola operativa real: runtime local, entrenamiento visible, internet por prompt y una sola interfaz principal.'
                              : 'The entry state now behaves like a real operations console: local runtime, visible training, prompt-level internet, and one primary interface.'}
                          </p>
                        </div>

                        <div className="grid gap-3 sm:grid-cols-3">
                          {heroCards.map((card) => (
                            <div key={card.label} className="rounded-[2rem] border border-black/[0.08] bg-white/80 px-5 py-5 shadow-[0_24px_60px_-38px_rgba(15,23,42,0.45)] backdrop-blur-2xl dark:border-white/10 dark:bg-white/5 dark:shadow-[0_28px_70px_-45px_rgba(0,0,0,0.85)]">
                              <p className="text-[10px] font-black uppercase tracking-[0.32em] text-muted-foreground">{card.label}</p>
                              <p className="mt-3 text-lg font-black tracking-tight">{card.value}</p>
                            </div>
                          ))}
                        </div>

                        <div className="flex flex-wrap items-center gap-4">
                          <motion.button whileHover={{ scale: 1.04, y: -2, boxShadow: '0 24px 60px -28px rgba(0,194,255,0.55)' }} whileTap={{ scale: 0.96 }} onClick={handleLoadDemo} className="flex items-center gap-4 rounded-full bg-foreground px-8 py-4 text-[10px] font-black uppercase tracking-[0.32em] text-background transition-all dark:bg-primary dark:text-primary-foreground">
                            {t.initialize_vortex}
                          </motion.button>
                          <motion.button whileHover={{ y: -2, borderColor: 'hsla(var(--primary) / 0.35)' }} whileTap={{ scale: 0.98 }} onClick={() => handleSelectView('analysis')} className="rounded-full border border-border/70 bg-background/70 px-7 py-4 text-[10px] font-black uppercase tracking-[0.28em] text-foreground shadow-sm backdrop-blur-2xl">
                            {settings.language === 'es' ? 'Abrir control' : 'Open control'}
                          </motion.button>
                        </div>

                        <p className="max-w-2xl text-sm font-medium text-muted-foreground">
                          {operationalStatus?.ok
                            ? (settings.language === 'es' ? 'Stack local listo para sesiones manuales, búsqueda puntual y entrenamiento controlado.' : 'Local stack is ready for manual sessions, prompt-level browsing, and controlled training.')
                            : sendDisabledReason}
                        </p>
                      </div>

                      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: 'easeOut' }} className="relative isolate overflow-hidden rounded-[2.45rem] border border-primary/[0.15] bg-white/72 p-5 text-foreground shadow-[0_50px_120px_-50px_rgba(15,23,42,0.45)] dark:border-white/10 dark:bg-[#02060d] dark:text-white dark:shadow-[0_60px_140px_-42px_rgba(0,0,0,0.75)] lg:p-7">
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_28%_24%,rgba(0,194,255,0.18),transparent_34%),radial-gradient(circle_at_76%_16%,rgba(255,255,255,0.7),transparent_18%),linear-gradient(180deg,rgba(255,255,255,0.45),transparent)] dark:bg-[radial-gradient(circle_at_28%_24%,rgba(0,194,255,0.34),transparent_34%),radial-gradient(circle_at_76%_16%,rgba(255,255,255,0.14),transparent_18%),linear-gradient(180deg,rgba(255,255,255,0.05),transparent)]" />
                        <div className="absolute inset-4 rounded-[2rem] border border-black/[0.08] dark:border-white/10" />
                        <div className="absolute inset-[22%] rounded-full border border-primary/[0.15] dark:border-cyan-300/20" />
                        <div className="absolute inset-[31%] rounded-full border border-black/[0.06] dark:border-white/10" />
                        <div className="absolute left-1/2 top-1/2 h-[68%] w-[68%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/[0.14] blur-3xl" />

                        <div className="relative z-10 flex min-h-[390px] flex-col justify-between">
                          <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-[0.35em] text-foreground/45 dark:text-white/55">
                            <span>{settings.language === 'es' ? 'Núcleo activo' : 'Active core'}</span>
                            <span>{readyLabel}</span>
                          </div>

                          <div className="flex flex-1 items-center justify-center py-6">
                            <motion.div animate={{ rotate: [0, 4, -4, 0], scale: [1, 1.015, 1] }} transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}>
                              <VortexLogo size={220} alt="Vortex mark" className="max-w-full" />
                            </motion.div>
                          </div>

                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-[1.7rem] border border-black/10 bg-white/82 p-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.06]">
                              <p className="text-[10px] font-black uppercase tracking-[0.28em] text-foreground/45 dark:text-white/45">
                                {settings.language === 'es' ? 'Engine' : 'Engine'}
                              </p>
                              <p className="mt-3 text-base font-black tracking-tight text-foreground dark:text-white">{activeEngineLabel}</p>
                              <p className="mt-2 text-xs text-foreground/55 dark:text-white/55">
                                {operationalStatus?.engine_base_url || '127.0.0.1'}
                              </p>
                            </div>
                            <div className="rounded-[1.7rem] border border-black/10 bg-white/82 p-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.06]">
                              <p className="text-[10px] font-black uppercase tracking-[0.28em] text-foreground/45 dark:text-white/45">
                                {settings.language === 'es' ? 'Modelo activo' : 'Active model'}
                              </p>
                              <p className="mt-3 text-base font-black tracking-tight text-foreground break-words dark:text-white">{activeModelLabel}</p>
                              <p className="mt-2 text-xs text-foreground/55 dark:text-white/55">
                                {operationalStatus?.web_disabled
                                  ? (settings.language === 'es' ? 'Internet solo al activarlo en el prompt' : 'Internet only when enabled on the prompt')
                                  : (settings.language === 'es' ? 'Política web editable' : 'Editable web policy')}
                              </p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </div>
                  ) : (
                    <div className="pb-40">
                      <VirtualizedMessageList 
                        messages={currentSession.messages} 
                        fontSize={settings.fontSize} 
                        codeTheme={settings.codeTheme}
                        onShowReasoning={messageId => { setActiveThoughtMessageId(messageId); setIsReasoningOpen(true); setIsSidebarOpen(false); }} 
                        onOpenModificationExplorer={handleOpenModificationExplorer} 
                        onSuggestPatch={messageId => { void suggestPatchFromMessage(messageId, 'manual'); }}
                        isLoading={isLoading} 
                        language={settings.language} 
                        containerRef={mainScrollRef} 
                      />
                      {isSearching && (
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-5 px-8 py-5 mt-6 bg-primary/5 border border-primary/20 rounded-[2.5rem] text-primary shadow-xl w-fit glass-card accelerated"><Globe size={22} className="animate-spin-slow" /><p className="text-[12px] font-black uppercase tracking-widest">{settings.language === 'es' ? 'Capas de conocimiento activas...' : 'Active knowledge layers...'}</p></motion.div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
              {activeView === 'analysis' && <motion.div key="analysis" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><AnalysisView sessions={sessions} onNavigateToChat={handleNavigateToChat} onAddLog={addLog} language={settings.language} controlStatus={controlStatus} operationalStatus={operationalStatus} onBootstrap={() => controlService.bootstrap(false)} onModelInit={() => controlService.initModel()} onRestartRuntime={() => controlService.restartRuntime()} onReloadInstructions={() => controlService.reloadInstructions()} onStartTraining={(trainingMode) => controlService.startTraining(trainingMode)} onSaveAllowlist={(domains) => controlService.saveAllowlist(domains)} focusTab={analysisFocusTab} /></motion.div>}
              {activeView === 'edits' && <motion.div key="edits" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><SelfEditsView language={settings.language} onAddLog={addLog} onPendingCountChange={setSelfEditsPendingCount} /></motion.div>}
              {activeView === 'terminal' && <motion.div key="terminal" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><TerminalView logs={logs} onClear={() => setLogs([])} language={settings.language} /></motion.div>}
            </AnimatePresence>
          </div>

          {!activeModificationFiles && activeView === 'chat' && (
            <motion.div initial={false} animate={{ y: footerVisible ? 0 : 200, opacity: footerVisible ? 1 : 0 }} transition={{ type: 'spring', damping: 30, stiffness: 200 }} className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t from-background via-background/95 to-transparent pt-12 pb-8 z-30 pointer-events-auto accelerated ${mode === 'agent' ? 'from-primary/5' : ''}`}>
              <div className="pointer-events-auto">
                <ChatInput
                  onSend={handleSendMessageLocalFirst}
                  isLoading={isLoading}
                  isDarkMode={isDarkMode}
                  canUseInternet={canUseInternet}
                  allowAutoTrain={canStartTraining}
                  sendDisabledReason={sendDisabledReason}
                  onStop={() => { abortControllerRef.current = true; }}
                  language={settings.language}
                  onInteraction={() => { resetInactivityTimer(); if (!footerVisible) setFooterVisible(true); }}
                  onFocusChange={setIsComposerFocused}
                  onDraftChange={setHasComposerDraft}
                />
              </div>
            </motion.div>
          )}
        </main>
        
        <AnimatePresence>{isReasoningOpen && !activeModificationFiles && (
            <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 400, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full border-l border-border/50 shrink-0 z-50 overflow-hidden bg-zinc-950/95 shadow-[-20px_0_50px_rgba(0,0,0,0.5)]"><ReasoningDrawer isOpen={isReasoningOpen} onClose={() => setIsReasoningOpen(false)} thought={activeThought} language={settings.language} isStreaming={isCurrentThoughtStreaming} /></motion.div>
        )}</AnimatePresence>
      </div>

      <AnimatePresence>
        {activeModificationFiles && (
          <ModificationExplorerModal 
            fileChanges={activeModificationFiles} 
            onClose={() => { setActiveModificationFiles(null); setHeaderVisible(true); setFooterVisible(true); }} 
            language={settings.language} 
          />
        )}
      </AnimatePresence>

      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={settings} onUpdateSettings={setSettings} />
      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} isDarkMode={isDarkMode} language={settings.language} />
    </div>
  );
};

export default App;

import React, { useMemo, useState } from 'react';
import {
  BarChart3,
  FileCode2,
  MessageSquare,
  Moon,
  PanelLeftClose,
  Plus,
  Settings,
  Sun,
  TerminalSquare,
  Trash2,
  AlertTriangle,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { ChatSession, Language, ViewType } from '../types';
import { translations } from '../translations';
import VortexLogo from './VortexLogo';

interface SidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  activeView: ViewType;
  onSelectSession: (id: string) => void;
  onSelectView: (view: ViewType) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  onClose: () => void;
  onOpenSettings: () => void;
  isOpen: boolean;
  language: Language;
  selfEditsPendingCount?: number;
}

const springTransition = { type: 'spring' as const, stiffness: 420, damping: 34 };

const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  currentSessionId,
  activeView,
  onSelectSession,
  onSelectView,
  onNewChat,
  onDeleteSession,
  isDarkMode,
  toggleDarkMode,
  onClose,
  onOpenSettings,
  language,
  selfEditsPendingCount = 0,
}) => {
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const t = translations[language];

  const themeLabel = language === 'es'
    ? (isDarkMode ? 'Cambiar a claro' : 'Cambiar a oscuro')
    : (isDarkMode ? 'Switch to light' : 'Switch to dark');

  const navigation = useMemo(
    () => [
      { id: 'chat' as ViewType, label: t.nav_chat, icon: MessageSquare },
      { id: 'analysis' as ViewType, label: t.nav_analysis, icon: BarChart3 },
      { id: 'edits' as ViewType, label: t.nav_edits, icon: FileCode2, badge: selfEditsPendingCount },
      { id: 'terminal' as ViewType, label: t.nav_terminal, icon: TerminalSquare },
    ],
    [selfEditsPendingCount, t.nav_analysis, t.nav_chat, t.nav_edits, t.nav_terminal],
  );

  const orderedSessions = useMemo(
    () => [...sessions].sort((left, right) => right.updatedAt - left.updatedAt),
    [sessions],
  );

  const formatSessionTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString(language === 'es' ? 'es-ES' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const handleDeleteConfirm = () => {
    if (!sessionToDelete) return;
    onDeleteSession(sessionToDelete);
    setSessionToDelete(null);
  };

  return (
    <>
      <div className="relative flex h-full w-full flex-col overflow-hidden border-r border-border/60 bg-background/90 backdrop-blur-3xl transition-colors duration-500 dark:bg-[#030812]/95">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(0,194,255,0.16),transparent_32%),linear-gradient(180deg,rgba(255,255,255,0.05),transparent)] dark:bg-[radial-gradient(circle_at_top_left,rgba(0,194,255,0.18),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />
          <div className="absolute inset-0 opacity-[0.05] vortex-grid-bg" />
        </div>

        <div className="relative z-10 border-b border-border/50 px-5 pb-5 pt-6">
          <div className="flex items-start justify-between gap-3">
            <button
              type="button"
              onClick={() => onSelectView('chat')}
              className="group flex min-w-0 items-center gap-3 rounded-[1.6rem] border border-border/50 bg-background/75 px-3 py-3 text-left shadow-sm transition-all hover:border-primary/30 hover:bg-background/95"
            >
              <VortexLogo size={34} alt="Vortex" />
              <div className="min-w-0">
                <p className="text-[9px] font-black uppercase tracking-[0.28em] text-primary/80">
                  {language === 'es' ? 'Local core' : 'Local core'}
                </p>
                <p className="mt-1 truncate text-sm font-black tracking-tight">Vortex</p>
                <p className="mt-1 text-[10px] font-semibold text-muted-foreground">
                  {language === 'es' ? 'Frontend principal' : 'Primary frontend'}
                </p>
              </div>
            </button>

            <motion.button
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              onClick={onClose}
              className="rounded-2xl p-2.5 text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
              aria-label={language === 'es' ? 'Cerrar lateral' : 'Close sidebar'}
            >
              <PanelLeftClose size={20} />
            </motion.button>
          </div>

          <button
            type="button"
            onClick={onNewChat}
            className="mt-5 flex w-full items-center justify-between rounded-[1.35rem] border border-primary/20 bg-primary/[0.08] px-4 py-3.5 text-left shadow-[0_16px_40px_-26px_rgba(0,194,255,0.55)] transition-all hover:border-primary/40 hover:bg-primary/[0.12]"
          >
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.24em] text-primary">
                {language === 'es' ? 'Nueva sesión' : 'New session'}
              </p>
              <p className="mt-1 text-sm font-semibold text-foreground">
                {language === 'es' ? 'Abrir chat limpio' : 'Start a fresh chat'}
              </p>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
              <Plus size={18} />
            </div>
          </button>
        </div>

        <div className="relative z-10 px-4 pt-4">
          <div className="space-y-1.5">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectView(item.id)}
                  className={`relative flex w-full items-center gap-3 rounded-[1.15rem] px-4 py-3 text-left transition-all ${
                    isActive
                      ? 'bg-primary text-primary-foreground shadow-[0_18px_40px_-28px_rgba(0,194,255,0.65)]'
                      : 'text-foreground/75 hover:bg-background/80 hover:text-foreground'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-active-pill"
                      transition={springTransition}
                      className="absolute inset-0 rounded-[1.15rem] bg-primary shadow-[0_18px_40px_-28px_rgba(0,194,255,0.65)]"
                    />
                  )}
                  <span className="relative z-10 flex h-9 w-9 items-center justify-center rounded-xl bg-background/10">
                    <Icon size={18} />
                  </span>
                  <span className="relative z-10 flex-1 text-sm font-semibold tracking-tight">
                    {item.label}
                  </span>
                  {Boolean(item.badge) && (
                    <span className="relative z-10 rounded-full bg-red-500 px-2 py-0.5 text-[10px] font-black text-white">
                      {item.badge! > 99 ? '99+' : item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <div className="relative z-10 mt-5 flex-1 overflow-y-auto px-4 pb-5 custom-scrollbar">
          <div className="mb-3 flex items-center justify-between px-2">
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.28em] text-muted-foreground">
                {t.smart_sessions}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {language === 'es'
                  ? `${orderedSessions.length} activas`
                  : `${orderedSessions.length} active`}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            {orderedSessions.map((session) => {
              const isCurrent = activeView === 'chat' && currentSessionId === session.id;
              return (
                <div
                  key={session.id}
                  className={`group relative rounded-[1.25rem] border px-4 py-3 transition-all ${
                    isCurrent
                      ? 'border-primary/30 bg-primary/[0.07] shadow-[0_16px_36px_-26px_rgba(0,194,255,0.6)]'
                      : 'border-transparent bg-transparent hover:border-border/50 hover:bg-background/80'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => {
                      onSelectSession(session.id);
                      onSelectView('chat');
                    }}
                    className="flex w-full items-start gap-3 text-left"
                  >
                    <div className={`mt-0.5 h-2.5 w-2.5 rounded-full ${isCurrent ? 'bg-primary' : 'bg-muted-foreground/30'}`} />
                    <div className="min-w-0 flex-1">
                      <p className={`truncate text-sm font-semibold tracking-tight ${isCurrent ? 'text-foreground' : 'text-foreground/80'}`}>
                        {session.title}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {session.messages.length} {language === 'es' ? 'mensajes' : 'messages'} · {formatSessionTime(session.updatedAt)}
                      </p>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      setSessionToDelete(session.id);
                    }}
                    className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-xl text-muted-foreground opacity-0 transition-all hover:bg-red-500/10 hover:text-red-500 group-hover:opacity-100"
                    aria-label={language === 'es' ? 'Eliminar sesión' : 'Delete session'}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div className="relative z-10 border-t border-border/50 px-4 py-4">
          <div className="space-y-2">
            <button
              type="button"
              onClick={toggleDarkMode}
              className="flex w-full items-center gap-3 rounded-[1.15rem] border border-border/50 bg-background/70 px-4 py-3 text-left transition-all hover:border-primary/25 hover:bg-background"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-muted/60 text-foreground">
                {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold tracking-tight">{themeLabel}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {isDarkMode ? t.interface_light : t.interface_dark}
                </p>
              </div>
            </button>

            <button
              type="button"
              onClick={onOpenSettings}
              className="flex w-full items-center gap-3 rounded-[1.15rem] border border-border/50 bg-background/70 px-4 py-3 text-left transition-all hover:border-primary/25 hover:bg-background"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-muted/60 text-foreground">
                <Settings size={18} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold tracking-tight">{t.configuration}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {language === 'es' ? 'Tema, idioma y paneles' : 'Theme, language, and panels'}
                </p>
              </div>
            </button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {sessionToDelete && (
          <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/45 px-6 backdrop-blur-md">
            <motion.button
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0"
              onClick={() => setSessionToDelete(null)}
              aria-label={language === 'es' ? 'Cerrar diálogo' : 'Close dialog'}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.94, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: 20 }}
              transition={springTransition}
              className="relative w-full max-w-md rounded-[2.2rem] border border-border bg-background px-8 py-8 shadow-2xl"
            >
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-[1.8rem] bg-red-500/10 text-red-500">
                <AlertTriangle size={28} />
              </div>
              <h3 className="mt-6 text-center text-2xl font-black tracking-tight">
                {language === 'es' ? 'Eliminar sesión' : 'Delete session'}
              </h3>
              <p className="mt-3 text-center text-sm leading-7 text-muted-foreground">
                {language === 'es'
                  ? 'Se borrará esta conversación del historial local. No se eliminarán los datos del runtime ni del modelo.'
                  : 'This will remove the conversation from local history. Runtime and model data will not be deleted.'}
              </p>

              <div className="mt-8 flex flex-col gap-3">
                <button
                  type="button"
                  onClick={handleDeleteConfirm}
                  className="rounded-[1.2rem] bg-red-500 px-5 py-3 text-sm font-black text-white transition-all hover:bg-red-600"
                >
                  {language === 'es' ? 'Eliminar sesión' : 'Delete session'}
                </button>
                <button
                  type="button"
                  onClick={() => setSessionToDelete(null)}
                  className="rounded-[1.2rem] border border-border px-5 py-3 text-sm font-semibold text-foreground transition-all hover:bg-muted/50"
                >
                  {language === 'es' ? 'Cancelar' : 'Cancel'}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
};

export default Sidebar;

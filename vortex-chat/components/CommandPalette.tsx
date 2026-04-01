import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Command,
  Download,
  Eraser,
  HelpCircle,
  Layout,
  MessageSquare,
  Moon,
  Plus,
  Search,
  Settings,
  Sparkles,
  Sun,
  Type,
  X,
  ChevronRight,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { ChatSession, FontSize, Language } from '../types';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onClearHistory: () => void;
  onExportChat: () => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  onOpenSettings: () => void;
  onOpenHelp: () => void;
  categoryOrder: string[];
  onSetFontSize: (size: FontSize) => void;
  language: Language;
}

type PaletteItem = {
  id: string;
  title: string;
  icon: React.ReactNode;
  action: () => void;
  category: string;
  disabled?: boolean;
};

const text = {
  es: {
    quick: 'Acciones Rápidas',
    preferences: 'Preferencias',
    interface: 'Interfaz',
    data: 'Datos',
    recent: 'Chats Recientes',
    system: 'Sistema',
    placeholder: 'Busca un comando o una sesión...',
    empty: 'No se encontraron coincidencias',
    navigate: 'Navegar',
    run: 'Ejecutar',
    footer: 'Consola local',
  },
  en: {
    quick: 'Quick Actions',
    preferences: 'Preferences',
    interface: 'Interface',
    data: 'Data',
    recent: 'Recent Chats',
    system: 'System',
    placeholder: 'Search for a command or session...',
    empty: 'No matches found',
    navigate: 'Navigate',
    run: 'Run',
    footer: 'Local console',
  },
};

const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onClearHistory,
  onExportChat,
  isDarkMode,
  toggleDarkMode,
  isSidebarOpen,
  onToggleSidebar,
  onOpenSettings,
  onOpenHelp,
  categoryOrder,
  onSetFontSize,
  language,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const copy = text[language];
  const themeToggleTitle = language === 'es'
    ? (isDarkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro')
    : (isDarkMode ? 'Switch to light mode' : 'Switch to dark mode');

  const categories = {
    quick: categoryOrder.includes(copy.quick) ? copy.quick : text.es.quick,
    preferences: categoryOrder.includes(copy.preferences) ? copy.preferences : text.es.preferences,
    interface: categoryOrder.includes(copy.interface) ? copy.interface : text.es.interface,
    data: categoryOrder.includes(copy.data) ? copy.data : text.es.data,
    recent: categoryOrder.includes(copy.recent) ? copy.recent : text.es.recent,
    system: categoryOrder.includes(copy.system) ? copy.system : text.es.system,
  };

  const actions = useMemo<PaletteItem[]>(() => [
    { id: 'new-chat', title: language === 'es' ? 'Nueva conversación' : 'New conversation', icon: <Plus size={18} />, action: onNewChat, category: categories.quick },
    { id: 'toggle-dark', title: themeToggleTitle, icon: isDarkMode ? <Sun size={18} /> : <Moon size={18} />, action: toggleDarkMode, category: categories.preferences },
    { id: 'toggle-sidebar', title: language === 'es' ? (isSidebarOpen ? 'Ocultar lateral' : 'Mostrar lateral') : (isSidebarOpen ? 'Hide sidebar' : 'Show sidebar'), icon: <Layout size={18} />, action: onToggleSidebar, category: categories.interface },
    { id: 'font-small', title: language === 'es' ? 'Fuente: pequeña' : 'Font: small', icon: <Type size={14} />, action: () => onSetFontSize('small'), category: categories.interface },
    { id: 'font-medium', title: language === 'es' ? 'Fuente: normal' : 'Font: medium', icon: <Type size={18} />, action: () => onSetFontSize('medium'), category: categories.interface },
    { id: 'font-large', title: language === 'es' ? 'Fuente: grande' : 'Font: large', icon: <Type size={22} />, action: () => onSetFontSize('large'), category: categories.interface },
    { id: 'export-chat', title: language === 'es' ? 'Exportar chat (Markdown)' : 'Export chat (Markdown)', icon: <Download size={18} />, action: onExportChat, category: categories.data, disabled: !currentSessionId },
    { id: 'clear-history', title: language === 'es' ? 'Borrar historial' : 'Clear history', icon: <Eraser size={18} />, action: onClearHistory, category: categories.data, disabled: sessions.length === 0 },
    { id: 'settings', title: language === 'es' ? 'Configuración avanzada' : 'Advanced settings', icon: <Settings size={18} />, action: onOpenSettings, category: categories.system },
    { id: 'help', title: language === 'es' ? 'Ayuda y atajos' : 'Help and shortcuts', icon: <HelpCircle size={18} />, action: onOpenHelp, category: categories.system },
  ], [categories.data, categories.interface, categories.preferences, categories.quick, categories.system, currentSessionId, isDarkMode, isSidebarOpen, language, onClearHistory, onExportChat, onNewChat, onOpenHelp, onOpenSettings, onSetFontSize, onToggleSidebar, sessions.length, themeToggleTitle, toggleDarkMode]);

  const filteredItems = useMemo(() => {
    const search = query.toLowerCase();
    const matchingActions = actions.filter((item) => !item.disabled && item.title.toLowerCase().includes(search));
    const matchingSessions = sessions
      .filter((session) => session.title.toLowerCase().includes(search))
      .map<PaletteItem>((session) => ({
        id: session.id,
        title: session.title,
        icon: <MessageSquare size={18} />,
        action: () => onSelectSession(session.id),
        category: categories.recent,
      }));

    return [...matchingActions, ...matchingSessions].sort((left, right) => categoryOrder.indexOf(left.category) - categoryOrder.indexOf(right.category));
  }, [actions, categories.recent, categoryOrder, onSelectSession, query, sessions]);

  const orderedCategories = useMemo(() => {
    const activeCategories = new Set(filteredItems.map((item) => item.category));
    return categoryOrder.filter((category) => activeCategories.has(category));
  }, [categoryOrder, filteredItems]);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        setSelectedIndex((current) => (current + 1) % Math.max(1, filteredItems.length));
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        setSelectedIndex((current) => (current - 1 + filteredItems.length) % Math.max(1, filteredItems.length));
      } else if (event.key === 'Enter') {
        event.preventDefault();
        if (filteredItems[selectedIndex]) {
          filteredItems[selectedIndex].action();
          onClose();
        }
      } else if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [filteredItems, isOpen, onClose, selectedIndex]);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-start justify-center px-4 pt-[15vh]">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/28 backdrop-blur-md"
            onClick={onClose}
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 40, filter: 'blur(10px)' }}
            animate={{ opacity: 1, scale: 1, y: 0, filter: 'blur(0px)' }}
            exit={{ opacity: 0, scale: 0.95, y: 20, filter: 'blur(10px)' }}
            className="surface-panel relative flex w-full max-w-2xl flex-col overflow-hidden rounded-[1.6rem]"
          >
            <div className="relative flex h-20 items-center border-b border-border/60 bg-muted/20 px-8">
              <Search size={22} className="mr-6 shrink-0 text-primary opacity-60" />
              <input
                ref={inputRef}
                type="text"
                className="flex-1 bg-transparent text-xl font-bold tracking-tight text-foreground outline-none placeholder:text-muted-foreground/45 dark:placeholder:text-zinc-500"
                placeholder={copy.placeholder}
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setSelectedIndex(0);
                }}
              />
              <button
                type="button"
                onClick={onClose}
                aria-label={language === 'es' ? 'Cerrar' : 'Close'}
                className="rounded-full p-3 transition-all hover:bg-muted"
              >
                <X size={20} className="text-muted-foreground" />
              </button>
            </div>

            <div className="max-h-[50vh] overflow-y-auto p-5 custom-scrollbar">
              <AnimatePresence mode="popLayout">
                {filteredItems.length === 0 ? (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center gap-4 py-16 text-muted-foreground">
                    <Sparkles size={32} className="opacity-20" />
                    <p className="text-xs font-black uppercase tracking-[0.3em] opacity-40">{copy.empty}</p>
                  </motion.div>
                ) : (
                  <div className="space-y-8 py-4">
                    {orderedCategories.map((category) => (
                      <div key={category} className="space-y-3">
                        <h3 className="mb-4 px-6 text-[10px] font-black uppercase tracking-[0.4em] text-primary/60">
                          {category}
                        </h3>
                        <div className="space-y-1.5">
                          {filteredItems
                            .map((item, index) => ({ item, index }))
                            .filter(({ item }) => item.category === category)
                            .map(({ item, index }) => {
                              const isSelected = index === selectedIndex;
                              return (
                                <motion.button
                                  key={item.id}
                                  layout
                                  onMouseEnter={() => setSelectedIndex(index)}
                                  onClick={() => {
                                    item.action();
                                    onClose();
                                  }}
                                  className={`relative flex w-full items-center gap-5 overflow-hidden rounded-[1rem] px-5 py-4 text-left transition-all ${
                                    isSelected
                                      ? 'z-10 bg-primary text-primary-foreground shadow-sm'
                                      : 'text-foreground/70 hover:bg-muted/60 dark:text-zinc-400'
                                  }`}
                                >
                                  {isSelected && (
                                    <motion.div
                                      layoutId="command-highlight"
                                      className="absolute inset-0 -z-10 bg-primary"
                                      transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                                    />
                                  )}
                                  <div className={`shrink-0 transition-transform duration-500 ${isSelected ? 'scale-110 rotate-3' : 'opacity-40'}`}>
                                    {item.icon}
                                  </div>
                                  <div className="flex min-w-0 flex-1 items-center justify-between">
                                    <p className={`text-[14.5px] font-bold tracking-tight ${isSelected ? 'text-white' : ''}`}>
                                      {item.title}
                                    </p>
                                    {isSelected && <ChevronRight size={16} />}
                                  </div>
                                </motion.button>
                              );
                            })}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </AnimatePresence>
            </div>

            <div className="flex items-center justify-between border-t border-border/40 bg-muted/10 px-10 py-5 text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/40 dark:bg-zinc-900/60">
              <div className="flex gap-6">
                <span className="flex items-center gap-2">
                  <kbd className="rounded-lg border border-border bg-background px-2 py-0.5 dark:bg-zinc-800">↑↓</kbd>
                  {copy.navigate}
                </span>
                <span className="flex items-center gap-2">
                  <kbd className="rounded-lg border border-border bg-background px-2 py-0.5 dark:bg-zinc-800">↵</kbd>
                  {copy.run}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Command size={14} />
                <span>{copy.footer}</span>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default CommandPalette;

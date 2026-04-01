import React, { useMemo, useState } from 'react';
import {
  AtSign,
  Check,
  CircleUserRound,
  Globe,
  GripVertical,
  Mail,
  Moon,
  Palette,
  Plus,
  SlidersHorizontal,
  Sun,
  Type,
  UserRound,
  X,
} from 'lucide-react';
import { Reorder, AnimatePresence, motion } from 'framer-motion';
import { FontSize, Language, LocalAccount, UserSettings } from '../types';
import { translations } from '../translations';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: SettingsTab;
  settings: UserSettings;
  onUpdateSettings: (settings: UserSettings) => void;
  accounts: LocalAccount[];
  currentAccountId: string | null;
  onSelectAccount: (id: string) => void;
  onCreateAccount: (draft: { name: string; email: string; handle?: string }) => void;
}

export type SettingsTab = 'general' | 'appearance' | 'profiles';

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  initialTab = 'general',
  settings,
  onUpdateSettings,
  accounts,
  currentAccountId,
  onSelectAccount,
  onCreateAccount,
}) => {
  const t = translations[settings.language];
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);
  const [draftName, setDraftName] = useState('');
  const [draftEmail, setDraftEmail] = useState('');
  const [draftHandle, setDraftHandle] = useState('');

  const currentAccount = useMemo(
    () => accounts.find((account) => account.id === currentAccountId) || accounts[0] || null,
    [accounts, currentAccountId],
  );

  const tabs = [
    {
      id: 'general' as const,
      label: settings.language === 'es' ? 'General' : 'General',
      icon: <SlidersHorizontal size={16} />,
    },
    {
      id: 'appearance' as const,
      label: settings.language === 'es' ? 'Apariencia' : 'Appearance',
      icon: <Palette size={16} />,
    },
    {
      id: 'profiles' as const,
      label: settings.language === 'es' ? 'Perfiles' : 'Profiles',
      icon: <UserRound size={16} />,
    },
  ];

  const handleReorder = (newOrder: string[]) => {
    onUpdateSettings({ ...settings, categoryOrder: newOrder });
  };

  const handleCodeThemeChange = (theme: 'dark' | 'light' | 'match-app') => {
    onUpdateSettings({ ...settings, codeTheme: theme });
  };

  const handleFontSizeChange = (size: FontSize) => {
    onUpdateSettings({ ...settings, fontSize: size });
  };

  const handleLanguageChange = (lang: Language) => {
    onUpdateSettings({ ...settings, language: lang });
  };

  const handleCreateAccount = () => {
    const name = draftName.trim();
    const email = draftEmail.trim();
    const handle = draftHandle.trim();
    if (!name || !email) return;
    onCreateAccount({ name, email, handle });
    setDraftName('');
    setDraftEmail('');
    setDraftHandle('');
  };

  React.useEffect(() => {
    if (isOpen) setActiveTab(initialTab);
  }, [initialTab, isOpen]);

  const OptionCard = ({
    active,
    title,
    description,
    icon,
    onClick,
  }: {
    active: boolean;
    title: string;
    description: string;
    icon: React.ReactNode;
    onClick: () => void;
  }) => (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-2xl border p-4 text-left transition-all ${
        active
          ? 'border-primary/50 bg-primary/[0.12] shadow-[0_16px_38px_-28px_rgba(0,174,255,0.9)]'
          : 'border-border/70 bg-background hover:border-primary/25 hover:bg-muted/20'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${active ? 'bg-primary text-primary-foreground' : 'bg-muted/40 text-foreground'}`}>
          {icon}
        </div>
        {active && (
          <span className="rounded-full border border-primary/30 bg-primary/15 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-primary">
            <Check size={11} />
          </span>
        )}
      </div>
      <p className="mt-4 text-sm font-bold tracking-tight text-foreground">{title}</p>
      <p className="mt-1 text-xs leading-6 text-muted-foreground">{description}</p>
    </button>
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/55 backdrop-blur-md"
            onClick={onClose}
          />

          <motion.div
            initial={{ scale: 0.98, opacity: 0, y: 18 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.98, opacity: 0, y: 18 }}
            transition={{ type: 'spring', damping: 28, stiffness: 260 }}
            className="relative flex max-h-[84vh] w-full max-w-[880px] overflow-hidden rounded-[1.75rem] border border-border/70 bg-background shadow-[0_36px_110px_-52px_rgba(0,0,0,0.7)]"
          >
            <aside className="flex w-[230px] shrink-0 flex-col border-r border-border/60 bg-muted/10 p-3">
              <div className="flex items-center justify-between px-3 py-3">
                <div>
                  <p className="text-sm font-bold tracking-tight text-foreground">{t.settings_title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {settings.language === 'es' ? 'Workspace local' : 'Local workspace'}
                  </p>
                </div>
                <button
                  onClick={onClose}
                  aria-label={t.settings_close}
                  className="rounded-full p-2 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
                >
                  <X size={17} />
                </button>
              </div>

              <nav className="mt-3 space-y-1">
                {tabs.map((tab) => {
                  const active = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-all ${
                        active
                          ? 'bg-primary/[0.12] text-foreground'
                          : 'text-muted-foreground hover:bg-muted/30 hover:text-foreground'
                      }`}
                    >
                      <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${active ? 'bg-primary text-primary-foreground' : 'bg-background border border-border/50'}`}>
                        {tab.icon}
                      </span>
                      <span className="text-sm font-semibold tracking-tight">{tab.label}</span>
                    </button>
                  );
                })}
              </nav>

              <div className="mt-auto rounded-[1rem] border border-border/60 bg-background px-3 py-3">
                <div className="flex items-center gap-3">
                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-full text-sm font-black text-white"
                    style={{ background: `linear-gradient(135deg, hsl(${currentAccount?.avatarHue ?? 198} 100% 58%), hsl(${(currentAccount?.avatarHue ?? 198) + 18} 100% 46%))` }}
                  >
                    {(currentAccount?.name || 'V').slice(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-foreground">{currentAccount?.name || 'Vortex Local'}</p>
                    <p className="truncate text-xs text-muted-foreground">{currentAccount?.email || 'local@vortex.dev'}</p>
                  </div>
                </div>
              </div>
            </aside>

            <section className="flex-1 overflow-y-auto p-7 custom-scrollbar">
              {activeTab === 'general' && (
                <div className="max-w-[640px] space-y-6">
                  <div>
                    <h3 className="text-lg font-bold tracking-tight text-foreground">
                      {settings.language === 'es' ? 'General' : 'General'}
                    </h3>
                    <p className="mt-1 text-sm leading-6 text-muted-foreground">
                      {settings.language === 'es'
                        ? 'Configura idioma y el orden de navegación del workspace.'
                        : 'Configure language and the workspace navigation order.'}
                    </p>
                  </div>

                  <div className="space-y-5">
                    <div>
                      <p className="mb-3 text-[11px] font-black uppercase tracking-[0.14em] text-muted-foreground">{t.language}</p>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <OptionCard
                          active={settings.language === 'es'}
                          title={t.lang_es}
                          description={settings.language === 'es' ? 'Interfaz principal y sistema en español.' : 'Spanish workspace language.'}
                          icon={<Globe size={18} />}
                          onClick={() => handleLanguageChange('es')}
                        />
                        <OptionCard
                          active={settings.language === 'en'}
                          title={t.lang_en}
                          description={settings.language === 'es' ? 'Cambia todo el workspace a inglés.' : 'Switch the workspace to English.'}
                          icon={<Globe size={18} />}
                          onClick={() => handleLanguageChange('en')}
                        />
                      </div>
                    </div>

                    <div className="rounded-[1.3rem] border border-border/60 bg-muted/10 p-4">
                      <p className="mb-2 text-[11px] font-black uppercase tracking-[0.14em] text-muted-foreground">{t.settings_section_order}</p>
                      <p className="mb-4 text-sm leading-6 text-muted-foreground">{t.settings_section_desc}</p>
                      <Reorder.Group axis="y" values={settings.categoryOrder} onReorder={handleReorder} className="space-y-2">
                        {settings.categoryOrder.map((category) => (
                          <Reorder.Item key={category} value={category} className="relative">
                            <motion.div
                              className="flex items-center gap-3 rounded-xl border border-border/60 bg-background px-4 py-3 shadow-sm"
                              whileDrag={{
                                scale: 1.02,
                                boxShadow: '0 18px 40px -26px rgba(0, 174, 255, 0.5)',
                                zIndex: 1,
                              }}
                            >
                              <GripVertical size={14} className="shrink-0 text-muted-foreground/50" />
                              <span className="flex-1 text-sm font-semibold text-foreground">{category}</span>
                            </motion.div>
                          </Reorder.Item>
                        ))}
                      </Reorder.Group>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'appearance' && (
                <div className="max-w-[640px] space-y-6">
                  <div>
                    <h3 className="text-lg font-bold tracking-tight text-foreground">
                      {settings.language === 'es' ? 'Apariencia' : 'Appearance'}
                    </h3>
                    <p className="mt-1 text-sm leading-6 text-muted-foreground">
                      {settings.language === 'es'
                        ? 'La base visual sigue una estética ChatGPT con la paleta azul eléctrica de Vortex.'
                        : 'The visual system follows a ChatGPT-like aesthetic with the Vortex electric blue palette.'}
                    </p>
                  </div>

                  <div className="space-y-5">
                    <div className="rounded-[1.3rem] border border-border/60 bg-muted/10 p-5">
                      <p className="mb-3 text-[11px] font-black uppercase tracking-[0.14em] text-muted-foreground">{t.settings_code_theme}</p>
                      <div className="flex flex-wrap gap-2">
                        {[
                          { key: 'light', label: settings.language === 'es' ? 'Claro' : 'Light', icon: <Sun size={16} /> },
                          { key: 'dark', label: settings.language === 'es' ? 'Oscuro' : 'Dark', icon: <Moon size={16} /> },
                          { key: 'match-app', label: settings.language === 'es' ? 'Sistema' : 'Match app', icon: <Palette size={16} /> },
                        ].map((option) => {
                          const active = settings.codeTheme === option.key;
                          return (
                            <button
                              key={option.key}
                              type="button"
                              onClick={() => handleCodeThemeChange(option.key as 'light' | 'dark' | 'match-app')}
                              className={`inline-flex items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-semibold transition-all ${
                                active
                                  ? 'border-primary/40 bg-primary/[0.10] text-foreground'
                                  : 'border-border/70 bg-background text-muted-foreground hover:border-primary/25 hover:text-foreground'
                              }`}
                            >
                              {option.icon}
                              {option.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="rounded-[1.3rem] border border-border/60 bg-muted/10 p-5">
                      <p className="mb-3 text-[11px] font-black uppercase tracking-[0.14em] text-muted-foreground">{t.settings_font_size}</p>
                      <div className="flex flex-wrap gap-2">
                        {[
                          { key: 'small', label: t.font_small },
                          { key: 'medium', label: t.font_medium },
                          { key: 'large', label: t.font_large },
                        ].map((option) => {
                          const active = settings.fontSize === option.key;
                          return (
                            <button
                              key={option.key}
                              type="button"
                              onClick={() => handleFontSizeChange(option.key as FontSize)}
                              className={`inline-flex items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-semibold transition-all ${
                                active
                                  ? 'border-primary/40 bg-primary/[0.10] text-foreground'
                                  : 'border-border/70 bg-background text-muted-foreground hover:border-primary/25 hover:text-foreground'
                              }`}
                            >
                              <Type size={16} />
                              {option.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="rounded-2xl border border-primary/20 bg-[linear-gradient(135deg,rgba(0,174,255,0.16),rgba(0,119,255,0.08))] p-5">
                      <p className="text-[11px] font-black uppercase tracking-[0.14em] text-primary">
                        {settings.language === 'es' ? 'Paleta activa' : 'Active palette'}
                      </p>
                      <div className="mt-4 flex gap-2">
                        {['#00AEFF', '#0077FF', '#07111F', '#F7FBFF', '#171717'].map((color) => (
                          <span key={color} className="h-10 w-10 rounded-xl border border-white/15 shadow-sm" style={{ backgroundColor: color }} />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'profiles' && (
                <div className="max-w-[660px] space-y-6">
                  <div>
                    <h3 className="text-lg font-bold tracking-tight text-foreground">
                      {settings.language === 'es' ? 'Perfiles' : 'Profiles'}
                    </h3>
                    <p className="mt-1 text-sm leading-6 text-muted-foreground">
                      {settings.language === 'es'
                        ? 'Perfiles locales para separar sesiones y preferencias visuales.'
                        : 'Local profiles to separate sessions and visual preferences.'}
                    </p>
                  </div>

                  <div className="rounded-[1.3rem] border border-border/60 bg-muted/10 p-4">
                    <p className="mb-3 text-[11px] font-black uppercase tracking-[0.14em] text-muted-foreground">
                      {settings.language === 'es' ? 'Perfil activo' : 'Current profile'}
                    </p>
                    <div className="flex items-center gap-4 rounded-[1rem] border border-primary/20 bg-primary/[0.08] px-4 py-4">
                      <div
                        className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full text-sm font-black text-white"
                        style={{ background: `linear-gradient(135deg, hsl(${currentAccount?.avatarHue ?? 198} 100% 58%), hsl(${(currentAccount?.avatarHue ?? 198) + 18} 100% 46%))` }}
                      >
                        {(currentAccount?.name || 'V').slice(0, 2).toUpperCase()}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold tracking-tight text-foreground">{currentAccount?.name || 'Vortex Local'}</p>
                        <p className="truncate text-xs text-muted-foreground">{currentAccount?.email || 'local@vortex.dev'}</p>
                      </div>
                      <span className="rounded-full bg-primary px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-primary-foreground">
                        {t.account_current}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {accounts.map((account) => {
                      const active = account.id === currentAccountId;
                      return (
                        <button
                          key={account.id}
                          type="button"
                          onClick={() => onSelectAccount(account.id)}
                          className={`flex w-full items-center gap-4 rounded-[1.1rem] border px-4 py-4 text-left transition-all ${
                            active
                              ? 'border-primary/40 bg-primary/[0.10]'
                              : 'border-border/60 bg-background hover:border-primary/20 hover:bg-muted/20'
                          }`}
                        >
                          <div
                            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-sm font-black text-white"
                            style={{ background: `linear-gradient(135deg, hsl(${account.avatarHue} 100% 58%), hsl(${account.avatarHue + 18} 100% 46%))` }}
                          >
                            {account.name.slice(0, 2).toUpperCase()}
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-semibold tracking-tight text-foreground">{account.name}</p>
                            <p className="truncate text-xs text-muted-foreground">{account.email}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs font-semibold text-muted-foreground">{account.handle}</p>
                            {active && (
                              <span className="mt-1 inline-flex rounded-full bg-primary px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-primary-foreground">
                                {t.account_current}
                              </span>
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>

                  <div className="rounded-[1.3rem] border border-border/60 bg-muted/10 p-5">
                    <p className="text-[11px] font-black uppercase tracking-[0.14em] text-primary">{t.account_create}</p>
                    <div className="mt-4 grid gap-3">
                      <label className="block">
                        <span className="mb-2 flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                          <CircleUserRound size={12} /> {t.account_name}
                        </span>
                        <input
                          value={draftName}
                          onChange={(event) => setDraftName(event.target.value)}
                          placeholder="Marc Vortex"
                          className="w-full rounded-xl border border-border/70 bg-background px-4 py-3 text-sm text-foreground outline-none transition-all focus:border-primary/40 focus:ring-4 focus:ring-primary/10"
                        />
                      </label>
                      <label className="block">
                        <span className="mb-2 flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                          <Mail size={12} /> {t.account_email}
                        </span>
                        <input
                          value={draftEmail}
                          onChange={(event) => setDraftEmail(event.target.value)}
                          placeholder="marc@vortex.local"
                          className="w-full rounded-xl border border-border/70 bg-background px-4 py-3 text-sm text-foreground outline-none transition-all focus:border-primary/40 focus:ring-4 focus:ring-primary/10"
                        />
                      </label>
                      <label className="block">
                        <span className="mb-2 flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                          <AtSign size={12} /> {t.account_handle}
                        </span>
                        <input
                          value={draftHandle}
                          onChange={(event) => setDraftHandle(event.target.value)}
                          placeholder="@marc"
                          className="w-full rounded-xl border border-border/70 bg-background px-4 py-3 text-sm text-foreground outline-none transition-all focus:border-primary/40 focus:ring-4 focus:ring-primary/10"
                        />
                      </label>
                      <button
                        type="button"
                        onClick={handleCreateAccount}
                        className="mt-2 flex w-fit items-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm font-bold text-primary-foreground shadow-[0_16px_36px_-24px_rgba(0,174,255,0.95)] transition-transform hover:scale-[1.01] active:scale-[0.99]"
                      >
                        <Plus size={16} />
                        {t.account_create}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </section>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default SettingsModal;

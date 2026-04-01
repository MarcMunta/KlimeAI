import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  Cpu,
  FlaskConical,
  Globe,
  HardDriveDownload,
  RefreshCw,
  TriangleAlert,
} from 'lucide-react';
import { ControlStatus, Language, OperationalStatus } from '../types';

interface TopBarStackStatusProps {
  status: OperationalStatus | null;
  controlStatus: ControlStatus | null;
  language: Language;
  onBootstrap?: () => Promise<unknown> | unknown;
  onModelInit?: () => Promise<unknown> | unknown;
  onRestartRuntime?: () => Promise<unknown> | unknown;
  onStartTraining?: () => Promise<unknown> | unknown;
  onOpenTraining?: () => void;
}

const TopBarStackStatus: React.FC<TopBarStackStatusProps> = ({
  status,
  controlStatus,
  language,
  onBootstrap,
  onModelInit,
  onRestartRuntime,
  onStartTraining,
  onOpenTraining,
}) => {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, []);

  const bootstrapRunning = Boolean(controlStatus?.bootstrap?.running);
  const runtimeReady = Boolean(controlStatus?.runtime?.runtime_ready || status?.engine_ready);
  const modelReady = Boolean(
    controlStatus?.model?.cached
    || status?.model_ready
    || controlStatus?.runtime?.runtime_ready
  );
  const ready = Boolean(status?.ok && runtimeReady && modelReady);
  const runtimeUrl = status?.engine_base_url || 'http://127.0.0.1:30000';
  const engineLabel = (status?.engine_kind || 'sglang').toUpperCase();
  const modelLabel = status?.active_model || controlStatus?.model?.model_id || 'Qwen/Qwen2.5-Coder-14B-Instruct-AWQ';
  const reason = status?.degraded_reason
    || controlStatus?.bootstrap?.message
    || controlStatus?.docker?.reason
    || (language === 'es' ? 'Stack local pendiente.' : 'Local stack pending.');

  const summary = useMemo(() => {
    if (ready) {
      return {
        label: language === 'es' ? 'Local listo' : 'Local ready',
        tone: 'ready',
        icon: <CheckCircle2 size={15} />,
      };
    }
    if (bootstrapRunning) {
      return {
        label: language === 'es' ? 'Arrancando' : 'Starting',
        tone: 'progress',
        icon: <Activity size={15} />,
      };
    }
    return {
      label: language === 'es' ? 'Revisar stack' : 'Check stack',
      tone: 'warning',
      icon: <TriangleAlert size={15} />,
    };
  }, [bootstrapRunning, language, ready]);

  const toneClasses =
    summary.tone === 'ready'
      ? 'border-emerald-500/25 bg-emerald-500/[0.08] text-emerald-500'
      : summary.tone === 'progress'
        ? 'border-primary/25 bg-primary/[0.08] text-primary'
        : 'border-amber-500/25 bg-amber-500/[0.10] text-amber-500';

  const items = [
    {
      key: 'runtime',
      icon: <Cpu size={14} />,
      label: language === 'es' ? 'Runtime' : 'Runtime',
      value: runtimeReady ? (language === 'es' ? 'Activo' : 'Live') : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: engineLabel,
    },
    {
      key: 'model',
      icon: <HardDriveDownload size={14} />,
      label: language === 'es' ? 'Modelo' : 'Model',
      value: modelReady ? (language === 'es' ? 'Listo' : 'Ready') : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: modelLabel,
    },
    {
      key: 'internet',
      icon: <Globe size={14} />,
      label: language === 'es' ? 'Internet' : 'Internet',
      value: status?.web_disabled ? (language === 'es' ? 'Por prompt' : 'Per prompt') : (language === 'es' ? 'Abierto' : 'Open'),
      caption: status?.web_disabled
        ? (language === 'es' ? 'Local por defecto' : 'Local by default')
        : (language === 'es' ? 'Revisar politica' : 'Review policy'),
    },
    {
      key: 'training',
      icon: <FlaskConical size={14} />,
      label: language === 'es' ? 'Entreno' : 'Training',
      value: status?.training_ready ? (language === 'es' ? 'Listo' : 'Ready') : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: bootstrapRunning
        ? String(controlStatus?.bootstrap?.stage || 'bootstrap')
        : (language === 'es' ? 'Control manual' : 'Manual control'),
    },
  ];

  const ActionButton = ({
    label,
    onClick,
    primary = false,
  }: {
    label: string;
    onClick?: () => Promise<unknown> | unknown;
    primary?: boolean;
  }) => (
    <button
      type="button"
      onClick={() => {
        void onClick?.();
      }}
      className={`rounded-full border px-3 py-2 text-[10px] font-black uppercase tracking-[0.2em] transition-all ${
        primary
          ? 'border-primary/30 bg-primary/[0.10] text-primary hover:bg-primary/[0.16]'
          : 'border-border/60 bg-background/75 text-foreground/75 hover:border-primary/25 hover:text-foreground'
      }`}
    >
      {label}
    </button>
  );

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className={`flex items-center gap-3 rounded-2xl border px-4 py-3 shadow-sm transition-all hover:-translate-y-0.5 ${toneClasses}`}
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-background/80 text-current">
          {summary.icon}
        </span>
        <div className="hidden min-w-0 text-left md:block">
          <p className="text-[10px] font-black uppercase tracking-[0.24em]">
            {summary.label}
          </p>
          <p className="mt-1 max-w-[200px] truncate text-xs text-foreground/70 dark:text-white/70">
            {engineLabel} - {runtimeReady ? runtimeUrl : reason}
          </p>
        </div>
        <ChevronDown
          size={14}
          className={`transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div className="absolute right-0 top-[calc(100%+12px)] z-50 w-[360px] rounded-[1.8rem] border border-border/60 bg-background/95 p-5 shadow-[0_32px_90px_-42px_rgba(0,0,0,0.55)] backdrop-blur-3xl">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.26em] text-primary">
                {language === 'es' ? 'Stack local' : 'Local stack'}
              </p>
              <h3 className="mt-2 text-lg font-black tracking-tight text-foreground">
                {ready
                  ? (language === 'es' ? 'Runtime operativo.' : 'Runtime operational.')
                  : (language === 'es' ? 'Revisar arranque local.' : 'Check local startup.')}
              </h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                {ready
                  ? (language === 'es' ? 'Vortex esta listo para responder.' : 'Vortex is ready to answer.')
                  : reason}
              </p>
            </div>
            <span className={`rounded-full px-3 py-1 text-[10px] font-black uppercase tracking-[0.2em] ${toneClasses}`}>
              {summary.label}
            </span>
          </div>

          <div className="mt-4 grid gap-3">
            {items.map((item) => (
              <div key={item.key} className="rounded-[1.2rem] border border-border/50 bg-muted/20 px-4 py-3">
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground">
                  {item.icon}
                  <span>{item.label}</span>
                </div>
                <p className="mt-2 text-sm font-black tracking-tight text-foreground">{item.value}</p>
                <p className="mt-1 break-words text-xs text-muted-foreground">{item.caption}</p>
              </div>
            ))}
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <ActionButton
              label={language === 'es' ? 'Modelo' : 'Model'}
              onClick={onModelInit}
              primary={!modelReady}
            />
            <ActionButton
              label={language === 'es' ? 'Runtime' : 'Runtime'}
              onClick={onBootstrap}
              primary={!runtimeReady}
            />
            <ActionButton
              label={language === 'es' ? 'Reiniciar' : 'Restart'}
              onClick={onRestartRuntime}
            />
            <ActionButton
              label={language === 'es' ? 'Panel train' : 'Train panel'}
              onClick={onOpenTraining || onStartTraining}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default TopBarStackStatus;

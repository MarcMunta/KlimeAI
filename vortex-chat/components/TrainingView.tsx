import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Bot,
  BrainCircuit,
  Clock3,
  FileCode2,
  FlaskConical,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Workflow,
} from 'lucide-react';
import {
  AutonomyEvent,
  AutonomyStatus,
  ChatSession,
  ControlStatus,
  Language,
  LogEntry,
  Role,
  TrainingRunSummary,
  TrainingStreamPayload,
} from '../types';
import { controlService } from '../services/controlService';

interface TrainingViewProps {
  sessions: ChatSession[];
  language: Language;
  controlStatus: ControlStatus | null;
  onAddLog: (level: LogEntry['level'], message: string) => void;
  onStartTraining: (mode: 'quick' | 'full') => Promise<unknown> | unknown;
  onStartAutonomy: () => Promise<unknown> | unknown;
  onStopAutonomy: () => Promise<unknown> | unknown;
  onConfigureAutonomy: (config: {
    enabled?: boolean;
    reflection_enabled?: boolean;
    training_enabled?: boolean;
    autoedit_enabled?: boolean;
  }) => Promise<unknown> | unknown;
}

const Panel: React.FC<{ title: string; eyebrow?: string; children: React.ReactNode; className?: string }> = ({
  title,
  eyebrow,
  children,
  className = '',
}) => (
  <section className={`surface-panel rounded-[1.5rem] p-6 ${className}`}>
    {(eyebrow || title) && (
      <header className="mb-5">
        {eyebrow && <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{eyebrow}</p>}
        <h3 className="mt-2 text-2xl font-extrabold tracking-tight">{title}</h3>
      </header>
    )}
    {children}
  </section>
);

const formatTime = (ts: number | null | undefined, language: Language): string => {
  if (!ts) return language === 'es' ? 'Sin actividad' : 'No activity';
  return new Date(ts * 1000).toLocaleString(language === 'es' ? 'es-ES' : 'en-US', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: 'short',
  });
};

const TrainingView: React.FC<TrainingViewProps> = ({
  sessions,
  language,
  controlStatus,
  onAddLog,
  onStartTraining,
  onStartAutonomy,
  onStopAutonomy,
  onConfigureAutonomy,
}) => {
  const [trainingStream, setTrainingStream] = useState<TrainingStreamPayload | null>(null);
  const [autonomyStream, setAutonomyStream] = useState<{ status: AutonomyStatus; events: AutonomyEvent[] } | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  useEffect(() => {
    const closeTraining = controlService.subscribeTrainingStream((payload) => setTrainingStream(payload));
    const closeAutonomy = controlService.subscribeAutonomyStream((payload) => {
      setAutonomyStream({ status: payload.status, events: payload.events || [] });
    });
    return () => {
      closeTraining();
      closeAutonomy();
    };
  }, []);

  const autonomy = autonomyStream?.status || controlStatus?.autonomy || null;
  const runs = (trainingStream?.runs && trainingStream.runs.length > 0)
    ? trainingStream.runs
    : (controlStatus?.runs || []);
  const activeRunId = trainingStream?.active_run_id || controlStatus?.active_run_id || null;
  const activeRun = runs.find((run) => run.run_id === activeRunId) || runs[0] || null;
  const timeline = autonomyStream?.events || autonomy?.latest_events || [];

  const promotedResponses = useMemo(() => {
    const items: Array<{ id: string; content: string; ts: number; sessionId: string; messageId: string }> = [];
    for (const session of sessions) {
      for (const message of session.messages) {
        if (message.role === Role.AI && message.trainingEvent) {
          items.push({
            id: `${session.id}:${message.id}`,
            content: message.content.slice(0, 220).trim(),
            ts: message.timestamp,
            sessionId: session.id,
            messageId: message.id,
          });
        }
      }
    }
    return items.sort((left, right) => right.ts - left.ts).slice(0, 8);
  }, [sessions]);

  const metrics = [
    {
      label: language === 'es' ? 'Estado' : 'State',
      value: autonomy?.state || (language === 'es' ? 'Esperando' : 'Waiting'),
      caption: autonomy?.enabled
        ? (language === 'es' ? 'Autonomia activa al arrancar' : 'Autonomy active on boot')
        : (language === 'es' ? 'Bucle en pausa' : 'Loop paused'),
      icon: <Activity size={18} />,
    },
    {
      label: language === 'es' ? 'Ciclo' : 'Cycle',
      value: autonomy?.current_cycle || 'idle',
      caption: formatTime(autonomy?.last_reflection_at, language),
      icon: <Workflow size={18} />,
    },
    {
      label: language === 'es' ? 'Run activo' : 'Active run',
      value: activeRun?.mode === 'full'
        ? (language === 'es' ? 'Entreno completo' : 'Full training')
        : activeRun?.mode === 'quick'
          ? (language === 'es' ? 'Aprendizaje rapido' : 'Quick learning')
          : (language === 'es' ? 'Sin run' : 'No run'),
      caption: activeRun?.status || (language === 'es' ? 'Sin ejecucion' : 'No execution'),
      icon: <FlaskConical size={18} />,
    },
    {
      label: language === 'es' ? 'Rollback' : 'Rollback',
      value: autonomy?.last_rollback?.status || (language === 'es' ? 'Sin rollback' : 'No rollback'),
      caption: autonomy?.last_rollback?.target || (language === 'es' ? 'Repo versionado' : 'Versioned repo'),
      icon: <RotateCcw size={18} />,
    },
  ];

  const runAction = async (
    label: string,
    action: () => Promise<unknown> | unknown,
    successMessage: string,
  ) => {
    setBusyAction(label);
    try {
      await action();
      onAddLog('LEARN', successMessage);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      onAddLog('SYSTEM', detail);
    } finally {
      setBusyAction(null);
    }
  };

  const renderRunCard = (run: TrainingRunSummary) => (
    <div key={run.run_id} className="rounded-[1.25rem] border border-border/50 bg-muted/15 px-4 py-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-black tracking-tight">
            {run.mode === 'full'
              ? (language === 'es' ? 'Entreno completo' : 'Full training')
              : (language === 'es' ? 'Aprendizaje rapido' : 'Quick learning')}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">{run.run_id}</p>
        </div>
        <span className="rounded-full border border-primary/20 bg-primary/[0.10] px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] text-primary">
          {run.status}
        </span>
      </div>
      <div className="mt-4 grid gap-2 text-xs text-muted-foreground">
        <p>{run.adapter_dir || (language === 'es' ? 'Adapter pendiente' : 'Adapter pending')}</p>
        <p>{run.dataset_hash || (language === 'es' ? 'Dataset sin hash' : 'Dataset without hash')}</p>
        <p>{run.promotion?.decision || (language === 'es' ? 'Revision manual' : 'Manual review')}</p>
      </div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="mx-auto w-full max-w-[1320px] space-y-8 px-6 pb-32 pt-24 lg:px-8"
    >
      <header className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-3 rounded-full border border-border/60 bg-muted/15 px-4 py-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary">
            <Sparkles size={14} />
            <span>{language === 'es' ? 'Entrenamiento continuo' : 'Continuous training'}</span>
          </div>
          <div className="space-y-3">
            <h2 className="max-w-3xl text-4xl font-extrabold tracking-[-0.05em] text-foreground lg:text-5xl">
              {language === 'es'
                ? 'Dos agentes internos mejoran Vortex de forma continua.'
                : 'Two internal agents keep improving Vortex continuously.'}
            </h2>
            <p className="max-w-2xl text-base leading-8 text-muted-foreground lg:text-lg">
              {language === 'es'
                ? 'Analista y Constructor vigilan sesiones, detectan gaps, lanzan aprendizaje, entrenamiento y autoedicion con snapshot y rollback visibles.'
                : 'Analyst and Builder watch sessions, detect gaps, launch learning, training, and self-editing with visible snapshots and rollback.'}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => { void runAction('autonomy-start', () => onStartAutonomy(), language === 'es' ? 'Autonomia reactivada.' : 'Autonomy resumed.'); }}
              className="rounded-full border border-primary/25 bg-primary/[0.10] px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-primary transition-all hover:bg-primary/[0.16]"
            >
              <span className="inline-flex items-center gap-2"><PlayCircle size={14} />{language === 'es' ? 'Reanudar' : 'Resume'}</span>
            </button>
            <button
              type="button"
              onClick={() => { void runAction('autonomy-stop', () => onStopAutonomy(), language === 'es' ? 'Autonomia pausada.' : 'Autonomy paused.'); }}
              className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"
            >
              <span className="inline-flex items-center gap-2"><PauseCircle size={14} />{language === 'es' ? 'Pausar todo' : 'Pause all'}</span>
            </button>
            <button
              type="button"
              onClick={() => { void runAction('quick', () => onStartTraining('quick'), language === 'es' ? 'Aprendizaje rapido lanzado.' : 'Quick learning launched.'); }}
              className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"
            >
              <span className="inline-flex items-center gap-2"><RefreshCw size={14} />{language === 'es' ? 'Quick learning' : 'Quick learning'}</span>
            </button>
            <button
              type="button"
              onClick={() => { void runAction('full', () => onStartTraining('full'), language === 'es' ? 'Entreno completo lanzado.' : 'Full training launched.'); }}
              className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"
            >
              <span className="inline-flex items-center gap-2"><Bot size={14} />{language === 'es' ? 'Full training' : 'Full training'}</span>
            </button>
          </div>
        </div>

        <Panel title={language === 'es' ? 'Resumen vivo' : 'Live summary'} eyebrow={language === 'es' ? 'Autonomia' : 'Autonomy'}>
          <div className="grid gap-3 sm:grid-cols-2">
            {metrics.map((card) => (
              <div key={card.label} className="rounded-[1.1rem] border border-border/50 bg-muted/15 px-4 py-4">
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.18em] text-primary">
                  {card.icon}
                  <span>{card.label}</span>
                </div>
                <p className="mt-3 text-lg font-black tracking-tight">{card.value}</p>
                <p className="mt-1 break-words text-xs text-muted-foreground">{card.caption}</p>
              </div>
            ))}
          </div>
        </Panel>
      </header>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Panel title={language === 'es' ? 'Controles de autonomia' : 'Autonomy controls'} eyebrow={language === 'es' ? 'Interruptores' : 'Switches'}>
          <div className="grid gap-3">
            {[
              {
                key: 'reflection',
                title: language === 'es' ? 'Solo reflexion' : 'Reflection only',
                description: language === 'es'
                  ? 'Mantiene preguntas internas y aprendizaje, pero sin lanzar training ni autoedicion.'
                  : 'Keeps internal reflection and learning active, but skips training and self-editing.',
                action: () => onConfigureAutonomy({ reflection_enabled: true, training_enabled: false, autoedit_enabled: false }),
                icon: <BrainCircuit size={16} />,
              },
              {
                key: 'training',
                title: language === 'es' ? 'Pausar solo training' : 'Pause training only',
                description: language === 'es'
                  ? 'La autonomia sigue analizando y proponiendo, pero no lanzara nuevos runs.'
                  : 'Autonomy keeps analyzing and proposing, but will not launch new runs.',
                action: () => onConfigureAutonomy({ training_enabled: false }),
                icon: <FlaskConical size={16} />,
              },
              {
                key: 'autoedit',
                title: language === 'es' ? 'Pausar solo autoedicion' : 'Pause self-editing only',
                description: language === 'es'
                  ? 'Mantiene reflexion y entrenamiento, pero bloquea cambios directos al repo.'
                  : 'Keeps reflection and training active, but blocks direct repo changes.',
                action: () => onConfigureAutonomy({ autoedit_enabled: false }),
                icon: <FileCode2 size={16} />,
              },
              {
                key: 'resume-all',
                title: language === 'es' ? 'Reactivar todo' : 'Resume everything',
                description: language === 'es'
                  ? 'Vuelve a poner reflexion, training y autoedicion en modo continuo.'
                  : 'Turns reflection, training, and self-editing back on.',
                action: () => onConfigureAutonomy({ reflection_enabled: true, training_enabled: true, autoedit_enabled: true, enabled: true }),
                icon: <Workflow size={16} />,
              },
            ].map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => { void runAction(item.key, item.action, item.title); }}
                className="rounded-[1.15rem] border border-border/60 bg-muted/15 px-4 py-4 text-left transition-all hover:border-primary/20 hover:bg-background"
              >
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary">
                  {item.icon}
                  <span>{item.title}</span>
                </div>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">{item.description}</p>
              </button>
            ))}
          </div>
          {busyAction && (
            <p className="mt-4 text-xs text-primary">
              {language === 'es' ? `Aplicando: ${busyAction}` : `Applying: ${busyAction}`}
            </p>
          )}
        </Panel>

        <Panel title={language === 'es' ? 'Timeline operativa' : 'Operational timeline'} eyebrow={language === 'es' ? 'Analista + Constructor' : 'Analyst + Builder'}>
          <div className="space-y-3">
            {timeline.length > 0 ? timeline.map((event) => (
              <div key={event.id} className="rounded-[1.15rem] border border-border/50 bg-muted/15 px-4 py-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.18em]">
                      <span className={event.agent === 'builder' ? 'text-primary' : 'text-foreground/80'}>
                        {event.agent === 'builder'
                          ? (language === 'es' ? 'Constructor' : 'Builder')
                          : event.agent === 'analyst'
                            ? (language === 'es' ? 'Analista' : 'Analyst')
                            : (language === 'es' ? 'Sistema' : 'System')}
                      </span>
                      <span className="text-muted-foreground">{event.kind}</span>
                    </div>
                    <p className="mt-2 text-sm font-black tracking-tight">{event.title}</p>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">{event.detail}</p>
                  </div>
                  <span className="text-xs text-muted-foreground">{formatTime(event.ts, language)}</span>
                </div>
              </div>
            )) : (
              <div className="rounded-[1.15rem] border border-dashed border-border/60 bg-muted/10 px-4 py-8 text-sm text-muted-foreground">
                {language === 'es'
                  ? 'Aun no hay eventos de autonomia. En cuanto el stack este listo, aqui veras el ciclo entre Analista y Constructor.'
                  : 'There are no autonomy events yet. As soon as the stack is ready, the Analyst/Builder loop will appear here.'}
              </div>
            )}
          </div>
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <Panel title={language === 'es' ? 'Runs y adapters' : 'Runs and adapters'} eyebrow={language === 'es' ? 'Training' : 'Training'}>
          <div className="space-y-3">
            {runs.length > 0 ? runs.slice(0, 8).map(renderRunCard) : (
              <div className="rounded-[1.15rem] border border-dashed border-border/60 bg-muted/10 px-4 py-8 text-sm text-muted-foreground">
                {language === 'es'
                  ? 'Todavia no hay runs. Lanza quick learning o full training desde aqui o deja que la autonomia los programe.'
                  : 'There are no runs yet. Launch quick learning or full training from here, or let autonomy schedule them.'}
              </div>
            )}
          </div>
        </Panel>

        <Panel title={language === 'es' ? 'Lo que aprende del chat' : 'What it learns from chat'} eyebrow={language === 'es' ? 'Promociones recientes' : 'Recent promotions'}>
          <div className="grid gap-3">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-[1.1rem] border border-border/50 bg-muted/15 px-4 py-4">
                <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Promovidas' : 'Promoted'}</p>
                <p className="mt-3 text-2xl font-black tracking-tight">{promotedResponses.length}</p>
              </div>
              <div className="rounded-[1.1rem] border border-border/50 bg-muted/15 px-4 py-4">
                <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Ultimo train' : 'Last train'}</p>
                <p className="mt-3 text-sm font-black tracking-tight">{formatTime(autonomy?.last_train_at, language)}</p>
              </div>
              <div className="rounded-[1.1rem] border border-border/50 bg-muted/15 px-4 py-4">
                <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Ultimo patch' : 'Last patch'}</p>
                <p className="mt-3 text-sm font-black tracking-tight">{formatTime(autonomy?.last_patch_at, language)}</p>
              </div>
            </div>

            <div className="space-y-3">
              {promotedResponses.length > 0 ? promotedResponses.map((item) => (
                <div key={item.id} className="rounded-[1.15rem] border border-border/50 bg-muted/15 px-4 py-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary">
                    <Clock3 size={14} />
                    <span>{formatTime(item.ts / 1000, language)}</span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-muted-foreground">{item.content}</p>
                </div>
              )) : (
                <div className="rounded-[1.15rem] border border-dashed border-border/60 bg-muted/10 px-4 py-8 text-sm text-muted-foreground">
                  {language === 'es'
                    ? 'Todavia no hay respuestas promovidas desde el chat para aprendizaje continuo.'
                    : 'There are no promoted chat answers for continuous learning yet.'}
                </div>
              )}
            </div>
          </div>
        </Panel>
      </div>
    </motion.div>
  );
};

export default TrainingView;

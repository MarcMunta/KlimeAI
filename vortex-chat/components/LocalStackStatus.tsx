import React from 'react';
import { Activity, AlertTriangle, Cpu, ShieldCheck } from 'lucide-react';
import { OperationalStatus, Language } from '../types';

interface LocalStackStatusProps {
  status: OperationalStatus | null;
  language: Language;
}

const resolveReason = (status: OperationalStatus | null, language: Language): string => {
  if (!status) return language === 'es' ? 'Verificando stack local…' : 'Checking local stack...';
  return (
    status.degraded_reason ||
    status.engine_reason ||
    status.model_reason ||
    status.docker_reason ||
    status.offline_reason ||
    status.training_reason ||
    (status.ok
      ? (language === 'es' ? 'Stack local listo.' : 'Local stack ready.')
      : (language === 'es' ? 'Stack local degradado.' : 'Local stack degraded.'))
  );
};

const LocalStackStatus: React.FC<LocalStackStatusProps> = ({ status, language }) => {
  const ready = Boolean(status?.ok && status?.engine_ready && status?.model_ready);
  const reason = resolveReason(status, language);
  const model = status?.active_model || (language === 'es' ? 'Modelo no detectado' : 'Model unavailable');
  const engine = status?.engine_kind || 'unknown';

  return (
    <div
      className={`mx-auto mb-4 max-w-[840px] rounded-3xl border px-5 py-4 shadow-xl backdrop-blur-2xl ${
        ready
          ? 'border-emerald-500/20 bg-emerald-500/8'
          : 'border-amber-500/20 bg-amber-500/8'
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.2em] text-foreground/70">
            {ready ? <ShieldCheck size={14} /> : <AlertTriangle size={14} />}
            <span>{language === 'es' ? 'Stack Local' : 'Local Stack'}</span>
          </div>
          <p className="text-sm font-semibold text-foreground">
            {ready
              ? (language === 'es' ? 'Vortex listo para responder en local.' : 'Vortex is ready to answer locally.')
              : (language === 'es' ? 'Chat bloqueado hasta que el runtime esté listo.' : 'Chat is blocked until the runtime is ready.')}
          </p>
          <p className="text-xs text-muted-foreground">{reason}</p>
        </div>
        <div className="grid grid-cols-1 gap-2 text-right text-xs text-muted-foreground sm:grid-cols-2">
          <div className="rounded-2xl border border-border/50 bg-background/60 px-3 py-2">
            <div className="flex items-center justify-end gap-2 font-semibold text-foreground">
              <Cpu size={14} />
              <span>{engine}</span>
            </div>
            <div>{model}</div>
          </div>
          <div className="rounded-2xl border border-border/50 bg-background/60 px-3 py-2">
            <div className="flex items-center justify-end gap-2 font-semibold text-foreground">
              <Activity size={14} />
              <span>{status?.engine_base_url || 'n/a'}</span>
            </div>
            <div>
              {language === 'es'
                ? `Docker ${status?.docker_ready ? 'listo' : 'pendiente'} · Web ${status?.web_disabled ? 'off' : 'on'}`
                : `Docker ${status?.docker_ready ? 'ready' : 'pending'} · Web ${status?.web_disabled ? 'off' : 'on'}`}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LocalStackStatus;

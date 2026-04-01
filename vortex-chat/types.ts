
export enum Role {
  USER = 'user',
  AI = 'ai'
}

export type ViewType = 'chat' | 'analysis' | 'edits' | 'terminal';
export type AppMode = 'ask' | 'agent';
export type FontSize = 'small' | 'medium' | 'large';
export type Language = 'es' | 'en';

export interface Source {
  title: string;
  url: string;
  domain: string;
  kind: 'web' | 'file' | 'unknown';
  index: number;
}

export interface GroundingSupport {
  segmentText: string;
  startIndex: number;
  endIndex: number;
  sourceIndices: number[];
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  thought?: string;
  requestId?: string;
  trainingEvent?: boolean;
  sources?: Source[];
  groundingSupports?: GroundingSupport[];
  timestamp: number;
  fileChanges?: { path: string; diff: string }[];
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

export interface UserSettings {
  categoryOrder: string[];
  codeTheme: 'dark' | 'light' | 'match-app';
  fontSize: FontSize;
  language: Language;
}

export interface LogEntry {
  id: string;
  timestamp: number;
  level: 'INFO' | 'LEARN' | 'SEARCH' | 'SYSTEM';
  message: string;
}

export interface OperationalStatus {
  ok: boolean;
  offline_ready: boolean;
  engine_ready: boolean;
  engine_kind?: string | null;
  engine_base_url?: string | null;
  model_ready: boolean;
  active_backend?: string | null;
  active_model?: string | null;
  training_ready: boolean;
  web_disabled: boolean;
  docker_ready?: boolean;
  degraded_reason?: string | null;
  offline_reason?: string | null;
  engine_reason?: string | null;
  model_reason?: string | null;
  training_reason?: string | null;
  docker_reason?: string | null;
  wsl_ready?: boolean;
  wsl_reason?: string | null;
  instructions?: {
    digest?: string | null;
    sources?: string[];
  };
}

export interface TrainingRunSummary {
  run_id: string;
  mode: 'quick' | 'full' | string;
  status: string;
  stage?: string;
  created_at?: number;
  updated_at?: number;
  profile?: string;
  base_model?: string;
  served_model?: string;
  dataset_hash?: string;
  adapter_dir?: string;
  log_path?: string;
  eval_log_path?: string;
  bench_log_path?: string;
  promotion?: {
    manual_only?: boolean;
    decision?: string;
    eval_ok?: boolean;
    bench_ok?: boolean;
  };
  train_result?: Record<string, unknown>;
  eval_result?: Record<string, unknown>;
  bench_result?: Record<string, unknown>;
}

export interface TrainingStreamPayload {
  ts: number;
  active_run_id?: string | null;
  runs?: TrainingRunSummary[];
}

export interface ControlStatus {
  ok: boolean;
  bootstrap?: {
    running?: boolean;
    stage?: string;
    message?: string;
    updated_at?: number;
    error?: unknown;
    tail?: string[];
  };
  docker?: {
    ready?: boolean;
    reason?: string;
    detail?: string;
    server_version?: string | null;
  };
  model?: {
    model_id?: string;
    cache_dir?: string;
    repo_dir?: string;
    cached?: boolean;
    snapshot_count?: number;
    last_snapshot?: string | null;
  };
  runtime?: {
    api_ready?: boolean;
    runtime_ready?: boolean;
    readyz?: { ok?: boolean };
    status?: OperationalStatus | null;
  };
  frontend?: {
    ready?: boolean;
    port?: number;
    url?: string;
  };
  internet?: {
    allowlist?: string[];
  };
  instructions?: {
    digest?: string | null;
    sources?: string[];
  };
  active_run_id?: string | null;
  runs?: TrainingRunSummary[];
}

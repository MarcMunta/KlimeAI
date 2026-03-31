
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
